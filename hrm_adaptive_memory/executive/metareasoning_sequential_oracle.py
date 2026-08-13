"""Sequential information-state oracle for V2B-I3.2.

Unlike I3.1's opening-observation diagnostic, this module plans over the full
observable history.  A state is a posterior-weighted set of concrete latent
states; policy feedback and every resulting public observation split that
belief exactly as the deterministic runtime does.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import json
import math
import resource
from time import perf_counter
from typing import Iterable, Mapping

from hrm_adaptive_memory.cognitive_control.actions import V2B_ACTIONS
from hrm_adaptive_memory.cognitive_control.core import DecisionAction, PolicyEffect
from hrm_adaptive_memory.cognitive_control.state import DecisionSummary

from .metareasoning_controller import (
    ControllerObservation, ObservationMask, PolicyFeedback, apply_observation_mask)
from .metareasoning_executor import (
    DeterministicMetareasoningExecutor, I3Runtime, build_observable_snapshot, policy_facts,
    runtime_state_hash)
from .metareasoning_state import canonicalize_runtime_state, runtime_from_oracle_state
from .metareasoning_transition_table import OraclePolicyTable
from .metareasoning_utility import MetareasoningUtility, frozen_action_cost_hash
from .policy import FrozenPolicy
from .resources import ResourceExhausted


SEQUENTIAL_ORACLE_SCHEMA = "DAPH_V2B_I3_2_SEQUENTIAL_INFORMATION_TABLE_V1"
SEQUENTIAL_ORACLE_REVISION = "v2b-i3.2.2-sequential-information-oracle-v1"
PRIOR_DEFINITION = "UNIFORM_BY_INITIAL_OBSERVATION_CLASS_V1"
POLICY_FEEDBACK_VISIBILITY = {
    "effect": True,
    "required_action": True,
    "reason_class": "COARSE_EFFECT_ONLY_V1",
    "detailed_reason_codes": False,
    "rejection_consumes_control_step": True,
}
DEFAULT_MAX_INFORMATION_STATES = 40_000
DEFAULT_MAX_INFORMATION_TRANSITIONS = 240_000
DEFAULT_MAX_MEMBERS_PER_BELIEF = 256


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def policy_feedback_visibility_hash() -> str:
    return _hash(POLICY_FEEDBACK_VISIBILITY)


def _reason_class(effect: PolicyEffect) -> str:
    return {
        PolicyEffect.ALLOW: "POLICY_ALLOWED",
        PolicyEffect.REQUIRE: "POLICY_REQUIRED",
        PolicyEffect.DENY: "POLICY_DENIED",
    }[effect]


@dataclass(frozen=True)
class InformationHistoryEvent:
    proposed_action: DecisionAction
    policy_effect: PolicyEffect
    resolved_action: DecisionAction | None
    execution_status: str

    def as_dict(self) -> dict[str, str | None]:
        return {"proposed_action": self.proposed_action.value,
                "policy_effect": self.policy_effect.value,
                "resolved_action": None if self.resolved_action is None else self.resolved_action.value,
                "execution_status": self.execution_status}


@dataclass(frozen=True)
class LatentMember:
    task_id: str
    table_identity_sha256: str
    state_id: str
    posterior_weight: Fraction

    @property
    def key(self) -> str:
        return f"{self.task_id}:{self.table_identity_sha256}:{self.state_id}"

    def as_dict(self) -> dict[str, str]:
        return {"task_id": self.task_id, "table_identity_sha256": self.table_identity_sha256,
                "state_id": self.state_id, "posterior_weight": str(self.posterior_weight)}


@dataclass(frozen=True)
class InformationState:
    observation_mask_id: str
    history: tuple[InformationHistoryEvent, ...]
    history_hash: str
    members: tuple[LatentMember, ...]
    posterior_weights: tuple[str, ...]
    resource_state: Mapping[str, int]
    observation_hash: str

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("information state needs at least one latent member")
        if sum(member.posterior_weight for member in self.members) != Fraction(1, 1):
            raise ValueError("information-state posterior weights must sum to one")
        if len(self.members) != len(self.posterior_weights):
            raise ValueError("information-state weights must align with members")

    def state_id(self) -> str:
        return _hash(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        return {
            "observation_mask_id": self.observation_mask_id,
            "history_hash": self.history_hash,
            "history": [item.as_dict() for item in self.history],
            "members": [item.as_dict() for item in self.members],
            "posterior_weights": list(self.posterior_weights),
            "resource_state": dict(sorted(self.resource_state.items())),
            "observation_hash": self.observation_hash,
        }

    @property
    def entropy_bits(self) -> float:
        return -sum(float(member.posterior_weight) * math.log2(float(member.posterior_weight))
                    for member in self.members if member.posterior_weight)


@dataclass(frozen=True)
class MemberTransition:
    member_key: str
    next_member: LatentMember | None
    next_runtime_state_hash: str | None
    terminal: bool
    terminal_utility: float | None
    action_cost: float
    immediate_reward: float
    feedback: PolicyFeedback
    history_event: InformationHistoryEvent
    observation_hash: str | None


@dataclass(frozen=True)
class InformationOutcome:
    outcome_id: str
    probability: Fraction
    next_information_state_id: str | None
    terminal: bool
    expected_utility: float | None
    expected_action_cost: float
    expected_immediate_reward: float
    member_keys: tuple[str, ...]
    entropy_after_bits: float | None


@dataclass(frozen=True)
class InformationTransition:
    proposed_action: DecisionAction
    outcomes: tuple[InformationOutcome, ...]
    expected_information_gain_bits: float


@dataclass(frozen=True)
class SequentialObservablePolicyTable:
    observation_mask_hash: str
    initial_information_state_id: str
    identity_sha256: str
    information_states: Mapping[str, InformationState]
    transitions: Mapping[tuple[str, DecisionAction], InformationTransition]
    member_transitions: Mapping[tuple[str, DecisionAction, str], MemberTransition]
    belief_values: Mapping[str, float]
    q_values: Mapping[tuple[str, DecisionAction], float]
    optimal_actions: Mapping[str, tuple[DecisionAction, ...]]
    expected_latent_values: Mapping[str, float]
    build_metrics: Mapping[str, float | int]

    def information_gap(self, state_id: str) -> float:
        return self.expected_latent_values[state_id] - self.belief_values[state_id]

    def action_regret(self, state_id: str, proposed: DecisionAction) -> float:
        value = self.q_values.get((state_id, proposed))
        if value is None:
            return float("inf")
        return max(0.0, self.belief_values[state_id] - value)

    def serializable(self) -> dict[str, object]:
        return {
            "schema": SEQUENTIAL_ORACLE_SCHEMA,
            "observation_mask_hash": self.observation_mask_hash,
            "initial_information_state_id": self.initial_information_state_id,
            "identity_sha256": self.identity_sha256,
            "information_state_count": len(self.information_states),
            "information_transition_count": len(self.transitions),
            "belief_values": dict(sorted(self.belief_values.items())),
            "q_values": {f"{state}:{action.value}": value for (state, action), value in sorted(
                self.q_values.items(), key=lambda item: (item[0][0], item[0][1].value))},
            "expected_latent_values": dict(sorted(self.expected_latent_values.items())),
            "optimal_actions": {state: [action.value for action in actions]
                                for state, actions in sorted(self.optimal_actions.items())},
            "information_states": {state: item.as_dict()
                                   for state, item in sorted(self.information_states.items())},
            "transitions": {
                f"{state}:{action.value}": {
                    "expected_information_gain_bits": transition.expected_information_gain_bits,
                    "outcomes": [{"outcome_id": outcome.outcome_id,
                                  "probability": str(outcome.probability),
                                  "next_information_state_id": outcome.next_information_state_id,
                                  "terminal": outcome.terminal,
                                  "expected_utility": outcome.expected_utility,
                                  "expected_action_cost": outcome.expected_action_cost,
                                  "expected_immediate_reward": outcome.expected_immediate_reward,
                                  "member_keys": list(outcome.member_keys),
                                  "entropy_after_bits": outcome.entropy_after_bits}
                                 for outcome in transition.outcomes],
                } for (state, action), transition in sorted(
                    self.transitions.items(), key=lambda item: (item[0][0], item[0][1].value))
            },
            "build_metrics": dict(self.build_metrics),
        }

    @property
    def table_sha256(self) -> str:
        material = self.serializable()
        material.pop("build_metrics", None)
        return _hash(material)


@dataclass(frozen=True)
class SequentialObservableOracleSet:
    observation_mask_hash: str
    tables: Mapping[str, SequentialObservablePolicyTable]
    member_to_initial_table: Mapping[str, str]

    def table_for_member(self, member: LatentMember) -> SequentialObservablePolicyTable:
        return self.tables[self.member_to_initial_table[member.key]]

    @property
    def table_sha256(self) -> str:
        return _hash({"observation_mask_hash": self.observation_mask_hash,
                      "tables": {key: table.table_sha256 for key, table in sorted(self.tables.items())}})


@dataclass(frozen=True)
class _Context:
    initial_runtime: I3Runtime
    latent_table: OraclePolicyTable


@dataclass(frozen=True)
class _RuntimeOutcome:
    runtime: I3Runtime | None
    terminal: bool
    terminal_utility: float | None
    task_success: bool | None
    action_cost: float
    immediate_reward: float
    feedback: PolicyFeedback
    history_event: InformationHistoryEvent


def _history_hash(history: tuple[InformationHistoryEvent, ...]) -> str:
    return _hash([item.as_dict() for item in history])


def _append_history(history: tuple[InformationHistoryEvent, ...],
                    event: InformationHistoryEvent) -> tuple[InformationHistoryEvent, ...]:
    """Append one public event to the canonical observable history.

    I3.2's information state is conditioned on the complete public sequence
    ``O0, A0, O1, ...``.  Repeated proposals and policy feedback are therefore
    intentionally retained rather than overwritten by action name.  Resource
    counters and the bounded horizon keep this history finite, while retaining
    it prevents policy-probe counts and repeated failed actions from becoming
    hidden state that the sequential oracle cannot represent.
    """
    return history + (event,)


def _history_view(history: tuple[InformationHistoryEvent, ...]) -> tuple[tuple[DecisionAction, ...],
                                                                           tuple[DecisionAction, ...],
                                                                           tuple[PolicyFeedback, ...],
                                                                           tuple[DecisionSummary, ...],
                                                                           tuple[str, ...]]:
    executed = tuple(item.resolved_action for item in history
                     if item.execution_status == "EXECUTED" and item.resolved_action is not None)
    rejected = tuple(item.proposed_action for item in history
                     if item.execution_status in {"POLICY_REJECTED", "RESOURCE_REJECTED"})
    feedback = tuple(PolicyFeedback(item.policy_effect.value, item.resolved_action,
                                    _reason_class(item.policy_effect)) for item in history)
    decisions = tuple(DecisionSummary(f"step-{index}", item.resolved_action.value
                                       if item.resolved_action else item.proposed_action.value,
                                       _reason_class(item.policy_effect), item.execution_status)
                      for index, item in enumerate(history))
    outcomes = tuple(item.execution_status for item in history)
    return executed, rejected, feedback, decisions, outcomes


def controller_observation(*, runtime: I3Runtime, history: tuple[InformationHistoryEvent, ...],
                           mask: ObservationMask) -> ControllerObservation:
    """Single public packet construction shared by oracle and I3.2 runtime."""
    executed, rejected, feedback, decisions, outcomes = _history_view(history)
    snapshot = apply_observation_mask(build_observable_snapshot(
        runtime, prior_decisions=decisions, prior_outcomes=outcomes), mask)
    return ControllerObservation(
        task_id=runtime.task.controller_instance_id or "opaque-instance",
        task_summary=runtime.task.task_summary,
        resource_state=runtime.resources.as_dict(),
        allowed_actions=tuple(action for action in V2B_ACTIONS if runtime.resources.can_execute(action)),
        executed_actions=executed, rejected_actions=rejected, cognitive_state=snapshot,
        policy_feedback=feedback)


def canonical_packet(observation: ControllerObservation) -> dict[str, object]:
    snapshot = observation.cognitive_state
    return {
        "instance_id": observation.task_id, "task_summary": observation.task_summary,
        "resource_state": dict(sorted(observation.resource_state.items())),
        "allowed_actions": [action.value for action in observation.allowed_actions],
        "executed_actions": [action.value for action in observation.executed_actions],
        "rejected_actions": [action.value for action in observation.rejected_actions],
        "policy_feedback": [{"effect": item.effect,
                             "resolved_action": None if item.resolved_action is None else item.resolved_action.value,
                             "reason_class": item.reason_class} for item in observation.policy_feedback],
        "cognitive_state": None if snapshot is None else {
            "verification_states": [item.state.value for item in snapshot.verification_states],
            "provenance_summaries": list(snapshot.provenance_summaries),
            "temporal_status": snapshot.temporal_status.value,
            "conflicts": [{"id": item.conflict_id, "status": item.status}
                          for item in snapshot.unresolved_conflicts],
            "prior_decisions": [{"action": item.selected_action,
                                 "outcome": item.outcome} for item in snapshot.prior_decisions],
            "prior_outcomes": list(snapshot.prior_outcomes),
            "observation_signals": list(snapshot.observation_signals),
        },
    }


def observation_hash(observation: ControllerObservation) -> str:
    return _hash(canonical_packet(observation))


def _apply_proposal(*, runtime: I3Runtime, proposed: DecisionAction, policy: FrozenPolicy,
                    utility: MetareasoningUtility) -> _RuntimeOutcome:
    """Exact runtime semantics, including visible policy/resource feedback."""
    decision = policy.gate.evaluate(runtime.task.task_id, proposed, policy_facts(runtime))
    resolved = decision.required_action if decision.effect is PolicyEffect.REQUIRE else proposed
    feedback = PolicyFeedback(decision.effect.value,
                              None if decision.effect is PolicyEffect.DENY else resolved,
                              _reason_class(decision.effect))
    if decision.effect is PolicyEffect.DENY:
        event = InformationHistoryEvent(proposed, decision.effect, None, "POLICY_REJECTED")
        try:
            next_runtime = replace(runtime, resources=runtime.resources.consume_policy_rejection())
        except ResourceExhausted:
            return _RuntimeOutcome(None, True, utility.incorrect_defer, False, 0.0, 0.0, feedback,
                                   InformationHistoryEvent(proposed, decision.effect, None,
                                                           "RESOURCE_EXHAUSTED"))
        cost = utility.action_cost(runtime.resources, next_runtime.resources)
        reward = utility.immediate_reward(before=runtime.resources, after=next_runtime.resources)
        return _RuntimeOutcome(next_runtime, False, None, None, cost, reward, feedback, event)
    assert resolved is not None
    if not runtime.resources.can_execute(resolved):
        event = InformationHistoryEvent(proposed, decision.effect, resolved, "RESOURCE_REJECTED")
        try:
            next_runtime = replace(runtime, resources=runtime.resources.consume_policy_rejection())
        except ResourceExhausted:
            return _RuntimeOutcome(None, True, utility.incorrect_defer, False, 0.0, 0.0, feedback,
                                   InformationHistoryEvent(proposed, decision.effect, resolved,
                                                           "RESOURCE_EXHAUSTED"))
        cost = utility.action_cost(runtime.resources, next_runtime.resources)
        reward = utility.immediate_reward(before=runtime.resources, after=next_runtime.resources)
        return _RuntimeOutcome(next_runtime, False, None, None, cost, reward, feedback, event)
    execution = DeterministicMetareasoningExecutor().execute(runtime, resolved)
    cost = utility.action_cost(runtime.resources, execution.runtime.resources)
    reward = utility.immediate_reward(before=runtime.resources, after=execution.runtime.resources)
    event = InformationHistoryEvent(proposed, decision.effect, resolved, "EXECUTED")
    if execution.terminal:
        assert execution.task_success is not None
        return _RuntimeOutcome(execution.runtime, True,
                               utility.terminal_reward(resolved, execution.task_success), execution.task_success,
                               cost, reward,
                               feedback, event)
    return _RuntimeOutcome(execution.runtime, False, None, None, cost, reward, feedback, event)


def _epistemic_signature(runtime: I3Runtime) -> tuple[object, ...]:
    """Fields whose change can create decision-relevant public information."""
    return (runtime.verification_state, runtime.temporal_status, runtime.unresolved_conflict,
            runtime.conflict_resolvable, runtime.composition_complete,
            runtime.provenance_count, runtime.prior_outcomes)


def _member_runtime(member: LatentMember, contexts: Mapping[str, _Context]) -> I3Runtime:
    context = contexts[member.task_id]
    return runtime_from_oracle_state(context.initial_runtime, context.latent_table.states[member.state_id])


def _make_information_state(*, members: Iterable[LatentMember], history: tuple[InformationHistoryEvent, ...],
                            mask: ObservationMask, contexts: Mapping[str, _Context],
                            max_members_per_belief: int) -> InformationState:
    ordered = tuple(sorted(members, key=lambda item: item.key))
    if len(ordered) > max_members_per_belief:
        raise RuntimeError("INFORMATION_STATE_MEMBER_LIMIT")
    runtime = _member_runtime(ordered[0], contexts)
    packet = controller_observation(runtime=runtime, history=history, mask=mask)
    packet_hash = observation_hash(packet)
    # A group is valid only if the exact public packet is equal for each member.
    for member in ordered[1:]:
        candidate = controller_observation(runtime=_member_runtime(member, contexts), history=history, mask=mask)
        if observation_hash(candidate) != packet_hash:
            raise RuntimeError("information state mixes non-equivalent observations")
    return InformationState(
        observation_mask_id=mask.sha256(), history=history, history_hash=_history_hash(history),
        members=ordered, posterior_weights=tuple(str(item.posterior_weight) for item in ordered),
        resource_state=packet.resource_state, observation_hash=packet_hash)


def _initial_groups(contexts: Mapping[str, _Context], mask: ObservationMask) -> tuple[tuple[LatentMember, ...], ...]:
    groups: dict[str, list[LatentMember]] = {}
    for task_id, context in sorted(contexts.items()):
        state_id = context.latent_table.initial_state_id
        provisional = LatentMember(task_id, context.latent_table.identity_sha256, state_id, Fraction(1, 1))
        packet = controller_observation(runtime=context.initial_runtime, history=(), mask=mask)
        groups.setdefault(observation_hash(packet), []).append(provisional)
    output = []
    for members in groups.values():
        weight = Fraction(1, len(members))
        output.append(tuple(replace(member, posterior_weight=weight) for member in members))
    return tuple(output)


def _build_table(*, initial_members: tuple[LatentMember, ...], contexts: Mapping[str, _Context],
                 mask: ObservationMask, policy: FrozenPolicy, utility: MetareasoningUtility,
                 max_information_states: int, max_information_transitions: int,
                 max_members_per_belief: int, benchmark_hash: str) -> SequentialObservablePolicyTable:
    started = perf_counter(); rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # A belief table can only reach members present in its initial class.
    # Binding every unrelated task table here added quadratic hashing work
    # without adding a semantic dependency to this policy table.
    relevant_task_ids = tuple(sorted({member.task_id for member in initial_members}))
    initial = _make_information_state(
        members=initial_members, history=(), mask=mask, contexts=contexts,
        max_members_per_belief=max_members_per_belief)
    initial_id = initial.state_id()
    states: dict[str, InformationState] = {initial_id: initial}
    transitions: dict[tuple[str, DecisionAction], InformationTransition] = {}
    transitions_by_state: dict[str, list[tuple[DecisionAction, InformationTransition]]] = {}
    member_transitions: dict[tuple[str, DecisionAction, str], MemberTransition] = {}
    # Runtime evolution depends on latent member state and proposal, not on
    # the public history used to group beliefs. Reusing this exact result
    # avoids re-running policy/Datalog and execution for the same Markov
    # transition across multiple information states.
    proposal_outcome_cache: dict[tuple[str, DecisionAction], _RuntimeOutcome] = {}
    queue: deque[str] = deque([initial_id])

    while queue:
        information_id = queue.popleft(); state = states[information_id]
        public_observation = controller_observation(
            runtime=_member_runtime(state.members[0], contexts), history=state.history, mask=mask)
        for proposed in public_observation.allowed_actions:
            groups: dict[tuple[str, str], list[tuple[LatentMember, _RuntimeOutcome]]] = {}
            terminal_groups: dict[tuple[str, str | None, str],
                                  list[tuple[LatentMember, _RuntimeOutcome]]] = {}
            for member in state.members:
                cache_key = (member.key, proposed)
                outcome = proposal_outcome_cache.get(cache_key)
                if outcome is None:
                    outcome = _apply_proposal(
                        runtime=_member_runtime(member, contexts), proposed=proposed,
                        policy=policy, utility=utility)
                    proposal_outcome_cache[cache_key] = outcome
                if outcome.terminal:
                    key = (outcome.feedback.effect,
                           None if outcome.feedback.resolved_action is None
                           else outcome.feedback.resolved_action.value,
                           outcome.history_event.execution_status)
                    terminal_groups.setdefault(key, []).append((member, outcome))
                    continue
                assert outcome.runtime is not None
                history = _append_history(state.history, outcome.history_event)
                observation = controller_observation(runtime=outcome.runtime, history=history, mask=mask)
                groups.setdefault((observation_hash(observation), _history_hash(history)), []).append((member, outcome))
            # A deterministic nonterminal action that changes no public
            # epistemic field in any compatible member and yields identical
            # feedback is strictly dominated: it only spends a resource and
            # cannot improve any later belief or utility. Pruning it is exact
            # for I3.2's history-independent transition/policy semantics and
            # prevents permutations of redundant calls from exploding the
            # information-state graph.
            nonterminal_pairs = [pair for pairs in groups.values() for pair in pairs]
            if nonterminal_pairs and not terminal_groups:
                signatures_unchanged = all(
                    _epistemic_signature(_member_runtime(member, contexts))
                    == _epistemic_signature(outcome.runtime)  # type: ignore[arg-type]
                    for member, outcome in nonterminal_pairs)
                feedbacks = {(outcome.feedback.effect, outcome.feedback.resolved_action,
                              outcome.history_event.execution_status)
                             for _, outcome in nonterminal_pairs}
                already_attempted = any(
                    (item.execution_status in {"POLICY_REJECTED", "RESOURCE_REJECTED"}
                     and item.proposed_action is proposed)
                    or (item.execution_status == "EXECUTED" and item.resolved_action is proposed)
                    for item in state.history)
                if signatures_unchanged and len(feedbacks) == 1 and already_attempted:
                    continue
            outcomes: list[InformationOutcome] = []
            for (packet_hash, _), members_outcomes in sorted(groups.items()):
                total = sum(member.posterior_weight for member, _ in members_outcomes)
                successor_members = []
                history = _append_history(state.history, members_outcomes[0][1].history_event)
                for member, outcome in members_outcomes:
                    assert outcome.runtime is not None
                    context = contexts[member.task_id]
                    next_state_id = canonicalize_runtime_state(outcome.runtime).state_id()
                    successor_members.append(LatentMember(
                        member.task_id, context.latent_table.identity_sha256, next_state_id,
                        member.posterior_weight / total))
                successor = _make_information_state(members=successor_members, history=history,
                                                    mask=mask, contexts=contexts,
                                                    max_members_per_belief=max_members_per_belief)
                successor_id = successor.state_id()
                if successor_id not in states:
                    if len(states) >= max_information_states:
                        raise RuntimeError("INFORMATION_STATE_SPACE_LIMIT")
                    if successor.resource_state["executive_steps_remaining"] >= state.resource_state["executive_steps_remaining"]:
                        raise RuntimeError("INFORMATION_STATE_ZERO_COST_CYCLE")
                    states[successor_id] = successor; queue.append(successor_id)
                for member, outcome in members_outcomes:
                    assert outcome.runtime is not None
                    context = contexts[member.task_id]
                    next_member = LatentMember(member.task_id, context.latent_table.identity_sha256,
                                                canonicalize_runtime_state(outcome.runtime).state_id(), Fraction(1, 1))
                    member_transitions[(information_id, proposed, member.key)] = MemberTransition(
                        member.key, next_member, runtime_state_hash(outcome.runtime), False, None,
                        outcome.action_cost, outcome.immediate_reward,
                        outcome.feedback, outcome.history_event, packet_hash)
                outcomes.append(InformationOutcome(
                    outcome_id=_hash({"packet": packet_hash, "history": successor.history_hash}),
                    probability=total, next_information_state_id=successor_id, terminal=False,
                    expected_utility=None,
                    expected_action_cost=sum(float(member.posterior_weight) * outcome.action_cost
                                             for member, outcome in members_outcomes) / float(total),
                    expected_immediate_reward=sum(
                        float(member.posterior_weight) * outcome.immediate_reward
                        for member, outcome in members_outcomes) / float(total),
                    member_keys=tuple(sorted(member.key for member, _ in members_outcomes)),
                    entropy_after_bits=successor.entropy_bits))
            for terminal_key, members_outcomes in sorted(
                    terminal_groups.items(), key=lambda item: json.dumps(item[0])):
                probability = sum(member.posterior_weight for member, _ in members_outcomes)
                expected_terminal = sum(
                    float(member.posterior_weight) * float(outcome.terminal_utility or 0.0)
                    for member, outcome in members_outcomes) / float(probability)
                expected_cost = sum(
                    float(member.posterior_weight) * outcome.action_cost
                    for member, outcome in members_outcomes) / float(probability)
                expected_reward = sum(
                    float(member.posterior_weight) * outcome.immediate_reward
                    for member, outcome in members_outcomes) / float(probability)
                posterior = tuple(member.posterior_weight / probability
                                  for member, _ in members_outcomes)
                terminal_entropy = -sum(
                    float(weight) * math.log2(float(weight)) for weight in posterior if weight)
                for member, outcome in members_outcomes:
                    member_transitions[(information_id, proposed, member.key)] = MemberTransition(
                        member.key, None,
                        None if outcome.runtime is None else runtime_state_hash(outcome.runtime),
                        True, outcome.terminal_utility, outcome.action_cost, outcome.immediate_reward,
                        outcome.feedback, outcome.history_event, None)
                outcomes.append(InformationOutcome(
                    outcome_id=_hash({"terminal_public_feedback": terminal_key}), probability=probability,
                    next_information_state_id=None, terminal=True,
                    expected_utility=expected_terminal,
                    expected_action_cost=expected_cost,
                    expected_immediate_reward=expected_reward,
                    member_keys=tuple(sorted(member.key for member, _ in members_outcomes)),
                    entropy_after_bits=terminal_entropy))
            if outcomes:
                transitions[(information_id, proposed)] = InformationTransition(
                    proposed, tuple(sorted(outcomes, key=lambda item: item.outcome_id)),
                    state.entropy_bits - sum(float(item.probability) * (item.entropy_after_bits or 0.0)
                                               for item in outcomes))
                transitions_by_state.setdefault(information_id, []).append(
                    (proposed, transitions[(information_id, proposed)]))
                if len(transitions) > max_information_transitions:
                    raise RuntimeError("INFORMATION_TRANSITION_LIMIT")

    values: dict[str, float] = {}; q_values: dict[tuple[str, DecisionAction], float] = {}
    optimal: dict[str, tuple[DecisionAction, ...]] = {}; latent_values: dict[str, float] = {}
    for information_id in sorted(states, key=lambda item: (states[item].resource_state["executive_steps_remaining"], item)):
        state = states[information_id]
        latent_values[information_id] = sum(
            float(member.posterior_weight) * contexts[member.task_id].latent_table.state_values[member.state_id]
            for member in state.members)
        candidates = {}
        for action, transition in transitions_by_state.get(information_id, ()):
            candidates[action] = sum(float(outcome.probability) * (
                outcome.expected_immediate_reward - outcome.expected_action_cost
                + (outcome.expected_utility if outcome.terminal
                else values[outcome.next_information_state_id])) for outcome in transition.outcomes)
            q_values[(information_id, action)] = candidates[action]
        if candidates:
            value = max(candidates.values()); values[information_id] = value
            optimal[information_id] = tuple(sorted((action for action, candidate in candidates.items()
                                                    if abs(candidate - value) <= 1e-12),
                                                   key=lambda action: action.value))
        else:
            values[information_id] = utility.incorrect_defer; optimal[information_id] = ()

    root_transitions = [transition for (origin, _), transition in transitions.items()
                        if origin == initial_id]
    resolving_first_actions = sum(
        bool(transition.outcomes)
        and all(not outcome.terminal
                and outcome.next_information_state_id is not None
                and len(states[outcome.next_information_state_id].members) == 1
                for outcome in transition.outcomes)
        for transition in root_transitions)
    one_step_full_resolution_rate = (
        resolving_first_actions / len(root_transitions)
        if len(initial.members) > 1 and root_transitions else 0.0)

    identity = _hash({
        "benchmark_hash": benchmark_hash, "mask_sha256": mask.sha256(), "policy_sha256": policy.sha256,
        "utility_sha256": utility.sha256, "action_cost_sha256": frozen_action_cost_hash(),
        "prior_definition": PRIOR_DEFINITION, "policy_feedback_visibility_sha256": policy_feedback_visibility_hash(),
        "max_members_per_belief": max_members_per_belief,
        "latent_oracles": [contexts[task_id].latent_table.table_sha256
                            for task_id in relevant_task_ids],
        "revision": SEQUENTIAL_ORACLE_REVISION,
    })
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    table = SequentialObservablePolicyTable(
        observation_mask_hash=mask.sha256(), initial_information_state_id=initial_id,
        identity_sha256=identity, information_states=states, transitions=transitions,
        member_transitions=member_transitions, belief_values=values, q_values=q_values,
        optimal_actions=optimal, expected_latent_values=latent_values,
        build_metrics={"information_state_count": len(states),
                       "information_transition_count": len(transitions),
                       "sequential_oracle_build_seconds": perf_counter() - started,
                       "belief_peak_resident_memory_delta_kib": max(0, rss_after - rss_before),
                       "max_belief_cardinality": max(len(item.members) for item in states.values()),
                       "ambiguous_information_state_fraction": sum(len(item.members) > 1 for item in states.values()) / len(states),
                       # Fraction of root actions whose every outcome is
                       # nonterminal and a singleton posterior belief.
                       "one_step_full_resolution_rate": one_step_full_resolution_rate})
    return table


def build_sequential_observable_oracle(*, runtime_tables: Iterable[tuple[I3Runtime, OraclePolicyTable]],
                                       mask: ObservationMask, policy: FrozenPolicy,
                                       utility: MetareasoningUtility, benchmark_hash: str,
                                       max_information_states: int = DEFAULT_MAX_INFORMATION_STATES,
                                       max_information_transitions: int = DEFAULT_MAX_INFORMATION_TRANSITIONS,
                                       max_members_per_belief: int = DEFAULT_MAX_MEMBERS_PER_BELIEF) -> SequentialObservableOracleSet:
    """Construct exact full-trajectory observable policy tables for one mask."""
    contexts = {runtime.task.task_id: _Context(runtime, table) for runtime, table in runtime_tables}
    tables: dict[str, SequentialObservablePolicyTable] = {}
    member_to_table: dict[str, str] = {}
    for members in _initial_groups(contexts, mask):
        table = _build_table(initial_members=members, contexts=contexts, mask=mask, policy=policy,
                             utility=utility, max_information_states=max_information_states,
                             max_information_transitions=max_information_transitions,
                             max_members_per_belief=max_members_per_belief,
                             benchmark_hash=benchmark_hash)
        tables[table.initial_information_state_id] = table
        for member in members:
            member_to_table[member.key] = table.initial_information_state_id
    return SequentialObservableOracleSet(mask.sha256(), tables, member_to_table)
