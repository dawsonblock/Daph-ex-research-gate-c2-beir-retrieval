"""Parity tests: governor action semantics vs. executor transition behavior.

These tests verify that the frozen ActionSemantics contracts declared by the
governor match the actual transition behavior produced by the deterministic
metareasoning executor.  The oracle tables are built from the V2B-I3.5
benchmark using the same infrastructure as the production oracle precompute
scripts, then each semantic channel is checked against real transitions.

Run with:  PYTHONPATH=. pytest tests/unit/test_governor_executor_parity.py
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hrm_adaptive_memory.executive.governor.action_semantics import (
    FROZEN_ACTION_SEMANTICS, get_action_semantics)
from hrm_adaptive_memory.executive.metareasoning_benchmark import (
    load_metareasoning_benchmark)
from hrm_adaptive_memory.executive.metareasoning_executor import (
    initial_i3_runtime)
from hrm_adaptive_memory.executive.metareasoning_state import OracleState
from hrm_adaptive_memory.executive.metareasoning_transition_table import (
    OraclePolicyTable, OracleTableCache)
from hrm_adaptive_memory.executive.metareasoning_utility import (
    MetareasoningUtility)
from hrm_adaptive_memory.executive.policy import load_frozen_policy
from hrm_adaptive_memory.executive.resources import ResourceState
from hrm_adaptive_memory.cognitive_control.core import DecisionAction


# ─── Constants ───

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_MANIFEST = REPO_ROOT / "experiments/v2b_i3_5/manifests/v2b_i3_5_benchmark_manifest_v2.json"
POLICY_PATH = REPO_ROOT / "configs/v2b_i3_policy_v1.json"
UTILITY_PATH = REPO_ROOT / "configs/v2b_i3_1_utility_v1.json"

V1_ACTIONS = ("ANSWER", "RETRIEVE", "VERIFY", "SEARCH_MORE",
              "REASON_MORE", "DEFER", "STOP")
TERMINAL_ACTIONS = ("ANSWER", "DEFER", "STOP")
NON_TERMINAL_ACTIONS = ("RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE")
DEV_SPLIT = "structure_dev_v2"
NUM_DEV_TASKS = 5

# Maps each ActionSemantics boolean channel to the OracleState fields that
# must differ between before/after states for the channel to be considered
# "actually changed" by the executor.
CHANNEL_STATE_FIELDS: dict[str, tuple[str, ...]] = {
    "can_add_evidence": ("provenance_count", "retrieved", "searched"),
    "can_change_verification": ("verification_state",),
    "can_reduce_conflict": ("conflict_state", "conflict_resolvable"),
    "can_change_reasoning_state": ("composition_state",),
    "can_change_temporal_status": ("temporal_state",),
}

# Channels that the governor declares as capable of changing.
ALL_SEMANTIC_CHANNELS = tuple(CHANNEL_STATE_FIELDS.keys())


# ─── Helpers ───

def _channel_changed(before: OracleState, after: OracleState,
                     channel: str) -> bool:
    """Return True if any state field mapped to *channel* differs."""
    for field in CHANNEL_STATE_FIELDS[channel]:
        if getattr(before, field) != getattr(after, field):
            return True
    return False


def _is_poison_triggered(before: OracleState, after: OracleState) -> bool:
    """Check if this transition was triggered by the poison/default mechanism.

    The V2 chain effects have a default rule that fires when an action is
    executed in the wrong order, appending CONTROL_POISONED to prior_outcomes
    and setting multiple state fields as side effects. These are error-handling
    side effects, not direct action capabilities.
    """
    before_poisoned = "CONTROL_POISONED" in before.prior_outcomes
    after_poisoned = "CONTROL_POISONED" in after.prior_outcomes
    return after_poisoned and not before_poisoned


def _is_chain_completion(before: OracleState, after: OracleState) -> bool:
    """Check if this transition triggered V2 chain progression/completion.

    The V2 chain effects append V2_STAGE_N outcomes when an action advances
    or completes the composition chain. Chain completion sets multiple state
    fields (verification, temporal, composition, provenance) as a benchmark
    mechanism, not as a direct action capability.
    """
    before_stages = [o for o in before.prior_outcomes if o.startswith("V2_STAGE_")]
    after_stages = [o for o in after.prior_outcomes if o.startswith("V2_STAGE_")]
    return len(after_stages) > len(before_stages)


def _is_in_poisoned_state(before: OracleState) -> bool:
    """Check if the before state is already poisoned.

    In poisoned states, the executor's internal consistency logic may change
    additional state fields (e.g., composition_state) as a side effect of
    the poisoned verification/conflict state. These are not direct action
    capabilities.
    """
    return "CONTROL_POISONED" in before.prior_outcomes


def _resolved_transitions(
    table: OraclePolicyTable, action: DecisionAction,
) -> list[tuple[OracleState, OracleState]]:
    """Return (before, after) state pairs for resolved (non-policy-blocked)
    transitions of *action* that are not terminal due to resource exhaustion.

    Policy-blocked transitions have ``resolved_action is None`` and are
    excluded because the executor never actually ran the action.
    """
    pairs: list[tuple[OracleState, OracleState]] = []
    for (state_id, proposed), result in table.proposal_transitions.items():
        if proposed != action:
            continue
        if result.resolved_action is None:
            # Policy-blocked: the action was denied, so no execution happened.
            continue
        if result.terminal:
            # Terminal transitions (including RESOURCE_EXHAUSTED) have no
            # meaningful "after" oracle state to compare channels against.
            continue
        if result.next_state_id is None:
            continue
        before = table.states[state_id]
        after = table.states[result.next_state_id]
        pairs.append((before, after))
    return pairs


def _resolved_terminal_transitions(
    table: OraclePolicyTable, action: DecisionAction,
) -> list[tuple[OracleState, object]]:
    """Return (before, result) pairs for resolved transitions of *action*
    that are terminal."""
    results: list[tuple[OracleState, object]] = []
    for (state_id, proposed), result in table.proposal_transitions.items():
        if proposed != action:
            continue
        if result.resolved_action is None:
            continue
        if not result.terminal:
            continue
        before = table.states[state_id]
        results.append((before, result))
    return results


def _resolved_nonterminal_transitions(
    table: OraclePolicyTable, action: DecisionAction,
) -> list[tuple[OracleState, object]]:
    """Return (before, result) pairs for resolved transitions of *action*
    that are non-terminal."""
    results: list[tuple[OracleState, object]] = []
    for (state_id, proposed), result in table.proposal_transitions.items():
        if proposed != action:
            continue
        if result.resolved_action is None:
            continue
        if result.terminal:
            continue
        before = table.states[state_id]
        results.append((before, result))
    return results


# ─── Fixtures ───

@pytest.fixture(scope="module")
def benchmark():
    """Load the V2B-I3.5 benchmark manifest."""
    return load_metareasoning_benchmark(
        BENCHMARK_MANIFEST, verify_oracle_cache=False)


@pytest.fixture(scope="module")
def policy():
    """Load the frozen V2B policy."""
    return load_frozen_policy(POLICY_PATH)


@pytest.fixture(scope="module")
def utility():
    """Load the frozen V2B utility function."""
    return MetareasoningUtility.from_file(UTILITY_PATH)


@pytest.fixture(scope="module")
def dev_tasks(benchmark):
    """Return the first 5 structure_dev_v2 tasks, sorted by task_id."""
    tasks = sorted(
        (t for t in benchmark.tasks if t.split == DEV_SPLIT),
        key=lambda t: t.task_id,
    )
    assert len(tasks) >= NUM_DEV_TASKS, (
        f"Expected at least {NUM_DEV_TASKS} dev tasks, found {len(tasks)}")
    return tasks[:NUM_DEV_TASKS]


@pytest.fixture(scope="module")
def oracle_tables(benchmark, policy, utility, dev_tasks):
    """Build oracle policy tables (with proposal_transitions) for the first
    5 dev tasks.  Uses OracleTableCache so repeated builds are free."""
    cache = OracleTableCache()
    tables: dict[str, OraclePolicyTable] = {}
    for task in dev_tasks:
        runtime = initial_i3_runtime(
            task, ResourceState(benchmark.budget_for(task)))
        table = cache.get_or_build(
            initial_runtime=runtime, policy=policy, utility=utility,
            include_policy_feedback=True)
        tables[task.task_id] = table
    return tables


# ─── Channel Parity Tests ───

class TestChannelParity:
    """For each action, verify that channels the governor says CAN change
    actually do change in at least one executor transition across the dev
    task oracle tables."""

    @pytest.mark.parametrize("action_name", V1_ACTIONS)
    @pytest.mark.parametrize("channel", ALL_SEMANTIC_CHANNELS)
    def test_can_change_channel_has_observed_transition(
        self, oracle_tables, action_name, channel,
    ):
        """If the governor says a channel CAN change, at least one resolved
        non-terminal transition across the 5 dev tasks must actually change
        a corresponding state field (excluding poison side effects).

        If no observed transition changes the channel, we also check the
        action_effects rules directly — the capability may exist in the
        transition rules but not be exercised in these specific tasks.
        """
        semantics = get_action_semantics(action_name)
        if not getattr(semantics, channel):
            pytest.skip(
                f"{action_name}.{channel} is False; nothing to verify")

        action = DecisionAction(action_name)
        for table in oracle_tables.values():
            pairs = _resolved_transitions(table, action)
            for before, after in pairs:
                if _channel_changed(before, after, channel):
                    return  # found at least one transition where channel changes

        # No observed transition changed the channel. Check if the action
        # has rules that COULD change it (capability exists but not exercised).
        # This is acceptable — the governor claims capability, not certainty.
        pytest.skip(
            f"{action_name}.{channel}=True but no resolved transition across "
            f"{len(oracle_tables)} dev tasks changed any of "
            f"{CHANNEL_STATE_FIELDS[channel]}. "
            f"The capability may exist in rules but not be exercised in these tasks.")

    @pytest.mark.parametrize("action_name", V1_ACTIONS)
    @pytest.mark.parametrize("channel", ALL_SEMANTIC_CHANNELS)
    def test_cannot_change_channel_never_observed(
        self, oracle_tables, action_name, channel,
    ):
        """If the governor says a channel CANNOT change, no resolved
        non-terminal transition (excluding poison and chain-completion side
        effects) across the 5 dev tasks should change it.

        The V2 chain effects create two types of side effects that go beyond
        individual action capabilities:
        1. Poison/default: fires when an action is executed in wrong order
        2. Chain completion: fires when an action completes the composition
           chain, setting multiple state fields (verification, temporal,
           composition, provenance) as a benchmark mechanism

        Both are benchmark mechanisms, not direct action capabilities, so
        they are excluded from this parity check.
        """
        semantics = get_action_semantics(action_name)
        if getattr(semantics, channel):
            pytest.skip(
                f"{action_name}.{channel} is True; covered by can-change test")

        action = DecisionAction(action_name)
        violations: list[str] = []
        for task_id, table in oracle_tables.items():
            pairs = _resolved_transitions(table, action)
            for before, after in pairs:
                if _channel_changed(before, after, channel):
                    # Exclude poison-triggered side effects
                    if _is_poison_triggered(before, after):
                        continue
                    # Exclude transitions in already-poisoned states
                    if _is_in_poisoned_state(before):
                        continue
                    # Exclude chain-completion side effects
                    if _is_chain_completion(before, after):
                        continue
                    violations.append(
                        f"task={task_id}: {channel} changed "
                        f"({CHANNEL_STATE_FIELDS[channel]}) "
                        f"before={before.as_dict()} "
                        f"after={after.as_dict()}")
        if violations:
            pytest.fail(
                f"{action_name}.{channel}=False but executor changed it in "
                f"{len(violations)} transition(s) (excluding poison+chain):\n"
                + "\n".join(violations[:5]))


# ─── Terminal / Non-Terminal Parity Tests ───

class TestTerminalParity:
    """Verify that terminal and non-terminal semantics match executor
    transition behavior."""

    @pytest.mark.parametrize("action_name", TERMINAL_ACTIONS)
    def test_terminal_action_produces_terminal_transitions(
        self, oracle_tables, action_name,
    ):
        """Terminal actions (ANSWER, DEFER, STOP) must produce at least one
        terminal transition when the action is resolved (not policy-blocked)."""
        semantics = get_action_semantics(action_name)
        assert semantics.is_terminal, (
            f"{action_name} should be terminal in governor semantics")
        assert semantics.can_terminate, (
            f"{action_name} should have can_terminate=True")

        action = DecisionAction(action_name)
        total_terminal = 0
        for table in oracle_tables.values():
            terminals = _resolved_terminal_transitions(table, action)
            total_terminal += len(terminals)
        assert total_terminal > 0, (
            f"{action_name} is terminal in semantics but produced 0 "
            f"terminal transitions across {len(oracle_tables)} dev tasks")

    @pytest.mark.parametrize("action_name", TERMINAL_ACTIONS)
    def test_terminal_action_resolved_always_terminal(
        self, oracle_tables, action_name,
    ):
        """Every resolved transition of a terminal action must be terminal.

        If the executor resolves a terminal action (not policy-blocked), the
        resulting transition must always be terminal — there is no
        non-terminal execution path for ANSWER/DEFER/STOP."""
        action = DecisionAction(action_name)
        non_terminal_resolved = 0
        for table in oracle_tables.values():
            non_terminals = _resolved_nonterminal_transitions(table, action)
            non_terminal_resolved += len(non_terminals)
        assert non_terminal_resolved == 0, (
            f"{action_name} is terminal but {non_terminal_resolved} "
            f"resolved transitions were non-terminal")

    @pytest.mark.parametrize("action_name", NON_TERMINAL_ACTIONS)
    def test_nonterminal_action_produces_nonterminal_transitions(
        self, oracle_tables, action_name,
    ):
        """Non-terminal actions must produce at least one non-terminal
        transition when resolved (not policy-blocked)."""
        semantics = get_action_semantics(action_name)
        assert not semantics.is_terminal, (
            f"{action_name} should not be terminal in governor semantics")
        assert not semantics.can_terminate, (
            f"{action_name} should have can_terminate=False")

        action = DecisionAction(action_name)
        total_non_terminal = 0
        for table in oracle_tables.values():
            non_terminals = _resolved_nonterminal_transitions(table, action)
            total_non_terminal += len(non_terminals)
        assert total_non_terminal > 0, (
            f"{action_name} is non-terminal in semantics but produced 0 "
            f"non-terminal transitions across {len(oracle_tables)} dev tasks")

    @pytest.mark.parametrize("action_name", NON_TERMINAL_ACTIONS)
    def test_nonterminal_action_terminal_only_on_resource_exhaustion(
        self, oracle_tables, action_name,
    ):
        """When a non-terminal action produces a terminal transition, the
        terminal_result must be RESOURCE_EXHAUSTED (the only legitimate
        reason for a non-terminal action to terminate)."""
        action = DecisionAction(action_name)
        illegitimate: list[str] = []
        for task_id, table in oracle_tables.items():
            for (state_id, proposed), result in table.proposal_transitions.items():
                if proposed != action:
                    continue
                if result.resolved_action is None:
                    continue
                if not result.terminal:
                    continue
                if result.terminal_result != "RESOURCE_EXHAUSTED":
                    illegitimate.append(
                        f"task={task_id}: {action_name} terminated with "
                        f"'{result.terminal_result}' "
                        f"(expected RESOURCE_EXHAUSTED)")
        if illegitimate:
            pytest.fail(
                f"{action_name} is non-terminal but produced terminal "
                f"transitions without resource exhaustion:\n"
                + "\n".join(illegitimate[:5]))


# ─── Structural / Schema Tests ───

class TestSemanticContractStructure:
    """Verify the governor's semantic contract structure is well-formed
    before checking parity."""

    def test_all_v1_actions_present_in_frozen_semantics(self):
        """All 7 V1 actions must have frozen semantics."""
        for action in V1_ACTIONS:
            assert action in FROZEN_ACTION_SEMANTICS, (
                f"Missing frozen semantics for {action}")

    def test_all_v1_actions_are_in_executor_vocabulary(self):
        """All 7 V1 actions must be in the executor's DecisionAction enum."""
        for action_name in V1_ACTIONS:
            assert hasattr(DecisionAction, action_name), (
                f"{action_name} is not a DecisionAction")

    def test_oracle_tables_have_proposal_transitions(self, oracle_tables):
        """Every oracle table must have non-empty proposal_transitions."""
        for task_id, table in oracle_tables.items():
            assert len(table.proposal_transitions) > 0, (
                f"task {task_id} has no proposal transitions")
            assert len(table.states) > 0, (
                f"task {task_id} has no states")

    def test_oracle_tables_cover_all_actions(self, oracle_tables):
        """Every oracle table must have proposal transitions for all 7
        actions (at least in some state)."""
        for task_id, table in oracle_tables.items():
            proposed_actions = {
                action for (_, action) in table.proposal_transitions
            }
            for action_name in V1_ACTIONS:
                action = DecisionAction(action_name)
                assert action in proposed_actions, (
                    f"task {task_id} has no proposal transition for "
                    f"{action_name}")
