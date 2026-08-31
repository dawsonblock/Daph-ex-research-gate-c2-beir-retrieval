"""Procedural causal benchmark generator for DAPH-X M4.

Replaces handcrafted templates with a dimension-sampling generator.
Generates states by sampling independent dimensions rather than
choosing from a handful of named templates.

Supports paired worlds: two states with the same observable coarse
structure but different deep causal properties (evidence reliability,
latent truth, transition dynamics) so that ΔU flips sign.

Dimensions varied:
  - hypothesis count
  - evidence count
  - graph connectivity (support/contradiction patterns)
  - verification state
  - evidence reliability (source_reliability, independence, ambiguity, noise)
  - source dependence (correlated evidence sources)
  - resource budgets (steps, verify, search, retrieve)
  - candidate-action targets
  - belief concentration
  - world-model accuracy
  - required continuation depth
  - harm mechanism
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceHypothesis, EvidenceItem, EvidenceTask,
)
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)

from daph_x.graph.epistemic_graph import (
    EpistemicGraph, GraphNode, GraphEdge, NodeType, EdgeType, EvidenceReliability,
)
from daph_x.benchmark.novelty_signatures import (
    compute_all_signatures, NoveltySignatures,
)


# Harm mechanism taxonomy
HARM_MECHANISMS = [
    "misleading_support",
    "bad_verify_target",
    "resource_depletion",
    "weak_evidence_dependence",
    "near_value_inversion",
    "world_model_error",
    "belief_overconfidence",
    "novel_topology",
    "correct_clear",
]

# Mechanism families for train/test split
MECHANISM_FAMILIES = {
    "evidence_quality": ["misleading_support", "weak_evidence_dependence"],
    "action_selection": ["bad_verify_target", "near_value_inversion"],
    "resource": ["resource_depletion"],
    "model_error": ["world_model_error", "belief_overconfidence"],
    "structural": ["novel_topology"],
    "benign": ["correct_clear"],
}


@dataclass(frozen=True)
class GeneratedState:
    """A procedurally generated benchmark state."""
    task: EvidenceTask
    graph: EpistemicGraph
    correct_hypothesis_id: str
    harm_mechanism: str
    mechanism_family: str
    signatures: NoveltySignatures
    # Paired world metadata
    pair_id: str = ""
    pair_polarity: str = ""  # "beneficial" or "harmful" or "neutral"
    # World model configuration for this state
    world_model_config: dict = field(default_factory=dict)
    # Generator parameters for reproducibility
    generator_seed: int = 0
    generator_params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task.task_id,
            "correct_hypothesis_id": self.correct_hypothesis_id,
            "harm_mechanism": self.harm_mechanism,
            "mechanism_family": self.mechanism_family,
            "signatures": self.signatures.to_dict(),
            "pair_id": self.pair_id,
            "pair_polarity": self.pair_polarity,
            "world_model_config": self.world_model_config,
            "generator_seed": self.generator_seed,
            "generator_params": self.generator_params,
        }


@dataclass
class GeneratorConfig:
    """Configuration for the procedural generator."""
    n_hyp_range: tuple[int, int] = (2, 7)
    n_ev_range: tuple[int, int] = (1, 8)
    steps_range: tuple[int, int] = (2, 8)
    verify_range: tuple[int, int] = (0, 4)
    search_range: tuple[int, int] = (0, 2)
    retrieve_range: tuple[int, int] = (0, 2)
    # Probability of various structural patterns
    p_competing_support: float = 0.3
    p_misleading: float = 0.25
    p_unverified: float = 0.4
    p_dependent_evidence: float = 0.2
    # Reliability variation
    reliability_variation: float = 0.3
    # World model accuracy
    wm_accuracy_range: tuple[float, float] = (0.5, 0.95)
    # Paired world generation
    enable_paired_worlds: bool = True
    p_paired: float = 0.3


def _sample_int(rng: random.Random, lo: int, hi: int) -> int:
    """Sample an integer in [lo, hi] inclusive."""
    return rng.randint(lo, hi)


def _sample_float(rng: random.Random, lo: float, hi: float) -> float:
    """Sample a float in [lo, hi)."""
    return rng.uniform(lo, hi)


def _sample_reliability(rng: random.Random, variation: float) -> EvidenceReliability:
    """Sample evidence reliability with variation."""
    base = 1.0 - variation
    return EvidenceReliability(
        source_reliability=_sample_float(rng, base, 1.0),
        verification_confidence=_sample_float(rng, base, 1.0),
        independence_score=_sample_float(rng, base, 1.0),
        ambiguity=_sample_float(rng, 0.0, variation),
        freshness=_sample_float(rng, base, 1.0),
        observation_noise=_sample_float(rng, 0.0, variation * 0.5),
    )


def _pick_mechanism(rng: random.Random, allowed_mechanisms: list[str]) -> str:
    """Pick a harm mechanism from the allowed set."""
    return rng.choice(allowed_mechanisms)


def _mechanism_to_family(mechanism: str) -> str:
    """Map a mechanism to its family."""
    for family, mechs in MECHANISM_FAMILIES.items():
        if mechanism in mechs:
            return family
    return "unknown"


def generate_state(
    seed: int,
    config: GeneratorConfig,
    allowed_mechanisms: list[str] | None = None,
    pair_id: str = "",
    pair_polarity: str = "",
    force_mechanism: str | None = None,
) -> GeneratedState:
    """Generate a single benchmark state procedurally.

    Args:
        seed: RNG seed for reproducibility
        config: Generator configuration
        allowed_mechanisms: Restrict to these mechanisms (for split control)
        pair_id: If part of a paired world, the pair identifier
        pair_polarity: "beneficial", "harmful", or ""
        force_mechanism: Force a specific harm mechanism (for paired worlds)
    """
    rng = random.Random(seed)

    if allowed_mechanisms is None:
        allowed_mechanisms = HARM_MECHANISMS

    # Pick harm mechanism
    if force_mechanism:
        harm_mechanism = force_mechanism
    else:
        harm_mechanism = _pick_mechanism(rng, allowed_mechanisms)
    mechanism_family = _mechanism_to_family(harm_mechanism)

    # Sample dimensions
    n_hyp = _sample_int(rng, *config.n_hyp_range)
    n_ev = _sample_int(rng, *config.n_ev_range)
    steps = _sample_int(rng, *config.steps_range)
    verify_budget = _sample_int(rng, *config.verify_range)
    search_budget = _sample_int(rng, *config.search_range)
    retrieve_budget = _sample_int(rng, *config.retrieve_range)

    # Pick correct hypothesis
    correct_idx = rng.randint(0, n_hyp - 1)

    # Build hypotheses
    hypotheses = []
    for i in range(n_hyp):
        h_id = f"H{i+1}"
        is_correct = (i == correct_idx)
        if i == n_hyp - 1:
            action = "DEFER"
        else:
            action = "ANSWER"
        prop = f"Hypothesis {i+1}"
        if is_correct and harm_mechanism == "misleading_support":
            prop = f"Hypothesis {i+1} (correct, appears wrong)"
        elif not is_correct and harm_mechanism == "misleading_support":
            prop = f"Hypothesis {i+1} (wrong, appears correct)"
        hypotheses.append((h_id, prop, action))

    # Build evidence — the key dimension that varies by mechanism
    evidence = []
    n_verified = 0
    n_unverified = 0

    for j in range(n_ev):
        e_id = f"E{j+1}"
        # Decide what this evidence does
        if harm_mechanism == "misleading_support":
            # Evidence supports wrong hypothesis, contradicts correct
            if j == 0:
                supports = (f"H{correct_idx + 2}" if correct_idx + 2 <= n_hyp else "H1",)
                contradicts = (f"H{correct_idx + 1}",)
                vstate = "SUFFICIENT"
            else:
                target = rng.randint(0, n_hyp - 1)
                supports = (f"H{target + 1}",) if rng.random() < 0.5 else ()
                contradicts = (f"H{target + 1}",) if not supports else ()
                vstate = rng.choice(["SUFFICIENT", "UNVERIFIED"])
        elif harm_mechanism == "bad_verify_target":
            # Mix of useful and useless unverified evidence
            if j == 0:
                # Useful unverified evidence for correct hypothesis
                supports = (f"H{correct_idx + 1}",)
                contradicts = ()
                vstate = "UNVERIFIED"
            else:
                # Useless unverified evidence
                target = rng.randint(0, n_hyp - 1)
                supports = () if rng.random() < 0.5 else (f"H{target + 1}",)
                contradicts = () if supports else (f"H{target + 1}",)
                vstate = "UNVERIFIED"
        elif harm_mechanism == "resource_depletion":
            # Many unverified items, limited budget
            target = rng.randint(0, n_hyp - 1)
            supports = (f"H{target + 1}",) if rng.random() < 0.5 else ()
            contradicts = () if supports else (f"H{target + 1}",)
            vstate = "UNVERIFIED"
            verify_budget = min(verify_budget, 1)  # Very limited
        elif harm_mechanism == "weak_evidence_dependence":
            # Multiple evidence items from same source, all supporting wrong
            target = rng.randint(0, n_hyp - 1)
            if target == correct_idx:
                target = (target + 1) % n_hyp
            supports = (f"H{target + 1}",)
            contradicts = ()
            vstate = "SUFFICIENT"
        elif harm_mechanism == "near_value_inversion":
            # Correct and wrong both have similar support
            if j == 0:
                supports = (f"H{correct_idx + 1}",)
                contradicts = ()
                vstate = "SUFFICIENT"
            elif j == 1:
                wrong_idx = (correct_idx + 1) % n_hyp
                supports = (f"H{wrong_idx + 1}",)
                contradicts = ()
                vstate = "SUFFICIENT"
            else:
                target = rng.randint(0, n_hyp - 1)
                supports = (f"H{target + 1}",) if rng.random() < 0.5 else ()
                contradicts = () if supports else (f"H{target + 1}",)
                vstate = rng.choice(["SUFFICIENT", "UNVERIFIED"])
        elif harm_mechanism == "world_model_error":
            # Evidence looks fine but world model will mispredict
            target = rng.randint(0, n_hyp - 1)
            supports = (f"H{target + 1}",) if rng.random() < 0.5 else ()
            contradicts = () if supports else (f"H{target + 1}",)
            vstate = rng.choice(["SUFFICIENT", "UNVERIFIED", "FALSIFIED"])
        elif harm_mechanism == "belief_overconfidence":
            # Strong support for wrong hypothesis
            wrong_idx = (correct_idx + 1) % n_hyp
            if j == 0:
                supports = (f"H{wrong_idx + 1}",)
                contradicts = (f"H{correct_idx + 1}",)
                vstate = "SUFFICIENT"
            else:
                target = rng.randint(0, n_hyp - 1)
                supports = (f"H{target + 1}",) if rng.random() < 0.3 else ()
                contradicts = () if supports else (f"H{target + 1}",)
                vstate = rng.choice(["SUFFICIENT", "UNVERIFIED"])
        elif harm_mechanism == "novel_topology":
            # Unusual graph structure
            target = rng.randint(0, n_hyp - 1)
            if rng.random() < 0.5:
                supports = (f"H{target + 1}",)
                contradicts = ()
            else:
                supports = ()
                contradicts = (f"H{target + 1}",)
            vstate = rng.choice(["SUFFICIENT", "UNVERIFIED", "FALSIFIED"])
        else:  # correct_clear
            # Clear support for correct hypothesis
            if j == 0:
                supports = (f"H{correct_idx + 1}",)
                contradicts = ()
                vstate = "SUFFICIENT"
            else:
                target = rng.randint(0, n_hyp - 1)
                if target == correct_idx:
                    supports = (f"H{target + 1}",)
                    contradicts = ()
                    vstate = "SUFFICIENT"
                else:
                    supports = ()
                    contradicts = (f"H{target + 1}",)
                    vstate = rng.choice(["SUFFICIENT", "FALSIFIED"])

        if vstate != "UNVERIFIED":
            n_verified += 1
        else:
            n_unverified += 1

        evidence.append((
            e_id,
            f"Evidence {j+1}",
            "initial",
            supports,
            contradicts,
            vstate,
            "CURRENT",
        ))

    # Determine expected terminal and oracle path based on mechanism
    if harm_mechanism in ("misleading_support", "belief_overconfidence"):
        # Should DEFER because the supported hypothesis is wrong
        expected_terminal = "DEFER"
        oracle_path = ("DEFER",)
    elif harm_mechanism == "near_value_inversion":
        # Should DEFER because competing support
        expected_terminal = "DEFER"
        oracle_path = ("DEFER",)
    elif harm_mechanism == "bad_verify_target":
        # Should VERIFY the useful evidence first
        expected_terminal = "ANSWER"
        oracle_path = ("VERIFY", "ANSWER")
    elif harm_mechanism == "resource_depletion":
        # Should DEFER because not enough budget to resolve
        expected_terminal = "DEFER"
        oracle_path = ("DEFER",)
    elif harm_mechanism == "correct_clear":
        # Should ANSWER with correct hypothesis
        expected_terminal = "ANSWER"
        oracle_path = ("ANSWER",)
    else:
        # Default: answer if unique support, defer otherwise
        if n_verified > 0:
            expected_terminal = "ANSWER"
            oracle_path = ("ANSWER",)
        else:
            expected_terminal = "DEFER"
            oracle_path = ("DEFER",)

    # Build the task
    task_id = f"m4_{harm_mechanism}_{seed:06d}"
    if pair_id:
        task_id = f"m4_{harm_mechanism}_pair{pair_id}_{pair_polarity}_{seed:06d}"

    hyp_objects = []
    for h_id, prop, action_str in hypotheses:
        hyp_objects.append(EvidenceHypothesis(
            hypothesis_id=h_id,
            proposition=prop,
            answer_action=DecisionAction(action_str),
            answer_payload=f"{action_str}:{h_id}:{prop}",
        ))

    ev_objects = []
    for ev_id, prop, source, supports, contradicts, vstate_str, tstatus_str in evidence:
        ev_objects.append(EvidenceItem(
            evidence_id=ev_id,
            proposition=prop,
            source_class=source,
            supports=supports,
            contradicts=contradicts,
            verification_state=VerificationState(vstate_str),
            temporal_status=TemporalStatus(tstatus_str),
            retrieved=True,
            verify_result=vstate_str if vstate_str != "UNVERIFIED" else None,
        ))

    task = EvidenceTask(
        task_id=task_id,
        split="m4",
        category=harm_mechanism,
        task_summary=f"Procedural {harm_mechanism}",
        high_stakes=True,
        budget_profile=f"M4_{steps}_{verify_budget}_{search_budget}",
        hypotheses=tuple(hyp_objects),
        evidence_items=tuple(ev_objects),
        retrieve_exposes=(),
        search_exposes=(),
        oracle_resolution_path=oracle_path,
        expected_terminal=DecisionAction(expected_terminal),
        correct_hypothesis_id=f"H{correct_idx + 1}",
    )

    # Build the graph with reliability
    graph = _build_graph_with_reliability(task, rng, config, harm_mechanism)

    # Compute signatures
    resources = {
        "steps": steps,
        "verify": verify_budget,
        "retrieve": retrieve_budget,
        "search": search_budget,
    }
    signatures = compute_all_signatures(
        graph, f"H{correct_idx + 1}", harm_mechanism, resources
    )

    # World model config
    wm_accuracy = _sample_float(rng, *config.wm_accuracy_range)
    world_model_config = {
        "verify_sufficient_prob": wm_accuracy if harm_mechanism != "world_model_error" else 1.0 - wm_accuracy,
        "verify_falsified_prob": 0.2,
        "verify_inconclusive_prob": 0.1,
        "search_found_prob": 0.5,
        "retrieve_found_prob": 0.5,
    }

    return GeneratedState(
        task=task,
        graph=graph,
        correct_hypothesis_id=f"H{correct_idx + 1}",
        harm_mechanism=harm_mechanism,
        mechanism_family=mechanism_family,
        signatures=signatures,
        pair_id=pair_id,
        pair_polarity=pair_polarity,
        world_model_config=world_model_config,
        generator_seed=seed,
        generator_params={
            "n_hyp": n_hyp,
            "n_ev": n_ev,
            "steps": steps,
            "verify_budget": verify_budget,
            "search_budget": search_budget,
            "retrieve_budget": retrieve_budget,
            "correct_idx": correct_idx,
        },
    )


def _build_graph_with_reliability(
    task: EvidenceTask,
    rng: random.Random,
    config: GeneratorConfig,
    harm_mechanism: str,
) -> EpistemicGraph:
    """Build an epistemic graph with varied reliability."""
    nodes = {}
    edges = []

    # Add hypothesis nodes
    for h in task.hypotheses:
        nodes[h.hypothesis_id] = GraphNode(
            node_id=h.hypothesis_id,
            node_type=NodeType.HYPOTHESIS,
            label=h.proposition,
            answer_action=h.answer_action.value,
        )

    # Add evidence nodes with reliability
    for e in task.evidence_items:
        if harm_mechanism == "weak_evidence_dependence":
            # Low independence, high source reliability (looks reliable but isn't)
            reliability = EvidenceReliability(
                source_reliability=0.9,
                verification_confidence=0.9,
                independence_score=0.3,  # Low independence
                ambiguity=0.2,
                freshness=0.8,
                observation_noise=0.1,
            )
        elif harm_mechanism == "world_model_error":
            # High noise
            reliability = EvidenceReliability(
                source_reliability=0.7,
                verification_confidence=0.5,
                independence_score=0.8,
                ambiguity=0.4,
                freshness=0.6,
                observation_noise=0.3,
            )
        elif harm_mechanism == "belief_overconfidence":
            # Looks very reliable but supports wrong
            reliability = EvidenceReliability(
                source_reliability=0.95,
                verification_confidence=0.95,
                independence_score=0.9,
                ambiguity=0.05,
                freshness=0.95,
                observation_noise=0.02,
            )
        else:
            reliability = _sample_reliability(rng, config.reliability_variation)

        nodes[e.evidence_id] = GraphNode(
            node_id=e.evidence_id,
            node_type=NodeType.EVIDENCE,
            label=e.proposition,
            verification_state=e.verification_state.value,
            temporal_status=e.temporal_status.value,
            reliability=reliability,
        )
        for h_id in e.supports:
            edges.append(GraphEdge(
                source_id=e.evidence_id,
                target_id=h_id,
                edge_type=EdgeType.SUPPORTS,
            ))
        for h_id in e.contradicts:
            edges.append(GraphEdge(
                source_id=e.evidence_id,
                target_id=h_id,
                edge_type=EdgeType.CONTRADICTS,
            ))

    # Parse resources from budget profile
    parts = task.budget_profile.split("_")
    steps = int(parts[1]) if len(parts) > 1 else 4
    verify = int(parts[2]) if len(parts) > 2 else 2
    search = int(parts[3]) if len(parts) > 3 else 0

    return EpistemicGraph(
        nodes=nodes,
        edges=tuple(edges),
        steps_remaining=steps,
        verify_remaining=verify,
        retrieve_remaining=0,
        search_remaining=search,
    )


def generate_paired_worlds(
    seed: int,
    config: GeneratorConfig,
    allowed_mechanisms: list[str] | None = None,
) -> tuple[GeneratedState, GeneratedState]:
    """Generate a pair of states with same structure but opposite ΔU.

    World A: intervention is beneficial (ΔU > 0)
    World B: intervention is harmful (ΔU < 0)

    Both worlds share the SAME observable coarse structure:
      - same hypothesis count
      - same evidence count
      - same resource budget
      - same coarse topology family
      - same candidate action types

    They differ ONLY in the latent causal variable:
      - which hypothesis is correct (truth is flipped)
      - or evidence reliability is degraded

    This prevents shortcut learning from coarse observable features.
    """
    rng = random.Random(seed)
    pair_id = f"{seed:06d}"

    # Generate World A (beneficial) — correct_clear mechanism
    state_a = generate_state(
        seed=seed,
        config=config,
        allowed_mechanisms=allowed_mechanisms,
        pair_id=pair_id,
        pair_polarity="beneficial",
        force_mechanism="correct_clear",
    )

    # Build World B by CLONING state_a's structure and flipping
    # only the correct hypothesis (truth inversion).
    # This keeps: same n_hyp, n_ev, resources, topology, action types
    # But changes: which hypothesis is correct → ΔU flips
    state_b = _clone_and_flip_truth(state_a, seed, pair_id)

    return state_a, state_b


def _clone_and_flip_truth(
    state_a: GeneratedState,
    seed: int,
    pair_id: str,
) -> GeneratedState:
    """Clone a state's structure but flip which hypothesis is correct.

    This creates a matched pair where:
      - Same graph structure (same nodes, edges, verification states)
      - Same resource budgets
      - Same candidate action types
      - DIFFERENT correct hypothesis (truth is inverted)

    In World A: the uniquely supported hypothesis is correct → ANSWER is good
    In World B: the uniquely supported hypothesis is WRONG → ANSWER is harmful
    """
    rng = random.Random(seed + 1)

    # Get the original correct hypothesis
    original_correct = state_a.correct_hypothesis_id
    hyp_ids = sorted(state_a.graph.hypothesis_ids())

    # Pick a different hypothesis as correct for World B
    # Prefer one that is NOT uniquely supported (so the supported one is wrong)
    other_hyps = [h for h in hyp_ids if h != original_correct]
    if not other_hyps:
        # Can't flip if only one hypothesis — just return a copy
        return state_a
    new_correct = rng.choice(other_hyps)

    # Clone the task but change the correct hypothesis
    from hrm_adaptive_memory.cognitive_control.core import DecisionAction
    from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
        EvidenceHypothesis, EvidenceItem, EvidenceTask,
    )
    from hrm_adaptive_memory.cognitive_control.state import (
        TemporalStatus, VerificationState,
    )

    task_a = state_a.task
    task_b_id = f"m4_misleading_support_pair{pair_id}_harmful_{seed+1:06d}"

    # Clone hypotheses — same structure, same answer actions
    hyp_objects = []
    for h in task_a.hypotheses:
        hyp_objects.append(EvidenceHypothesis(
            hypothesis_id=h.hypothesis_id,
            proposition=h.proposition,
            answer_action=h.answer_action,
            answer_payload=h.answer_payload,
        ))

    # Clone evidence — identical
    ev_objects = []
    for e in task_a.evidence_items:
        ev_objects.append(EvidenceItem(
            evidence_id=e.evidence_id,
            proposition=e.proposition,
            source_class=e.source_class,
            supports=e.supports,
            contradicts=e.contradicts,
            verification_state=e.verification_state,
            temporal_status=e.temporal_status,
            retrieved=e.retrieved,
            verify_result=e.verify_result,
        ))

    # The key change: different correct hypothesis
    # This means the uniquely supported hypothesis is now WRONG
    # → ANSWER(supported) is harmful
    # → DEFER is correct (should defer because the supported hyp is wrong)
    task_b = EvidenceTask(
        task_id=task_b_id,
        split="m4",
        category="misleading_support",
        task_summary=f"Paired world (harmful) — truth flipped from {original_correct} to {new_correct}",
        high_stakes=True,
        budget_profile=task_a.budget_profile,  # Same resources
        hypotheses=tuple(hyp_objects),
        evidence_items=tuple(ev_objects),
        retrieve_exposes=(),
        search_exposes=(),
        oracle_resolution_path=("DEFER",),  # Should defer — supported hyp is wrong
        expected_terminal=DecisionAction("DEFER"),
        correct_hypothesis_id=new_correct,  # Flipped!
    )

    # Clone the graph — same structure, same reliability
    # (The graph doesn't change — only the correct hypothesis metadata changes)
    graph_b = state_a.graph  # Frozen, safe to share

    # Compute new signatures
    from daph_x.benchmark.novelty_signatures import compute_all_signatures
    signatures_b = compute_all_signatures(
        graph_b, new_correct, "misleading_support",
    )

    # World model config — same as World A (structure is identical)
    wm_config = dict(state_a.world_model_config)

    return GeneratedState(
        task=task_b,
        graph=graph_b,
        correct_hypothesis_id=new_correct,
        harm_mechanism="misleading_support",
        mechanism_family="evidence_quality",
        signatures=signatures_b,
        pair_id=pair_id,
        pair_polarity="harmful",
        world_model_config=wm_config,
        generator_seed=seed + 1,
        generator_params=dict(state_a.generator_params),
    )
