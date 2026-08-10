"""Feature-availability schema for executive/controller state features.

Per the research-lead directive following the Executive Opportunity Study v1
(evidence/gate_executive/opportunity_execute.json): a candidate feature like
graph path count or packet coherence is only observable AFTER G2 graph
construction (or later) has already run -- which costs nearly as much as the
MEMORY action itself. Using such a feature to decide WHETHER to invoke MEMORY
in the first place would be circular: the "cheap" decision would secretly
require having already paid for the "expensive" action.

This module makes that distinction mechanical rather than a design note that
can be silently forgotten. Every feature an executive experiment wants to use
must be declared here with its AvailabilityStage; require_stage_available()
is the fail-closed gate a controller's feature-selection code calls before
consuming a feature.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AvailabilityStage(str, Enum):
    #: Computable from the question text alone, before EITHER action has
    #: done any work.
    PRE_DECISION = "PRE_DECISION"
    #: AMENDMENT (Executive v0 design, after the EOB-v1/v2 template-overlap
    #: finding showed PRE_DECISION alone carries near-zero signal between
    #: this project's two training families): computable after ANSWER_NOW
    #: has run -- which is legitimate for an ANSWER_NOW-vs-MEMORY gate
    #: because ANSWER_NOW is, by definition, the cheap action; using its own
    #: output/confidence to decide whether to ALSO pay for MEMORY is
    #: "uncertainty-gated routing", not circular in the way POST_RETRIEVAL/
    #: POST_GRAPH would be (those require part of MEMORY -- the EXPENSIVE
    #: action -- to have already run). Executive v0 is admissible at this
    #: stage AND PRE_DECISION; still inadmissible at POST_RETRIEVAL/
    #: POST_GRAPH, which remain reserved for a later RETRIEVE_MORE/
    #: GRAPH_REFINE/VERIFY/STOP controller where memory work has already
    #: been committed.
    POST_ANSWER_NOW_PRE_MEMORY = "POST_ANSWER_NOW_PRE_MEMORY"
    #: Requires C2 retrieval (BM25+BGE+fusion) to have already run.
    POST_RETRIEVAL = "POST_RETRIEVAL"
    #: Requires G2 graph construction and/or path enumeration to have
    #: already run -- this is most of the cost of the MEMORY action itself.
    POST_GRAPH = "POST_GRAPH"
    #: Only observable after HRM generation has already happened -- either
    #: ANSWER_NOW's (see POST_ANSWER_NOW_PRE_MEMORY, now the intended path
    #: for that) or MEMORY's own (still inadmissible for the ANSWER_NOW-vs-
    #: MEMORY gate).
    POST_GENERATION = "POST_GENERATION"


#: Stages a controller choosing only between ANSWER_NOW and MEMORY (no
#: intermediate actions like RETRIEVE_MORE/GRAPH_REFINE) may legitimately
#: condition on. POST_RETRIEVAL/POST_GRAPH remain excluded -- both require
#: part of MEMORY, the expensive action, to have already run.
STAGES_ADMISSIBLE_FOR_ANSWER_VS_MEMORY = frozenset(
    {AvailabilityStage.PRE_DECISION, AvailabilityStage.POST_ANSWER_NOW_PRE_MEMORY})


class FeatureAvailabilityError(RuntimeError):
    """A feature was used at a decision point that could not legitimately
    have computed it yet. Fail closed -- this is exactly the mistake this
    module exists to prevent, not a warning-level concern."""


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    availability_stage: AvailabilityStage
    #: Free-text note on what computing this feature actually costs (e.g.
    #: "one C2 retrieval call" vs "full G2 graph + path enumeration") --
    #: informational, not machine-enforced, but required so a reader isn't
    #: left guessing why a feature is PRE_DECISION vs POST_RETRIEVAL.
    cost_to_observe: str
    #: Whether this feature can be computed in a real online deployment
    #: (True) or only retrospectively from a completed run's receipts
    #: (False, e.g. "actual latency spent" is only known after the fact).
    runtime_safe: bool


def require_admissible_for_answer_vs_memory(spec: FeatureSpec) -> None:
    """Call this before a controller choosing only ANSWER_NOW vs MEMORY
    consumes any feature. Raises FeatureAvailabilityError (fail-closed) for
    anything past PRE_DECISION."""
    if spec.availability_stage not in STAGES_ADMISSIBLE_FOR_ANSWER_VS_MEMORY:
        raise FeatureAvailabilityError(
            f"feature {spec.name!r} has availability_stage="
            f"{spec.availability_stage.value}, which requires part or all of "
            "the MEMORY action to have already run. It is not admissible for "
            "a controller deciding WHETHER to invoke MEMORY in the first "
            "place -- that would make the 'cheap' decision secretly depend "
            "on having already paid for the 'expensive' action. Valid for a "
            "LATER controller choosing among RETRIEVE_MORE/GRAPH_REFINE/"
            "VERIFY/STOP, where memory work has already been committed.")


#: The features captured by scripts/run_executive_opportunity_study.py's
#: state_features block, and by any EOB-v1 successor, classified once here so
#: every consumer agrees on what stage each one belongs to.
KNOWN_FEATURES: dict[str, FeatureSpec] = {
    "identity_status": FeatureSpec(
        "identity_status", AvailabilityStage.POST_RETRIEVAL,
        "requires run_identity_stage over the full C2 candidate pool, which "
        "itself requires C2 retrieval (BM25+BGE+fusion) to have already run",
        runtime_safe=True),
    "retrieval_score_margin": FeatureSpec(
        "retrieval_score_margin", AvailabilityStage.POST_RETRIEVAL,
        "top1-top2 fused retrieval score -- requires C2 retrieval",
        runtime_safe=True),
    "graph_reachability": FeatureSpec(
        "graph_reachability", AvailabilityStage.POST_GRAPH,
        "fraction of candidate pool with >=1 incident graph edge -- requires "
        "full G2 runtime graph construction",
        runtime_safe=True),
    "working_set_size": FeatureSpec(
        "working_set_size", AvailabilityStage.POST_GRAPH,
        "size of the G2 working set after g2_prefilter -- requires G2",
        runtime_safe=True),
    "n_complete_paths": FeatureSpec(
        "n_complete_paths", AvailabilityStage.POST_GRAPH,
        "requires G2 path enumeration over the constructed graph",
        runtime_safe=True),
    "path_competition_bucket": FeatureSpec(
        "path_competition_bucket", AvailabilityStage.POST_GRAPH,
        "bucketed n_complete_paths -- same cost as n_complete_paths",
        runtime_safe=True),
    "structural_competition_ratio": FeatureSpec(
        "structural_competition_ratio", AvailabilityStage.POST_GRAPH,
        "n_complete_paths / working_set_size -- requires both",
        runtime_safe=True),
    "bridge_availability_estimate": FeatureSpec(
        "bridge_availability_estimate", AvailabilityStage.POST_GRAPH,
        "checks working-set membership against oracle bridge record ids -- "
        "requires the G2 working set to exist",
        runtime_safe=False),
    "terminal_availability_estimate": FeatureSpec(
        "terminal_availability_estimate", AvailabilityStage.POST_GRAPH,
        "same as bridge_availability_estimate",
        runtime_safe=False),
    "packet_coherence": FeatureSpec(
        "packet_coherence", AvailabilityStage.POST_GRAPH,
        "requires the composed A1 packet and complete_paths to already exist",
        runtime_safe=True),
    "cost_already_spent": FeatureSpec(
        "cost_already_spent", AvailabilityStage.POST_GENERATION,
        "A0's own actual latency/tokens -- only known after A0 has already "
        "run to completion",
        runtime_safe=False),
    #: PRE_DECISION features -- admissible, but EMPIRICALLY WEAK for
    #: Executive v0's actual training split (data/hrm/exec_training_v1):
    #: both families ("What is the {relation} for {subject}?") share the
    #: same b3-native template shape, so length/digit-presence carry
    #: near-zero discriminative signal between them. Kept declared for
    #: completeness/honesty about what was considered, not because they are
    #: expected to matter here.
    "question_length_tokens": FeatureSpec(
        "question_length_tokens", AvailabilityStage.PRE_DECISION,
        "whitespace token count of the raw question string",
        runtime_safe=True),
    "question_has_explicit_numeric_literal": FeatureSpec(
        "question_has_explicit_numeric_literal", AvailabilityStage.PRE_DECISION,
        "regex check for a numeric literal in the question -- was a "
        "candidate signal for the original arithmetic-flavored D0 family, "
        "which Executive v0's training split does not include",
        runtime_safe=True),
    #: POST_ANSWER_NOW_PRE_MEMORY features -- Executive v0's actual signal.
    #: hrm_adaptive_memory.executive.confidence.generate_with_confidence()
    #: computes these from ANSWER_NOW's own greedy generation.
    "answer_now_mean_token_confidence": FeatureSpec(
        "answer_now_mean_token_confidence", AvailabilityStage.POST_ANSWER_NOW_PRE_MEMORY,
        "mean per-step softmax probability of ANSWER_NOW's own greedily-"
        "chosen tokens -- requires ANSWER_NOW's generation to have run "
        "(cheap: no retrieval/graph/composition), not MEMORY's",
        runtime_safe=True),
    "answer_now_min_token_confidence": FeatureSpec(
        "answer_now_min_token_confidence", AvailabilityStage.POST_ANSWER_NOW_PRE_MEMORY,
        "the single lowest per-step confidence during ANSWER_NOW's "
        "generation -- same cost as the mean variant",
        runtime_safe=True),
}
