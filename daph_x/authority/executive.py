"""Executive scoring and authority selection for DAPH-X.

The executive scores each candidate action and selects the best one
with an appropriate authority mode.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.actions.typed_actions import Action, ActionType
from daph_x.actions.candidate_generator import generate_and_prune
from daph_x.authority.executive_decision import ExecutiveDecision, AuthorityMode
from daph_x.belief.belief_engine import BeliefState, compute_belief_state
from daph_x.graph.epistemic_graph import EpistemicGraph
from daph_x.world_model.transition_model import transition_model


class ExecutiveConfig:
    """Configuration for the DAPH-X executive."""
    def __init__(
        self,
        lambda_sigma: float = 1.0,   # Model uncertainty weight
        lambda_i: float = 1.0,       # Information value weight
        lambda_c: float = 0.5,       # Cost weight
        lambda_r: float = 1.0,       # Risk weight
        force_threshold: float = 5.0,  # LCB threshold for FORCE
        risk_threshold: float = 0.1,   # Risk threshold for FORCE
    ):
        self.lambda_sigma = lambda_sigma
        self.lambda_i = lambda_i
        self.lambda_c = lambda_c
        self.lambda_r = lambda_r
        self.force_threshold = force_threshold
        self.risk_threshold = risk_threshold


def score_action(
    graph: EpistemicGraph,
    action: Action,
    belief: BeliefState,
    config: ExecutiveConfig,
) -> float:
    """Score a candidate action.

    Score(s,a) = μ_Q(s,a) - λ_σ·σ_Q(s,a) + λ_I·IG(s,a) - λ_C·C(s,a) - λ_R·Risk(s,a)

    For now, uses simple heuristics. Will be replaced by learned models.
    """
    # Base value from action type
    base_values = {
        ActionType.ANSWER: 100.0 if belief.readiness.value == "ANSWER_READY" else -50.0,
        ActionType.DEFER: 50.0 if belief.readiness.value == "DEFER_READY" else -30.0,
        ActionType.STOP: -10.0,
        ActionType.VERIFY: 30.0,
        ActionType.RETRIEVE: 20.0,
        ActionType.SEARCH: 15.0,
        ActionType.COMPARE: 10.0,
        ActionType.CHECK_CONSISTENCY: 10.0,
    }
    base_value = base_values.get(action.action_type, 0.0)

    # Information gain proxy (target-specific)
    ig = 0.0
    if action.action_type == ActionType.VERIFY:
        # IG proportional to how much this evidence discriminates
        evidence_id = action.target
        if isinstance(evidence_id, str):
            edges = graph.evidence_edges(evidence_id)
            n_supported = sum(1 for e in edges if e.edge_type.value == "supports")
            n_contradicted = sum(1 for e in edges if e.edge_type.value == "contradicts")
            # Higher IG if evidence discriminates between hypotheses
            ig = (n_supported + n_contradicted) * 10.0

    # Cost
    cost = action.expected_cost

    # Risk (placeholder — will be learned)
    risk = 0.0

    # Uncertainty penalty (placeholder)
    uncertainty = 0.0

    score = (
        base_value
        + config.lambda_i * ig
        - config.lambda_c * cost
        - config.lambda_r * risk
        - config.lambda_sigma * uncertainty
    )
    return score


def select_action(
    graph: EpistemicGraph,
    llm_proposal: str | None = None,
    config: ExecutiveConfig | None = None,
) -> ExecutiveDecision:
    """Select the best action with appropriate authority mode.

    This is the main entry point for the DAPH-X executive.
    """
    if config is None:
        config = ExecutiveConfig()

    # Compute belief state
    belief = compute_belief_state(graph)

    # Generate and prune candidates
    candidates = generate_and_prune(graph)

    if not candidates:
        return ExecutiveDecision(
            selected_action=Action(action_type=ActionType.STOP, target="no_candidates"),
            authority_mode=AuthorityMode.ABSTAIN,
            expected_value=0.0,
            value_margin=0.0,
            value_lcb=0.0,
            model_uncertainty=1.0,
            epistemic_uncertainty=1.0,
            transition_uncertainty=1.0,
            intervention_risk=1.0,
            structural_certificate=False,
            certificate_type=None,
            rationale="No candidates generated",
            state_hash=graph.graph_hash(),
            candidate_count=0,
        )

    # Score all candidates
    scores = {}
    for action in candidates:
        scores[str(action)] = score_action(graph, action, belief, config)

    # Select best action
    best_action_str = max(scores, key=scores.get)
    best_action = next(a for a in candidates if str(a) == best_action_str)
    best_score = scores[best_action_str]

    # Compute value margin
    sorted_scores = sorted(scores.values(), reverse=True)
    value_margin = best_score - sorted_scores[1] if len(sorted_scores) > 1 else best_score

    # Determine authority mode
    authority_mode = _determine_authority_mode(
        best_score, value_margin, belief, config,
    )

    # Check structural certificate
    structural_certificate = False
    certificate_type = None
    if belief.unique_supported and belief.readiness.value == "ANSWER_READY":
        structural_certificate = True
        certificate_type = "unique_verified_support_answer"

    # LLM proposal comparison
    llm_proposal_score = None
    intervention_advantage = None
    if llm_proposal and llm_proposal in scores:
        llm_proposal_score = scores[llm_proposal]
        intervention_advantage = best_score - llm_proposal_score

    return ExecutiveDecision(
        selected_action=best_action,
        authority_mode=authority_mode,
        expected_value=best_score,
        value_margin=value_margin,
        value_lcb=best_score - config.lambda_sigma * 1.0,  # Placeholder LCB
        model_uncertainty=0.1,  # Placeholder
        epistemic_uncertainty=belief.entropy,
        transition_uncertainty=0.1,  # Placeholder
        intervention_risk=0.05,  # Placeholder
        structural_certificate=structural_certificate,
        certificate_type=certificate_type,
        rationale=f"Best action by score: {best_action_str} ({best_score:.1f})",
        state_hash=graph.graph_hash(),
        candidate_count=len(candidates),
        action_scores=scores,
        llm_proposal=llm_proposal,
        llm_proposal_score=llm_proposal_score,
        intervention_advantage=intervention_advantage,
    )


def _determine_authority_mode(
    best_score: float,
    value_margin: float,
    belief: BeliefState,
    config: ExecutiveConfig,
) -> AuthorityMode:
    """Determine the appropriate authority mode."""
    # FORCE if high confidence and low risk
    if (value_margin > config.force_threshold
            and belief.readiness.value == "ANSWER_READY"):
        return AuthorityMode.FORCE

    # ADVISE if we have a clear preference but not enough confidence
    if value_margin > 0:
        return AuthorityMode.ADVISE

    # ABSTAIN if no clear preference
    return AuthorityMode.ABSTAIN
