"""DAPH-X: Parameterized Model-Based Executive.

A clean-break redesign of DAPH. The LLM is no longer the primary policy.
The executive selects from typed parameterized actions using:
  - Canonical symbolic topology (inherited from V3R2)
  - Belief engine (calibrated P(H|E))
  - Model-based transition layer (P(o|s,a))
  - Target-specific information value
  - Uncertainty + risk estimation
  - Structural certificates as invariants (not the policy)

Architecture:
  Observations → Epistemic Graph → Canonical State → Candidate Actions
    → Consequence Evaluator → Executive Selector → Action

The executive chooses among concrete parameterized actions:
  ANSWER(h), DEFER(r), VERIFY(e), RETRIEVE(q), SEARCH(q),
  TEST(t), COMPARE(h1,h2), CHECK_CONSISTENCY(target), STOP(r)

Authority modes:
  OBSERVE, ADVISE, CONSTRAIN, FORCE, ABSTAIN

V3R2 is preserved as the frozen control baseline, not a design constraint.
"""

__version__ = "0.1.0"
