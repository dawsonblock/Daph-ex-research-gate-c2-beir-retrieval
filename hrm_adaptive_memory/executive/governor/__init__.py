"""General Governor: model-based executive layer for structure-general metareasoning.

The governor sits between ControllerObservation and the model's ActionProposal.
It constructs a decision frame that helps the model reason about action
consequences rather than reacting to state labels.

Architecture:
    ControllerObservation
            ↓
    GeneralGovernor.assess()
        ├── enumerate legal actions
        ├── detect current bottleneck
        ├── predict bounded outcomes per action
        ├── estimate information gain / progress / cost / risk
        ├── detect redundancy (repeated no-gain actions)
        ├── score candidates with topology-invariant features
        └── build GovernorDecisionFrame
            ↓
    Model (DeepSeek) chooses from the frame
            ↓
    ActionProposal
            ↓
    policy/resource gate → executor

The governor uses ONLY controller-visible information (G_M(s)).
It never accesses latent state, oracle values, or topology IDs.
"""
