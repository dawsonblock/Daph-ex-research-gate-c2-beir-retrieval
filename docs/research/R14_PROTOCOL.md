# R14 Protocol — External Cognitive Operator Qualification

## 1. Executive Summary & Scientific Intent

### 1.1 The Shift in Architectural Role
DAPH-X is not a reasoning engine; it does not need to invent bespoke samplers, verifiers, critics, or uncertainty heuristics. Mature external test-time reasoning systems already provide sophisticated implementations:
- **ThinkBooster**: Unified test-time compute engine packaging Best-of-N, Beam Search, DeepConf, MUR, phi-decoding, uncertainty CoT, and adaptive scaling with PRM scorers and an OpenAI-compatible service.
- **OptiLLM**: OpenAI-compatible inference proxy exposing 20+ test-time strategies (Best-of-N, MCTS, PlanSearch, self-consistency, prover/verifier, Mixture-of-Agents).
- **DeepConf**: Official implementation of confidence-aware parallel reasoning and dynamic early stopping.
- **MUR**: Stepwise and temporal uncertainty tracking for dynamic compute allocation.
- **Plan-and-Budget**: Training-free structured decomposition and local token-budget allocation.
- **PaCoRe**: Coordinated multi-trajectory parallel reasoning and learned synthesis (expensive escalation tier).

### 1.2 The Core Question
DAPH-X is an **external, cost-aware cognitive compute executive**. It answers:
> Given the current problem, observable model state, uncertainty, available budget, and prior evidence, which existing reasoning mechanism should be invoked, if any?

R14 establishes whether external test-time reasoning systems materially dominate DAPH-X native operators, and whether an external heterogeneous oracle demonstrates sufficient state-dependent complementary value over the best single external strategy to justify an executive:

$$\Delta_{\text{executive}} = J_{\text{external heterogeneous oracle}} - J_{\text{best single external strategy}}$$

---

## 2. Preserved Harness Assets

The methodological and verification assets built across R12 and R13 remain strictly in force:
- `RuntimeState` with strict isolation from evaluation ground truth.
- `EvaluationLabels` boundary and canonical answer judging (`daph_x/evaluation/answer_judge.py`).
- Immutable checkpoint hashing (`state_hash`, `checkpoint_sha256`).
- Canonical candidate representation and R12 selector (`select_r12_maxcal`).
- Deterministic append-only receipts with backend provenance and execution IDs.
- Resumable tournament runners with idempotent deduplication.
- Multidimensional cost accounting and fail-closed integrity checks.
- Task-clustered bootstrap with explicit multiplicity weighting.
- Multi-seed replicated qualification protocols (`{42, 123, 2024}`).

---

## 3. Native Operator Reclassification

Existing DAPH-X bespoke operators are preserved as frozen reference baselines:
- `STOP`: Permanent zero-cost baseline.
- `VERIFY_TARGETED_V2`: Native reference targeted verifier.
- `SAMPLE_STANDARD_V2`: Native reference sampling continuation.
- `CRITIQUE_RETRY_V2`: Native reference critic.
- `SAMPLE_DIVERSE_V2`: Retired from active candidate pools (0 oracle selections in R13-A v2.3).

No further engineering effort will be spent modifying native operators until external operators are qualified.

---

## 4. Generic Cognitive Operator ABI

All reasoning mechanisms—native and external—must satisfy a uniform contract.

### 4.1 Operator Specification (`OperatorSpec`)
```python
@dataclass(frozen=True)
class OperatorSpec:
    operator_id: str
    operator_version: str
    provider: str                      # "daph_native", "thinkbooster", "optillm", "deepconf", etc.
    strategy: str                      # "stop", "verify", "best_of_n", "deepconf", "mur", "plansearch", etc.
    strategy_version: str
    state_mode: StateMode              # FRESH_SOLVE, STATE_CONDITIONED, CANDIDATE_RERANK, CONTINUATION, VERIFICATION, PARALLEL_SEARCH
    tier: int                          # 0 (free/stop), 1 (cheap/fast), 2 (moderate), 3 (expensive)
    backend_requirements: tuple[str, ...]  # ("openai_compatible",) or ("vllm_whitebox", "logprobs")
    supports_budget: bool
    supports_seed: bool
    requires_logprobs: bool
    requires_whitebox: bool
    max_context_tokens: int
    provenance: dict[str, Any]
```

### 4.2 State Modes (`StateMode`)
External systems do not all behave as simple continuations of an existing generation prefix:
- `FRESH_SOLVE`: Solves the problem from the raw task prompt (e.g. Plan-and-Budget, PaCoRe, independent Best-of-N).
- `STATE_CONDITIONED`: Uses the problem prompt plus observable state summary / prior candidates as context.
- `CANDIDATE_RERANK`: Takes existing candidates from `RuntimeState` and reranks / selects without new generation.
- `CONTINUATION`: Generates additional candidate trajectories appending to state (e.g. R12 sampling).
- `VERIFICATION`: Audits the current baseline answer and optionally suggests repair (e.g. `VERIFY_TARGETED`).
- `PARALLEL_SEARCH`: Runs parallel exploration trajectories and synthesizes an answer (e.g. Beam, ToT, MCTS).

### 4.3 Multidimensional Cost Vector (`CostVector`)
Cost is tracked across multiple physical and economic dimensions:
```python
@dataclass(frozen=True)
class CostVector:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    model_calls: int = 1
    wall_ms: float = 0.0
    gpu_ms: float | None = None
    gpu_memory_peak_mb: float | None = None
    api_cost_usd: float | None = None
    estimated_flops: float | None = None
```
**Rule**: Unmeasured dimensions must be recorded as `None` (`NOT_MEASURED`), never as `0.0`. Only free operations (like `STOP`) record explicit zero cost.

### 4.4 Standardized Operator Result (`OperatorResult`)
```python
@dataclass(frozen=True)
class OperatorResult:
    terminal_answer: str
    candidates: list[Candidate]
    reasoning_artifacts: dict[str, Any]
    diagnostics: dict[str, Any]
    status: str                        # "SUCCESS", "FAILURE", "TIMEOUT", "NOT_SUPPORTED", "BUDGET_EXCEEDED"
    error_code: str | None = None
    error_message: str | None = None
    cost: CostVector = field(default_factory=CostVector)
    provenance: dict[str, Any] = field(default_factory=dict)
```

### 4.5 Base Interface (`CognitiveOperator`)
```python
class CognitiveOperator(ABC):
    @property
    @abstractmethod
    def spec(self) -> OperatorSpec: ...

    @abstractmethod
    def estimate_cost(self, state: RuntimeState, budget: BudgetEnvelope | None = None) -> CostVector: ...

    @abstractmethod
    def execute(
        self,
        state: RuntimeState,
        backend: ExternalBackend,
        budget: BudgetEnvelope | None = None,
        replicate_id: int = 42,
    ) -> OperatorResult: ...
```

---

## 5. Execution Lanes & Backend Routing

Two distinct execution lanes are supported:
- **Lane A: Black-Box / OpenAI-Compatible (Primary for R14-A/B)**
  - Operates against any OpenAI-compatible API endpoint (`v1/chat/completions` or `v1/completions`).
  - Supports local `llama-server` / `llama.cpp`, ThinkBooster service, OptiLLM proxy, Ollama, vLLM OpenAI server, and remote providers.
  - Required for initial screening on local Apple Silicon hardware.
- **Lane B: White-Box / Engine-Integrated (Secondary for deep MUR/DeepConf)**
  - Requires logprob distributions, KV-cache intervention, or token-level confidence access.
  - Enabled when vLLM or Hugging Face engines with CUDA/MPS logprob support are available.

---

## 6. Discrete Operator Profiles

To prevent combinatorial hyperparameter explosion, operators are exposed to the executive as discrete, frozen profiles:
- `STOP`: Permanent baseline, 0 tokens.
- `VERIFY_NATIVE`: DAPH native targeted verifier.
- `SAMPLE_NATIVE`: DAPH native standard 2-sample continuation.
- `TB_BON_LOW`: ThinkBooster Best-of-N ($N=4$, temperature=0.7, majority vote).
- `TB_BON_MED`: ThinkBooster Best-of-N ($N=8$, temperature=0.7, majority vote).
- `TB_DEEPCONF_LOW`: ThinkBooster DeepConf (early stopping threshold 0.6).
- `TB_MUR_LOW`: ThinkBooster MUR (stepwise uncertainty threshold).
- `TB_BEAM_LOW`: ThinkBooster Beam Search (width=3, depth=3).
- `OPT_PLANSEARCH_LOW`: OptiLLM PlanSearch (budget=3).
- `OPT_MOA_LOW`: OptiLLM Mixture-of-Agents (layers=2, temperature=0.7).
- `PLAN_BUDGET_DEFAULT`: Plan-and-Budget default decomposition.
- `PACORE_EXPENSIVE`: PaCoRe multi-round parallel reasoning (Tier 3 escalation).

---

## 7. Admission Gate for External Operators

An external operator cannot be routed or included in tournament evaluations until it passes:
- **G0 (Adapter Conformance)**: 100% valid `OperatorResult` schema on test prompts, seed reproducibility, error handling.
- **G1 (Zero Oracle Leakage)**: No task ground truth accessed, imported, or used in execution or selection.
- **G2 (Deterministic Provenance)**: Model ID, server URL, strategy version, git commit recorded in receipts.
- **G3 (Complete Cost Metering)**: Token counts, model calls, and wall clock measured and non-zero (for active operators).
- **G4 (Non-Dominance)**: Not strictly dominated by an existing cheaper operator in initial screening.

---

## 8. Experimental Roadmap: R14

### 8.1 R14-A: Adapter Conformance & Smoke Qualification
- Implement `daph_x/operators/external/base.py`.
- Implement `ThinkBooster` service adapter (`daph_x/operators/external/thinkbooster.py`).
- Implement `OptiLLM` proxy adapter (`daph_x/operators/external/optillm.py`).
- Implement `OperatorRegistry` (`daph_x/executive/registry.py`) and `BudgetEnvelope` (`daph_x/executive/budget.py`).
- Run conformance test suite verifying G0–G3 across 5 smoke checkpoints.

### 8.2 R14-B: Broad Pareto Screening (Development Tasks)
- Screen candidate profiles on the 90 R13-A development checkpoints (replicate 42).
- Measure multi-objective metrics: accuracy, token cost, model calls, wall latency, rescue/break/waste rates.
- Identify the empirical Pareto frontier across Accuracy vs. Tokens vs. Latency.
- Strictly eliminate dominated profiles.

### 8.3 R14-C: Replicated Tournament & Existence Gate
- Evaluate the surviving non-dominated profiles across the 90 frozen checkpoints × 3 seeds (`{42, 123, 2024}`).
- Compute replicated $\bar{Q}(s, a)$ and $\bar{c}(s, a)$.
- Evaluate the DAPH-X Existence Gate:
  $$\Delta_{\text{executive}}(\lambda) = J_{\text{heterogeneous external oracle}}(\lambda) - J_{\text{best single external strategy}}(\lambda)$$

### 8.4 Outcome Branching & Decision Rules
- **Outcome 1 (Single External Strategy Dominates)**:
  If $J_{\text{oracle}} - J_{\text{best}} < 0.005$ with $UCB_{95} < 0.005$ across all $\lambda$, terminate learned executive development. Adopt the winning external strategy directly. DAPH-X serves purely as an evaluation harness and budget gateway.
- **Outcome 2 (Heterogeneity Exists but is Unpredictable)**:
  If $J_{\text{oracle}} - J_{\text{best}} \ge 0.005$ but state features show no predictive association with action advantages, adopt a static or simple heuristic allocation rule. Do not build a learned router.
- **Outcome 3 (Predictable Heterogeneity)**:
  If $J_{\text{oracle}} - J_{\text{best}} \ge 0.005$ and state features reliably predict action value differences, proceed to R15 (Constrained-Compute Executive Learning).

---

## 9. Audit Trail & Provenance
- `R13A_v2_3_REPLICATED_RESULT.md`: Frozen R13-A results (`86b898f`).
- `R13C_PROTOCOL.md` & `Addendum 1`: Frozen reference branch (`3a66741`).
- `R14_PROTOCOL.md`: This document.
