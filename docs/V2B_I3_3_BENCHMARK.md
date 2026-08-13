# V2B-I3.3 Scaled Frozen Metareasoning Benchmark

V2B-I3.3 scales the deterministic seven-action environment after the I3.2.2
methodology freeze. It does not add a model controller and is not a scientific
V2B result.

The concrete corpus contains 750 immutable task instances: 300 development,
150 validation, and 300 held-out. It covers verification, temporal validity,
provenance lineage, conflict, decision history, composition, irreducible
partial observability, state-irrelevant controls, and budget-conditioned
pairs. Concrete JSON is authoritative; the deterministic generator and seed
are provenance for the frozen bytes rather than an instruction to regenerate
held-out evaluation data.

The benchmark uses the I3.2.2 protocol's seven actions and utility semantics.
Each controller condition changes only the observation mask. Oracle cache
records bind latent and sequential observable ground truth without enabling a
model-controller claim.

Status: **FROZEN BENCHMARK, NOT A SCIENTIFIC RESULT**.
