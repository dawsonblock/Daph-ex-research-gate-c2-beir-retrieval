# V2B-I3.2.2 protocol finalization

V2B-I3.2.2 freezes metareasoning evaluation semantics without issuing a V2B
scientific verdict. The primary benchmark-level prior is `TASK_UNIFORM`; a
`CLASS_UNIFORM` view is retained as a representation diagnostic. Every
aggregate information gap, decision gap, and total regret is labeled with its
prior, and both views enforce `TR = IG + DG`.

Step accounting separates nonnegative `action_cost`, `immediate_reward`, and
`terminal_reward`. Net utility is reward minus cost. Policy telemetry describes
observable interventions rather than inferring controller intent. The
one-step resolution diagnostic is named `one_step_full_resolution_rate` and
means the fraction of root actions whose every outcome is nonterminal and a
singleton posterior belief.

The canonical artifact resolver closes the benchmark graph over its manifest,
private environment, public packets, extensions, protocol, masks, policy,
utility, and resource profiles. Qualification must hash that resolved graph;
handwritten parallel artifact lists are not authoritative.

Status:

```text
METHODOLOGY IMPLEMENTED       yes
DEVELOPMENT TESTED            yes
SCIENTIFICALLY QUALIFIED      no
MODEL EXECUTIVE ENABLED       no
V2B SCIENTIFIC VERDICT        no
```
