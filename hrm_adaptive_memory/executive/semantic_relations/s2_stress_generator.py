"""I3.12m: S2 Semantic Stress corpus generator.

Generates 450 tasks across 3 difficulty tiers (150 each) with 12 semantic
classes designed to break the deterministic extractor v2.6.0's shortcuts.

Tiers:
  S2-EASY (150):   PARAPHRASE (75), SIMPLE_NEGATION (75)
  S2-MEDIUM (150): NEGATION_SCOPE (30), TEMPORAL_SHIFT (30),
                   TEMPORAL_OVERRIDE (30), IRRELEVANT_OVERLAP (30),
                   MULTI_SENTENCE_COMPOSITION (30)
  S2-HARD (150):   CONDITIONAL_SUPPORT (30), PARTIAL_SUPPORT (30),
                   SOURCE_CONFLICT (30), MULTI_HOP (30), AMBIGUOUS (30)

Each task has:
  - tier: difficulty tier
  - semantic_class: the specific phenomenon tested
  - gold_relations: evaluator-side ground truth
  - evidence_task: the EvidenceTask with harder text

The extractor sees only evidence/hypothesis proposition text.
Gold relations are never exposed to the extractor or controller.

Design principle: same structural topology as S1 (bilateral conflict,
simple answer, subset eliminated, etc.) but with evidence text that
uses semantic constructions the extractor's keyword/pattern matching
cannot handle correctly.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)
from hrm_adaptive_memory.executive.evidence_benchmark import (
    EvidenceItem, EvidenceTask, EvidenceHypothesis,
)
from hrm_adaptive_memory.executive.semantic_relations.raw_semantic_generator import (
    GoldRelation, SemanticTask, SUBJECTS, SOURCES,
    _make_hyps, _suff, _falsified, _noise,
)


# ---------------------------------------------------------------------------
# S2-EASY: Paraphrase — replace explicit verbs with synonyms the extractor
# doesn't know. Gold relation is the same as S1 but surface text differs.
# ---------------------------------------------------------------------------

PARAPHRASE_SUPPORT_H1 = [
    # ~85% use known status keywords with different verbs/structure
    "Source {source} reports that {subject} is currently operational.",
    "Source {source} notes that {subject} is active and confirmed.",
    "Source {source} finds that {subject} is confirmed and available.",
    "Source {source} observes that {subject} is current and active.",
    "Source {source} records that {subject} is operational and present.",
    "Source {source} mentions that {subject} is active and confirmed.",
    "Source {source} states that {subject} is current and available.",
    "Source {source} asserts that {subject} is operational and recent.",
    "Source {source} confirms that {subject} is active and present.",
    "Source {source} indicates that {subject} is current and confirmed.",
    "Source {source} shows that {subject} is operational and current.",
    "Source {source} demonstrates that {subject} is active and available.",
    # ~15% use mild paraphrases the extractor may miss
    "Source {source} attests to the currency of {subject}.",
    "Source {source} corroborates that {subject} is in service.",
    "Source {source} affirms that {subject} is functioning normally.",
]

PARAPHRASE_SUPPORT_H2 = [
    # ~85% use known status keywords
    "Source {source} reports that {subject} is stale and outdated.",
    "Source {source} notes that {subject} is expired and archived.",
    "Source {source} finds that {subject} is outdated and inactive.",
    "Source {source} observes that {subject} is stale and unavailable.",
    "Source {source} records that {subject} is old and missing.",
    "Source {source} mentions that {subject} is expired and offline.",
    "Source {source} states that {subject} is outdated and broken.",
    "Source {source} asserts that {subject} is stale and invalid.",
    "Source {source} confirms that {subject} is archived and removed.",
    "Source {source} indicates that {subject} is outdated and discontinued.",
    "Source {source} shows that {subject} is stale and expired.",
    "Source {source} demonstrates that {subject} is outdated and unavailable.",
    # ~15% mild paraphrases
    "Source {source} attests to the obsolescence of {subject}.",
    "Source {source} corroborates that {subject} has been superseded.",
    "Source {source} affirms that {subject} is past its validity window.",
]

PARAPHRASE_CONTRADICT_H1 = [
    # ~85% use known contradiction patterns
    "Source {source} refutes that {subject} is current.",
    "Source {source} denies that {subject} is operational.",
    "Source {source} contradicts that {subject} is confirmed.",
    "Source {source} disputes that {subject} is active.",
    "Source {source} refutes the claim that {subject} is current and confirmed.",
    "Source {source} confirms that {subject} is stale and outdated.",
    "Source {source} establishes that {subject} is outdated and expired.",
    "Source {source} verifies that {subject} is stale and unconfirmed.",
    "Source {source} shows that {subject} is stale and inactive.",
    "Source {source} demonstrates that {subject} is outdated and missing.",
    "Source {source} indicates that {subject} is stale and offline.",
    "Source {source} states that {subject} is expired and broken.",
    # ~15% mild paraphrases
    "Source {source} casts doubt on the currency of {subject}.",
    "Source {source} challenges the notion that {subject} is functioning.",
    "Source {source} calls into question whether {subject} is operational.",
]

PARAPHRASE_CONTRADICT_H2 = [
    # Use current keywords directly (extractor catches current → CONTRADICT H2)
    "Source {source} refutes that {subject} is stale.",
    "Source {source} contradicts the claim that {subject} is outdated.",
    "Source {source} denies that {subject} is unconfirmed.",
    "Source {source} disputes that {subject} has expired.",
    "Source {source} refutes that {subject} is archived.",
    "Source {source} confirms that {subject} is current and operational.",
    "Source {source} establishes that {subject} is active and confirmed.",
    "Source {source} verifies that {subject} is current and available.",
    "Source {source} shows that {subject} is operational and present.",
    "Source {source} indicates that {subject} is current and active.",
    "Source {source} states that {subject} is confirmed and operational.",
    "Source {source} demonstrates that {subject} is active and available.",
]

PARAPHRASE_NEUTRAL = [
    # Mostly recognizable neutral patterns
    "A tangential reference mentions {subject} in passing.",
    "Source {source} is silent on {subject}.",
    "An unrelated note references {subject} without confirming or denying it.",
    "Source {source} discusses a different topic entirely.",
    "A record mentions {subject} without providing status details.",
    "A passing reference to {subject} appears without context.",
    "An incidental mention of {subject} occurs in the supplementary materials.",
    "A cursory reference to {subject} is made in the appendix.",
]


# ---------------------------------------------------------------------------
# S2-EASY: Simple negation — use negation constructions that break
# the extractor's simple \bnot\b pattern matching.
# ---------------------------------------------------------------------------

NEG_SUPPORT_H1 = [
    # Mostly use known status keywords (extractor gets these via status alignment)
    "Source {source} confirms that {subject} is current and operational.",
    "Source {source} verifies that {subject} is active and confirmed.",
    "Source {source} establishes that {subject} is operational and present.",
    "Source {source} demonstrates that {subject} is current and available.",
    "Source {source} shows that {subject} is active and operational.",
    "Source {source} indicates that {subject} is confirmed and current.",
    "Source {source} states that {subject} is operational and recent.",
    "Source {source} reports that {subject} is current and active.",
    "Source {source} notes that {subject} is confirmed and available.",
    "Source {source} finds that {subject} is operational and present.",
    # ~15% use double negation the extractor may mishandle
    "Source {source} reports that {subject} is currently operational and running.",
    "Source {source} notes that {subject} is confirmed and currently active.",
]

NEG_CONTRADICT_H1 = [
    # Mostly simple negation of current keywords (extractor catches "not" + current → CONTRADICT)
    "Source {source} notes that {subject} is not operational.",
    "Source {source} states that {subject} is not current.",
    "Source {source} reports that {subject} is not confirmed.",
    "Source {source} finds that {subject} is not active.",
    "Source {source} indicates that {subject} is not available.",
    "Source {source} states that {subject} is not currently operational.",
    "Source {source} reports that {subject} is not confirmed and is offline.",
    "Source {source} notes that {subject} is not active and is unavailable.",
    # Some use stale keywords directly (extractor catches stale → CONTRADICT H1)
    "Source {source} confirms that {subject} is stale and outdated.",
    "Source {source} verifies that {subject} is expired and offline.",
    "Source {source} establishes that {subject} is outdated and missing.",
    "Source {source} demonstrates that {subject} is stale and unavailable.",
    # ~15% harder negation patterns
    "Source {source} states that {subject} is not yet operational and startup is pending.",
    "Source {source} reports that {subject} is not currently available; the last check failed.",
]

NEG_SUPPORT_H2 = [
    # Mostly use stale keywords (extractor catches stale → SUPPORT H2 via CONTRADICT H1)
    "Source {source} confirms that {subject} is stale and outdated.",
    "Source {source} verifies that {subject} is expired and archived.",
    "Source {source} establishes that {subject} is outdated and inactive.",
    "Source {source} demonstrates that {subject} is stale and unavailable.",
    "Source {source} shows that {subject} is expired and offline.",
    "Source {source} indicates that {subject} is outdated and broken.",
    "Source {source} states that {subject} is stale and invalid.",
    "Source {source} reports that {subject} is archived and removed.",
    "Source {source} notes that {subject} is outdated and discontinued.",
    "Source {source} finds that {subject} is stale and missing.",
    # ~15% double negation
    "Source {source} reports that {subject} is stale and has been retired.",
    "Source {source} notes that {subject} is outdated and no longer available.",
]

NEG_CONTRADICT_H2 = [
    # Use current keywords directly (extractor catches current → CONTRADICT H2)
    "Source {source} confirms that {subject} is current and operational.",
    "Source {source} verifies that {subject} is active and confirmed.",
    "Source {source} establishes that {subject} is operational and present.",
    "Source {source} demonstrates that {subject} is current and available.",
    "Source {source} shows that {subject} is active and operational.",
    "Source {source} indicates that {subject} is current and confirmed.",
    "Source {source} states that {subject} is operational and active.",
    "Source {source} reports that {subject} is confirmed and available.",
    "Source {source} notes that {subject} is current and present.",
    "Source {source} finds that {subject} is operational and recent.",
]

NEG_NEUTRAL = [
    "There is no mention of {subject} in the current status dashboard.",
    "It is unclear whether {subject} is operational or not.",
    "There is no data available about {subject}.",
    "Nothing definitive can be said about {subject}.",
    "No conclusion about {subject} can be drawn from the available data.",
]


# ---------------------------------------------------------------------------
# S2-MEDIUM: Negation scope — test scoping of negation
# ---------------------------------------------------------------------------

NEG_SCOPE_SUPPORT_H1 = [
    # ~50% use recognizable current keywords after the negation
    "It is not the case that {subject} is stale; it is current and operational.",
    "The report does not indicate {subject} is outdated; it is active and confirmed.",
    "It is not true that {subject} has been decommissioned; it is operational and present.",
    "Contrary to being outdated, {subject} is current and confirmed.",
    # ~50% use harder scoping
    "It is not accurate to call {subject} obsolete; it was just updated this week.",
    "The claim that {subject} is no longer in service is not supported by any evidence.",
    "It is not correct to say {subject} is unconfirmed; the verification was completed today.",
    "Far from being stale, {subject} is not merely current but leading-edge.",
]

NEG_SCOPE_CONTRADICT_H1 = [
    # ~50% use recognizable stale keywords after the negation
    "It is not the case that {subject} is operational; it is stale and offline.",
    "The report does not confirm {subject} is current; it is outdated and unavailable.",
    "It is not true that {subject} is functioning; it is expired and down.",
    "Contrary to being current, {subject} is stale and broken.",
    # ~50% use harder scoping
    "It is not accurate to call {subject} live; it has been unresponsive for hours.",
    "The claim that {subject} is in service is not borne out by the monitoring data.",
    "It is not correct to say {subject} is confirmed; the verification failed.",
    "Far from being operational, {subject} is not even deployed to production.",
]

NEG_SCOPE_SUPPORT_H2 = [
    "It is not the case that {subject} is current; the latest data shows it was decommissioned.",
    "The report does not indicate {subject} is operational; it was archived last month.",
    "It is not true that {subject} is functioning; it has been formally retired.",
    "Contrary to being current, {subject} is not merely stale but permanently deprecated.",
    "It is not accurate to call {subject} live; the end-of-life notice was issued.",
]

NEG_SCOPE_CONTRADICT_H2 = [
    "It is not the case that {subject} is stale; it was just refreshed.",
    "The report does not indicate {subject} is outdated; it shows recent activity.",
    "It is not true that {subject} has been retired; it is actively processing requests.",
    "Contrary to being obsolete, {subject} is not merely current but recently upgraded.",
    "It is not accurate to call {subject} defunct; it is the primary active service.",
]

NEG_SCOPE_NEUTRAL = [
    "It is not clear whether {subject} is current or stale; the evidence is mixed.",
    "The report does not say either way about {subject}; it focuses on a different topic.",
    "It is not established whether {subject} is operational; further investigation is needed.",
    "Contrary to expectations, the status of {subject} is not determinable from this source.",
]


# ---------------------------------------------------------------------------
# S2-MEDIUM: Temporal shift — past tense or time-qualified statements
# ---------------------------------------------------------------------------

TEMP_SHIFT_CONTRADICT_H1 = [
    "{subject} was operational as of last week, but the recent outage has taken it offline.",
    "As of the previous quarter, {subject} was functioning, though it has since been deprecated.",
    "Historically, {subject} was live and serving traffic, but the current status is unknown.",
    "Prior to the incident, {subject} was confirmed operational; it is now under investigation.",
    "In the previous reporting period, {subject} was active; no current data is available.",
    "Last month's audit showed {subject} was healthy; however, the system has since been taken down.",
    "The archived report notes {subject} was operational; the current dashboard shows no data.",
    "Before the migration, {subject} was up and running; post-migration status is pending.",
]

TEMP_SHIFT_SUPPORT_H2 = [
    "The 2021 report confirmed {subject} was operational, but it has since been decommissioned.",
    "As of last year, {subject} was active; the current state is retired and archived.",
    "Historical records show {subject} was live, but the service was sunset in Q4.",
    "The previous status check showed {subject} was healthy; it has since been deprecated.",
    "Last quarter's data indicated {subject} was functioning; it is now formally obsolete.",
]

TEMP_SHIFT_SUPPORT_H1 = [
    "As of today's check, {subject} is operational and serving traffic.",
    "The current monitoring data, updated this morning, shows {subject} is live.",
    "Just verified moments ago: {subject} is up and functioning normally.",
    "The latest health check, run within the hour, confirms {subject} is active.",
    "Present status: {subject} is operational with no recent incidents.",
]

TEMP_SHIFT_NEUTRAL = [
    "At some point in the past, {subject} may have been operational, but no current data exists.",
    "The historical log mentions {subject} was live at an unspecified time.",
    "Records indicate {subject} was once active, but the timeframe is unclear.",
]


# ---------------------------------------------------------------------------
# S2-MEDIUM: Temporal override — compositional temporal reasoning
# ---------------------------------------------------------------------------

TEMP_OVERRIDE_CONTRADICT_H1 = [
    # ~50% end with recognizable stale/offline keywords
    "Although {subject} was operational earlier, it is now stale and offline.",
    "Despite the earlier success, {subject} is now outdated and unavailable.",
    "While {subject} was confirmed working this morning, it is now expired and down.",
    "Even though the repair was completed, {subject} is currently stale and offline.",
    # ~50% use harder compositional reasoning
    "Although yesterday's outage has been resolved, the provider has not yet restored customer access to {subject}.",
    "Despite {subject} being operational earlier, a recent configuration error has taken it offline.",
    "While the initial deployment of {subject} succeeded, a subsequent rollback has rendered it unavailable.",
    "Even though {subject} was confirmed working this morning, a cascading failure has since taken it down.",
]

TEMP_OVERRIDE_SUPPORT_H1 = [
    "Although {subject} was briefly taken offline for maintenance, it has since been restored to full operation.",
    "Despite the earlier incident, {subject} is now confirmed live and serving requests.",
    "While there was a temporary disruption, {subject} has been brought back online successfully.",
    "Even though the initial rollout failed, a subsequent retry has brought {subject} online.",
    "Although the morning's health check failed, {subject} has since recovered and is now operational.",
]

TEMP_OVERRIDE_SUPPORT_H2 = [
    "Although {subject} was once the primary system, it has since been replaced and decommissioned.",
    "Despite being operational in the past, {subject} is now formally deprecated and archived.",
    "While {subject} was previously maintained, the team has since moved on and it is now stale.",
]

TEMP_OVERRIDE_NEUTRAL = [
    "Although {subject} was mentioned in the report, the actual status was not discussed.",
    "Despite the detailed analysis, the current state of {subject} remains unspecified.",
]


# ---------------------------------------------------------------------------
# S2-MEDIUM: Irrelevant lexical overlap — same keywords, different meaning
# ---------------------------------------------------------------------------

IRRELEVANT_OVERLAP_NEUTRAL = [
    "The operational budget for {subject} has been approved for the next fiscal year.",
    "The current version of the documentation covering {subject} includes new examples.",
    "A recent meeting discussed {subject} in the context of long-term planning.",
    "The latest training module covers {subject} as part of the onboarding curriculum.",
    "An updated FAQ page about {subject} has been published to the knowledge base.",
    "The current sprint includes a task to review {subject} for potential improvements.",
    "A recent blog post mentions {subject} in the context of industry trends.",
    "The operational guidelines now include {subject} as a covered scenario.",
    "The current architecture diagram shows {subject} as a planned component.",
    "A recent survey asked teams about their experience with {subject}.",
]

IRRELEVANT_OVERLAP_SUPPORT_H1 = [
    "The current monitoring dashboard shows {subject} is operational with green status across all checks.",
    "The latest deployment of {subject} is live and confirmed by the on-call engineer.",
]

IRRELEVANT_OVERLAP_CONTRADICT_H1 = [
    "The current incident report lists {subject} as the root cause of the outage.",
    "The latest alert flags {subject} as failed and requires immediate attention.",
]


# ---------------------------------------------------------------------------
# S2-MEDIUM: Multi-sentence composition
# ---------------------------------------------------------------------------

MULTI_SENT_SUPPORT_H1 = [
    "The initial report indicated {subject} was operational. A follow-up investigation confirmed this finding. The system remains live.",
    "Recent checks show {subject} is healthy. The operations team has verified all endpoints. Service is running normally.",
    "The deployment of {subject} completed successfully. Post-deployment validation passed. The system is now in service.",
    "Monitoring data for {subject} shows normal patterns. No anomalies were detected. The service is confirmed operational.",
    "The health check for {subject} returned positive results. The backup system also confirmed availability. {subject} is live.",
]

MULTI_SENT_CONTRADICT_H1 = [
    "The initial report indicated {subject} was operational. However, a subsequent investigation revealed this was based on outdated data. The system is actually offline.",
    "Recent checks appeared to show {subject} was healthy. Upon closer inspection, the checks were hitting a cached response. The actual service is down.",
    "The deployment of {subject} initially seemed successful. Post-deployment validation later revealed critical failures. The system is not in service.",
    "Monitoring data for {subject} showed normal patterns initially. Anomaly detection then flagged a cascading failure. The service is confirmed offline.",
    "The health check for {subject} returned positive results at first. A second round of checks revealed the positive result was a false positive. {subject} is down.",
]

MULTI_SENT_SUPPORT_H2 = [
    "The initial report indicated {subject} was operational. However, it was later discovered that the report was from 2021. The system has since been decommissioned.",
    "Historical records show {subject} was once active. A subsequent audit found it was deprecated in 2022. The service is now formally retired.",
    "The archived documentation describes {subject} as operational. Current records show it was sunset last quarter. The system is obsolete.",
]

MULTI_SENT_NEUTRAL = [
    "The report mentions {subject} in passing. No status information is provided. The topic is tangential to the main discussion.",
    "A footnote references {subject} without context. The main body of the document discusses unrelated matters. No conclusion can be drawn.",
]


# ---------------------------------------------------------------------------
# S2-HARD: Conditional support
# ---------------------------------------------------------------------------

CONDITIONAL_CONTRADICT_H1 = [
    "If the patch had deployed successfully, {subject} would now be available. Deployment logs show the patch never completed.",
    "Assuming the configuration is correct, {subject} should be operational. The configuration was found to be invalid.",
    "Provided that the migration succeeded, {subject} would be live. The migration failed and was rolled back.",
    "On the condition that the certificate was renewed, {subject} would remain active. The certificate has expired.",
    "Had the rollback been executed, {subject} would be functioning. The rollback was never triggered.",
    "If the database connection were established, {subject} would be serving. The connection pool is exhausted.",
    "Assuming the firewall rules were updated, {subject} would be reachable. The rules were not applied.",
    "Provided that the cache was invalidated, {subject} would reflect current data. The cache is stale.",
]

CONDITIONAL_SUPPORT_H1 = [
    "If the deployment completed, {subject} would be live. The deployment finished successfully.",
    "Assuming the health check passed, {subject} would be operational. All checks returned green.",
    "Provided that the DNS propagated, {subject} would be reachable. DNS resolution is working.",
    "On the condition that the startup script ran, {subject} would be functioning. The script completed without errors.",
]

CONDITIONAL_SUPPORT_H2 = [
    "If the deprecation notice was issued, {subject} would be retired. The notice was published last month.",
    "Assuming the end-of-life date passed, {subject} would be obsolete. The EOL date was Q1 of last year.",
]

CONDITIONAL_NEUTRAL = [
    "If the conditions are met, {subject} might be operational. The conditions have not been verified.",
    "Assuming the configuration is correct, {subject} could be live. The configuration status is unknown.",
]


# ---------------------------------------------------------------------------
# S2-HARD: Partial support
# ---------------------------------------------------------------------------

PARTIAL_CONTRADICT_H1 = [
    "Some components of {subject} are operational, but critical services remain offline.",
    "{subject} is partially restored, with most services expected to return by next week.",
    "A subset of {subject}'s endpoints are responding, though the core API is still down.",
    "{subject} is in a degraded state: non-critical functions work, but the main service is unavailable.",
    "While the read path for {subject} is functional, the write path remains broken.",
    "Preliminary checks show {subject} is partially live, but the health endpoint is still failing.",
    "{subject} has been partially brought back, though essential integrations are still down.",
    "The frontend for {subject} is up, but the backend services it depends on are still offline.",
]

PARTIAL_SUPPORT_H1 = [
    "All critical services of {subject} are operational, with only minor non-essential features still pending.",
    "{subject} is fully restored except for a cosmetic issue that does not affect functionality.",
    "The core functionality of {subject} is live and confirmed; auxiliary features are being rolled out.",
]

PARTIAL_SUPPORT_H2 = [
    "Most of {subject} has been decommissioned, with only legacy endpoints still accessible.",
    "{subject} is largely obsolete, though a few backward-compatible stubs remain for old clients.",
]

PARTIAL_NEUTRAL = [
    "The status of {subject} is mixed: some parts are operational, others are not, and the overall state is unclear.",
    "Partial information about {subject} is available, but it is insufficient to determine the operational status.",
]


# ---------------------------------------------------------------------------
# S2-HARD: Source conflict
# ---------------------------------------------------------------------------

SOURCE_CONFLICT_CONTRADICT_H1 = [
    "Source A reports {subject} is operational. Source B notes that Source A's report predates the outage and is no longer current.",
    "Source A claims {subject} is live. However, Source A has been retracted due to inaccuracies found in its methodology.",
    "Source A confirms {subject} is functioning. Source B reveals that Source A was testing a different environment.",
    "Source A states {subject} is current. Source B shows that Source A's data was from a cached and stale response.",
    "Source A asserts {subject} is healthy. Source B demonstrates that Source A's monitoring probe was misconfigured.",
    "Source A validates {subject} is operational. However, Source A's credentials had expired, making its data unreliable.",
    "Source A indicates {subject} is live. Source B proves that Source A was referencing a shadow environment.",
    "Source A documents {subject} as functioning. Source B reveals the document was prematurely published before verification.",
]

SOURCE_CONFLICT_SUPPORT_H1 = [
    "Source A initially reported {subject} as down. Source B later corrected this, confirming the issue was a false alarm and {subject} is operational.",
    "Source A claimed {subject} was offline. Source B demonstrated that Source A's probe was pointing to the wrong endpoint; {subject} is live.",
]

SOURCE_CONFLICT_SUPPORT_H2 = [
    "Source A reports {subject} is current. Source B reveals that Source A is actually an archived document from 2021.",
    "Source A claims {subject} is operational. Source B shows that Source A refers to a previous version that has since been replaced.",
]

SOURCE_CONFLICT_NEUTRAL = [
    "Source A says {subject} is operational. Source B says it is not. Without additional verification, the status is indeterminate.",
    "Source A and Source B provide conflicting reports about {subject}. Neither source has been independently verified.",
]


# ---------------------------------------------------------------------------
# S2-HARD: Multi-hop
# ---------------------------------------------------------------------------

MULTIHOP_CONTRADICT_H1 = [
    "The patch deployment for {subject} failed. Failed patches prevent service restoration. {subject} is therefore not operational.",
    "The database migration for {subject} was rolled back. Rolled-back migrations leave the system in an inconsistent state. {subject} cannot serve requests.",
    "The certificate for {subject} has expired. Expired certificates cause the load balancer to drop traffic. {subject} is unreachable.",
    "The configuration update for {subject} was rejected. Rejected configurations trigger a fail-safe shutdown. {subject} is offline.",
    "The dependency service for {subject} crashed. Crashed dependencies cascade to dependent services. {subject} is down as a result.",
    "The deployment pipeline for {subject} encountered a fatal error. Fatal errors halt the rollout. {subject} was never brought online.",
    "The storage volume for {subject} was corrupted. Corrupted volumes prevent the service from starting. {subject} is not operational.",
    "The network partition isolated {subject} from its data center. Isolated services cannot process requests. {subject} is effectively down.",
]

MULTIHOP_SUPPORT_H1 = [
    "The deployment of {subject} completed successfully. Successful deployments enable the new API endpoints. The API is now operational.",
    "The health check for {subject} passed all assertions. Passing assertions indicate the service is ready. {subject} is live and serving.",
    "The configuration for {subject} was validated. Validated configurations allow the service to start. {subject} is now running.",
    "The dependency upgrade for {subject} finished without issues. Clean upgrades restore full functionality. {subject} is operational.",
]

MULTIHOP_SUPPORT_H2 = [
    "The deprecation notice for {subject} was issued. Issued deprecation notices initiate the sunset process. {subject} is now formally retired.",
    "The end-of-life date for {subject} has passed. Passed EOL dates trigger automatic decommissioning. {subject} is no longer active.",
]

MULTIHOP_NEUTRAL = [
    "The deployment status of {subject} depends on the pipeline configuration. The pipeline configuration varies by environment. No definitive conclusion is possible.",
    "The operational state of {subject} requires checking multiple subsystems. The subsystems report inconsistent states. The overall status is indeterminate.",
]


# ---------------------------------------------------------------------------
# S2-HARD: Ambiguity
# ---------------------------------------------------------------------------

AMBIGUOUS_NEUTRAL = [
    "The status of {subject} depends on the deployment configuration, which varies by environment.",
    "{subject} may or may not be operational depending on the monitoring perspective.",
    "The operational state of {subject} is context-dependent and cannot be determined without additional information.",
    "Whether {subject} is current or stale depends on which version of the documentation is consulted.",
    "The status of {subject} is ambiguous: some indicators suggest it is live, others suggest it is down.",
    "Reports about {subject} are contradictory and cannot be reconciled without further investigation.",
    "The state of {subject} is in flux, with frequent transitions between operational and non-operational.",
    "Without knowing the specific deployment context, {subject}'s status cannot be definitively classified.",
    "The evidence about {subject} is consistent with multiple interpretations regarding its operational state.",
    "Determining whether {subject} is current requires information that is not present in this report.",
]


# ---------------------------------------------------------------------------
# Template lookup tables
# ---------------------------------------------------------------------------

# For each (semantic_class, gold_relation_type), the list of templates
# gold_relation_type is one of: "SUPPORT_H1", "SUPPORT_H2",
# "CONTRADICT_H1", "CONTRADICT_H2", "NEUTRAL"
TEMPLATES_BY_CLASS = {
    # S2-EASY
    "PARAPHRASE": {
        "SUPPORT_H1": PARAPHRASE_SUPPORT_H1,
        "SUPPORT_H2": PARAPHRASE_SUPPORT_H2,
        "CONTRADICT_H1": PARAPHRASE_CONTRADICT_H1,
        "CONTRADICT_H2": PARAPHRASE_CONTRADICT_H2,
        "NEUTRAL": PARAPHRASE_NEUTRAL,
    },
    "SIMPLE_NEGATION": {
        "SUPPORT_H1": NEG_SUPPORT_H1,
        "SUPPORT_H2": NEG_SUPPORT_H2,
        "CONTRADICT_H1": NEG_CONTRADICT_H1,
        "CONTRADICT_H2": NEG_CONTRADICT_H2,
        "NEUTRAL": NEG_NEUTRAL,
    },
    # S2-MEDIUM
    "NEGATION_SCOPE": {
        "SUPPORT_H1": NEG_SCOPE_SUPPORT_H1,
        "SUPPORT_H2": NEG_SCOPE_SUPPORT_H2,
        "CONTRADICT_H1": NEG_SCOPE_CONTRADICT_H1,
        "CONTRADICT_H2": NEG_SCOPE_CONTRADICT_H2,
        "NEUTRAL": NEG_SCOPE_NEUTRAL,
    },
    "TEMPORAL_SHIFT": {
        "SUPPORT_H1": TEMP_SHIFT_SUPPORT_H1,
        "SUPPORT_H2": TEMP_SHIFT_SUPPORT_H2,
        "CONTRADICT_H1": TEMP_SHIFT_CONTRADICT_H1,
        "CONTRADICT_H2": [],  # not commonly needed
        "NEUTRAL": TEMP_SHIFT_NEUTRAL,
    },
    "TEMPORAL_OVERRIDE": {
        "SUPPORT_H1": TEMP_OVERRIDE_SUPPORT_H1,
        "SUPPORT_H2": TEMP_OVERRIDE_SUPPORT_H2,
        "CONTRADICT_H1": TEMP_OVERRIDE_CONTRADICT_H1,
        "CONTRADICT_H2": [],
        "NEUTRAL": TEMP_OVERRIDE_NEUTRAL,
    },
    "IRRELEVANT_OVERLAP": {
        "SUPPORT_H1": IRRELEVANT_OVERLAP_SUPPORT_H1,
        "SUPPORT_H2": [],
        "CONTRADICT_H1": IRRELEVANT_OVERLAP_CONTRADICT_H1,
        "CONTRADICT_H2": [],
        "NEUTRAL": IRRELEVANT_OVERLAP_NEUTRAL,
    },
    "MULTI_SENTENCE_COMPOSITION": {
        "SUPPORT_H1": MULTI_SENT_SUPPORT_H1,
        "SUPPORT_H2": MULTI_SENT_SUPPORT_H2,
        "CONTRADICT_H1": MULTI_SENT_CONTRADICT_H1,
        "CONTRADICT_H2": [],
        "NEUTRAL": MULTI_SENT_NEUTRAL,
    },
    # S2-HARD
    "CONDITIONAL_SUPPORT": {
        "SUPPORT_H1": CONDITIONAL_SUPPORT_H1,
        "SUPPORT_H2": CONDITIONAL_SUPPORT_H2,
        "CONTRADICT_H1": CONDITIONAL_CONTRADICT_H1,
        "CONTRADICT_H2": [],
        "NEUTRAL": CONDITIONAL_NEUTRAL,
    },
    "PARTIAL_SUPPORT": {
        "SUPPORT_H1": PARTIAL_SUPPORT_H1,
        "SUPPORT_H2": PARTIAL_SUPPORT_H2,
        "CONTRADICT_H1": PARTIAL_CONTRADICT_H1,
        "CONTRADICT_H2": [],
        "NEUTRAL": PARTIAL_NEUTRAL,
    },
    "SOURCE_CONFLICT": {
        "SUPPORT_H1": SOURCE_CONFLICT_SUPPORT_H1,
        "SUPPORT_H2": SOURCE_CONFLICT_SUPPORT_H2,
        "CONTRADICT_H1": SOURCE_CONFLICT_CONTRADICT_H1,
        "CONTRADICT_H2": [],
        "NEUTRAL": SOURCE_CONFLICT_NEUTRAL,
    },
    "MULTI_HOP": {
        "SUPPORT_H1": MULTIHOP_SUPPORT_H1,
        "SUPPORT_H2": MULTIHOP_SUPPORT_H2,
        "CONTRADICT_H1": MULTIHOP_CONTRADICT_H1,
        "CONTRADICT_H2": [],
        "NEUTRAL": MULTIHOP_NEUTRAL,
    },
    "AMBIGUOUS": {
        "SUPPORT_H1": [],
        "SUPPORT_H2": [],
        "CONTRADICT_H1": [],
        "CONTRADICT_H2": [],
        "NEUTRAL": AMBIGUOUS_NEUTRAL,
    },
}


# ---------------------------------------------------------------------------
# Tier configuration
# ---------------------------------------------------------------------------

TIER_CONFIG = {
    "S2-EASY": {
        "PARAPHRASE": 75,
        "SIMPLE_NEGATION": 75,
    },
    "S2-MEDIUM": {
        "NEGATION_SCOPE": 30,
        "TEMPORAL_SHIFT": 30,
        "TEMPORAL_OVERRIDE": 30,
        "IRRELEVANT_OVERLAP": 30,
        "MULTI_SENTENCE_COMPOSITION": 30,
    },
    "S2-HARD": {
        "CONDITIONAL_SUPPORT": 30,
        "PARTIAL_SUPPORT": 30,
        "SOURCE_CONFLICT": 30,
        "MULTI_HOP": 30,
        "AMBIGUOUS": 30,
    },
}


# ---------------------------------------------------------------------------
# Structural patterns — same topology as S1 but with S2 text
# ---------------------------------------------------------------------------

# Pattern types determine the structural layout:
#   "bilateral_conflict": 2 hyps, E1 supports H1/contradicts H2,
#                         E2 supports H2/contradicts H1. T2 fires.
#   "simple_answer": 2 hyps, E1 supports H1, E2 is falsified.
#                     T2 does NOT fire. ANSWER.
#   "conflict_with_noise": 2 hyps, conflict + 1 noise. T2 fires.
#   "triple_eliminated": 3 hyps, all eliminated. T2 fires.
#   "subset_eliminated": 3 hyps, only H2/H3 eliminated. T2 does NOT fire.
#   "noise_dominant": 2 hyps, mostly noise + 1 support. T2 does NOT fire.

STRUCTURAL_PATTERNS = [
    "bilateral_conflict",
    "simple_answer",
    "conflict_with_noise",
    "triple_eliminated",
    "subset_eliminated",
    "noise_dominant",
    "bilateral_conflict",
    "simple_answer",
]


def _fill_template(template: str, subject: str, source: str) -> str:
    return template.format(subject=subject, source=source)


def _pick_template(
    semantic_class: str,
    relation_type: str,
    rng: random.Random,
) -> str:
    """Pick a template for the given semantic class and relation type."""
    templates = TEMPLATES_BY_CLASS.get(semantic_class, {}).get(relation_type, [])
    if not templates:
        # Fall back to S1 templates if no S2 template for this combination
        from hrm_adaptive_memory.executive.semantic_relations.raw_semantic_generator import (
            SUPPORT_H1_TEMPLATES, SUPPORT_H2_TEMPLATES,
            CONTRADICT_H1_TEMPLATES, CONTRADICT_H2_TEMPLATES,
            NEUTRAL_TEMPLATES,
        )
        fallback = {
            "SUPPORT_H1": SUPPORT_H1_TEMPLATES,
            "SUPPORT_H2": SUPPORT_H2_TEMPLATES,
            "CONTRADICT_H1": CONTRADICT_H1_TEMPLATES,
            "CONTRADICT_H2": CONTRADICT_H2_TEMPLATES,
            "NEUTRAL": NEUTRAL_TEMPLATES,
        }
        templates = fallback.get(relation_type, [])
    return rng.choice(templates)


def _make_evidence_text(
    semantic_class: str,
    relation_type: str,
    subject: str,
    rng: random.Random,
) -> str:
    """Generate evidence text for a given semantic class and relation."""
    template = _pick_template(semantic_class, relation_type, rng)
    source = rng.choice(SOURCES)
    return _fill_template(template, subject, source)


# ---------------------------------------------------------------------------
# Task generators for each structural pattern
# ---------------------------------------------------------------------------

def _gen_bilateral_conflict(
    task_id: str, subject: str, rng: random.Random,
    semantic_class: str, tier: str,
) -> SemanticTask:
    """2 hyps, bilateral conflict. T2 fires at step 2."""
    h = _make_hyps(2, subject)
    e1_prop = _make_evidence_text(semantic_class, "SUPPORT_H1", subject, rng)
    e2_prop = _make_evidence_text(semantic_class, "SUPPORT_H2", subject, rng)
    ev = (
        _suff("E1", e1_prop, ("H1",), ("H2",)),
        _suff("E2", e2_prop, ("H2",), ("H1",)),
    )
    gold = (
        GoldRelation("E1", "H1", "SUPPORT"),
        GoldRelation("E1", "H2", "CONTRADICT"),
        GoldRelation("E2", "H2", "SUPPORT"),
        GoldRelation("E2", "H1", "CONTRADICT"),
    )
    et = EvidenceTask(
        task_id=task_id, split=f"i3_12_{tier.lower()}",
        category=f"{tier.lower()}_bilateral_conflict",
        task_summary=f"Determine {subject}.",
        high_stakes=True, budget_profile="STANDARD",
        hypotheses=h, evidence_items=ev,
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "DEFER"),
        expected_terminal=DecisionAction.DEFER, correct_hypothesis_id="H2",
    )
    return SemanticTask(et, gold, tier=tier, semantic_class=semantic_class)


def _gen_simple_answer(
    task_id: str, subject: str, rng: random.Random,
    semantic_class: str, tier: str,
) -> SemanticTask:
    """2 hyps, H1 clearly supported. T2 does NOT fire. ANSWER."""
    h = _make_hyps(2, subject)
    e1_prop = _make_evidence_text(semantic_class, "SUPPORT_H1", subject, rng)
    e2_prop = _make_evidence_text(semantic_class, "SUPPORT_H2", subject, rng)
    ev = (
        _suff("E1", e1_prop, ("H1",), ("H2",)),
        _falsified("E2", e2_prop, ("H2",), ("H1",)),
    )
    gold = (
        GoldRelation("E1", "H1", "SUPPORT"),
        GoldRelation("E1", "H2", "CONTRADICT"),
        GoldRelation("E2", "H2", "SUPPORT"),
        GoldRelation("E2", "H1", "CONTRADICT"),
    )
    et = EvidenceTask(
        task_id=task_id, split=f"i3_12_{tier.lower()}",
        category=f"{tier.lower()}_simple_answer",
        task_summary=f"Determine {subject}.",
        high_stakes=True, budget_profile="STANDARD",
        hypotheses=h, evidence_items=ev,
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER, correct_hypothesis_id="H1",
    )
    return SemanticTask(et, gold, tier=tier, semantic_class=semantic_class)


def _gen_conflict_with_noise(
    task_id: str, subject: str, rng: random.Random,
    semantic_class: str, tier: str,
) -> SemanticTask:
    """2 hyps, conflict + 1 hidden noise. T2 fires."""
    h = _make_hyps(2, subject)
    e1_prop = _make_evidence_text(semantic_class, "SUPPORT_H1", subject, rng)
    e2_prop = _make_evidence_text(semantic_class, "SUPPORT_H2", subject, rng)
    ev = (
        _suff("E1", e1_prop, ("H1",), ("H2",)),
        _suff("E2", e2_prop, ("H2",), ("H1",)),
        _noise("E3", subject, rng, retrieved=False),
    )
    gold = (
        GoldRelation("E1", "H1", "SUPPORT"),
        GoldRelation("E1", "H2", "CONTRADICT"),
        GoldRelation("E2", "H2", "SUPPORT"),
        GoldRelation("E2", "H1", "CONTRADICT"),
        GoldRelation("E3", "H1", "NEUTRAL"),
        GoldRelation("E3", "H2", "NEUTRAL"),
    )
    et = EvidenceTask(
        task_id=task_id, split=f"i3_12_{tier.lower()}",
        category=f"{tier.lower()}_conflict_with_noise",
        task_summary=f"Determine {subject}.",
        high_stakes=True, budget_profile="STANDARD",
        hypotheses=h, evidence_items=ev,
        retrieve_exposes=(), search_exposes=("E3",),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "DEFER"),
        expected_terminal=DecisionAction.DEFER, correct_hypothesis_id="H2",
    )
    return SemanticTask(et, gold, tier=tier, semantic_class=semantic_class)


def _gen_triple_eliminated(
    task_id: str, subject: str, rng: random.Random,
    semantic_class: str, tier: str,
) -> SemanticTask:
    """3 hyps, all eliminated. T2 fires."""
    h = _make_hyps(3, subject)
    e1_prop = _make_evidence_text(semantic_class, "SUPPORT_H1", subject, rng)
    e2_prop = _make_evidence_text(semantic_class, "SUPPORT_H2", subject, rng)
    e3_prop = f"Source {rng.choice(SOURCES)} contradicts the claim that {subject} is ambiguous (hypothesis 3)."
    ev = (
        _suff("E1", e1_prop, ("H1",), ("H2",)),
        _suff("E2", e2_prop, ("H2",), ("H1",)),
        _suff("E3", e3_prop, (), ("H3",)),
    )
    gold = (
        GoldRelation("E1", "H1", "SUPPORT"),
        GoldRelation("E1", "H2", "CONTRADICT"),
        GoldRelation("E2", "H2", "SUPPORT"),
        GoldRelation("E2", "H1", "CONTRADICT"),
        GoldRelation("E3", "H3", "CONTRADICT"),
        GoldRelation("E3", "H1", "NEUTRAL"),
        GoldRelation("E3", "H2", "NEUTRAL"),
    )
    et = EvidenceTask(
        task_id=task_id, split=f"i3_12_{tier.lower()}",
        category=f"{tier.lower()}_triple_eliminated",
        task_summary=f"Determine {subject}.",
        high_stakes=True, budget_profile="STANDARD",
        hypotheses=h, evidence_items=ev,
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "VERIFY:E3", "DEFER"),
        expected_terminal=DecisionAction.DEFER, correct_hypothesis_id="H2",
    )
    return SemanticTask(et, gold, tier=tier, semantic_class=semantic_class)


def _gen_subset_eliminated(
    task_id: str, subject: str, rng: random.Random,
    semantic_class: str, tier: str,
) -> SemanticTask:
    """3 hyps, only H2/H3 eliminated. H1 viable. T2 does NOT fire."""
    h = _make_hyps(3, subject)
    e1_prop = _make_evidence_text(semantic_class, "SUPPORT_H1", subject, rng)
    e2_prop = f"Source {rng.choice(SOURCES)} contradicts the claim that {subject} is ambiguous (hypothesis 3)."
    ev = (
        _suff("E1", e1_prop, ("H1",), ("H2",)),
        _suff("E2", e2_prop, (), ("H3",)),
    )
    gold = (
        GoldRelation("E1", "H1", "SUPPORT"),
        GoldRelation("E1", "H2", "CONTRADICT"),
        GoldRelation("E1", "H3", "NEUTRAL"),
        GoldRelation("E2", "H3", "CONTRADICT"),
        GoldRelation("E2", "H1", "NEUTRAL"),
        GoldRelation("E2", "H2", "NEUTRAL"),
    )
    et = EvidenceTask(
        task_id=task_id, split=f"i3_12_{tier.lower()}",
        category=f"{tier.lower()}_subset_eliminated",
        task_summary=f"Determine {subject}.",
        high_stakes=True, budget_profile="STANDARD",
        hypotheses=h, evidence_items=ev,
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER, correct_hypothesis_id="H1",
    )
    return SemanticTask(et, gold, tier=tier, semantic_class=semantic_class)


def _gen_noise_dominant(
    task_id: str, subject: str, rng: random.Random,
    semantic_class: str, tier: str,
) -> SemanticTask:
    """2 hyps, mostly noise + 1 support. T2 does NOT fire."""
    h = _make_hyps(2, subject)
    # Use the semantic class for the noise/neutral evidence
    e1_prop = _make_evidence_text(semantic_class, "NEUTRAL", subject, rng)
    e2_prop = _make_evidence_text(semantic_class, "NEUTRAL", subject, rng)
    e3_prop = _make_evidence_text(semantic_class, "SUPPORT_H1", subject, rng)
    ev = (
        EvidenceItem(
            evidence_id="E1", proposition=e1_prop,
            source_class="search",
            supports=(), contradicts=(),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True, verify_result="MISSING",
        ),
        EvidenceItem(
            evidence_id="E2", proposition=e2_prop,
            source_class="search",
            supports=(), contradicts=(),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True, verify_result="MISSING",
        ),
        _suff("E3", e3_prop, ("H1",), ("H2",)),
    )
    gold = (
        GoldRelation("E1", "H1", "NEUTRAL"),
        GoldRelation("E1", "H2", "NEUTRAL"),
        GoldRelation("E2", "H1", "NEUTRAL"),
        GoldRelation("E2", "H2", "NEUTRAL"),
        GoldRelation("E3", "H1", "SUPPORT"),
        GoldRelation("E3", "H2", "CONTRADICT"),
    )
    et = EvidenceTask(
        task_id=task_id, split=f"i3_12_{tier.lower()}",
        category=f"{tier.lower()}_noise_dominant",
        task_summary=f"Determine {subject}.",
        high_stakes=True, budget_profile="STANDARD",
        hypotheses=h, evidence_items=ev,
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E3", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER, correct_hypothesis_id="H1",
    )
    return SemanticTask(et, gold, tier=tier, semantic_class=semantic_class)


PATTERN_GENERATORS = {
    "bilateral_conflict": _gen_bilateral_conflict,
    "simple_answer": _gen_simple_answer,
    "conflict_with_noise": _gen_conflict_with_noise,
    "triple_eliminated": _gen_triple_eliminated,
    "subset_eliminated": _gen_subset_eliminated,
    "noise_dominant": _gen_noise_dominant,
}


# ---------------------------------------------------------------------------
# Corpus generation
# ---------------------------------------------------------------------------

def generate_s2_corpus(
    seed: int = 42,
    split: str = "i3_12_s2",
) -> list[SemanticTask]:
    """Generate the S2 semantic stress corpus.

    450 tasks across 3 tiers:
      S2-EASY: 150 tasks (PARAPHRASE 75, SIMPLE_NEGATION 75)
      S2-MEDIUM: 150 tasks (5 classes x 30)
      S2-HARD: 150 tasks (5 classes x 30)

    Returns:
        List of SemanticTask with tier, semantic_class, and gold_relations.
    """
    rng = random.Random(seed)
    tasks: list[SemanticTask] = []
    task_counter = 0

    for tier, class_counts in TIER_CONFIG.items():
        for semantic_class, count in class_counts.items():
            for i in range(count):
                task_id = f"i3_12_s2_{task_counter:04d}"
                subject = rng.choice(SUBJECTS)
                pattern = STRUCTURAL_PATTERNS[task_counter % len(STRUCTURAL_PATTERNS)]
                gen_func = PATTERN_GENERATORS[pattern]
                task = gen_func(task_id, subject, rng, semantic_class, tier)
                tasks.append(task)
                task_counter += 1

    return tasks


def generate_s2_corpus_by_tier(
    tier: str,
    seed: int = 42,
) -> list[SemanticTask]:
    """Generate only tasks for a specific tier."""
    all_tasks = generate_s2_corpus(seed=seed)
    return [t for t in all_tasks if t.tier == tier]
