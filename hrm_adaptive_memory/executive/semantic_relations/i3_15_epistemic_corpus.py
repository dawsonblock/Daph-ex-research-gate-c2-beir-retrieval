"""I3.15: Epistemically demanding document corpus.

A corpus designed to separate retrieval difficulty from epistemic difficulty.
Evidence passages require multi-step reasoning, temporal tracking, authority
comparison, or conditional implication — not just direct text reading.

Design axes:
  - Retrieval difficulty: Easy (domain terms in query) / Hard (abstract query)
  - Epistemic difficulty: Easy / Medium / Hard

All evidence for a task is in the same domain neighborhood so retrieval is
fair, but the decision requires reasoning across passages.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class I3_15Passage:
    """A document passage with temporal metadata and evaluator-side gold relations."""
    passage_id: str
    text: str
    source: str
    domain: str
    timestamp: str  # e.g., "2024-06-04T14:00Z" or "" for undated
    # Evaluator-side gold: does this passage support/contradict H1 (operational)?
    # "SUPPORT", "CONTRADICT", "NEUTRAL", "CONDITIONAL", "TEMPORAL"
    gold_relation: str = "NEUTRAL"
    # For multi-step chains: which passage this depends on
    depends_on: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# EpistemicEasy: Direct status statements
# Same as I3.13 style — passage directly says "operational" or "not operational"
# ---------------------------------------------------------------------------

EASY_PASSAGES: list[I3_15Passage] = [
    # --- API Gateway (easy) ---
    I3_15Passage("E001", "The API gateway is currently operational. All endpoints are responding within normal parameters.", "status_report", "api_gateway", "2024-06-10T09:00Z", "SUPPORT"),
    I3_15Passage("E002", "The API gateway is currently experiencing elevated error rates. The service is degraded and not fully operational.", "incident_report", "api_gateway", "2024-06-10T09:00Z", "CONTRADICT"),

    # --- Database (easy) ---
    I3_15Passage("E003", "Database replication lag has returned to normal levels. The database is currently operational.", "status_report", "database", "2024-06-10T09:00Z", "SUPPORT"),
    I3_15Passage("E004", "The database is currently experiencing connectivity issues. New connections are being rejected. The database is not operational.", "incident_report", "database", "2024-06-10T09:00Z", "CONTRADICT"),

    # --- CDN (easy) ---
    I3_15Passage("E005", "CDN edge locations are all reporting healthy status. The CDN is currently operational.", "status_report", "cdn", "2024-06-10T09:00Z", "SUPPORT"),
    I3_15Passage("E006", "A configuration error caused the CDN to serve stale content. The CDN is currently not fully operational during cache warmup.", "incident_report", "cdn", "2024-06-10T09:00Z", "CONTRADICT"),

    # --- Kubernetes (easy) ---
    I3_15Passage("E007", "All Kubernetes nodes are reporting Ready status. The Kubernetes cluster is currently operational.", "status_report", "kubernetes", "2024-06-10T09:00Z", "SUPPORT"),
    I3_15Passage("E008", "Three Kubernetes nodes went into NotReady state after a network partition. The Kubernetes cluster is currently not fully operational.", "incident_report", "kubernetes", "2024-06-10T09:00Z", "CONTRADICT"),

    # --- Security (easy) ---
    I3_15Passage("E009", "The security vulnerability patch has been applied to all production services. The security posture is currently confirmed.", "security_advisory", "security", "2024-06-10T09:00Z", "SUPPORT"),
    I3_15Passage("E010", "An active security incident is affecting the authentication service. The security posture is currently not confirmed.", "security_incident", "security", "2024-06-10T09:00Z", "CONTRADICT"),

    # --- Deployment (easy) ---
    I3_15Passage("E011", "The deployment of version 2.4.1 completed successfully. The deployment is currently operational.", "deployment_report", "deployment", "2024-06-10T09:00Z", "SUPPORT"),
    I3_15Passage("E012", "The deployment was rolled back due to a regression. The deployment status is currently unconfirmed.", "rollback_report", "deployment", "2024-06-10T09:00Z", "CONTRADICT"),

    # --- Monitoring (easy) ---
    I3_15Passage("E013", "All monitoring systems are operational. Dashboards are updating in real-time.", "monitoring_status", "monitoring", "2024-06-10T09:00Z", "SUPPORT"),
    I3_15Passage("E014", "The monitoring system experienced a data ingestion delay. The monitoring system is currently not fully operational.", "monitoring_incident", "monitoring", "2024-06-10T09:00Z", "CONTRADICT"),

    # --- Cache (easy) ---
    I3_15Passage("E015", "Redis cluster health check passed. The Redis cache is currently operational.", "status_report", "cache", "2024-06-10T09:00Z", "SUPPORT"),
    I3_15Passage("E016", "The Redis cache was flushed to resolve a data consistency issue. The Redis cache is currently not fully operational during warmup.", "cache_incident", "cache", "2024-06-10T09:00Z", "CONTRADICT"),

    # --- Message queue (easy) ---
    I3_15Passage("E017", "The message queue is processing normally with no backlog. The message queue is currently operational.", "status_report", "message_queue", "2024-06-10T09:00Z", "SUPPORT"),
    I3_15Passage("E018", "A consumer group has fallen behind with a backlog of 50,000 messages. The message queue is currently not fully operational.", "queue_incident", "message_queue", "2024-06-10T09:00Z", "CONTRADICT"),

    # --- Load balancer (easy) ---
    I3_15Passage("E019", "The load balancer is distributing traffic evenly across all backend servers. The load balancer is currently operational.", "status_report", "load_balancer", "2024-06-10T09:00Z", "SUPPORT"),
    I3_15Passage("E020", "Two backend servers behind the load balancer have been marked as unhealthy. The load balancer is currently not fully operational.", "lb_incident", "load_balancer", "2024-06-10T09:00Z", "CONTRADICT"),
]


# ---------------------------------------------------------------------------
# EpistemicMedium: Temporal supersession and authority conflict
# Requires comparing timestamps or source authority
# ---------------------------------------------------------------------------

MEDIUM_PASSAGES: list[I3_15Passage] = [
    # --- API Gateway temporal chain ---
    I3_15Passage("M001", "The API gateway was taken offline for emergency maintenance at 22:00 UTC on June 3. The service is not operational.", "maintenance_notice", "api_gateway", "2024-06-03T22:00Z", "CONTRADICT"),
    I3_15Passage("M002", "The API gateway maintenance was completed at 02:00 UTC on June 4. The service has been restored.", "maintenance_resolution", "api_gateway", "2024-06-04T02:00Z", "SUPPORT"),
    I3_15Passage("M003", "The API gateway is experiencing elevated error rates following the recent maintenance. The service is degraded.", "incident_report", "api_gateway", "2024-06-04T06:00Z", "CONTRADICT"),

    # --- Database temporal supersession ---
    I3_15Passage("M004", "The database was taken offline for emergency maintenance. The database is not operational.", "maintenance_notice", "database", "2024-06-05T01:00Z", "CONTRADICT"),
    I3_15Passage("M005", "Database maintenance completed at 04:00 UTC. All services have been restored. The database is operational.", "maintenance_resolution", "database", "2024-06-05T04:00Z", "SUPPORT"),
    I3_15Passage("M006", "Partial database restoration has begun. Some tables are still being recovered.", "recovery_report", "database", "2024-06-05T03:00Z", "CONTRADICT"),

    # --- CDN authority conflict ---
    I3_15Passage("M007", "The CDN status page reports that all edge locations are healthy and the service is fully operational.", "status_page", "cdn", "2024-06-06T12:00Z", "SUPPORT"),
    I3_15Passage("M008", "Internal monitoring indicates that the CDN status page report was premature. Cache miss rates are elevated and some edge nodes are still serving stale content.", "internal_report", "cdn", "2024-06-06T12:20Z", "CONTRADICT"),

    # --- Kubernetes scope qualifier ---
    I3_15Passage("M009", "All Kubernetes cluster regions except APAC are reporting Ready status. The cluster is operational in US, EU, and ME regions.", "status_report", "kubernetes", "2024-06-07T10:00Z", "SUPPORT"),
    I3_15Passage("M010", "The APAC Kubernetes region experienced a network partition. Nodes in APAC are in NotReady state.", "incident_report", "kubernetes", "2024-06-07T10:00Z", "CONTRADICT"),

    # --- Security temporal ---
    I3_15Passage("M011", "A critical security vulnerability was identified in the authentication library. The system is currently vulnerable.", "security_advisory", "security", "2024-06-08T08:00Z", "CONTRADICT"),
    I3_15Passage("M012", "The security patch for the authentication vulnerability has been applied to all production services. The vulnerability has been remediated.", "security_resolution", "security", "2024-06-08T14:00Z", "SUPPORT"),

    # --- Deployment rollback chain ---
    I3_15Passage("M013", "Version 2.5.0 was deployed to production at 16:00 UTC. All health checks passed initially.", "deployment_report", "deployment", "2024-06-09T16:00Z", "SUPPORT"),
    I3_15Passage("M014", "Version 2.5.0 was rolled back at 17:30 UTC due to a memory leak. Version 2.4.9 has been restored.", "rollback_report", "deployment", "2024-06-09T17:30Z", "CONTRADICT"),

    # --- Monitoring authority ---
    I3_15Passage("M015", "The monitoring dashboard shows all systems green. No active alerts.", "dashboard_screenshot", "monitoring", "2024-06-09T15:00Z", "SUPPORT"),
    I3_15Passage("M016", "The monitoring agent was misconfigured and was checking a shadow environment. Production monitoring data is stale.", "config_audit", "monitoring", "2024-06-09T15:30Z", "CONTRADICT"),

    # --- Cache temporal ---
    I3_15Passage("M017", "The Redis cache was flushed at 10:00 UTC to resolve a data consistency issue. The cache is not operational during warmup.", "cache_incident", "cache", "2024-06-08T10:00Z", "CONTRADICT"),
    I3_15Passage("M018", "The Redis cache has finished repopulating at 10:45 UTC. Cache hit rates are returning to normal. The cache is operational.", "cache_resolution", "cache", "2024-06-08T10:45Z", "SUPPORT"),

    # --- Message queue scope ---
    I3_15Passage("M019", "The message queue is processing normally in all regions except EU. Consumer lag is at zero in US and APAC.", "status_report", "message_queue", "2024-06-07T14:00Z", "SUPPORT"),
    I3_15Passage("M020", "The EU message queue consumer group has fallen behind with a backlog of 50,000 messages. Processing is delayed in EU.", "incident_report", "message_queue", "2024-06-07T14:00Z", "CONTRADICT"),

    # --- Load balancer authority ---
    I3_15Passage("M021", "The load balancer health check endpoint reports all targets as healthy.", "health_check", "load_balancer", "2024-06-06T11:00Z", "SUPPORT"),
    I3_15Passage("M022", "The load balancer health check was probing a deprecated endpoint that always returns 200. Two backend servers are actually unhealthy and returning 502 errors.", "config_audit", "load_balancer", "2024-06-06T11:15Z", "CONTRADICT"),
]


# ---------------------------------------------------------------------------
# EpistemicHard: Multi-step reasoning chains and conditional implications
# No single passage directly answers the question
# ---------------------------------------------------------------------------

HARD_PASSAGES: list[I3_15Passage] = [
    # --- Chain 1: API Gateway incident -> validation -> DNS -> operational ---
    I3_15Passage("H001", "The API gateway operations were suspended following the database incident at 14:00 UTC. The API gateway is not accepting requests.", "incident_report", "api_gateway", "2024-06-04T14:00Z", "CONTRADICT"),
    I3_15Passage("H002", "The API gateway replacement database cluster passed validation at 16:00 UTC. However, API gateway customer traffic remains blocked pending DNS propagation.", "validation_report", "api_gateway", "2024-06-04T16:00Z", "CONTRADICT", depends_on=("H001",)),
    I3_15Passage("H003", "The API gateway DNS propagation completed at 18:20 UTC. API gateway customer traffic is now being routed to the new endpoints.", "dns_update", "api_gateway", "2024-06-04T18:20Z", "SUPPORT", depends_on=("H002",)),

    # --- Chain 2: Database migration -> writes -> operational ---
    I3_15Passage("H004", "The database migration was initiated at 02:00 UTC. Database write operations are suspended during the migration process.", "migration_notice", "database", "2024-06-05T02:00Z", "CONTRADICT"),
    I3_15Passage("H005", "If the database migration completes successfully, database write operations will resume automatically without manual intervention.", "migration_plan", "database", "2024-06-05T02:00Z", "CONDITIONAL", depends_on=("H004",)),
    I3_15Passage("H006", "The database migration completed successfully at 05:30 UTC. All database data has been transferred and validated.", "migration_report", "database", "2024-06-05T05:30Z", "SUPPORT", depends_on=("H005",)),

    # --- Chain 3: CDN cache purge -> propagation -> operational ---
    I3_15Passage("H007", "A cache purge was initiated across all CDN edge locations at 09:00 UTC due to a content update. The CDN is serving stale content until propagation completes.", "cache_purge", "cdn", "2024-06-06T09:00Z", "CONTRADICT"),
    I3_15Passage("H008", "CDN cache propagation has completed in US and EU regions. CDN APAC edge nodes are still propagating due to network latency.", "propagation_report", "cdn", "2024-06-06T10:00Z", "CONTRADICT", depends_on=("H007",)),
    I3_15Passage("H009", "All CDN edge nodes including APAC have completed cache propagation as of 11:30 UTC. The CDN is serving fresh content globally.", "propagation_complete", "cdn", "2024-06-06T11:30Z", "SUPPORT", depends_on=("H008",)),

    # --- Chain 4: Kubernetes upgrade -> cordon -> drain -> operational ---
    I3_15Passage("H010", "The Kubernetes cluster upgrade from v1.28 to v1.29 has been initiated. Kubernetes nodes are being cordoned and drained sequentially.", "upgrade_notice", "kubernetes", "2024-06-07T01:00Z", "CONTRADICT"),
    I3_15Passage("H011", "The Kubernetes control plane upgrade completed at 02:30 UTC. Kubernetes worker node upgrades are in progress. Pods are being rescheduled.", "upgrade_progress", "kubernetes", "2024-06-07T02:30Z", "CONTRADICT", depends_on=("H010",)),
    I3_15Passage("H012", "All Kubernetes worker nodes have been upgraded and are reporting Ready status with v1.29. The Kubernetes cluster upgrade is complete.", "upgrade_complete", "kubernetes", "2024-06-07T04:00Z", "SUPPORT", depends_on=("H011",)),

    # --- Chain 5: Security incident -> investigation -> patch -> verified ---
    I3_15Passage("H013", "A security breach was detected in the payment processing module. The security posture is compromised and the module is isolated.", "security_incident", "security", "2024-06-08T03:00Z", "CONTRADICT"),
    I3_15Passage("H014", "The security investigation identified the breach vector as a compromised API key. The security team has revoked the key.", "investigation_report", "security", "2024-06-08T05:00Z", "CONTRADICT", depends_on=("H013",)),
    I3_15Passage("H015", "The security patch has been applied and the payment module restored with new API key rotation. Security verification passed at 08:00 UTC.", "security_resolution", "security", "2024-06-08T08:00Z", "SUPPORT", depends_on=("H014",)),

    # --- Chain 6: Deployment -> canary -> promotion -> operational ---
    I3_15Passage("H016", "A canary deployment of version 3.0 was initiated at 10:00 UTC. Deployment traffic is being gradually shifted to the new version.", "canary_report", "deployment", "2024-06-09T10:00Z", "CONTRADICT"),
    I3_15Passage("H017", "The deployment canary is performing well with error rates within acceptable bounds. Deployment traffic shift is at 50%.", "canary_progress", "deployment", "2024-06-09T11:00Z", "CONDITIONAL", depends_on=("H016",)),
    I3_15Passage("H018", "The deployment has been promoted to 100% of traffic at 12:00 UTC. Version 3.0 is now the active production deployment.", "deployment_complete", "deployment", "2024-06-09T12:00Z", "SUPPORT", depends_on=("H017",)),

    # --- Chain 7: Monitoring -> agent fix -> data recovery -> operational ---
    I3_15Passage("H019", "The monitoring agent crashed at 06:00 UTC due to an out-of-memory error. Monitoring data is not being collected.", "monitoring_incident", "monitoring", "2024-06-10T06:00Z", "CONTRADICT"),
    I3_15Passage("H020", "The monitoring agent has been restarted with increased memory limits. Monitoring data collection has resumed but historical data from the outage period is lost.", "monitoring_fix", "monitoring", "2024-06-10T07:00Z", "CONDITIONAL", depends_on=("H019",)),
    I3_15Passage("H021", "Monitoring data collection has been verified as stable for 30 minutes since the agent restart. The monitoring system is operational.", "monitoring_verified", "monitoring", "2024-06-10T07:30Z", "SUPPORT", depends_on=("H020",)),

    # --- Chain 8: Cache failover -> sync -> operational ---
    I3_15Passage("H022", "The primary Redis cache node failed at 14:00 UTC. The Redis cache failover to the replica has been initiated but is not yet complete.", "failover_notice", "cache", "2024-06-08T14:00Z", "CONTRADICT"),
    I3_15Passage("H023", "Redis cache failover completed at 14:05 UTC. The Redis cache replica is now the primary node. Cache data sync is in progress.", "failover_report", "cache", "2024-06-08T14:05Z", "CONDITIONAL", depends_on=("H022",)),
    I3_15Passage("H024", "Redis cache data sync completed at 14:20 UTC. The Redis cache is fully operational with the new primary node.", "sync_complete", "cache", "2024-06-08T14:20Z", "SUPPORT", depends_on=("H023",)),
]


ALL_PASSAGES = EASY_PASSAGES + MEDIUM_PASSAGES + HARD_PASSAGES


def get_corpus() -> list[I3_15Passage]:
    return list(ALL_PASSAGES)


def corpus_sha256() -> str:
    content = json.dumps(
        [{"passage_id": p.passage_id, "text": p.text, "source": p.source,
          "domain": p.domain, "timestamp": p.timestamp,
          "gold_relation": p.gold_relation, "depends_on": list(p.depends_on)}
         for p in ALL_PASSAGES],
        sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


def save_corpus(path: Path) -> None:
    data = {
        "corpus_id": "i3_15_epistemic_corpus_v1",
        "n_passages": len(ALL_PASSAGES),
        "sha256": corpus_sha256(),
        "n_easy": len(EASY_PASSAGES),
        "n_medium": len(MEDIUM_PASSAGES),
        "n_hard": len(HARD_PASSAGES),
        "passages": [
            {"passage_id": p.passage_id, "text": p.text, "source": p.source,
             "domain": p.domain, "timestamp": p.timestamp,
             "gold_relation": p.gold_relation, "depends_on": list(p.depends_on)}
            for p in ALL_PASSAGES
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    print(f"Corpus: {len(ALL_PASSAGES)} passages")
    print(f"  Easy: {len(EASY_PASSAGES)}")
    print(f"  Medium: {len(MEDIUM_PASSAGES)}")
    print(f"  Hard: {len(HARD_PASSAGES)}")
    print(f"SHA-256: {corpus_sha256()[:16]}")
    by_domain = {}
    for p in ALL_PASSAGES:
        by_domain.setdefault(p.domain, 0)
        by_domain[p.domain] += 1
    print(f"By domain: {by_domain}")
