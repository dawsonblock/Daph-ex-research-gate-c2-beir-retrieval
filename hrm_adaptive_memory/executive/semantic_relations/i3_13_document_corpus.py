"""I3.13: Frozen document corpus with recoverable ground truth.

Builds a corpus of real-world document passages where the
SUPPORT/CONTRADICT/NEUTRAL relation to hypotheses is recoverable
by an evaluator but requires actual language understanding from the extractor.

Design:
  - Each passage is a real-world status report or operational update
  - Passages contain explicit status language the model can understand
  - But natural language variety still challenges the deterministic extractor
  - Each passage has evaluator-side gold relation labels
  - Gold relations are never exposed to the controller

The corpus covers domains that match the hypothesis structure
(current/stale, operational/offline) with real language variety.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DocumentPassage:
    """A real-world document passage with evaluator-side metadata."""
    passage_id: str
    text: str
    source: str
    domain: str
    # Evaluator-side gold relations (never exposed to controller)
    # Each entry: (hypothesis_orientation, relation)
    # hypothesis_orientation: "current" or "stale"
    # relation: "SUPPORT", "CONTRADICT", or "NEUTRAL"
    gold_relations: tuple[tuple[str, str], ...] = ()


# ---------------------------------------------------------------------------
# Real-world document passages with explicit status language
# ---------------------------------------------------------------------------

PASSAGES: list[DocumentPassage] = [
    # --- API Gateway ---
    DocumentPassage(
        passage_id="P001",
        text="The API gateway is currently operational. All endpoints are responding within normal parameters. No incidents have been reported in the last 24 hours.",
        source="status_report",
        domain="api_gateway",
        gold_relations=(("current", "SUPPORT"), ("stale", "CONTRADICT")),
    ),
    DocumentPassage(
        passage_id="P002",
        text="The API gateway is currently experiencing elevated error rates. Some requests are failing with 503 errors. The service is degraded and not fully operational.",
        source="incident_report",
        domain="api_gateway",
        gold_relations=(("current", "CONTRADICT"), ("stale", "NEUTRAL")),
    ),
    DocumentPassage(
        passage_id="P003",
        text="The API gateway was restored at 14:32 UTC after a brief outage. All services are now functioning normally and the gateway is currently operational.",
        source="incident_resolution",
        domain="api_gateway",
        gold_relations=(("current", "SUPPORT"), ("stale", "CONTRADICT")),
    ),
    DocumentPassage(
        passage_id="P004",
        text="Scheduled maintenance for the API gateway is planned for this weekend. The service will be temporarily unavailable from 02:00 to 04:00 UTC on Saturday.",
        source="maintenance_notice",
        domain="api_gateway",
        gold_relations=(("current", "NEUTRAL"), ("stale", "NEUTRAL")),
    ),
    DocumentPassage(
        passage_id="P005",
        text="The legacy API gateway (v1) has been deprecated as of January 2024. The v1 endpoint is no longer operational and will be permanently removed in Q2.",
        source="deprecation_notice",
        domain="api_gateway",
        gold_relations=(("current", "CONTRADICT"), ("stale", "SUPPORT")),
    ),

    # --- Database ---
    DocumentPassage(
        passage_id="P006",
        text="Database replication lag has returned to normal levels after the configuration fix. All replicas are now in sync. The database is currently operational.",
        source="status_report",
        domain="database",
        gold_relations=(("current", "SUPPORT"), ("stale", "CONTRADICT")),
    ),
    DocumentPassage(
        passage_id="P007",
        text="The database migration completed successfully at 03:15 UTC. All tables have been migrated and validated. The database is currently operational with the new schema active.",
        source="migration_report",
        domain="database",
        gold_relations=(("current", "SUPPORT"), ("stale", "CONTRADICT")),
    ),
    DocumentPassage(
        passage_id="P008",
        text="The database is currently experiencing connectivity issues. The connection pool is exhausted and new connections are being rejected. The database is not operational at this time.",
        source="incident_report",
        domain="database",
        gold_relations=(("current", "CONTRADICT"), ("stale", "NEUTRAL")),
    ),
    DocumentPassage(
        passage_id="P009",
        text="The database backup from last night completed successfully. The backup file has been verified. The database remains operational.",
        source="backup_report",
        domain="database",
        gold_relations=(("current", "SUPPORT"), ("stale", "NEUTRAL")),
    ),
    DocumentPassage(
        passage_id="P010",
        text="The database was taken offline for emergency maintenance at 22:00 UTC. The database is currently not operational. Services depending on the database may experience errors.",
        source="maintenance_notice",
        domain="database",
        gold_relations=(("current", "CONTRADICT"), ("stale", "NEUTRAL")),
    ),

    # --- CDN ---
    DocumentPassage(
        passage_id="P011",
        text="CDN edge locations are all reporting healthy status. Cache hit rates are at 94.2% globally. The CDN is currently operational with no anomalies detected.",
        source="status_report",
        domain="cdn",
        gold_relations=(("current", "SUPPORT"), ("stale", "CONTRADICT")),
    ),
    DocumentPassage(
        passage_id="P012",
        text="A configuration error caused the CDN to serve stale content for approximately 45 minutes. The issue has been resolved and caches have been purged. The CDN is currently operational with fresh content.",
        source="incident_resolution",
        domain="cdn",
        gold_relations=(("current", "SUPPORT"), ("stale", "CONTRADICT")),
    ),
    DocumentPassage(
        passage_id="P013",
        text="The CDN provider has announced a scheduled maintenance window for their North American edge nodes. During this time, the CDN may experience brief interruptions.",
        source="maintenance_notice",
        domain="cdn",
        gold_relations=(("current", "NEUTRAL"), ("stale", "NEUTRAL")),
    ),

    # --- Kubernetes ---
    DocumentPassage(
        passage_id="P014",
        text="All Kubernetes nodes are reporting Ready status. The cluster autoscaler is functioning normally. The Kubernetes cluster is currently operational with 342 pods across 12 nodes.",
        source="status_report",
        domain="kubernetes",
        gold_relations=(("current", "SUPPORT"), ("stale", "CONTRADICT")),
    ),
    DocumentPassage(
        passage_id="P015",
        text="Three Kubernetes nodes went into NotReady state after a network partition. Pods on those nodes are being rescheduled. The Kubernetes cluster is currently not fully operational.",
        source="incident_report",
        domain="kubernetes",
        gold_relations=(("current", "CONTRADICT"), ("stale", "NEUTRAL")),
    ),
    DocumentPassage(
        passage_id="P016",
        text="The Kubernetes cluster upgrade from v1.27 to v1.28 has been completed. All control plane components are running v1.28. The cluster is currently operational.",
        source="upgrade_report",
        domain="kubernetes",
        gold_relations=(("current", "SUPPORT"), ("stale", "CONTRADICT")),
    ),
    DocumentPassage(
        passage_id="P017",
        text="The legacy Kubernetes cluster (cluster-prod-01) has been decommissioned. All workloads have been migrated. The old cluster is no longer operational.",
        source="decommission_report",
        domain="kubernetes",
        gold_relations=(("current", "CONTRADICT"), ("stale", "SUPPORT")),
    ),

    # --- Security ---
    DocumentPassage(
        passage_id="P018",
        text="A critical security vulnerability (CVE-2024-1234) has been identified in the logging library. The patch has been applied to all production services. The security posture is currently confirmed with no evidence of exploitation.",
        source="security_advisory",
        domain="security",
        gold_relations=(("current", "SUPPORT"), ("stale", "CONTRADICT")),
    ),
    DocumentPassage(
        passage_id="P019",
        text="We are aware of an active security incident affecting the authentication service. Some user tokens may have been compromised. The security posture is currently not confirmed.",
        source="security_incident",
        domain="security",
        gold_relations=(("current", "CONTRADICT"), ("stale", "NEUTRAL")),
    ),
    DocumentPassage(
        passage_id="P020",
        text="The SSL certificate for api.example.com was renewed successfully. The new certificate is valid until December 2025. The security posture is currently confirmed.",
        source="certificate_renewal",
        domain="security",
        gold_relations=(("current", "SUPPORT"), ("stale", "CONTRADICT")),
    ),
    DocumentPassage(
        passage_id="P021",
        text="The SSL certificate for legacy.example.com expired last week. The endpoint is no longer accessible via HTTPS. The security posture for this endpoint is stale and unconfirmed.",
        source="certificate_expiry",
        domain="security",
        gold_relations=(("current", "CONTRADICT"), ("stale", "SUPPORT")),
    ),

    # --- Deployment ---
    DocumentPassage(
        passage_id="P022",
        text="The deployment of version 2.4.1 to production completed successfully at 16:45 UTC. All health checks passed. The deployment is currently operational and serving 100% of traffic.",
        source="deployment_report",
        domain="deployment",
        gold_relations=(("current", "SUPPORT"), ("stale", "CONTRADICT")),
    ),
    DocumentPassage(
        passage_id="P023",
        text="The deployment of version 2.4.0 was rolled back due to a regression in the payment processing module. Version 2.3.9 has been restored. The deployment is currently operational on the previous version.",
        source="rollback_report",
        domain="deployment",
        gold_relations=(("current", "SUPPORT"), ("stale", "CONTRADICT")),
    ),
    DocumentPassage(
        passage_id="P024",
        text="Deployment is currently blocked due to a failing integration test in the CI pipeline. The team is investigating. No new code has been deployed. The deployment status is unconfirmed.",
        source="ci_status",
        domain="deployment",
        gold_relations=(("current", "NEUTRAL"), ("stale", "NEUTRAL")),
    ),
    DocumentPassage(
        passage_id="P025",
        text="The canary deployment of the new recommendation service is progressing well. Error rates are within acceptable bounds. The deployment is currently operational and traffic is being gradually increased.",
        source="canary_report",
        domain="deployment",
        gold_relations=(("current", "SUPPORT"), ("stale", "NEUTRAL")),
    ),

    # --- Monitoring ---
    DocumentPassage(
        passage_id="P026",
        text="All monitoring systems are operational. Dashboards are updating in real-time. No active alerts. The monitoring system is currently confirmed operational.",
        source="monitoring_status",
        domain="monitoring",
        gold_relations=(("current", "SUPPORT"), ("stale", "CONTRADICT")),
    ),
    DocumentPassage(
        passage_id="P027",
        text="The monitoring system experienced a data ingestion delay of approximately 30 minutes. The issue has been resolved. The monitoring system is currently operational and data is flowing normally.",
        source="monitoring_incident",
        domain="monitoring",
        gold_relations=(("current", "SUPPORT"), ("stale", "CONTRADICT")),
    ),
    DocumentPassage(
        passage_id="P028",
        text="A new alert has been configured for CPU utilization above 90% for 5 consecutive minutes. This is a configuration change. The monitoring system status is not directly affected.",
        source="alert_configuration",
        domain="monitoring",
        gold_relations=(("current", "NEUTRAL"), ("stale", "NEUTRAL")),
    ),

    # --- Cache ---
    DocumentPassage(
        passage_id="P029",
        text="Redis cluster health check passed. All 6 nodes are responding. The Redis cache is currently operational with memory usage at 45% of total capacity.",
        source="status_report",
        domain="cache",
        gold_relations=(("current", "SUPPORT"), ("stale", "CONTRADICT")),
    ),
    DocumentPassage(
        passage_id="P030",
        text="The Redis cache was flushed to resolve a data consistency issue. The cache is now repopulating. The Redis cache is currently not fully operational during the warmup period.",
        source="cache_incident",
        domain="cache",
        gold_relations=(("current", "CONTRADICT"), ("stale", "NEUTRAL")),
    ),
    DocumentPassage(
        passage_id="P031",
        text="The Redis instance has been upgraded from v6 to v7. All existing data has been preserved. The Redis cache is currently operational on the new version.",
        source="upgrade_report",
        domain="cache",
        gold_relations=(("current", "SUPPORT"), ("stale", "CONTRADICT")),
    ),

    # --- Message queue ---
    DocumentPassage(
        passage_id="P032",
        text="The message queue is processing normally with no backlog. Consumer lag is at 0 messages. The message queue is currently operational at 12,000 messages per second.",
        source="status_report",
        domain="message_queue",
        gold_relations=(("current", "SUPPORT"), ("stale", "CONTRADICT")),
    ),
    DocumentPassage(
        passage_id="P033",
        text="A consumer group has fallen behind and is experiencing a backlog of 50,000 messages. The message queue is currently not fully operational. Some messages may be delayed.",
        source="queue_incident",
        domain="message_queue",
        gold_relations=(("current", "CONTRADICT"), ("stale", "NEUTRAL")),
    ),
    DocumentPassage(
        passage_id="P034",
        text="The message queue service has been migrated from RabbitMQ to Apache Kafka. The old RabbitMQ instance has been shut down. The message queue is currently operational on Kafka.",
        source="migration_report",
        domain="message_queue",
        gold_relations=(("current", "SUPPORT"), ("stale", "CONTRADICT")),
    ),

    # --- Load balancer ---
    DocumentPassage(
        passage_id="P035",
        text="The load balancer is distributing traffic evenly across all 8 backend servers. Health checks are passing for all targets. The load balancer is currently operational.",
        source="status_report",
        domain="load_balancer",
        gold_relations=(("current", "SUPPORT"), ("stale", "CONTRADICT")),
    ),
    DocumentPassage(
        passage_id="P036",
        text="Two backend servers behind the load balancer have been marked as unhealthy after failing consecutive health checks. The load balancer is currently not fully operational with only 6 healthy targets.",
        source="lb_incident",
        domain="load_balancer",
        gold_relations=(("current", "CONTRADICT"), ("stale", "NEUTRAL")),
    ),

    # --- Feature flags ---
    DocumentPassage(
        passage_id="P037",
        text="The new checkout flow has been enabled for 100% of users. The feature flag has been fully rolled out. The feature flag configuration is currently operational and confirmed.",
        source="feature_flag_report",
        domain="feature_flags",
        gold_relations=(("current", "SUPPORT"), ("stale", "CONTRADICT")),
    ),
    DocumentPassage(
        passage_id="P038",
        text="The experimental search feature has been disabled due to performance issues. The feature flag has been turned off. The feature flag configuration is currently not confirmed.",
        source="feature_flag_report",
        domain="feature_flags",
        gold_relations=(("current", "CONTRADICT"), ("stale", "NEUTRAL")),
    ),

    # --- Historical/stale evidence ---
    DocumentPassage(
        passage_id="P039",
        text="As of Q3 2023, the system was running on the old infrastructure with no known issues. This report is from 2023 and the infrastructure status is stale. The current status is not reflected in this report.",
        source="historical_report",
        domain="infrastructure",
        gold_relations=(("current", "CONTRADICT"), ("stale", "SUPPORT")),
    ),
    DocumentPassage(
        passage_id="P040",
        text="The 2022 annual report noted that all systems were operational at the time of writing. This report is retained for historical reference only. The infrastructure status is stale and does not reflect current system status.",
        source="historical_report",
        domain="infrastructure",
        gold_relations=(("current", "CONTRADICT"), ("stale", "SUPPORT")),
    ),

    # --- Ambiguous / neutral passages ---
    DocumentPassage(
        passage_id="P041",
        text="The engineering team is currently evaluating whether to migrate from PostgreSQL to MySQL. A decision is expected by the end of the quarter. No changes have been made yet. The database status is not directly affected by this evaluation.",
        source="planning_document",
        domain="database",
        gold_relations=(("current", "NEUTRAL"), ("stale", "NEUTRAL")),
    ),
    DocumentPassage(
        passage_id="P042",
        text="A new SLO for API response time has been proposed: 99th percentile under 200ms. This SLO is under review. The API gateway status is not directly affected by this proposal.",
        source="planning_document",
        domain="api_gateway",
        gold_relations=(("current", "NEUTRAL"), ("stale", "NEUTRAL")),
    ),
    DocumentPassage(
        passage_id="P043",
        text="The team has published a new runbook for handling database failover scenarios. The runbook is available in the internal documentation portal. The database status is not directly affected by this documentation.",
        source="documentation",
        domain="database",
        gold_relations=(("current", "NEUTRAL"), ("stale", "NEUTRAL")),
    ),
    DocumentPassage(
        passage_id="P044",
        text="Training sessions for the new monitoring platform will be held next week. All engineers are encouraged to attend. The monitoring system status is not directly affected by this training schedule.",
        source="announcement",
        domain="monitoring",
        gold_relations=(("current", "NEUTRAL"), ("stale", "NEUTRAL")),
    ),

    # --- Complex compositional passages ---
    DocumentPassage(
        passage_id="P045",
        text="Although the initial deployment failed due to a missing dependency, a hotfix was applied and the service is now operational. The deployment is currently confirmed operational despite the earlier failure.",
        source="incident_resolution",
        domain="deployment",
        gold_relations=(("current", "SUPPORT"), ("stale", "CONTRADICT")),
    ),
    DocumentPassage(
        passage_id="P046",
        text="The service was operational as of this morning's health check, but a subsequent network partition at 10:30 UTC has isolated it from the database. The service is currently not operational due to the network partition.",
        source="incident_report",
        domain="api_gateway",
        gold_relations=(("current", "CONTRADICT"), ("stale", "NEUTRAL")),
    ),
    DocumentPassage(
        passage_id="P047",
        text="If the DNS propagation completes as expected, the new endpoint will be reachable within the next hour. However, DNS propagation is currently delayed. The CDN status is currently not confirmed due to the DNS delay.",
        source="deployment_report",
        domain="cdn",
        gold_relations=(("current", "CONTRADICT"), ("stale", "NEUTRAL")),
    ),
    DocumentPassage(
        passage_id="P048",
        text="Source A reports the service is healthy, but Source A's monitoring probe was misconfigured and was checking a shadow environment. The actual production service is currently not operational and is experiencing elevated error rates.",
        source="conflicting_reports",
        domain="api_gateway",
        gold_relations=(("current", "CONTRADICT"), ("stale", "NEUTRAL")),
    ),
    DocumentPassage(
        passage_id="P049",
        text="The patch deployment failed because the target server ran out of disk space. The service is still running the old version, which is functional but missing the security fix. The security posture is currently operational but unpatched.",
        source="deployment_report",
        domain="security",
        gold_relations=(("current", "SUPPORT"), ("stale", "NEUTRAL")),
    ),
    DocumentPassage(
        passage_id="P050",
        text="The certificate renewal process encountered an error and did not complete. The current certificate is still valid but will expire in 3 days. The security posture is currently confirmed but at risk.",
        source="certificate_status",
        domain="security",
        gold_relations=(("current", "SUPPORT"), ("stale", "NEUTRAL")),
    ),
]


def get_corpus() -> list[DocumentPassage]:
    """Return the frozen document corpus."""
    return list(PASSAGES)


def corpus_sha256() -> str:
    """Compute SHA-256 of the corpus content."""
    content = json.dumps(
        [{"passage_id": p.passage_id, "text": p.text, "source": p.source,
          "domain": p.domain, "gold_relations": list(p.gold_relations)}
         for p in PASSAGES],
        sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


def save_corpus(path: Path) -> None:
    """Save the corpus to a JSON file."""
    data = {
        "corpus_id": "i3_13_document_corpus_v1",
        "n_passages": len(PASSAGES),
        "sha256": corpus_sha256(),
        "passages": [
            {
                "passage_id": p.passage_id,
                "text": p.text,
                "source": p.source,
                "domain": p.domain,
                "gold_relations": list(p.gold_relations),
            }
            for p in PASSAGES
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    print(f"Corpus: {len(PASSAGES)} passages")
    print(f"SHA-256: {corpus_sha256()[:16]}")
    by_domain = {}
    for p in PASSAGES:
        by_domain.setdefault(p.domain, 0)
        by_domain[p.domain] += 1
    print(f"By domain: {by_domain}")
