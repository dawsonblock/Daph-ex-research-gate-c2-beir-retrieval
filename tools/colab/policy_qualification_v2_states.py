"""POLICY_QUALIFICATION_V2: Expanded frozen policy qualification states.

80 states balanced across 4 core actions (VERIFY, ANSWER, DEFER, RETRIEVE)
with A1/M3 balance within each class.

Includes hard DEFER-vs-STOP, DEFER-vs-ANSWER, DEFER-vs-SEARCH_MORE cases:
  - insufficient evidence with no affordances
  - all hypotheses eliminated by conflicting SUFFICIENT evidence
  - falsified support with no remaining options
  - retrieval exhausted
  - verification exhausted

These states are FROZEN and must not be modified after first use.
"""
from __future__ import annotations


def make_qualification_v2_states() -> list[dict]:
    """Build 80 frozen POLICY_QUALIFICATION_V2 states.

    Distribution:
      20 VERIFY   (10 A1, 10 M3)
      20 ANSWER    (10 A1, 10 M3)
      20 DEFER     (10 A1, 10 M3) — includes hard neighbors
      20 RETRIEVE  (10 A1, 10 M3)
    """
    states = []

    # ================================================================
    # VERIFY states (20: 10 A1, 10 M3)
    # ================================================================

    # A1 VERIFY (10)
    states.append({
        "id": "V2_A1_VERIFY_001", "representation": "A1", "expected_action": "VERIFY",
        "scenario": "Evidence E1 is present but UNVERIFIED. It may support or contradict H1.",
        "evidence": [{"id": "E1", "proposition": "Monitoring shows service is responding.", "verified": False, "state": "UNVERIFIED", "supports": [], "contradicts": []}],
        "hypotheses": [{"id": "H1", "proposition": "Service is operational.", "status": "VIABLE"}, {"id": "H2", "proposition": "Service is not operational.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": True},
    })
    states.append({
        "id": "V2_A1_VERIFY_002", "representation": "A1", "expected_action": "VERIFY",
        "scenario": "Two evidence items are UNVERIFIED. Both need checking before deciding.",
        "evidence": [{"id": "E1", "proposition": "Probe A says service is up.", "verified": False, "state": "UNVERIFIED", "supports": [], "contradicts": []}, {"id": "E2", "proposition": "Probe B says service is down.", "verified": False, "state": "UNVERIFIED", "supports": [], "contradicts": []}],
        "hypotheses": [{"id": "H1", "proposition": "Service is operational.", "status": "VIABLE"}, {"id": "H2", "proposition": "Service is not operational.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": True},
    })
    states.append({
        "id": "V2_A1_VERIFY_003", "representation": "A1", "expected_action": "VERIFY",
        "scenario": "One evidence is SUFFICIENT (eliminates H2). Another is UNVERIFIED and may change the picture.",
        "evidence": [{"id": "E1", "proposition": "Health check passed.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}, {"id": "E2", "proposition": "Latency spike detected.", "verified": False, "state": "UNVERIFIED", "supports": [], "contradicts": []}],
        "hypotheses": [{"id": "H1", "proposition": "Service is operational.", "status": "VIABLE"}, {"id": "H2", "proposition": "Service is not operational.", "status": "ELIMINATED"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": True},
    })
    states.append({
        "id": "V2_A1_VERIFY_004", "representation": "A1", "expected_action": "VERIFY",
        "scenario": "Three UNVERIFIED evidence items from different sources. Need to verify before concluding.",
        "evidence": [{"id": "E1", "proposition": "DNS resolves correctly.", "verified": False, "state": "UNVERIFIED", "supports": [], "contradicts": []}, {"id": "E2", "proposition": "TCP connection succeeds.", "verified": False, "state": "UNVERIFIED", "supports": [], "contradicts": []}, {"id": "E3", "proposition": "TLS handshake fails.", "verified": False, "state": "UNVERIFIED", "supports": [], "contradicts": []}],
        "hypotheses": [{"id": "H1", "proposition": "Service is operational.", "status": "VIABLE"}, {"id": "H2", "proposition": "Service is not operational.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": True},
    })
    states.append({
        "id": "V2_A1_VERIFY_005", "representation": "A1", "expected_action": "VERIFY",
        "scenario": "Evidence E1 is UNVERIFIED and could eliminate H1 if confirmed. Verification is available.",
        "evidence": [{"id": "E1", "proposition": "Error rate is 100%.", "verified": False, "state": "UNVERIFIED", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "Service is operational.", "status": "VIABLE"}, {"id": "H2", "proposition": "Service is not operational.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": True},
    })
    states.append({
        "id": "V2_A1_VERIFY_006", "representation": "A1", "expected_action": "VERIFY",
        "scenario": "One evidence is verified MISSING (inconclusive). Another is UNVERIFIED. Must verify E2.",
        "evidence": [{"id": "E1", "proposition": "Old log entry shows service was up.", "verified": True, "state": "MISSING", "supports": [], "contradicts": []}, {"id": "E2", "proposition": "Recent ping timed out.", "verified": False, "state": "UNVERIFIED", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "Service is operational.", "status": "VIABLE"}, {"id": "H2", "proposition": "Service is not operational.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": True},
    })
    states.append({
        "id": "V2_A1_VERIFY_007", "representation": "A1", "expected_action": "VERIFY",
        "scenario": "Evidence E1 is UNVERIFIED and directly contradicts the current leading hypothesis.",
        "evidence": [{"id": "E1", "proposition": "Memory usage is at 99%.", "verified": False, "state": "UNVERIFIED", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "Service is healthy.", "status": "VIABLE"}, {"id": "H2", "proposition": "Service is overloaded.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": True},
    })
    states.append({
        "id": "V2_A1_VERIFY_008", "representation": "A1", "expected_action": "VERIFY",
        "scenario": "Two UNVERIFIED items point in opposite directions. Must verify both.",
        "evidence": [{"id": "E1", "proposition": "CPU usage is normal.", "verified": False, "state": "UNVERIFIED", "supports": ["H1"], "contradicts": ["H2"]}, {"id": "E2", "proposition": "Disk I/O is saturated.", "verified": False, "state": "UNVERIFIED", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "System is healthy.", "status": "VIABLE"}, {"id": "H2", "proposition": "System is degraded.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": True},
    })
    states.append({
        "id": "V2_A1_VERIFY_009", "representation": "A1", "expected_action": "VERIFY",
        "scenario": "Evidence E1 is UNVERIFIED and is the only evidence available. Verification is the only affordance.",
        "evidence": [{"id": "E1", "proposition": "Queue depth is 5000.", "verified": False, "state": "UNVERIFIED", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "Pipeline is processing normally.", "status": "VIABLE"}, {"id": "H2", "proposition": "Pipeline is backlogged.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": True},
    })
    states.append({
        "id": "V2_A1_VERIFY_010", "representation": "A1", "expected_action": "VERIFY",
        "scenario": "E1 is SUFFICIENT for H1. E2 is UNVERIFIED and could contradict E1. Must verify E2 before answering.",
        "evidence": [{"id": "E1", "proposition": "API returns 200 OK.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}, {"id": "E2", "proposition": "Response body is malformed.", "verified": False, "state": "UNVERIFIED", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "API is functional.", "status": "VIABLE"}, {"id": "H2", "proposition": "API is broken.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": True},
    })

    # M3 VERIFY (10)
    states.append({
        "id": "V2_M3_VERIFY_001", "representation": "M3", "expected_action": "VERIFY",
        "scenario": "T2 has not fired yet. E1 is SUFFICIENT (eliminates H1), E2 is UNVERIFIED. Must verify E2 before T2 can fire.",
        "evidence": [{"id": "E1", "proposition": "API gateway is offline.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}, {"id": "E2", "proposition": "API gateway is online.", "verified": False, "state": "UNVERIFIED", "supports": ["H1"], "contradicts": ["H2"]}],
        "hypotheses": [{"id": "H1", "proposition": "API gateway is operational.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "API gateway is not operational.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": True},
        "t2_fired": False,
    })
    states.append({
        "id": "V2_M3_VERIFY_002", "representation": "M3", "expected_action": "VERIFY",
        "scenario": "T2 has not fired. E1 eliminates H1. E2 is UNVERIFIED and might eliminate H2. Must verify E2.",
        "evidence": [{"id": "E1", "proposition": "Database is rejecting connections.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}, {"id": "E2", "proposition": "Database is accepting connections.", "verified": False, "state": "UNVERIFIED", "supports": ["H1"], "contradicts": ["H2"]}],
        "hypotheses": [{"id": "H1", "proposition": "Database is operational.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Database is not operational.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": True},
        "t2_fired": False,
    })
    states.append({
        "id": "V2_M3_VERIFY_003", "representation": "M3", "expected_action": "VERIFY",
        "scenario": "T2 has not fired. H1 is eliminated. E2 is UNVERIFIED and could support H1 if verified. Must verify.",
        "evidence": [{"id": "E1", "proposition": "CDN returns 503.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}, {"id": "E2", "proposition": "CDN returns 200 from edge.", "verified": False, "state": "UNVERIFIED", "supports": ["H1"], "contradicts": ["H2"]}],
        "hypotheses": [{"id": "H1", "proposition": "CDN is operational.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "CDN is not operational.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": True},
        "t2_fired": False,
    })
    states.append({
        "id": "V2_M3_VERIFY_004", "representation": "M3", "expected_action": "VERIFY",
        "scenario": "T2 has not fired. E1 is SUFFICIENT for H2. E2 and E3 are UNVERIFIED. Must verify E2 first.",
        "evidence": [{"id": "E1", "proposition": "Load balancer has no healthy backends.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}, {"id": "E2", "proposition": "Backend server 1 is responding.", "verified": False, "state": "UNVERIFIED", "supports": ["H1"], "contradicts": []}, {"id": "E3", "proposition": "Backend server 2 is responding.", "verified": False, "state": "UNVERIFIED", "supports": ["H1"], "contradicts": []}],
        "hypotheses": [{"id": "H1", "proposition": "Load balancer is functional.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Load balancer is misconfigured.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": True},
        "t2_fired": False,
    })
    states.append({
        "id": "V2_M3_VERIFY_005", "representation": "M3", "expected_action": "VERIFY",
        "scenario": "T2 has not fired. E1 eliminates H1. E2 is UNVERIFIED and might also eliminate H2, triggering T2.",
        "evidence": [{"id": "E1", "proposition": "Redis is down.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}, {"id": "E2", "proposition": "Redis is up.", "verified": False, "state": "UNVERIFIED", "supports": ["H1"], "contradicts": ["H2"]}],
        "hypotheses": [{"id": "H1", "proposition": "Redis is operational.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Redis is not operational.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": True},
        "t2_fired": False,
    })
    states.append({
        "id": "V2_M3_VERIFY_006", "representation": "M3", "expected_action": "VERIFY",
        "scenario": "T2 has not fired. E1 is SUFFICIENT against H1. E2 is UNVERIFIED with unknown support.",
        "evidence": [{"id": "E1", "proposition": "Kafka brokers are unreachable.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}, {"id": "E2", "proposition": "Zookeeper session is active.", "verified": False, "state": "UNVERIFIED", "supports": [], "contradicts": []}],
        "hypotheses": [{"id": "H1", "proposition": "Kafka cluster is healthy.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Kafka cluster is down.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": True},
        "t2_fired": False,
    })
    states.append({
        "id": "V2_M3_VERIFY_007", "representation": "M3", "expected_action": "VERIFY",
        "scenario": "T2 has not fired. E1 eliminates H1. E2 is UNVERIFIED and contradictory to E1.",
        "evidence": [{"id": "E1", "proposition": "Elasticsearch cluster is red.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}, {"id": "E2", "proposition": "Elasticsearch cluster is green.", "verified": False, "state": "UNVERIFIED", "supports": ["H1"], "contradicts": ["H2"]}],
        "hypotheses": [{"id": "H1", "proposition": "Elasticsearch is healthy.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Elasticsearch is unhealthy.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": True},
        "t2_fired": False,
    })
    states.append({
        "id": "V2_M3_VERIFY_008", "representation": "M3", "expected_action": "VERIFY",
        "scenario": "T2 has not fired. Two UNVERIFIED items could change the hypothesis status. Must verify E2 first.",
        "evidence": [{"id": "E1", "proposition": "Service mesh proxy is crashing.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}, {"id": "E2", "proposition": "Sidecar is stable.", "verified": False, "state": "UNVERIFIED", "supports": ["H1"], "contradicts": ["H2"]}],
        "hypotheses": [{"id": "H1", "proposition": "Service mesh is operational.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Service mesh is degraded.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": True},
        "t2_fired": False,
    })
    states.append({
        "id": "V2_M3_VERIFY_009", "representation": "M3", "expected_action": "VERIFY",
        "scenario": "T2 has not fired. E1 is SUFFICIENT. E2 is UNVERIFIED and could revive H1.",
        "evidence": [{"id": "E1", "proposition": "Vault is sealed.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}, {"id": "E2", "proposition": "Vault auto-unseal triggered.", "verified": False, "state": "UNVERIFIED", "supports": ["H1"], "contradicts": ["H2"]}],
        "hypotheses": [{"id": "H1", "proposition": "Vault is accessible.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Vault is inaccessible.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": True},
        "t2_fired": False,
    })
    states.append({
        "id": "V2_M3_VERIFY_010", "representation": "M3", "expected_action": "VERIFY",
        "scenario": "T2 has not fired. E1 eliminates H1. E2 is UNVERIFIED. If E2 is confirmed, T2 fires.",
        "evidence": [{"id": "E1", "proposition": "Prometheus target is down.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}, {"id": "E2", "proposition": "Prometheus target is up.", "verified": False, "state": "UNVERIFIED", "supports": ["H1"], "contradicts": ["H2"]}],
        "hypotheses": [{"id": "H1", "proposition": "Monitoring is operational.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Monitoring is down.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": True},
        "t2_fired": False,
    })

    # ================================================================
    # ANSWER states (20: 10 A1, 10 M3)
    # ================================================================

    # A1 ANSWER (10)
    states.append({
        "id": "V2_A1_ANSWER_001", "representation": "A1", "expected_action": "ANSWER",
        "scenario": "H1 is supported by SUFFICIENT verified evidence. H2 is eliminated.",
        "evidence": [{"id": "E1", "proposition": "Service is operational.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}],
        "hypotheses": [{"id": "H1", "proposition": "Service is operational.", "status": "VIABLE"}, {"id": "H2", "proposition": "Service is not operational.", "status": "ELIMINATED"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "V2_A1_ANSWER_002", "representation": "A1", "expected_action": "ANSWER",
        "scenario": "H1 is confirmed by two SUFFICIENT verified evidence items. H2 is eliminated.",
        "evidence": [{"id": "E1", "proposition": "Health check passed.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}, {"id": "E2", "proposition": "Synthetic test succeeded.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}],
        "hypotheses": [{"id": "H1", "proposition": "Service is operational.", "status": "VIABLE"}, {"id": "H2", "proposition": "Service is not operational.", "status": "ELIMINATED"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "V2_A1_ANSWER_003", "representation": "A1", "expected_action": "ANSWER",
        "scenario": "Kubernetes cluster is confirmed operational by SUFFICIENT evidence. H2 eliminated.",
        "evidence": [{"id": "E1", "proposition": "All nodes are Ready and pods scheduled normally.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}],
        "hypotheses": [{"id": "H1", "proposition": "Kubernetes cluster is operational.", "status": "VIABLE"}, {"id": "H2", "proposition": "Kubernetes cluster is not operational.", "status": "ELIMINATED"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "V2_A1_ANSWER_004", "representation": "A1", "expected_action": "ANSWER",
        "scenario": "H2 is confirmed by SUFFICIENT evidence. H1 is eliminated. Answer with H2.",
        "evidence": [{"id": "E1", "proposition": "Service is returning 500 errors.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "Service is operational.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Service is not operational.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "V2_A1_ANSWER_005", "representation": "A1", "expected_action": "ANSWER",
        "scenario": "Three SUFFICIENT evidence items all support H1. H2 is eliminated.",
        "evidence": [{"id": "E1", "proposition": "API responds correctly.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}, {"id": "E2", "proposition": "Database queries succeed.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}, {"id": "E3", "proposition": "Cache hit rate is normal.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}],
        "hypotheses": [{"id": "H1", "proposition": "System is fully operational.", "status": "VIABLE"}, {"id": "H2", "proposition": "System is degraded.", "status": "ELIMINATED"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "V2_A1_ANSWER_006", "representation": "A1", "expected_action": "ANSWER",
        "scenario": "H1 is supported by SUFFICIENT evidence from multiple independent sources.",
        "evidence": [{"id": "E1", "proposition": "Uptime monitoring shows 100%.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}, {"id": "E2", "proposition": "User reports are positive.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}],
        "hypotheses": [{"id": "H1", "proposition": "Service is healthy.", "status": "VIABLE"}, {"id": "H2", "proposition": "Service is unhealthy.", "status": "ELIMINATED"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "V2_A1_ANSWER_007", "representation": "A1", "expected_action": "ANSWER",
        "scenario": "Single SUFFICIENT evidence confirms H2. H1 is eliminated.",
        "evidence": [{"id": "E1", "proposition": "Process crashed with OOM.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "Process is running.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Process is dead.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "V2_A1_ANSWER_008", "representation": "A1", "expected_action": "ANSWER",
        "scenario": "H1 is confirmed. All evidence is SUFFICIENT and verified. No UNVERIFIED items remain.",
        "evidence": [{"id": "E1", "proposition": "Deployment succeeded.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}, {"id": "E2", "proposition": "Smoke tests passed.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}],
        "hypotheses": [{"id": "H1", "proposition": "Deployment is successful.", "status": "VIABLE"}, {"id": "H2", "proposition": "Deployment failed.", "status": "ELIMINATED"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "V2_A1_ANSWER_009", "representation": "A1", "expected_action": "ANSWER",
        "scenario": "H2 is the only viable hypothesis. SUFFICIENT evidence eliminates H1.",
        "evidence": [{"id": "E1", "proposition": "Disk is full.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "System can write data.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "System cannot write data.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "V2_A1_ANSWER_010", "representation": "A1", "expected_action": "ANSWER",
        "scenario": "H1 is confirmed by SUFFICIENT evidence. H2 is eliminated. No further action needed.",
        "evidence": [{"id": "E1", "proposition": "SSL certificate is valid.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}, {"id": "E2", "proposition": "SSL handshake succeeds.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}],
        "hypotheses": [{"id": "H1", "proposition": "SSL is valid.", "status": "VIABLE"}, {"id": "H2", "proposition": "SSL is invalid.", "status": "ELIMINATED"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
    })

    # M3 ANSWER (10)
    states.append({
        "id": "V2_M3_ANSWER_001", "representation": "M3", "expected_action": "ANSWER",
        "scenario": "T2 has not fired. H1 is VIABLE with SUFFICIENT evidence. H2 is eliminated. Answer with H1.",
        "evidence": [{"id": "E1", "proposition": "API gateway returns 200.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}],
        "hypotheses": [{"id": "H1", "proposition": "API gateway is operational.", "status": "VIABLE"}, {"id": "H2", "proposition": "API gateway is not operational.", "status": "ELIMINATED"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
        "t2_fired": False,
    })
    states.append({
        "id": "V2_M3_ANSWER_002", "representation": "M3", "expected_action": "ANSWER",
        "scenario": "T2 has not fired. H2 is VIABLE with SUFFICIENT evidence. H1 is eliminated.",
        "evidence": [{"id": "E1", "proposition": "Database is down.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "Database is operational.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Database is not operational.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
        "t2_fired": False,
    })
    states.append({
        "id": "V2_M3_ANSWER_003", "representation": "M3", "expected_action": "ANSWER",
        "scenario": "T2 has not fired. H1 is confirmed by two SUFFICIENT items. H2 eliminated.",
        "evidence": [{"id": "E1", "proposition": "CDN is serving content.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}, {"id": "E2", "proposition": "CDN cache hit rate is high.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}],
        "hypotheses": [{"id": "H1", "proposition": "CDN is operational.", "status": "VIABLE"}, {"id": "H2", "proposition": "CDN is not operational.", "status": "ELIMINATED"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
        "t2_fired": False,
    })
    states.append({
        "id": "V2_M3_ANSWER_004", "representation": "M3", "expected_action": "ANSWER",
        "scenario": "T2 has not fired. H2 is confirmed. H1 is eliminated by SUFFICIENT evidence.",
        "evidence": [{"id": "E1", "proposition": "Redis is rejecting commands.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "Redis is operational.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Redis is not operational.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
        "t2_fired": False,
    })
    states.append({
        "id": "V2_M3_ANSWER_005", "representation": "M3", "expected_action": "ANSWER",
        "scenario": "T2 has not fired. H1 is VIABLE. Three SUFFICIENT items confirm it.",
        "evidence": [{"id": "E1", "proposition": "Kafka is producing.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}, {"id": "E2", "proposition": "Kafka is consuming.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}, {"id": "E3", "proposition": "Lag is zero.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}],
        "hypotheses": [{"id": "H1", "proposition": "Kafka is healthy.", "status": "VIABLE"}, {"id": "H2", "proposition": "Kafka is unhealthy.", "status": "ELIMINATED"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
        "t2_fired": False,
    })
    states.append({
        "id": "V2_M3_ANSWER_006", "representation": "M3", "expected_action": "ANSWER",
        "scenario": "T2 has not fired. H2 is the only viable hypothesis. H1 eliminated.",
        "evidence": [{"id": "E1", "proposition": "Elasticsearch is red.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "Elasticsearch is healthy.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Elasticsearch is unhealthy.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
        "t2_fired": False,
    })
    states.append({
        "id": "V2_M3_ANSWER_007", "representation": "M3", "expected_action": "ANSWER",
        "scenario": "T2 has not fired. H1 is confirmed by SUFFICIENT evidence. No UNVERIFIED items.",
        "evidence": [{"id": "E1", "proposition": "Load balancer is healthy.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}, {"id": "E2", "proposition": "All backends are up.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}],
        "hypotheses": [{"id": "H1", "proposition": "Load balancer is operational.", "status": "VIABLE"}, {"id": "H2", "proposition": "Load balancer is misconfigured.", "status": "ELIMINATED"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
        "t2_fired": False,
    })
    states.append({
        "id": "V2_M3_ANSWER_008", "representation": "M3", "expected_action": "ANSWER",
        "scenario": "T2 has not fired. H2 is confirmed. H1 eliminated by SUFFICIENT evidence.",
        "evidence": [{"id": "E1", "proposition": "Vault is sealed.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "Vault is accessible.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Vault is inaccessible.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
        "t2_fired": False,
    })
    states.append({
        "id": "V2_M3_ANSWER_009", "representation": "M3", "expected_action": "ANSWER",
        "scenario": "T2 has not fired. H1 is VIABLE. SUFFICIENT evidence from monitoring.",
        "evidence": [{"id": "E1", "proposition": "Prometheus is scraping.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}, {"id": "E2", "proposition": "Alerts are firing correctly.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}],
        "hypotheses": [{"id": "H1", "proposition": "Monitoring is operational.", "status": "VIABLE"}, {"id": "H2", "proposition": "Monitoring is down.", "status": "ELIMINATED"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
        "t2_fired": False,
    })
    states.append({
        "id": "V2_M3_ANSWER_010", "representation": "M3", "expected_action": "ANSWER",
        "scenario": "T2 has not fired. H2 is confirmed. H1 eliminated. SUFFICIENT evidence.",
        "evidence": [{"id": "E1", "proposition": "Service mesh is crashing.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "Service mesh is operational.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Service mesh is degraded.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
        "t2_fired": False,
    })

    # ================================================================
    # DEFER states (20: 10 A1, 10 M3) — includes hard neighbors
    # ================================================================

    # A1 DEFER (10) — hard cases
    states.append({
        "id": "V2_A1_DEFER_001", "representation": "A1", "expected_action": "DEFER",
        "scenario": "All hypotheses eliminated by SUFFICIENT contradicting evidence. Evidence set is inconsistent. No resolution possible.",
        "evidence": [{"id": "E1", "proposition": "Service is operational.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}, {"id": "E2", "proposition": "Service is not operational.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "Service is operational.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Service is not operational.", "status": "ELIMINATED"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "V2_A1_DEFER_002", "representation": "A1", "expected_action": "DEFER",
        "scenario": "Evidence is insufficient and no more retrieval or verification is possible.",
        "evidence": [{"id": "E1", "proposition": "Service might be operational.", "verified": True, "state": "MISSING", "supports": [], "contradicts": []}],
        "hypotheses": [{"id": "H1", "proposition": "Service is operational.", "status": "VIABLE"}, {"id": "H2", "proposition": "Service is not operational.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "V2_A1_DEFER_003", "representation": "A1", "expected_action": "DEFER",
        "scenario": "All evidence is verified but none is SUFFICIENT. Cannot retrieve or search more.",
        "evidence": [{"id": "E1", "proposition": "Service was up 1 hour ago.", "verified": True, "state": "MISSING", "supports": [], "contradicts": []}, {"id": "E2", "proposition": "No recent reports available.", "verified": True, "state": "MISSING", "supports": [], "contradicts": []}],
        "hypotheses": [{"id": "H1", "proposition": "Service is operational.", "status": "VIABLE"}, {"id": "H2", "proposition": "Service is not operational.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "V2_A1_DEFER_004", "representation": "A1", "expected_action": "DEFER",
        "scenario": "Retrieval exhausted. Verification exhausted. Evidence is MISSING. Cannot determine status.",
        "evidence": [{"id": "E1", "proposition": "Partial log shows intermittent errors.", "verified": True, "state": "MISSING", "supports": [], "contradicts": []}],
        "hypotheses": [{"id": "H1", "proposition": "Service is stable.", "status": "VIABLE"}, {"id": "H2", "proposition": "Service is flaky.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "V2_A1_DEFER_005", "representation": "A1", "expected_action": "DEFER",
        "scenario": "All hypotheses eliminated. Falsified support. No remaining options. NOT a STOP — evidence is genuinely insufficient.",
        "evidence": [{"id": "E1", "proposition": "Test A says pass.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}, {"id": "E2", "proposition": "Test B says fail.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "Feature works.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Feature is broken.", "status": "ELIMINATED"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "V2_A1_DEFER_006", "representation": "A1", "expected_action": "DEFER",
        "scenario": "Conflicting SUFFICIENT evidence. Both hypotheses eliminated. This is NOT an execution error — defer because evidence is contradictory.",
        "evidence": [{"id": "E1", "proposition": "CPU is at 10%.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}, {"id": "E2", "proposition": "CPU is at 95%.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "System is idle.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "System is overloaded.", "status": "ELIMINATED"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "V2_A1_DEFER_007", "representation": "A1", "expected_action": "DEFER",
        "scenario": "No evidence at all. No affordances. Cannot determine anything. Defer due to insufficient evidence.",
        "evidence": [],
        "hypotheses": [{"id": "H1", "proposition": "Service is operational.", "status": "VIABLE"}, {"id": "H2", "proposition": "Service is not operational.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "V2_A1_DEFER_008", "representation": "A1", "expected_action": "DEFER",
        "scenario": "All evidence is MISSING. Retrieval and search are exhausted. Hypotheses remain viable but undetermined.",
        "evidence": [{"id": "E1", "proposition": "Old metric shows normal.", "verified": True, "state": "MISSING", "supports": [], "contradicts": []}, {"id": "E2", "proposition": "Another old metric shows anomaly.", "verified": True, "state": "MISSING", "supports": [], "contradicts": []}],
        "hypotheses": [{"id": "H1", "proposition": "System is normal.", "status": "VIABLE"}, {"id": "H2", "proposition": "System is anomalous.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "V2_A1_DEFER_009", "representation": "A1", "expected_action": "DEFER",
        "scenario": "Verification exhausted. All evidence UNVERIFIED but no verification available. Evidence cannot be trusted. Defer.",
        "evidence": [{"id": "E1", "proposition": "Unverified report of outage.", "verified": False, "state": "UNVERIFIED", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "Service is operational.", "status": "VIABLE"}, {"id": "H2", "proposition": "Service is not operational.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "V2_A1_DEFER_010", "representation": "A1", "expected_action": "DEFER",
        "scenario": "Both hypotheses eliminated by conflicting SUFFICIENT evidence. NOT STOP — this is an epistemic insufficiency, not an execution error.",
        "evidence": [{"id": "E1", "proposition": "Memory is 20%.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}, {"id": "E2", "proposition": "Memory is 90%.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "Memory is fine.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Memory is pressured.", "status": "ELIMINATED"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
    })

    # M3 DEFER (10) — hard cases with T2 fired
    states.append({
        "id": "V2_M3_DEFER_001", "representation": "M3", "expected_action": "DEFER",
        "scenario": "T2 has fired. All hypotheses eliminated by SUFFICIENT contradicting evidence. Evidence set is inconsistent. No resolution possible.",
        "evidence": [{"id": "E1", "proposition": "Authoritative probe: service is NOT operational.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}, {"id": "E2", "proposition": "Authoritative probe: service IS operational.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}],
        "hypotheses": [{"id": "H1", "proposition": "Service is operational.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Service is not operational.", "status": "ELIMINATED"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
        "t2_fired": True,
    })
    states.append({
        "id": "V2_M3_DEFER_002", "representation": "M3", "expected_action": "DEFER",
        "scenario": "T2 has fired. Both hypotheses eliminated. Evidence is contradictory and SUFFICIENT. Must defer — hypothesis set cannot be resolved.",
        "evidence": [{"id": "E1", "proposition": "Database is rejecting all queries.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}, {"id": "E2", "proposition": "Database is accepting all queries.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}],
        "hypotheses": [{"id": "H1", "proposition": "Database is operational.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Database is not operational.", "status": "ELIMINATED"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
        "t2_fired": True,
    })
    states.append({
        "id": "V2_M3_DEFER_003", "representation": "M3", "expected_action": "DEFER",
        "scenario": "T2 has fired. All hypotheses eliminated. CDN conflict evidence is SUFFICIENT on both sides.",
        "evidence": [{"id": "E1", "proposition": "CDN edge check: all locations returning 5xx.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}, {"id": "E2", "proposition": "CDN edge test: all locations serving content.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}],
        "hypotheses": [{"id": "H1", "proposition": "CDN is operational.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "CDN is not operational.", "status": "ELIMINATED"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
        "t2_fired": True,
    })
    states.append({
        "id": "V2_M3_DEFER_004", "representation": "M3", "expected_action": "DEFER",
        "scenario": "T2 has fired. All hypotheses eliminated by conflicting SUFFICIENT evidence. NOT STOP — epistemic insufficiency.",
        "evidence": [{"id": "E1", "proposition": "Redis is responding.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}, {"id": "E2", "proposition": "Redis is timing out.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "Redis is operational.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Redis is not operational.", "status": "ELIMINATED"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
        "t2_fired": True,
    })
    states.append({
        "id": "V2_M3_DEFER_005", "representation": "M3", "expected_action": "DEFER",
        "scenario": "T2 has fired. Conflicting SUFFICIENT evidence eliminates all hypotheses. Defer — cannot resolve.",
        "evidence": [{"id": "E1", "proposition": "Kafka is producing normally.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}, {"id": "E2", "proposition": "Kafka is not producing.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "Kafka is healthy.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Kafka is unhealthy.", "status": "ELIMINATED"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
        "t2_fired": True,
    })
    states.append({
        "id": "V2_M3_DEFER_006", "representation": "M3", "expected_action": "DEFER",
        "scenario": "T2 has fired. All evidence SUFFICIENT but contradictory. No affordances. Defer due to epistemic insufficiency.",
        "evidence": [{"id": "E1", "proposition": "ES cluster is green.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}, {"id": "E2", "proposition": "ES cluster is red.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "Elasticsearch is healthy.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Elasticsearch is unhealthy.", "status": "ELIMINATED"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
        "t2_fired": True,
    })
    states.append({
        "id": "V2_M3_DEFER_007", "representation": "M3", "expected_action": "DEFER",
        "scenario": "T2 has fired. Both hypotheses eliminated. Falsified support. No resolution. Defer.",
        "evidence": [{"id": "E1", "proposition": "Load balancer is healthy.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}, {"id": "E2", "proposition": "Load balancer is unhealthy.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "Load balancer is operational.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Load balancer is misconfigured.", "status": "ELIMINATED"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
        "t2_fired": True,
    })
    states.append({
        "id": "V2_M3_DEFER_008", "representation": "M3", "expected_action": "DEFER",
        "scenario": "T2 has fired. Conflicting SUFFICIENT evidence. NOT STOP. Defer because evidence is contradictory and unresolvable.",
        "evidence": [{"id": "E1", "proposition": "Vault is unsealed.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}, {"id": "E2", "proposition": "Vault is sealed.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "Vault is accessible.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Vault is inaccessible.", "status": "ELIMINATED"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
        "t2_fired": True,
    })
    states.append({
        "id": "V2_M3_DEFER_009", "representation": "M3", "expected_action": "DEFER",
        "scenario": "T2 has fired. All hypotheses eliminated. SUFFICIENT evidence on both sides. NOT an execution error.",
        "evidence": [{"id": "E1", "proposition": "Prometheus is up.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}, {"id": "E2", "proposition": "Prometheus is down.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "Monitoring is operational.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Monitoring is down.", "status": "ELIMINATED"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
        "t2_fired": True,
    })
    states.append({
        "id": "V2_M3_DEFER_010", "representation": "M3", "expected_action": "DEFER",
        "scenario": "T2 has fired. Both eliminated by SUFFICIENT conflict. Retrieval and verification exhausted. Defer — epistemic dead end.",
        "evidence": [{"id": "E1", "proposition": "Service mesh is stable.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}, {"id": "E2", "proposition": "Service mesh is crashing.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "Service mesh is operational.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Service mesh is degraded.", "status": "ELIMINATED"}],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
        "t2_fired": True,
    })

    # ================================================================
    # RETRIEVE states (20: 10 A1, 10 M3)
    # ================================================================

    # A1 RETRIEVE (10)
    states.append({
        "id": "V2_A1_RETRIEVE_001", "representation": "A1", "expected_action": "RETRIEVE",
        "scenario": "No evidence has been retrieved yet. Retrieval is available.",
        "evidence": [],
        "hypotheses": [{"id": "H1", "proposition": "Service is operational.", "status": "VIABLE"}, {"id": "H2", "proposition": "Service is not operational.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": True, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "V2_A1_RETRIEVE_002", "representation": "A1", "expected_action": "RETRIEVE",
        "scenario": "Initial evidence is retrieved but insufficient. Retrieval is still available for more evidence.",
        "evidence": [{"id": "E1", "proposition": "Old report says service was up.", "verified": True, "state": "MISSING", "supports": [], "contradicts": []}],
        "hypotheses": [{"id": "H1", "proposition": "Service is operational.", "status": "VIABLE"}, {"id": "H2", "proposition": "Service is not operational.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": True, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "V2_A1_RETRIEVE_003", "representation": "A1", "expected_action": "RETRIEVE",
        "scenario": "One evidence item is MISSING. Retrieval available to get more data.",
        "evidence": [{"id": "E1", "proposition": "Partial metric available.", "verified": True, "state": "MISSING", "supports": [], "contradicts": []}],
        "hypotheses": [{"id": "H1", "proposition": "System is healthy.", "status": "VIABLE"}, {"id": "H2", "proposition": "System is unhealthy.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": True, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "V2_A1_RETRIEVE_004", "representation": "A1", "expected_action": "RETRIEVE",
        "scenario": "No evidence available. Retrieval is the only affordance. Must retrieve to get any data.",
        "evidence": [],
        "hypotheses": [{"id": "H1", "proposition": "API is functional.", "status": "VIABLE"}, {"id": "H2", "proposition": "API is broken.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": True, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "V2_A1_RETRIEVE_005", "representation": "A1", "expected_action": "RETRIEVE",
        "scenario": "Current evidence is all MISSING. Retrieval available to get SUFFICIENT evidence.",
        "evidence": [{"id": "E1", "proposition": "Stale cache entry.", "verified": True, "state": "MISSING", "supports": [], "contradicts": []}],
        "hypotheses": [{"id": "H1", "proposition": "Cache is valid.", "status": "VIABLE"}, {"id": "H2", "proposition": "Cache is invalid.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": True, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "V2_A1_RETRIEVE_006", "representation": "A1", "expected_action": "RETRIEVE",
        "scenario": "Empty evidence set. Retrieval available. Two viable hypotheses need evidence.",
        "evidence": [],
        "hypotheses": [{"id": "H1", "proposition": "Database is up.", "status": "VIABLE"}, {"id": "H2", "proposition": "Database is down.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": True, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "V2_A1_RETRIEVE_007", "representation": "A1", "expected_action": "RETRIEVE",
        "scenario": "One MISSING evidence. Retrieval available. Need more data to distinguish hypotheses.",
        "evidence": [{"id": "E1", "proposition": "Connection count unknown.", "verified": True, "state": "MISSING", "supports": [], "contradicts": []}],
        "hypotheses": [{"id": "H1", "proposition": "Connections are normal.", "status": "VIABLE"}, {"id": "H2", "proposition": "Connections are exhausted.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": True, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "V2_A1_RETRIEVE_008", "representation": "A1", "expected_action": "RETRIEVE",
        "scenario": "No evidence retrieved. Retrieval is available. Must get data before any decision.",
        "evidence": [],
        "hypotheses": [{"id": "H1", "proposition": "Queue is processing.", "status": "VIABLE"}, {"id": "H2", "proposition": "Queue is stalled.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": True, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "V2_A1_RETRIEVE_009", "representation": "A1", "expected_action": "RETRIEVE",
        "scenario": "Evidence is MISSING. Retrieval available. Need to get current status.",
        "evidence": [{"id": "E1", "proposition": "Last known status was OK.", "verified": True, "state": "MISSING", "supports": [], "contradicts": []}],
        "hypotheses": [{"id": "H1", "proposition": "Status is still OK.", "status": "VIABLE"}, {"id": "H2", "proposition": "Status has changed.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": True, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "V2_A1_RETRIEVE_010", "representation": "A1", "expected_action": "RETRIEVE",
        "scenario": "Empty evidence. Retrieval available. Two hypotheses about disk status.",
        "evidence": [],
        "hypotheses": [{"id": "H1", "proposition": "Disk has space.", "status": "VIABLE"}, {"id": "H2", "proposition": "Disk is full.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": True, "can_search": False, "can_verify": False},
    })

    # M3 RETRIEVE (10)
    states.append({
        "id": "V2_M3_RETRIEVE_001", "representation": "M3", "expected_action": "RETRIEVE",
        "scenario": "T2 has not fired. E1 is SUFFICIENT (eliminates H1). E2 is hidden — must retrieve it first.",
        "evidence": [{"id": "E1", "proposition": "CDN is offline.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "CDN is operational.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "CDN is not operational.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": True, "can_search": False, "can_verify": True},
        "t2_fired": False,
    })
    states.append({
        "id": "V2_M3_RETRIEVE_002", "representation": "M3", "expected_action": "RETRIEVE",
        "scenario": "T2 has not fired. E1 eliminates H1. More evidence needed. Retrieval available.",
        "evidence": [{"id": "E1", "proposition": "API is returning 500.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "API is functional.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "API is broken.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": True, "can_search": False, "can_verify": False},
        "t2_fired": False,
    })
    states.append({
        "id": "V2_M3_RETRIEVE_003", "representation": "M3", "expected_action": "RETRIEVE",
        "scenario": "T2 has not fired. E1 is SUFFICIENT. H2 is viable but needs more evidence. Retrieve.",
        "evidence": [{"id": "E1", "proposition": "Database is down.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "Database is operational.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Database is not operational.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": True, "can_search": False, "can_verify": False},
        "t2_fired": False,
    })
    states.append({
        "id": "V2_M3_RETRIEVE_004", "representation": "M3", "expected_action": "RETRIEVE",
        "scenario": "T2 has not fired. E1 eliminates H1. Retrieval available for more evidence.",
        "evidence": [{"id": "E1", "proposition": "Redis is down.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "Redis is operational.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Redis is not operational.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": True, "can_search": False, "can_verify": False},
        "t2_fired": False,
    })
    states.append({
        "id": "V2_M3_RETRIEVE_005", "representation": "M3", "expected_action": "RETRIEVE",
        "scenario": "T2 has not fired. E1 is SUFFICIENT. Need more evidence about H2. Retrieve.",
        "evidence": [{"id": "E1", "proposition": "Kafka is down.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "Kafka is healthy.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Kafka is unhealthy.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": True, "can_search": False, "can_verify": False},
        "t2_fired": False,
    })
    states.append({
        "id": "V2_M3_RETRIEVE_006", "representation": "M3", "expected_action": "RETRIEVE",
        "scenario": "T2 has not fired. E1 eliminates H1. Retrieval available. Need to confirm H2.",
        "evidence": [{"id": "E1", "proposition": "ES is red.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "Elasticsearch is healthy.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Elasticsearch is unhealthy.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": True, "can_search": False, "can_verify": False},
        "t2_fired": False,
    })
    states.append({
        "id": "V2_M3_RETRIEVE_007", "representation": "M3", "expected_action": "RETRIEVE",
        "scenario": "T2 has not fired. E1 is SUFFICIENT. H2 needs more evidence. Retrieve.",
        "evidence": [{"id": "E1", "proposition": "Load balancer has no healthy backends.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "Load balancer is operational.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Load balancer is misconfigured.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": True, "can_search": False, "can_verify": False},
        "t2_fired": False,
    })
    states.append({
        "id": "V2_M3_RETRIEVE_008", "representation": "M3", "expected_action": "RETRIEVE",
        "scenario": "T2 has not fired. E1 eliminates H1. Retrieval available for confirmation.",
        "evidence": [{"id": "E1", "proposition": "Vault is sealed.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "Vault is accessible.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Vault is inaccessible.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": True, "can_search": False, "can_verify": False},
        "t2_fired": False,
    })
    states.append({
        "id": "V2_M3_RETRIEVE_009", "representation": "M3", "expected_action": "RETRIEVE",
        "scenario": "T2 has not fired. E1 is SUFFICIENT. Need more data. Retrieve.",
        "evidence": [{"id": "E1", "proposition": "Prometheus is down.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "Monitoring is operational.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Monitoring is down.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": True, "can_search": False, "can_verify": False},
        "t2_fired": False,
    })
    states.append({
        "id": "V2_M3_RETRIEVE_010", "representation": "M3", "expected_action": "RETRIEVE",
        "scenario": "T2 has not fired. E1 eliminates H1. Retrieval available. Get more evidence.",
        "evidence": [{"id": "E1", "proposition": "Service mesh is crashing.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}],
        "hypotheses": [{"id": "H1", "proposition": "Service mesh is operational.", "status": "ELIMINATED"}, {"id": "H2", "proposition": "Service mesh is degraded.", "status": "VIABLE"}],
        "affordances": {"can_retrieve": True, "can_search": False, "can_verify": False},
        "t2_fired": False,
    })

    return states
