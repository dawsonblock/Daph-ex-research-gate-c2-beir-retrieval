"""Tests for I3.5.1 provenance DAG."""
import pytest
from hrm_adaptive_memory.executive.i3_5_1.provenance import (
    ProvenanceNode, build_results_provenance, build_scores_provenance,
    build_stats_provenance, build_report_provenance,
    verify_provenance_chain,
)


class TestProvenanceDAG:
    def test_results_provenance(self):
        node = build_results_provenance(
            results_sha256="res_sha",
            source_receipts_sha256="rec_sha",
            receipt_chain_root="root_sha",
            experiment_identity_sha256="exp_sha",
        )
        assert node.artifact_type == "results"
        assert node.artifact_sha256 == "res_sha"
        assert node.source_artifacts["receipts"] == "rec_sha"
        assert node.experiment_identity_sha256 == "exp_sha"

    def test_scores_provenance(self):
        node = build_scores_provenance(
            scores_sha256="sco_sha",
            source_results_sha256="res_sha",
            experiment_identity_sha256="exp_sha",
        )
        assert node.artifact_type == "scores"
        assert node.source_artifacts["results"] == "res_sha"

    def test_stats_provenance(self):
        node = build_stats_provenance(
            stats_sha256="sta_sha",
            source_results_sha256="res_sha",
            source_scores_sha256="sco_sha",
            statistics_implementation_sha256="impl_sha",
            experiment_identity_sha256="exp_sha",
        )
        assert node.artifact_type == "stats"
        assert node.source_artifacts["results"] == "res_sha"
        assert node.source_artifacts["scores"] == "sco_sha"

    def test_report_provenance(self):
        node = build_report_provenance(
            report_sha256="rep_sha",
            source_stats_sha256="sta_sha",
            source_analysis_sha256="ana_sha",
            source_run_id="run_001",
            experiment_identity_sha256="exp_sha",
        )
        assert node.artifact_type == "report"
        assert node.source_artifacts["stats"] == "sta_sha"

    def test_valid_chain_verifies(self):
        nodes = [
            build_results_provenance(
                results_sha256="res_sha",
                source_receipts_sha256="rec_sha",
                receipt_chain_root="root",
                experiment_identity_sha256="exp",
            ),
            build_scores_provenance(
                scores_sha256="sco_sha",
                source_results_sha256="res_sha",
                experiment_identity_sha256="exp",
            ),
            build_stats_provenance(
                stats_sha256="sta_sha",
                source_results_sha256="res_sha",
                source_scores_sha256="sco_sha",
                statistics_implementation_sha256="impl",
                experiment_identity_sha256="exp",
            ),
        ]
        assert verify_provenance_chain(nodes) is True

    def test_broken_chain_detected(self):
        nodes = [
            build_results_provenance(
                results_sha256="res_sha",
                source_receipts_sha256="rec_sha",
                receipt_chain_root="root",
                experiment_identity_sha256="exp",
            ),
            build_scores_provenance(
                scores_sha256="sco_sha",
                source_results_sha256="WRONG_sha",  # Mismatch
                experiment_identity_sha256="exp",
            ),
        ]
        assert verify_provenance_chain(nodes) is False
