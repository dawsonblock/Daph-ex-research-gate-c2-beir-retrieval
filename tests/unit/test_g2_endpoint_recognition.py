"""G2-v2: leakage wall, import-boundary, and endpoint-recognition correctness.

Grounded in the corpus's own generator grammar (transcribed from
generalization_dataset_v4.py:_render, 6 styles x 4 variants) rather than
invented sentence forms, so a pass here says something about the real corpus,
not just about a toy fixture.
"""
from __future__ import annotations

import inspect
import io
import tokenize

import pytest

from hrm_adaptive_memory.c4.endpoint_recognition import (
    COMPLETION_MODES, k0_literal_completion, k1_entity_bound_exact_completion,
    k2_entity_bound_family_completion)
from hrm_adaptive_memory.c4.oracle_endpoint_ceiling import make_oracle_completion_fn
from hrm_adaptive_memory.c4.relation_grammar import (
    bindings_for_entity, parse_relation_bindings)


def _executable_source(module) -> str:
    source = inspect.getsource(module)
    kept: list[str] = []
    previous_type = tokenize.INDENT
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            continue
        if token.type == tokenize.STRING and previous_type in (
                tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE,
                tokenize.NL, tokenize.ENCODING):
            previous_type = token.type
            continue
        kept.append(token.string)
        if token.type not in (tokenize.NL, tokenize.NEWLINE):
            previous_type = token.type
    return " ".join(kept)


RUNTIME_MODULES = ["hrm_adaptive_memory.c4.endpoint_recognition",
                   "hrm_adaptive_memory.c4.relation_grammar",
                   "hrm_adaptive_memory.c4.g2_paths"]
FORBIDDEN = ["required_evidence_ids", "proof_edges", "record_kind", "latent_bridge",
             "answer_node", "oracle_bridge", "oracle_evidence_ids", "gold_path",
             "_required", "answer_key", "_oracle_metadata"]


class TestLeakageWall:
    @pytest.mark.parametrize("module_name", RUNTIME_MODULES)
    @pytest.mark.parametrize("forbidden", FORBIDDEN)
    def test_runtime_module_never_references_an_oracle_field(self, module_name, forbidden):
        import importlib
        mod = importlib.import_module(module_name)
        assert forbidden not in _executable_source(mod), (
            f"{module_name} references {forbidden}")

    def test_guard_the_guard(self):
        import hrm_adaptive_memory.c4.endpoint_recognition as mod
        stripped = _executable_source(mod)
        assert "k1_entity_bound_exact_completion" in stripped
        assert "k2_entity_bound_family_completion" in stripped


class TestImportBoundary:
    """Dependency direction: runtime <- evaluator, never runtime -> evaluator."""

    def test_endpoint_recognition_does_not_import_oracle_ceiling(self):
        import hrm_adaptive_memory.c4.endpoint_recognition as mod
        source = inspect.getsource(mod)
        assert "oracle_endpoint_ceiling" not in source

    def test_g2_paths_does_not_import_oracle_ceiling(self):
        import hrm_adaptive_memory.c4.g2_paths as mod
        source = inspect.getsource(mod)
        assert "oracle_endpoint_ceiling" not in source

    def test_relation_grammar_does_not_import_oracle_ceiling(self):
        import hrm_adaptive_memory.c4.relation_grammar as mod
        source = inspect.getsource(mod)
        assert "oracle_endpoint_ceiling" not in source

    def test_oracle_ceiling_may_import_runtime_endpoint_recognition(self):
        import hrm_adaptive_memory.c4.oracle_endpoint_ceiling as mod
        source = inspect.getsource(mod)
        assert "endpoint_recognition" in source

    def test_oracle_ceiling_docstring_declares_itself_evaluator_only(self):
        import hrm_adaptive_memory.c4.oracle_endpoint_ceiling as mod
        assert "EVALUATOR-ONLY" in (mod.__doc__ or "")

    def test_all_k_modes_share_one_call_signature(self):
        """g2_paths.py must be able to swap K0/K1/K2/K3 without knowing which
        is active -- that requires an identical call signature everywhere."""
        oracle_fn = make_oracle_completion_fn(
            required_evidence_ids=frozenset(), topology_record_ids=frozenset())
        for fn in (k0_literal_completion, k1_entity_bound_exact_completion,
                  k2_entity_bound_family_completion, oracle_fn):
            params = set(inspect.signature(fn).parameters)
            assert {"record_id", "entity", "relation", "texts"} <= params


# --- grammar grounded in the real generator's template forms ----------------
# Each string below is one of the 24 templates in
# generalization_dataset_v4.py:_render, with concrete entity/relation/value
# substituted in -- these are not invented sentence shapes.

REAL_TEMPLATES = {
    "formal_registry_0": 'The ownership tier registry records that Finch control module is assigned Gold.',
    "formal_registry_1": 'Registry entry: Finch control module — ownership tier — Gold.',
    "key_value_json": '{"subject": "Finch control module", "ownership tier": "Gold"}',
    "key_value_yaml": 'ownership tier:\n  subject: Finch control module\n  value: Gold',
    "key_value_bracket": '[ownership tier] Finch control module -> Gold',
    "change_log_set_to": 'Changelog: ownership tier for Finch control module set to Gold.',
    "change_log_revision": 'Revision applied — Finch control module ownership tier changed to Gold.',
    "change_log_bracket": '[change] Finch control module :: ownership tier := Gold',
    "message_quick_note": "Quick note — Finch control module's ownership tier is Gold, in case it comes up.",
}

WRONG_ENTITY_RECORD = ('Registry entry: Finch control module — ownership tier — '
                       'Falcon regulator.')
#: The generator emits exactly one fact per evidence record (one _render()
#: call each), so "multi-entity" risk in this corpus is cross-RECORD, not
#: cross-fact-within-a-record: does binding entity A's fact in record 1 ever
#: leak into a check against entity B in record 2?
MULTI_RECORD_FACTS = {
    "r_finch": "The registered asset registry records that Finch control module is assigned Sparrow module.",
    "r_egret": "The ownership tier registry records that Egret power cell is assigned Gold.",
}


class TestPositiveLiteralAndParaphrase:
    @pytest.mark.parametrize("name,text", REAL_TEMPLATES.items())
    def test_k0_recognizes_literal_relation_text(self, name, text):
        r = k0_literal_completion(record_id="r", entity="Finch control module",
                                  relation="ownership tier", texts={"r": text})
        assert r.completed, f"K0 should see the literal relation phrase in {name}"

    @pytest.mark.parametrize("name,text", REAL_TEMPLATES.items())
    def test_k1_recognizes_entity_bound_relation_across_all_real_templates(self, name, text):
        r = k1_entity_bound_exact_completion(
            record_id="r", entity="Finch control module", relation="ownership tier",
            texts={"r": text})
        assert r.completed, f"K1 should entity-bind the relation in {name}: {r.completion_reason}"
        assert r.entity_bound

    def test_k1_recognizes_value_shaped_records_k0_and_relational_state_cannot(self):
        """The construction defect this sprint targets: numeric/code values."""
        text = "Changelog: assigned category for Jacana pressure assembly set to GAMMA-BLUE."
        r = k1_entity_bound_exact_completion(
            record_id="r", entity="Jacana pressure assembly",
            relation="assigned category", texts={"r": text})
        assert r.completed


class TestWrongEntity:
    def test_k0_cannot_distinguish_entities_false_positive(self):
        """K0's known weakness, reproduced directly: it sees the relation
        phrase anywhere and binds it to whichever entity is asked about."""
        r = k0_literal_completion(record_id="r", entity="Falcon regulator",
                                  relation="ownership tier",
                                  texts={"r": WRONG_ENTITY_RECORD})
        assert r.completed  # K0's false positive -- this is the defect being fixed

    def test_k1_correctly_rejects_the_value_side_entity(self):
        r = k1_entity_bound_exact_completion(
            record_id="r", entity="Falcon regulator", relation="ownership tier",
            texts={"r": WRONG_ENTITY_RECORD})
        assert not r.completed
        assert not r.entity_bound

    def test_k1_correctly_accepts_the_true_subject(self):
        r = k1_entity_bound_exact_completion(
            record_id="r", entity="Finch control module", relation="ownership tier",
            texts={"r": WRONG_ENTITY_RECORD})
        assert r.completed


class TestMultiEntityAcrossRecords:
    def test_each_record_binds_only_its_own_entity(self):
        finch = k1_entity_bound_exact_completion(
            record_id="r_finch", entity="Finch control module",
            relation="registered asset", texts=MULTI_RECORD_FACTS)
        egret = k1_entity_bound_exact_completion(
            record_id="r_egret", entity="Egret power cell", relation="ownership tier",
            texts=MULTI_RECORD_FACTS)
        assert finch.completed and egret.completed

    def test_entity_does_not_bind_to_the_wrong_records_fact(self):
        wrong = k1_entity_bound_exact_completion(
            record_id="r_egret", entity="Finch control module",
            relation="registered asset", texts=MULTI_RECORD_FACTS)
        assert not wrong.completed


class TestK2FamilyMatch:
    def test_k1_rejects_a_synonym_but_k2_accepts_it(self):
        text = 'Changelog: category for Finch control module set to Gold.'
        k1 = k1_entity_bound_exact_completion(
            record_id="r", entity="Finch control module",
            relation="assigned category", texts={"r": text})
        k2 = k2_entity_bound_family_completion(
            record_id="r", entity="Finch control module",
            relation="assigned category", texts={"r": text})
        assert not k1.completed
        assert k2.completed
        assert k2.completion_reason == "k2_entity_bound_family_match"

    def test_k2_still_rejects_wrong_entity(self):
        k2 = k2_entity_bound_family_completion(
            record_id="r", entity="Falcon regulator", relation="ownership tier",
            texts={"r": WRONG_ENTITY_RECORD})
        assert not k2.completed


class TestOracleCeiling:
    def test_requires_both_reachable_and_required(self):
        fn = make_oracle_completion_fn(
            required_evidence_ids=frozenset({"r1"}),
            topology_record_ids=frozenset({"r1", "r2"}))
        both = fn(record_id="r1", entity="x", relation="y", texts={})
        reachable_not_required = fn(record_id="r2", entity="x", relation="y", texts={})
        required_not_reachable = fn(record_id="r3", entity="x", relation="y", texts={})
        assert both.completed
        assert not reachable_not_required.completed
        assert not required_not_reachable.completed

    def test_cannot_mark_a_record_complete_outside_the_topology(self):
        """K3 must not smuggle in topology the shared K0/K1 pass never reached
        -- it may only relabel completion for records already discovered."""
        fn = make_oracle_completion_fn(
            required_evidence_ids=frozenset({"anywhere"}),
            topology_record_ids=frozenset())
        r = fn(record_id="anywhere", entity="x", relation="y", texts={})
        assert not r.completed
        assert r.completion_reason == "k3_oracle_record_not_graph_reachable"


class TestCompletionModesRegistry:
    def test_k0_k1_k2_are_registered(self):
        assert set(COMPLETION_MODES) == {"K0", "K1", "K2"}
        for mode, fn in COMPLETION_MODES.items():
            r = fn(record_id="r", entity="Finch control module",
                   relation="ownership tier", texts={"r": REAL_TEMPLATES["change_log_set_to"]})
            assert r.completed, f"{mode} should recognize a literal exact-match record"


class TestRelationGrammarDirect:
    def test_parses_every_real_template_form(self):
        for name, text in REAL_TEMPLATES.items():
            # entity_hint resolves the one structurally ambiguous template
            # (subject and relation both multi-word with no delimiter between
            # them); every other template is unambiguous without it.
            bindings = parse_relation_bindings(text, "r", entity_hint="Finch control module")
            assert bindings, f"{name} failed to parse: {text!r}"
            assert bindings[0].relation == "ownership tier", f"{name}: {bindings[0]}"

    def test_non_relation_record_returns_empty(self):
        assert parse_relation_bindings(
            "JPA-5 is the short code for Jacana pressure assembly.", "r") == []

    def test_bindings_for_entity_is_substring_tolerant(self):
        text = REAL_TEMPLATES["change_log_set_to"]
        assert bindings_for_entity(text, "r", "Finch control module")
        assert not bindings_for_entity(text, "r", "Falcon regulator")
