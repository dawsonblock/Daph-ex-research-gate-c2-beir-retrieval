"""run_full()'s HRM calls are now grouped into batches (see _run_hrm_batch
in scripts/run_gate_c4.py). This restructured the per-arm loop: pre-HRM
stages now run for every task up front, pending (non-resumed) tasks are
grouped into HRM_BATCH_SIZE-sized batches, and results are written back by
original task index. This test exercises the real run_full() end to end
(real development-split tasks, real pre-HRM stages) against a stub HRM
model, so it never needs the actual 2.4GB checkpoint or a GPU, and checks
exactly the properties a batching-index bug would break: receipts land in
original task order, every task is accounted for exactly once, resumed
tasks never reach the model, and the model really is called with more than
one prompt per forward pass.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from hrm_adaptive_memory.hrm.model import PromptCondition

ROOT = Path(__file__).resolve().parents[2]


class _StubConfig:
    model_type = "hrm_text"
    hidden_size = 1536
    num_layers_per_stack = 16
    H_cycles = 2
    L_cycles = 3
    max_position_embeddings = 4096
    prefix_lm = True
    architectures = ("HrmTextForCausalLM",)


class _StubModel:
    """Echoes input plus deterministic continuation tokens; records batch sizes."""

    config = _StubConfig()
    device = None

    def __init__(self):
        self.batch_sizes: list[int] = []

    def generate(self, *, input_ids, attention_mask=None, token_type_ids=None,
                 max_new_tokens=8, do_sample=False, **kwargs):
        self.batch_sizes.append(int(input_ids.shape[0]))
        batch, _length = input_ids.shape
        continuation = torch.arange(1, max_new_tokens + 1).repeat(batch, 1)
        return torch.cat([input_ids, continuation], dim=1)


class _StubTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    padding_side = "right"

    def __call__(self, text, return_tensors=None, padding=False, **kwargs):
        texts = [text] if isinstance(text, str) else list(text)
        # Length keyed off actual text so real (varying-length) prompts
        # actually exercise different padding amounts, unlike a fixed stub.
        encoded = [[5 + i % 50 for i in range(min(len(t), 200) + 4)] for t in texts]
        width = max(len(row) for row in encoded)
        ids, mask = [], []
        for row in encoded:
            pad = width - len(row)
            if padding and self.padding_side == "left":
                ids.append([self.pad_token_id] * pad + row)
                mask.append([0] * pad + [1] * len(row))
            else:
                ids.append(row + [self.pad_token_id] * pad)
                mask.append([1] * len(row) + [0] * pad)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(int(v)) for v in ids)


def _load_run_gate_c4():
    spec = importlib.util.spec_from_file_location(
        "_run_gate_c4_batch_test", ROOT / "scripts/run_gate_c4.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_run_gate_c4_batch_test"] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop("_run_gate_c4_batch_test", None)
    return module


@pytest.fixture()
def mod(tmp_path):
    module = _load_run_gate_c4()
    module.OUT = tmp_path / "evidence_gate_c4"  # never touch the real repo tree
    return module


def _wire_stub(mod, n_tasks: int, batch_size: int = 3):
    """Patch _load_split to a small real slice and _load_hrm to the stub."""
    from hrm_adaptive_memory.hrm.model import HRMAdapter, HRMModelSpec

    original_load_split = mod._load_split

    def _small_load_split(split):
        tasks, evidence, texts = original_load_split(split)
        return tasks[:n_tasks], evidence, texts

    stub_model = _StubModel()
    adapter = HRMAdapter(stub_model, _StubTokenizer(), spec=HRMModelSpec())

    mod._load_split = _small_load_split
    mod._load_hrm = lambda: (adapter, PromptCondition.DIRECT)
    mod.HRM_BATCH_SIZE = batch_size
    return stub_model


class TestBatchedRunFullMatchesSequentialAccounting:
    def test_every_task_accounted_for_exactly_once_in_order(self, mod):
        stub_model = _wire_stub(mod, n_tasks=5, batch_size=3)
        mod.run_full(split="development", arm_ids=["C4_0"])

        arm_file = mod.OUT / "full" / "development" / "C4_0.jsonl"
        lines = [json.loads(l) for l in arm_file.read_text().splitlines() if l.strip()]
        tasks, _, _ = mod._load_split("development")
        assert [r["task_id"] for r in lines] == [t["task_id"] for t in tasks]

    def test_model_is_actually_called_in_batches_not_one_at_a_time(self, mod):
        stub_model = _wire_stub(mod, n_tasks=5, batch_size=3)
        mod.run_full(split="development", arm_ids=["C4_0"])
        # 5 tasks at batch size 3 -> two calls, sizes [3, 2].
        assert stub_model.batch_sizes == [3, 2]

    def test_batch_size_one_reproduces_one_call_per_task(self, mod):
        stub_model = _wire_stub(mod, n_tasks=4, batch_size=1)
        mod.run_full(split="development", arm_ids=["C4_0"])
        assert stub_model.batch_sizes == [1, 1, 1, 1]

    def test_resumed_tasks_never_reach_the_model(self, mod):
        stub_model = _wire_stub(mod, n_tasks=4, batch_size=4)
        mod.run_full(split="development", arm_ids=["C4_0"])
        assert stub_model.batch_sizes == [4]

        # Second run against the same output dir: everything should resume.
        stub_model.batch_sizes.clear()
        mod.run_full(split="development", arm_ids=["C4_0"])
        assert stub_model.batch_sizes == [], (
            "a fully-resumed run must not call the model at all")

    def test_prompt_binding_holds_across_a_batch(self, mod):
        """The exact bug class batching risks: result[i] must correspond to
        prompt[i], not some other member of the batch. _assert_prompt_binding
        inside _build_receipt would raise if indices were shuffled."""
        _wire_stub(mod, n_tasks=6, batch_size=6)
        mod.run_full(split="development", arm_ids=["C4_0"])  # must not raise

        arm_file = mod.OUT / "full" / "development" / "C4_0.jsonl"
        lines = [json.loads(l) for l in arm_file.read_text().splitlines() if l.strip()]
        assert len(lines) == 6
        for r in lines:
            assert r["runtime_payload"]["packet"]["prompt_hash"] == \
                r["runtime_payload"]["hrm"]["prompt_hash"]
