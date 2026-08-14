"""Cheap confidence extraction from ANSWER_NOW's own generation.

Per the research-lead directive: Executive v0 is uncertainty-gated -- it
runs the cheap ANSWER_NOW action first (by definition cheap, since it never
touches retrieval/graph/composition), then decides whether to ALSO invoke
CERTIFIED_MEMORY_V1 based on ANSWER_NOW's own confidence. This reframes
"PRE_DECISION" (hrm_adaptive_memory/experiment_integrity/executive_features.py)
as "before MEMORY specifically", not "before either action" -- see that
module's amended docstring for the explicit taxonomy update.

hrm_adaptive_memory/hrm/model.py's HRMAdapter.generate_batch() cannot be
reused for this: it accepts a generation_kwargs passthrough, but its own
body assumes model.generate() returns a plain tensor (sequences[index][...]
indexing) -- passing return_dict_in_generate=True through it would silently
break that indexing, not add a feature. This module calls the adapter's
model/tokenizer directly instead, replicating only the single-prompt
(non-batched) subset of generate_batch's logic needed for scored generation.
model.py itself is not modified.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    #: Mean, over generated steps, of the greedy-chosen token's own softmax
    #: probability at that step. In [0, 1]; higher means the model was more
    #: confident in its own (greedy) choice at each step. NOT calibrated
    #: probability of correctness -- a raw self-confidence signal only.
    mean_token_confidence: float
    #: The single lowest per-step confidence -- the model's worst moment of
    #: hesitation during generation, often more informative than the mean
    #: for detecting "the model is guessing" on at least one token.
    min_token_confidence: float
    #: Geometric mean of per-step confidence (exp(mean(log p_i))), i.e. the
    #: sequence's JOINT confidence normalized by length. Distinct from the
    #: arithmetic mean: one low-confidence step pulls this down much harder
    #: (a single p=0.01 step among nine p=0.99 steps drags the geometric mean
    #: far more than the arithmetic mean), closer in spirit to "would the
    #: whole answer survive if any one token had been sampled instead of
    #: argmax'd" than the plain average is.
    sequence_confidence: float
    #: Mean, over generated steps, of the full-distribution Shannon entropy
    #: (in nats) -(sum p_i log p_i) over the ENTIRE vocabulary at that step --
    #: not just the top-1 token. A genuinely different signal from
    #: confidence: a distribution can have high top-1 probability (confident
    #: in one answer) while still having moderate entropy if there's a
    #: long tail, or the reverse (low top-1 prob spread over just 2-3
    #: plausible tokens -> lower entropy than the confidence score alone
    #: would suggest). Lower entropy = more peaked = more certain.
    mean_entropy: float
    #: Alias for completion_tokens, named to match the research-lead's
    #: explicit feature list ("answer length") without requiring a caller to
    #: know it's the same value as completion_tokens.
    answer_length: int


def generate_with_confidence(adapter, condition, prompt: str,
                             max_new_tokens: int) -> ConfidenceResult:
    """Single-prompt greedy generation with per-step confidence extracted
    from output_scores. Not batched -- correctness over throughput, since
    this signal has never been extracted anywhere in this project before and
    the batched path's left-padding + dict-output interaction is untested."""
    import torch

    inputs = adapter.encode(prompt, condition)
    device = getattr(adapter.model, "device", None)
    if device is not None:
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.no_grad():
        out = adapter.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            output_scores=True, return_dict_in_generate=True)

    prompt_length = int(inputs["input_ids"].shape[-1])
    sequence = out.sequences[0]
    completion_ids = sequence[prompt_length:]
    text = adapter.tokenizer.decode(completion_ids, skip_special_tokens=False)

    # out.scores is a tuple of per-step logits, one tensor per generated
    # token, each shape (batch=1, vocab). Greedy decoding means the chosen
    # token IS the argmax at each step, so its own softmax probability is
    # exactly the model's confidence in the choice it actually made.
    import math
    step_confidences: list[float] = []
    step_entropies: list[float] = []
    _EPS = 1e-12  # guards log(0) for a near-degenerate (prob=1.0) distribution
    for step_logits in out.scores:
        probs = torch.softmax(step_logits[0].float(), dim=-1)
        step_confidences.append(float(probs.max().item()))
        entropy = float(-(probs * torch.log(probs + _EPS)).sum().item())
        step_entropies.append(entropy)

    if not step_confidences:
        # Zero-length completion (e.g. immediate EOS) -- NOT_COMPUTABLE in
        # spirit, but this signal is used as a real-valued feature, not a
        # rate, so 0.0/0.0 (minimum confidence, minimum entropy) is the
        # defensible fail-safe rather than a sentinel that would break
        # downstream arithmetic.
        mean_conf, min_conf, seq_conf, mean_ent = 0.0, 0.0, 0.0, 0.0
    else:
        mean_conf = sum(step_confidences) / len(step_confidences)
        min_conf = min(step_confidences)
        log_probs = [math.log(max(p, _EPS)) for p in step_confidences]
        seq_conf = math.exp(sum(log_probs) / len(log_probs))
        mean_ent = sum(step_entropies) / len(step_entropies)

    return ConfidenceResult(
        text=text, prompt_tokens=prompt_length, completion_tokens=len(completion_ids),
        mean_token_confidence=round(mean_conf, 6), min_token_confidence=round(min_conf, 6),
        sequence_confidence=round(seq_conf, 6), mean_entropy=round(mean_ent, 6),
        answer_length=len(completion_ids))
