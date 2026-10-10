"""Tests for eval/chat_coherence.py -- mostly tests that the eval CAN fail.

Each scorer test feeds a canned reply of a known-bad kind (boilerplate, loop,
ran to the token limit, echoed question) and asserts the matching criterion
flags it. If one of these ever passes a bad reply, the published scores are
measuring nothing.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.chat_coherence import (  # noqa: E402
    keyword_hits, load_prompts, score_replies, words,
)

DATA, _SHA = load_prompts()
PROMPTS = DATA["prompts"]


def _good(p):
    return {"text": " and ".join(p["keywords"][:p.get("min_hits", 1)]) + ".",
            "hit_eos": True}


# --- prompt file ------------------------------------------------------------

def test_prompt_set_shape():
    ids = [p["id"] for p in PROMPTS]
    assert len(ids) == len(set(ids)) and 28 <= len(ids) <= 40
    cats = {p["category"] for p in PROMPTS}
    assert {"factual", "followup", "instruction", "adversarial"} <= cats
    assert any(len(p["messages"]) > 1 for p in PROMPTS)       # real multi-turn
    for p in PROMPTS:
        assert p["keywords"] and p["messages"][-1]["role"] == "user"


def test_answer_keywords_not_in_question():
    """Echoing the question must not score on factual/follow-up prompts."""
    for p in PROMPTS:
        if p["category"] in ("factual", "followup"):
            q = p["messages"][-1]["content"]
            assert not keyword_hits(q, p["keywords"]), p["id"]


# --- scorer can fail --------------------------------------------------------

def test_good_replies_pass_all():
    s = score_replies(PROMPTS, [_good(p) for p in PROMPTS])["summary"]
    assert s["coherent"] == 1.0
    assert s["chance_on_topic"] < 0.1       # keywords are prompt-specific


def test_boilerplate_fails_on_topic():
    bp = {"text": "As an AI language model, I can provide you with information "
                  "and help you with a wide range of tasks.", "hit_eos": True}
    s = score_replies(PROMPTS, [bp] * len(PROMPTS))["summary"]
    assert s["on_topic"] == 0.0


def test_boilerplate_with_answer_still_flagged_generic():
    """Same long template for every prompt, only the keyword differs."""
    tmpl = ("As an AI language model, I can provide you with information and help "
            "you with a wide range of tasks and questions. {}")
    rows = score_replies(PROMPTS, [{"text": tmpl.format(p["keywords"][0]), "hit_eos": True}
                                   for p in PROMPTS])["rows"]
    assert not any(r["on_topic"] for r in rows)


def test_echo_fails_on_topic():
    replies = [{"text": p["messages"][-1]["content"], "hit_eos": True} for p in PROMPTS]
    rows = score_replies(PROMPTS, replies)["rows"]
    for r in rows:
        if r["category"] in ("factual", "followup"):
            assert not r["on_topic"], r["id"]


def test_loops_fail_non_repetitive():
    for loop in ("2+3+3+3+3+3+3+3+3+3+3+3+3+3+3+3", "island of the " * 12,
                 "hello " * 40):
        rows = score_replies(PROMPTS[:2], [{"text": loop, "hit_eos": False},
                                           _good(PROMPTS[1])])["rows"]
        assert not rows[0]["non_repetitive"], loop


def test_no_eos_fails_terminates():
    rows = score_replies(PROMPTS[:2], [dict(_good(PROMPTS[0]), hit_eos=False),
                                       _good(PROMPTS[1])])["rows"]
    assert not rows[0]["terminates"] and rows[1]["terminates"]


def test_numeric_keyword_boundary():
    assert keyword_hits("It is 4.", ["4"]) == ["4"]
    assert keyword_hits("It is 40.", ["4"]) == []
    assert keyword_hits("light scattering", ["scatter"]) == ["scatter"]
    assert words("2+2=4") == ["2", "+", "2", "=", "4"]


# --- generation loop (serve_openai.generate_ids) ----------------------------

class _Counter(torch.nn.Module):
    """Predicts token id == current context length; EOS at a fixed length."""
    def __init__(self, vocab=64, eos=0, stop_at=None):
        super().__init__()
        self.vocab, self.eos, self.stop_at = vocab, eos, stop_at

    def forward(self, x):
        n = x.shape[1]
        logits = torch.full((1, n, self.vocab), -5.0)
        logits[0, -1, self.eos if n == self.stop_at else n % self.vocab] = 5.0
        return logits


def test_generate_ids_context_grows_and_stops():
    from serve_openai import generate_ids
    kw = dict(device=torch.device("cpu"), temperature=0.0, top_p=1.0,
              repetition_penalty=1.0)
    out, stopped = generate_ids(_Counter(), [1, 2, 3], eos=0, max_new=10, **kw)
    # A frozen context would repeat one id; a growing one counts up.
    assert out == list(range(3, 13)) and not stopped
    out, stopped = generate_ids(_Counter(stop_at=6), [1, 2, 3], eos=0, max_new=10, **kw)
    assert out == [3, 4, 5] and stopped


def test_repetition_penalty_lowers_negative_logits_too():
    from serve_openai import generate_ids

    class _Neg(torch.nn.Module):   # every logit negative; id 1 best, id 2 close
        def forward(self, x):
            lg = torch.full((1, x.shape[1], 4), -10.0)
            lg[0, -1, 1], lg[0, -1, 2] = -1.0, -1.2
            return lg
    out, _ = generate_ids(_Neg(), [3], eos=0, device=torch.device("cpu"),
                          max_new=2, temperature=0.0, top_p=1.0,
                          repetition_penalty=1.3)
    assert out == [1, 2]       # penalised repeat of 1 (-1.3) loses to 2 (-1.2)


def test_build_prompt_multiturn_matches_training_format():
    import serve_openai
    serve_openai._STATE["index"] = None
    got = serve_openai.build_prompt([
        {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"}])
    assert got == "<|user|>\na\n<|assistant|>\nb<|endoftext|><|user|>\nc\n<|assistant|>\n"


# --- #167: no repetition penalty inside a [TOOL_CALL] block ------------------

_TC_VOCAB = {0: "", 1: "[TOOL_CALL]", 2: '{"expression":"', 3: "2", 4: "+",
             5: "22", 6: '"}', 7: "[/TOOL_CALL]", 8: " x"}


def _tc_decode(ids):
    return "".join(_TC_VOCAB[i] for i in ids)


class _ToolCallScript(torch.nn.Module):
    """Wants ``[TOOL_CALL]{"expression":"2+2"}[/TOOL_CALL]`` then a plain-text
    ``2``. Every ``2`` has the near-miss ``22`` just behind it (2.0 vs 1.8),
    so a 1.3 penalty on an already-emitted ``2`` flips it -- the #167 bug."""
    SCRIPT = [1, 2, 3, 4, 3, 6, 7, 3, 0]

    def forward(self, x):
        i = x.shape[1] - 1                       # prompt is one token
        lg = torch.full((1, x.shape[1], len(_TC_VOCAB)), -5.0)
        want = self.SCRIPT[min(i, len(self.SCRIPT) - 1)]
        lg[0, -1, want] = 2.0
        if want == 3:
            lg[0, -1, 5] = 1.8
        return lg


def test_tool_call_args_survive_repetition_penalty():
    from serve_openai import generate_ids
    kw = dict(eos=0, device=torch.device("cpu"), max_new=12,
              temperature=0.0, top_p=1.0, repetition_penalty=1.3)
    out, stopped = generate_ids(_ToolCallScript(), [8], decode=_tc_decode, **kw)
    text = _tc_decode(out)
    assert '{"expression":"2+2"}' in text, text
    # Outside the block the penalty still applies: the trailing 2 -> 22.
    assert text.endswith("[/TOOL_CALL]22") and stopped, text


def test_tool_call_script_reproduces_the_bug_without_span_awareness():
    """Guard against a vacuous test above: with no decode (old behaviour) the
    same script MUST corrupt the argument, or the harness proves nothing."""
    from serve_openai import generate_ids
    out, _ = generate_ids(_ToolCallScript(), [8], eos=0, device=torch.device("cpu"),
                          max_new=12, temperature=0.0, top_p=1.0,
                          repetition_penalty=1.3)
    assert '"2+22' in _tc_decode(out)
