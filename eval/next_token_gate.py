"""Is the 48M model useful as an EDITOR autocomplete, one token at a time?

The FIM gate says it cannot fill a two-line gap (1.3% against a 1.0% floor). But
first-token accuracy is 59.8%, which is a different and much easier task -- the
one an editor's ghost-text actually performs. This measures whether that 59.8%
is worth anything to a human.

PRE-REGISTERED, before any result is seen
-----------------------------------------
The obvious trap: most tokens in code are punctuation, indentation and closing
brackets. A model that only ever predicts `)`, `,` and newline scores well on
raw top-1 and is worth nothing on camera. So the headline number here is
accuracy on SUBSTANTIVE tokens only -- identifiers and keywords -- and the
trivial ones are reported separately rather than mixed in.

Thresholds, fixed now:
  * >= 50% on substantive tokens -> genuinely demoable as autocomplete
  * 30-50%                       -> visibly works, visibly wrong often
  * < 30%                        -> not demoable; do not record it

Floors, because a bare accuracy number means nothing:
  * most-frequent-token: always predict the corpus's single commonest token
  * copy-previous-token: predict whatever the previous token was
  A model must beat BOTH on the substantive slice or it has learned nothing an
  editor could not do with two lines of code.

Falsifiability check: if the model were useless here, would this catch it? Yes --
the substantive slice removes the punctuation cushion that would otherwise carry
a useless model to a respectable-looking score, and both floors are computed on
the identical token set.
"""
import os, sys, json, random, re
import torch, torch.nn.functional as F

sys.path.insert(0, r"C:\Users\komal\Documents\Codes\kramba-ai")
os.chdir(r"C:\Users\komal\Documents\Codes\kramba-ai")

from transformers import GPT2TokenizerFast
import train_edge_conversational as tec
from eval.fim_completion_gate import load_heldout, SEP

CKPT = "checkpoints/code_48m_v2/edge_conversational.pt"
TOK = "data_cache/tokenizer_32k_fim"
CORPUS = "data_cache/code_corpus/stack_code.txt"
N_DOCS, CTX, PER_DOC = 40, 512, 24

tok = GPT2TokenizerFast.from_pretrained(TOK)
dev = torch.device("cuda")

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def substantive(tok_id: int) -> bool:
    """Would a human notice this suggestion? Identifiers and keywords yes;
    brackets, commas, operators and whitespace no."""
    s = tok.decode([tok_id])
    return bool(IDENT.search(s))


docs = load_heldout(CORPUS, SEP, N_DOCS * 3, 20_000_000, 400, tok)[:N_DOCS]
print(f"{len(docs)} held-out files, {CTX}-token context, {PER_DOC} cursors each")

model, step, _ = tec.load_checkpoint_for_resume(CKPT, dev)
model.eval()
print(f"checkpoint step {step}\n")

rng = random.Random(0)
tot = {"all": [0, 0], "sub": [0, 0], "triv": [0, 0]}
floor_freq = {"all": [0, 0], "sub": [0, 0]}
floor_prev = {"all": [0, 0], "sub": [0, 0]}

# Commonest token, computed on the same held-out text the model is scored on.
counts: dict[int, int] = {}
encoded = []
for d in docs:
    ids = tok(d, add_special_tokens=False)["input_ids"][:CTX + PER_DOC + 8]
    if len(ids) < CTX + 8:
        continue
    encoded.append(ids)
    for i in ids:
        counts[i] = counts.get(i, 0) + 1
common = max(counts, key=counts.get)
print(f"most frequent token: {tok.decode([common])!r}\n")

with torch.no_grad():
    for ids in encoded:
        cuts = sorted(rng.sample(range(64, min(len(ids) - 1, CTX)),
                                 min(PER_DOC, min(len(ids) - 1, CTX) - 64)))
        x = torch.tensor([ids[:CTX]], device=dev)
        lg = model(x)
        if isinstance(lg, tuple):
            lg = lg[0]
        pred = lg[0].argmax(-1).tolist()
        for c in cuts:
            target = ids[c]
            got = pred[c - 1]
            kind = "sub" if substantive(target) else "triv"
            for k in ("all", kind):
                tot[k][1] += 1
                tot[k][0] += int(got == target)
            for k in ("all",) + (("sub",) if kind == "sub" else ()):
                floor_freq[k][1] += 1
                floor_freq[k][0] += int(common == target)
                floor_prev[k][1] += 1
                floor_prev[k][0] += int(ids[c - 1] == target)

pct = lambda a: 100.0 * a[0] / max(1, a[1])
print(f"{'slice':<28}{'n':>7}{'model':>9}{'freq floor':>12}{'prev floor':>12}")
print(f"{'ALL tokens':<28}{tot['all'][1]:>7}{pct(tot['all']):>8.1f}%"
      f"{pct(floor_freq['all']):>11.1f}%{pct(floor_prev['all']):>11.1f}%")
print(f"{'SUBSTANTIVE (names/keywords)':<28}{tot['sub'][1]:>7}{pct(tot['sub']):>8.1f}%"
      f"{pct(floor_freq['sub']):>11.1f}%{pct(floor_prev['sub']):>11.1f}%")
print(f"{'trivial (punct/space)':<28}{tot['triv'][1]:>7}{pct(tot['triv']):>8.1f}%"
      f"{'-':>11}{'-':>11}")

sub = pct(tot["sub"])
beats = sub > pct(floor_freq["sub"]) and sub > pct(floor_prev["sub"])
if not beats:
    verdict = "NOT DEMOABLE: does not beat both floors on substantive tokens."
elif sub >= 50:
    verdict = f"DEMOABLE: {sub:.1f}% on names and keywords, both floors beaten."
elif sub >= 30:
    verdict = (f"MARGINAL: {sub:.1f}% on names and keywords -- visibly works, "
               f"visibly wrong often.")
else:
    verdict = f"NOT DEMOABLE: {sub:.1f}% on names and keywords (threshold 30%)."
print(f"\nVERDICT: {verdict}")

os.makedirs("results/next_token", exist_ok=True)
with open("results/next_token/code48m_v2.json", "w") as fh:
    json.dump({"checkpoint": CKPT, "step": step,
               "all": tot["all"], "substantive": tot["sub"],
               "trivial": tot["triv"], "floor_most_frequent": floor_freq,
               "floor_prev_token": floor_prev, "verdict": verdict}, fh, indent=2)
print("wrote results/next_token/code48m_v2.json")
