# kramba-ai — agent guide

## How to talk to the maintainer (read this first)

- **Start every reply with the current date and time**, read from this PC
  (`Get-Date`), not from the session start. Sessions here run past midnight and
  the start-of-session date goes stale, which has already mis-dated a docs entry.
- **Use plain, simple words.** Short chunks, one idea per paragraph. No ornate
  phrasing, no essay structure, no bolded lead-ins on ordinary answers. Complex
  wording costs tokens and hides the point; the maintainer wants the finding, not
  the prose. Say the verdict first, in human units.
- Long technical detail belongs in docs/state.md, commits and MR text — not chat.

Model policy (strict):
- **Opus** — **orchestration, design, all coding, all debugging.** Everything where a
  wrong *decision* is the expensive failure.
  - **Design is the point, not typing.** The costly mistakes in this repo have not been
    buggy code — they were choosing the wrong experiment, trusting a proxy metric that
    couldn't measure the target, and guessing at a cause instead of diffing against a
    working reference. So Opus decides *what* to build and *what results mean*; handing
    it a plan someone else made defeats the purpose.
  - **All debugging, no size exemption.** Every unexplained failure goes to **debugger**,
    however small — "this one's simple enough to guess at" is exactly where it goes wrong.
  - **Before building anything:** check the plan against the issue's own wording, and
    check that the measurement can actually detect the thing being claimed.
  - **Before RUNNING any test or benchmark, ask: "if the thing I am testing were
    broken, would this catch it?"** Answer it before you see the result, not after.
    A test that cannot fail is worse than no test — it converts a bug into a
    published number. Every expensive mistake in this repo has this shape:
      - #138 shipped "6.8x" from a CUDA graph that could take exactly ONE correct
        step. The check compared one replay against one eager step from the same
        state; both do the same thing, so it passed at `max|dlogits| = 0.000e+00`
        while the model's context had silently stopped growing.
      - #117's cache served untrained weights for a whole epic because the only
        test watched aggregate loss, which still trended down.
      - #122's 5-item recall gate was retracted for gating a scale-up decision on
        a measurement dominated by luck.
      - #132's verdict was retracted for grading arms on **train** loss in a
        multi-pass regime, which measures memorisation, not quality.
    Practical rules that follow: compare a **sequence**, not one step, whenever
    state carries forward; assert the specific quantity that would freeze or go
    stale, not just the output; and add a test whose only job is to fail if the
    real test ever becomes vacuous (too few tokens, too few trials, a floored
    metric).
  - **Writing too** — docs/state.md entries, commit messages, MR text. Do NOT switch
    models mid-task to write these up. Writing about work just done needs the full
    session context, so it cannot be handed to a subagent, and toggling `/model` by hand
    for a docs turn is friction with no upside. Opus writes it inline.
- **Haiku** — reading: locating code, summarizing files/dirs, mapping the repo. The one
  thing that genuinely delegates cleanly (a fresh agent can grep/summarize without
  needing session history).

> **Cap note.** Opus is cap-limited on Pro and this policy routes nearly everything to it.
> If the cap becomes the binding constraint, the lever is to push *reading* onto
> **repo-reader** (Haiku) more aggressively — **not** to let coding or "small" debugging
> drift back to a cheaper model, which is what this policy exists to prevent.

## Hard-won gotchas that cost hours (read before touching training)

- **Compile the BLOCKS, not the model.** `torch.compile(model)` raises
  `InductorError: MemoryError` at 16 layers — `compute_ancestors` is O(N²) in graph nodes.
  Compile each block: +18%, and it is already wired into `build_model_and_config`.
  Full finding: docs/training.md § "COMPILE THE BLOCKS, NOT THE MODEL" (#158).
- **`tokenizer.vocab_size` EXCLUDES added tokens.** Use `len(tokenizer)`. Sizing an embedding
  table from `vocab_size` with a FIM tokenizer builds it 3 rows short and the sentinels index
  out of bounds — a bare CUDA device-side assert with no hint of the cause (#155).
- **A tokenized `.bin` cache is keyed on the string "code", not the corpus.** Pointing
  `--code-corpus` at a different file silently reused the old cache — a run announced 11.2 GB
  and trained on 121M tokens. Now guarded, loudly (#157).
- **Bare `--no-fast` reverts to fp32** (74.6 s/step vs 8.66). Re-add
  `--precision bf16 --tf32 --triton-scan --megakernel-scan` by hand.
- **Nothing prints until step 50** (`log_every_steps` is hardcoded). A shorter smoke run looks
  hung when it is working. Watch `training_log.csv`, which flushes per row.
- **A number from a different harness is not a measurement of this one.** Every wrong call in
  the 2026-08-11/12 session came from quoting one setup's result at another.

## Where knowledge lives — read ON DEMAND, don't preload all of it
- Doc hub / map of everything → docs/index.md
- Architecture & project structure → docs/architecture.md
- Project context, delivery workflow & rules → docs/ai_rules.md
- Current measured state (PPL, probes, live investigations) → docs/state.md
- Theory / math formalism → docs/theory.md
- Training flags & VRAM envelopes → docs/training.md

Have the **repo-reader** subagent read the specific file when a task needs it.
Do NOT re-explore the repo if docs/index.md already points to the answer.

## Subagents
- **repo-reader** (Haiku): locating code, summarizing files/dirs, mapping. Returns
  concise findings + `file:line` refs, never whole files. Use for all bulk reading.
- **coder** (Opus): implements, and owns the design judgement inside the task — sanity-checks
  the plan against the issue before building it.
- **debugger** (Opus): root-causes failures — wrong numbers, silent slowdowns, crashes,
  results that don't match expectations. Invoke it for **every** unexplained failure,
  however small — never triage one inline. Once it names the cause, **coder** fixes it.

There is no writer agent, deliberately: writing up work just done needs the full session
context a fresh subagent lacks. Opus writes docs/commits/MR text inline.

## Working rules (see docs/ai_rules.md for the full version)
- Long GPU benchmarks/monitors → run in background + tail logs, never block the session.
- **One run per log file, and check nothing else is on the GPU before launching.**
  Two processes redirected to the same log each keep their own write offset, so the
  later one overwrites the earlier in place: the file stops growing, `tail -f` goes
  silent, and a finished run looks hung. Two copies of a 12-layer model also OOM the
  6 GB card. Both happened on 2026-08-10 (see docs/state.md, code-vocab A/B).
- Feature branches only; never push to main (maintainer merges manually).
- Parallel Claude sessions share this working tree — prefer git worktrees, commit early.
