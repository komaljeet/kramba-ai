# kramba-ai — agent guide

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
- **Sonnet** — writing: docs/state.md entries, commit messages, MR descriptions, issue
  text. Prose about work already done and already understood.
- **Haiku** — reading: locating code, summarizing files/dirs, mapping the repo.

> **Cap note.** Opus is cap-limited on Pro and this policy routes the highest-volume work
> to it. If the cap becomes the binding constraint, the lever is to hand *writing* and
> *reading* away more aggressively (Sonnet/Haiku, via subagents) — **not** to let Sonnet
> take back coding or "small" debugging, which is what this policy exists to prevent.

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

Writing (docs/state.md, commit messages, MR text) is Sonnet's, but note it is usually
*not* delegable to a subagent: writing up work just done needs the full session context,
which is exactly what a fresh subagent lacks. In practice, switch the session model
(`/model`) for a writing turn rather than spawning an agent for it.

## Working rules (see docs/ai_rules.md for the full version)
- Long GPU benchmarks/monitors → run in background + tail logs, never block the session.
- Feature branches only; never push to main (maintainer merges manually).
- Parallel Claude sessions share this working tree — prefer git worktrees, commit early.
