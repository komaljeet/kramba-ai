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
- Feature branches only; never push to main (maintainer merges manually).
- Parallel Claude sessions share this working tree — prefer git worktrees, commit early.
