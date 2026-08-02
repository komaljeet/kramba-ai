# kramba-ai — agent guide

Model policy (strict):
- **Sonnet** — orchestration + all coding/implementation.
- **Haiku** — reading: locating code, summarizing files/dirs, mapping the repo.
- **Opus** — **ALL debugging, and only debugging.**
  - **All of it, no size exemption.** Any root-causing goes to the **debugger** agent:
    wrong numbers, silent slowdowns, crashes, results that don't match expectations —
    however small it looks. Sonnet does **not** debug inline, and does not decide a bug
    is "too trivial to escalate"; that judgement is exactly where cheap models guess.
    Sonnet may apply a fix once **debugger** has identified the cause.
  - **Only debugging.** Not implementation, not review, not "this seems hard" — those
    stay with Sonnet however tricky they look. Opus is cap-limited on Pro, so the cap
    is spent on diagnosis, where it demonstrably pays for itself.

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
- **coder** (Sonnet): implements from an explicit plan; does not redesign.
- **debugger** (Opus): root-causes failures — wrong numbers, silent slowdowns, crashes,
  results that don't match expectations. The only Opus agent. Invoke it for **every**
  unexplained failure, however small — never triage one inline to save the cap. Once it
  names the cause, **coder** writes the fix.

## Working rules (see docs/ai_rules.md for the full version)
- Long GPU benchmarks/monitors → run in background + tail logs, never block the session.
- Feature branches only; never push to main (maintainer merges manually).
- Parallel Claude sessions share this working tree — prefer git worktrees, commit early.
