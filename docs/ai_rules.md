# Project Context, Delivery Workflow & AI-Assistant Rules

> **Moved from README:** §12 Project Context & Delivery Workflow, plus the top-of-file
> "Maintainer note — keep this file current." These are the durable working conventions for
> anyone (human or AI assistant) contributing to the project.

## Maintainer note — keep the docs current

Whenever a session reaches a conclusion worth recording (a benchmark result, a
confirmed/dead-end optimization, a parity or speed gate outcome, an architecture decision, or a
closed issue's verdict), update the relevant doc **before ending the session**. Treat these
files as the durable log of what was tried and what was concluded — don't leave a settled
result only in a commit message or a session transcript.

> **Status legend:** ✅ Done · 🔄 In Progress · ⬜ Planned

## Original Design Brief

Act as a Principal AI Research Scientist and Theoretical Physicist specializing in
alternative, ultra-low-compute sequence modeling. Goal: build a novel, untrained LM
architecture ("Hamiltonian Vortex Network") achieving Attention's long-range associative
recall at <20% compute, O(1) inference, no KV cache. Do **not** use standard Attention,
Transformers, Mamba, or Test-Time Training gradient loops — synthesize Fluid Dynamics
(Lagrangian vortex particles) with Quantum Mechanics (Matrix Product States) in standard
PyTorch. Deliverables: theory + formalism, core engine, vocabulary decoder + runnable
pipeline, and a strict FLOP comparison vs MHA as N → 100,000.

## GPU / CUDA Requirement

All training, benchmarking, and evaluation **must run on GPU, never CPU**. Environment:
NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM), driver 596.36, PyTorch with CUDA
(`pip install torch --index-url https://download.pytorch.org/whl/cu126`). Verify
`torch.cuda.is_available()` before running; if False, fix the install rather than falling
back to CPU.

## Issue & Delivery Workflow

> **⚠️ If the maintainer says a feature exists but you can't find it in the code, check the
> GitLab issues first (`glab issue list --all`, `glab issue view <id>`).** Much of this project
> was scoped issue-by-issue, and an issue may describe intended behavior that was only *partially*
> delivered (e.g. Issue #30 designed a React/Vue chat client, but only the REST + WebSocket
> backend was committed — no frontend UI page exists). The issue is the source of truth for what
> was *meant* to be built; reconcile the gap before assuming the maintainer is mistaken.
>
> **Also consult the persistent assistant memory index (`MEMORY.md`) at the start of each
> session** (in the Claude project memory dir, e.g.
> `~/.claude/projects/<project>/memory/MEMORY.md`). It records durable workflow conventions and
> past feedback (branching discipline, push-to-feature-branch-only, merge/close etiquette) that
> aren't obvious from the code — check it alongside the GitLab issues before acting.

> **📝 Always create a GitLab issue BEFORE starting any non-trivial work** (`glab issue create
> --title "..." --description "..."`), even for work the maintainer just asked for verbally.
> The issue is the unit of follow-up here: it captures scope/acceptance up front, gives the
> branch and MR something to reference (`Closes #NN`), and leaves a durable trail so a later
> session can pick the thread back up. Don't start editing code for a feature until its issue
> exists.

**Always use the GitLab CLI (`glab`)** for all GitLab interactions (the web UI is behind a
login wall + Cloudflare challenge). Examples: `glab issue view <id> --repo komalbarun/kramba-ai`,
`glab issue list`, `glab work-items list`. Per-issue workflow:

1. **ALWAYS pull `main` and branch off `main` — no exceptions.** Even if a dependency issue
   isn't merged yet, still `git fetch` and branch from the updated remote `main`; **never**
   cut the branch off whatever branch happens to be checked out (that inherits unrelated
   in-flight work and produces a polluted MR diff). If a prerequisite has since merged, the
   dependency is already in `main` — that's exactly why you pull first:
   ```bash
   git fetch origin
   git checkout main && git pull origin main
   git checkout -b issue-NN-short-description
   ```
   Never work on `main` directly.
2. Implement the change; verify with a real test or runnable command on GPU.
3. Commit referencing the issue (`git commit -m "Fix: … for issue #NN"`).
4. Push the branch (`git push origin <branch>`).
5. **Always open a Merge Request** with `glab mr create` — target `main`, put `Closes #NN`
   in the description so the issue auto-closes on merge. Never push to or merge `main`
   directly; the maintainer reviews and merges the MR manually.
6. **On merge to `main`, always post a closing comment on the issue** with
   `glab issue note NN` — summarize what shipped and link the merged MR, then confirm the
   issue is closed (`Closes #NN` auto-closes it on merge; add the note regardless). Do this
   for every handled issue so each one has a durable record of the delivered change.

GitLab project: https://gitlab.com/komalbarun/kramba-ai
