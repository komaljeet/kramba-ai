# How I Made It Possible to Pretrain a 1B LLM on 6GB VRAM

## The problem

I wanted to train a language model from scratch. Not fine-tune one, not call an API for
one — actually watch a model learn from nothing, on my own machine, because I was curious
whether it was possible and because I couldn't afford to find out any other way.

Every forum, every blog post, every "getting started" guide said the same thing: you need
a real GPU for this. An A100, a cluster, at minimum a desktop card with 24GB of memory.
What I had was a gaming laptop with a 6GB graphics card in it — the kind of card that's
fine for games and a joke for anything people call "AI."

So I didn't do the sensible thing. I opened Claude and started asking: is that actually
true, or is it just what everyone assumes because nobody's tried hard to make it not true?

It took weeks. It is not a joke anymore. The training pipeline — building a real 1.1
billion parameter model, running real steps, saving a checkpoint, killing the process and
resuming it correctly — works, start to finish, on that same 6GB laptop GPU. I want to be
precise about what that does and doesn't mean: nobody, including me, has sat through an
entire training run yet. A full run takes months, not hours, so "it works end to end"
means the machinery is correct and measured, not that a finished model is sitting on my
disk. Every number below is something I actually watched happen on my own machine, over
real measured stretches of training — not a finished model, and not a projection dressed
up as one.

Here's the honest before-and-after, same laptop, from the first working version to now:

| | First working version | Now |
|---|---|---|
| Time for one training step | 21.81 seconds | **12.335 seconds** (-43%) |
| Tokens processed per second | 1,579 | **2,656** |
| Estimated time for a full pretrain | about 375 days | **about 132-211 days** |
| Peak memory used | 4.51 GB of 6 GB | **5.24 GB of 6 GB** |

That "132-211 days" range, not a single number, is itself an honest admission: the cheap
way of fitting the model (described next) costs some quality, and the range is "if you
accept that cost" versus "if you insist on matching full-quality training instead."

## Why a laptop GPU can't just "fit" a model this size

A model's size in parameters isn't really the problem. The problem is that *training* a
model needs a lot more memory than just storing it does. For every number in the model's
weights, training also needs to remember the direction that number should move in (the
gradient), and a running history of how it's been moving (the optimizer's memory of past
updates) — and on top of the model itself, there's the scratch space for the actual
calculation as it works through a chunk of text.

Add all that up with no tricks at all, and a billion-parameter model needs somewhere
around 16-17 times its own weight in GPU memory. A 6GB card can hold maybe 350-400 million
parameters that way. Not 1.1 billion. Not close.

This is the real reason "you need an A100" is the standard advice: it's true if you train
the obvious way. Everything that follows is about not training the obvious way.

## Idea one: only wake up the part of the model that's learning right now

Here's the trick that made the biggest difference. A deep model is really a stack of
blocks — layer after layer. The usual way to train is to wake all of them up at once:
every block sits in GPU memory, every block gets updated every step.

Instead: freeze all but one block. Train just that one for a while. Move on to the next
block, and freeze the one before it. Rotate through the whole stack, one block "live" at
a time.

The win is memory, not speed. A frozen block doesn't need its training-memory (the
gradient history, the optimizer's running average) sitting around — only the one active
block does. And since a frozen block isn't being touched, it doesn't even need to stay on
the graphics card at all; it can sit in ordinary system RAM until its turn comes back
around, and get swapped in just before it's needed.

That's two separate tricks working together: training one block at a time (so most of the
model needs no training-memory at any given moment), and parking the blocks that aren't
"live" in regular RAM instead of the graphics card's precious memory. Together they're
most of the reason a 6GB card can hold a model many times its own naive capacity.

There's a real cost to this, and I want to be upfront about it rather than bury it:
training one block at a time is not quite as good as training the whole model together.
Right now, measured honestly, it costs about 1.6x worse perplexity (a measure of how
surprised the model is by real text — lower is better) compared to training everything at
once with the same data and the same time. It used to cost about 4x worse, until I found
a dumb inefficiency in how the blocks were split up — one of them was sized to fit the
vocabulary table and ended up way bigger than the others, which wasted memory that could
have gone to training more of the model live at once. Fixing that split cut the quality
cost by more than half. It's still a real cost, not a free lunch, and I'd rather say that
plainly than let anyone think this is a trick with no tradeoff.

## Idea two: shrink the weights themselves

Separately from all of that: the model's weights don't need to be stored as full-precision
numbers. There's a technique (not mine — it's from a paper called BitNet) where each
weight, once trained, can be rounded down to one of exactly three values: minus one, zero,
or plus one, plus a single shared scale number per layer to stretch those three values
back out to the right range.

That sounds like it should destroy the model, and it would if you tried to do it after
training normally. The trick is to quantize *during* training, from the start, so the
model learns weights that are already comfortable being three values instead of a
continuous range. Done that way, it works — the model trains fine, and the compressed
weights take up a fraction of the space the original ones would have.

It doesn't shrink the memory used *during* training, to be clear — while the model is
actively learning, a full-precision shadow copy of each weight still has to exist so the
gradient updates have something precise to nudge. The saving shows up afterward: in the
saved checkpoint file, and later, when the model is actually being used to generate text
rather than being trained.

## Idea three: do the arithmetic in a cheaper number format, safely

Modern graphics cards can do their core multiplication-and-addition arithmetic in several
different levels of numerical precision. Lower precision is faster and uses less memory,
but every level down is a little less accurate.

Most of the training here runs in a middle-precision format that's a good balance already.
But most recently, I moved the single most common operation in the whole model — the big
matrix multiplications inside every layer — onto the graphics card's lowest-precision,
fastest tensor format, specifically for the parts of the computation where a few digits of
precision genuinely don't matter.

I didn't take that on faith. I measured the actual output of the model with and without
it, on real data, to make sure the answers didn't change in any way that mattered — and
they didn't. What changed was speed: about 8% faster across a full cycle through all the
blocks. Nothing about training the model got worse to get that.

Here's what each real idea above actually removes, side by side, plus two more small ones
that don't earn a whole section of their own: the input and output vocabulary tables are
really the same lookup used twice, so sharing one copy instead of two is a clean cut with
no real cost; and instead of keeping every intermediate result from the forward pass
sitting in memory until it's needed for the backward pass, most of it gets thrown away and
quickly recalculated when it's actually needed — more GPU work, far less memory held open
at once.

| Idea | What it removes | How much |
|---|---|---|
| Train one block at a time | Training-memory for every block but the active one | about 8x less optimizer memory |
| Park frozen blocks in system RAM | Those blocks leave the graphics card entirely | scales with how deep the model is |
| Three-value weights | Full-precision weight storage | 2.33x smaller, checkpoint and inference only |
| Share one vocabulary table instead of two | A whole duplicate table of weights | 103M parameters at this model's size |
| Recompute instead of store | Saved activations from the forward pass | activation memory down to about 50 MB |

## The part nobody finds exciting but that added up anyway

A lot of the rest of the speed came from doing dozens of small, boring things slightly
better: combining two or three separate small calculations into one so the graphics card
doesn't have to keep stopping and restarting; not recomputing something twice when the
first result is still sitting right there; moving data around in the order the hardware
actually likes instead of the order that was easiest to write first.

None of these individually sound like much. Stacked together, over weeks, they're a big
part of why the step time came down by more than 40% since the very first working version.
The lesson I keep re-learning: "I think there's nothing left to speed up" has been wrong
multiple times in a row now. I try to say it a lot less confidently these days.

## What doesn't work (and why I'm telling you)

I spent real time on things that didn't pan out, and I think that's worth being honest
about instead of only showing the wins.

Mixture-of-experts — splitting the model's "thinking" layer into several smaller specialist
pieces that only partially activate — looked promising on paper but cost almost twice the
compute for a modest quality gain here. Not worth it at this scale.

A scheme to automatically detect when the model was about to have a sudden jump in ability,
so training could react to it — it never found a real jump. Every "detection" also fired on
pure random noise, which means it wasn't detecting anything at all.

Compiling the model with PyTorch's automatic compiler, which usually speeds things up —
here it made things more than twice as slow, every single time I tried it, because this
particular architecture recompiles itself constantly in a way that eats the benefit. I
stopped trying it a while ago.

Teaching the model to read exact facts out of provided text turned out to need a
completely different ability than holding a fluent conversation. The chat-tuned model
responds naturally and correctly to small talk and simple factual questions, but asking it
to recall something specific from a document it was just given is a separate, much weaker
skill — fluency and grounded recall are not the same thing, and I'd misjudged how much one
implies the other.

And one I'll be fully honest about because it happened a few hours before I wrote this:
I tried to work out how much faster a nicer graphics card would make all of this, and I
got it wrong — twice, in two different ways, before getting a defensible number. The first
mistake assumed a much bigger card would run at exactly the same speed as this laptop's,
which badly understated it. The second mistake, trying to correct the first, used a lazy
shortcut — just scaling an old percentage — that implied a 24GB desktop card could train a
5-billion-parameter model in about a week, which is backwards: bigger models need more
compute, not less, so they should always take longer on the same hardware, never less time.
The real, carefully worked-out answer is that a strong desktop card would cut the time
down meaningfully — from months to something more like a few weeks for a model this size —
but nothing close to days. I'm leaving this paragraph in instead of quietly fixing the
numbers and moving on, because getting caught, correcting it in public, and explaining
exactly where the reasoning broke is more useful to anyone reading this than pretending it
never happened.

## So what does this actually get you?

On this laptop's 6GB graphics card, right now: a 1.1 billion parameter model trains
cleanly, using about 5.2GB of the card's 6GB — close to full, but not spilling over. It
processes a little over 2,600 words' worth of tokens per second. A full, proper training
run — enough real text for the model to actually learn something, not just memorize a
small sample — takes somewhere between four and a half and seven months, depending
on whether you're willing to accept the one-block-at-a-time quality cost described above
or insist on matching full-quality training exactly.

That's not fast. It's genuinely possible, on hardware nobody expected to be capable of it,
which is the whole point. If I had a stronger desktop card instead of this laptop, the
same training would very plausibly come down to a handful of weeks rather than months
— real spec sheets for current high-end desktop cards suggest something like four to
seven times faster raw number-crunching than this laptop chip, even before accounting for
not needing as many of these memory tricks in the first place.

Without any of these techniques at all — the straightforward way everyone recommends — this
same 6GB card tops out somewhere around a 350-400 million parameter model, and that's before
even trying to make it fast. The techniques here are roughly a 4-to-5x jump in how big a
model fits in the same amount of memory, which is the actual headline: it's not about
making training faster so much as making it possible in the first place on hardware this
small.

Only the 6GB row below is something I actually measured — the rest is a straight-line
projection from that one real point, assuming the same savings hold at bigger sizes. Treat
it as a reasonable guess for other cards, not a promise:

| Card's VRAM | Biggest model that fits | Rough pretrain time |
|---|---|---|
| **6GB** (what I have) | **~1.58B — measured** | **132-211 days — measured** |
| 8GB | ~2.2B | a floor guess, same speed as this laptop: 246-392 days |
| 12GB | ~3.5B | a floor guess, same speed as this laptop: 682-1,090 days |
| 16GB | ~4.8B | real desktop-card specs say roughly 333-532 days |
| 24GB | ~7.5B | real desktop-card specs say roughly 401-1,490 days |

Two honest caveats on that table. First, "biggest model that fits" past the 6GB row is a
simple ratio, not a measurement — nobody's run this on an 8, 12, 16, or 24GB card yet.
Second, the time estimates for the 8 and 12GB rows assume that bigger card runs at the
exact same speed as this laptop chip, which is almost certainly too pessimistic — I just
don't have reliable enough public specs for those specific chips to say by how much. The
16 and 24GB rows use real published numbers for those cards instead, which is why they
look more confident.

## After pretraining: turning it into something that talks back

A model that's only been shown raw web text doesn't know how to have a conversation — it
just continues whatever text you give it. Making it into something chat-like is a separate
step: showing it real conversations, with the model's own turns clearly marked, and only
grading it on the parts it's actually supposed to be learning to produce (its own replies,
not the human's side of the conversation).

I did this as a full update of every weight, not a lightweight patch bolted on top — partly
because the three-value weight trick above works better that way, and partly because it
keeps the story simple: everything about this model, start to finish, is trained the same
way. The result genuinely holds a basic conversation and answers simple factual questions
correctly. It is not, and shouldn't be mistaken for, a model that deeply knows things — that
turned out to be a meaningfully different skill that needs much more exposure to build,
separate from learning to talk fluently at all.

## What this is and isn't

The architecture underneath all of this — the way it processes sequences, the mix of a
constant-memory recurrent design with a couple of ordinary attention layers mixed in for
exact recall — is not something I invented. It's a combination of ideas already published
by other researchers. What's original here is the engineering: making all of these memory
and speed tricks work together, correctly, on genuinely constrained hardware, and being
rigorous about proving each one actually does what it claims instead of assuming it does.

Nothing much bigger than about 48 million parameters has been trained all the way to
convergence yet, so anything said here about how this scales up is a measured pattern at
small sizes, not a guarantee at a billion parameters and beyond. Every comparison in this
story is a single run against a single run — not yet repeated across several random seeds
— and I've measured, separately, that just changing the random starting point can shift a
result by an amount comparable to some of the differences reported here. Some of what I
call a "win" above might partly be luck; where that matters, the full writeup says so.

## The full timeline, for the record

Everything above is the story. This is the same project's complete measured history,
dated, every real win and every rejected idea, so nothing in the story above is taking
your word for it — or mine. "Before" and "After/Verdict" are what changed, or why an idea
died.

| Date | What was tried | Before | After / Verdict |
|---|---|---|---|
| 2026-07-12 | How much memory normal training needs | — | 16.8x the model's own weight — 1B needs ~16.8GB, 3x over this 6GB card |
| 2026-07-12 | Train one block at a time | 268M: 6.9GB; 1.04B: hard crash | 268M: 4.2GB (-39%); 1.04B: **runs** |
| 2026-07-12 | Recompute instead of storing activations | 1.04B still spilling past 6GB | 4.6GB peak, clean fit, no spill |
| 2026-07-24 | A startup check that doubled memory use | looked like block-training itself failing | root cause found, fixed for that case — the same bug resurfaced today in a slightly different form |
| 2026-07-24 | Combining block-training with PyTorch's auto-compiler | hoped it would help | slower, and it recompiles on every block switch |
| 2026-08-02 | Splitting the "thinking" layer into specialist sub-pieces | dense baseline | better loss, but 1.91x the compute cost — **rejected** |
| 2026-08-03 | Auto-detecting a sudden jump in ability | claimed real jumps | 0 real jumps found; fired on pure noise too — **rejected** |
| 2026-08-03 | Extrapolating weights forward instead of training them | — | **worse than doing nothing — rejected** |
| 2026-08-06 | Packing several short examples into one training row instead of padding | 1x, padded | **5.17x cheaper** per real token |
| 2026-08-07 | Same packing idea, re-checked on longer conversations | 5.17x on short examples | win shrinks to **1.45x** on longer ones |
| 2026-08-09 | A learning-rate schedule that reacts to the loss curve | fixed schedule | looked better, but a correctly-set fixed schedule still won in the end — **"the method doesn't work"** |
| 2026-08-09 | A learning-rate method with no manual tuning at all | hand-tuned schedule | wins for fine-tuning, fixed schedule still wins for pretraining — split by use case |
| 2026-08-10 | Letting the model search real documents for an answer | 0% correct closed-book | **22.5%** correct with the right amount of retrieved text (too much backfires) |
| 2026-08-10 | Sharing one vocabulary table instead of two | separate tables, slightly worse loss | **better loss, and 31% fewer parameters** |
| 2026-08-10/11 | PyTorch's automatic compiler, checked four separate times | normal speed | **more than twice as slow, every time — rejected** |
| 2026-08-11 | Checking whether the GPU was sitting idle waiting on anything | assumed there was slack to reclaim | **97% busy — there wasn't any** |
| 2026-08-11/12 | A faster scan calculation, 12 different designs tried | one design hoped to be faster | that one: **1.5-4x slower, dead**. A different design: **a real, small, kept win** |
| 2026-08-14 | A leaner optimizer as a substitute for block-training | block-training: fast and clean | spilled past the card anyway, **10x slower** |
| 2026-08-17 | Computing the loss in smaller pieces instead of all at once | a large model spilling past the card | **fits cleanly instead** |
| 2026-08-27 | A different way to compress certain weight matrices | — | **4x fewer parameters** there, same speed, quality not yet fully checked |
| 2026-09-29 | Fixing how blocks get split up for block-training | one lopsided split wasting memory | **cut the block-training quality cost by more than half** |
| 2026-10-12 | Combining several small internal calculations into one, three separate times | baseline | each a real few-percent win, stacked together |
| 2026-10-14 | Caching the three-value weight conversion instead of redoing it | baseline | four small real wins, stacked |
| 2026-10-15 | Combining a normalization step into the surrounding calculation | original baseline | a real step-time cut, measured honestly against the true original baseline |
| 2026-10-16/17 | Turning off a regularization setting that's usually left on | on by default | genuinely better, not just faster — checked twice to be sure it wasn't luck |
| 2026-10-17 | Four more small calculation-combining wins | before this batch | **a real, further 5.7% cut** |
| 2026-10-22/23 | Doing the biggest matrix multiplication in the lowest-precision tensor format | before this change | **confirmed 7.81% faster**, answers unaffected |
| 2026-10-23 | Using fewer data-loading worker processes | 7.16GB of RAM in use | **about 1.1GB saved, free, no slowdown** |
| 2026-10-24 | Trying the lowest-precision format on the recurrent calculation too | normal path: small, safe error | **5-18x worse error — rejected** (this calculation has no natural reset point, unlike the one that won above) |
| 2026-10-24 | Measuring, for real, how big a model fits with everything except block-training | assumed around 700M-1.1B from an old estimate | **measured ~340M** — the old estimate used the wrong optimizer as its stand-in |
| 2026-10-24 | The startup double-memory check, again | looked like the real training loop failing | same underlying bug as the July one, just the other half of it — fixed |

A few more real rejected ideas that don't have an exact date pinned down: an earlier,
"novel" version of the core architecture, which turned out to simply lose to the published
design it was replaced with; training several smaller copies of a layer and averaging
them, which lost badly to just having more real layers; gradually growing the training
sequence length, which fought the hardware instead of helping it; and a scheme to let the
model stop computing early on easy tokens, which never worked.

If any of this turns out to be wrong, the habit here is to say so in public and correct it
in place rather than quietly edit it away — this file already does that once, a few
sections up.

The code is Apache 2.0 — free to use, change, or build on. Any weights you train yourself
are yours.
