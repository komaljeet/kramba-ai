# How I Made It Possible to Pretrain a 1B LLM on 6GB VRAM

This is the story version. If you want the spreadsheet — every number, every date, every
dead end with a citation — that's `HF_BLOG_POST.md` in this same repo. This file is for
reading on a couch.

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

It took weeks. It is not a joke anymore. There's now a real 1.1 billion parameter model
that trains, end to end, on that same 6GB laptop GPU. It's slow — measured in months, not
hours — but it runs, it learns, and every number below is something I actually watched
happen on my own machine, not a projection.

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
small sample — takes somewhere between about four and a half and seven months, depending
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

## Where the actual numbers live

Every claim in this story has a measured number behind it, with the date it was measured
and the exact before/after figures, in `HF_BLOG_POST.md` and in this repository's
`docs/state.md`. If something here sounds too good, that's where to go check it — and if
it turns out to be wrong, the project's own habit is to say so in public rather than quietly
fix it, which is also where you'll find the handful of times that's already happened.

The code is Apache 2.0 — free to use, change, or build on. Any weights you train yourself
are yours.
