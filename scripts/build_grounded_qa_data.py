"""Build the grounded-QA training corpus for A4 (hybrid token-recall continue-FT).

A4 teaches the hybrid's fresh attention layers the one skill the A5 gate
(``eval/grounded_qa.py``) measures: **given a passage + a question, emit the
answer span that is in the passage** — i.e. read the context, locate the
answer, copy/synthesize it. Pure DeltaNet recurrence structurally cannot do
this; the 2 sliding-attention layers can.

This builder ships a **curated multi-domain core** of (passage, question, answer)
triples where ``answer`` is always a *verbatim span* of ``passage`` (so the model
is taught real extraction, not hallucination). The core is **disjoint from the 5
eval items** in ``eval/grounded_qa.py`` (no train/test leakage): different facts,
different passages. A `--validate` pass asserts both invariants.

Output: ``data/grounded_qa_train.jsonl`` — one JSON object per line::

    {"passage": "...", "question": "...", "answer": "..."}

The continue-FT entrypoint (``continue_ft_hybrid.py``) renders each triple in the
gate's *exact* open-book chat format with loss only on the answer turn.

Extension points for the next LLM (the curated core is a *seed*, ~40 passages):
  * Append more entries to ``CORE`` below (keep answers verbatim spans).
  * Add a Wiki/SlimPajama cloze augmenter (read ``data_cache/tokenized/*.bin``,
    detokenize, mask a content word, emit a cloze example) for scale + diversity.
    A cached SQuAD/TriviaQA arrow file read with pyarrow would be even better
    (``datasets.load_dataset`` segfaults in this env — see
    ``finetune_edge_conversational.py`` for the pyarrow-from-cache pattern).
The gate is binary (lift > 0), so the core is enough to *probe* whether the
architecture recalls; scale matters only if the probe is marginal.

Usage::

    python scripts/build_grounded_qa_data.py                  # write data/grounded_qa_train.jsonl
    python scripts/build_grounded_qa_data.py --validate        # also assert invariants
    python scripts/build_grounded_qa_data.py --out other.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# The 5 eval items from eval/grounded_qa.py — used ONLY for the leakage guard.
# (question, answer-keywords). A training triple "collides" if its question text
# is near-identical to an eval question; we keep the *facts* disjoint instead.
EVAL_QUESTIONS = {
    "what is the capital of france?",
    "who wrote romeo and juliet?",
    "what is the boiling point of water?",
    "what language is spoken in brazil?",
    "what is a black hole?",
}

# Curated multi-domain core. Each entry: (passage, [(question, answer), ...]).
# `answer` MUST be a case-insensitive verbatim substring of `passage` (validated).
# Domains span geography / physics / chemistry / biology / astronomy / history /
# literature / math / tech / art / music / economics / language — disjoint from
# the eval facts (no Paris, Shakespeare, Celsius, Portuguese, black-hole).
CORE: list[tuple[str, list[tuple[str, str]]]] = [
    ("Japan is an island country in East Asia. Its capital and largest city is "
     "Tokyo, located on the island of Honshu.",
     [("What is the capital of Japan?", "Tokyo")]),
    ("The Nile is a major river in northeastern Africa. It is generally regarded "
     "as the longest river in the world, flowing about 6,650 kilometers.",
     [("What is the longest river in the world?", "Nile")]),
    ("The Sahara is a desert on the African continent. It is the largest hot "
     "desert in the world, covering much of North Africa.",
     [("What is the largest hot desert in the world?", "Sahara")]),
    ("Australia is a country and continent in the Southern Hemisphere. Its capital "
     "city is Canberra, not Sydney or Melbourne as many people assume.",
     [("What is the capital of Australia?", "Canberra")]),
    ("Mount Everest is the highest mountain on Earth. It is located in the "
     "Himalayas on the border between Nepal and China, with a peak at 8,849 meters.",
     [("Where is Mount Everest located?", "Himalayas")]),
    ("Light travels in a vacuum at approximately 299,792 kilometers per second, a "
     "value known as the speed of light and denoted by the letter c.",
     [("What is the speed of light?", "299,792 kilometers per second")]),
    ("The chemical formula for water is H2O, meaning each molecule contains two "
     "hydrogen atoms and one oxygen atom.",
     [("What is the chemical formula for water?", "H2O")]),
    ("The structure of DNA was discovered in 1953 by James Watson and Francis "
     "Crick, who proposed the double helix model based on X-ray data.",
     [("Who discovered the structure of DNA?", "James Watson and Francis Crick")]),
    ("The Sun is the star at the center of the Solar System. It is the closest "
     "star to Earth and provides the energy that sustains life on our planet.",
     [("What is the closest star to Earth?", "Sun")]),
    ("Jupiter is the largest planet in the Solar System. It is a gas giant with a "
     "famous feature called the Great Red Spot, a huge storm.",
     [("What is the largest planet in the Solar System?", "Jupiter")]),
    ("Saturn is a gas giant famous for its prominent ring system, which is made "
     "mostly of ice and rock particles.",
     [("Which planet is famous for its rings?", "Saturn")]),
    ("Mars is the fourth planet from the Sun. It is often called the Red Planet "
     "because of its reddish appearance from iron oxide on its surface.",
     [("Which planet is called the Red Planet?", "Mars")]),
    ("The human genome is organized into 23 pairs of chromosomes, for a total of "
     "46 chromosomes in each normal cell.",
     [("How many pairs of chromosomes do humans have?", "23")]),
    ("The heart is a muscular organ that pumps blood through the circulatory "
     "system of the body.",
     [("What does the heart pump through the body?", "blood")]),
    ("Plants produce their own food through a process called photosynthesis, "
     "which uses sunlight to convert carbon dioxide and water into glucose.",
     [("What process do plants use to make food?", "photosynthesis")]),
    ("George Washington was the first president of the United States, serving "
     "from 1789 to 1797.",
     [("Who was the first president of the United States?", "George Washington")]),
    ("World War II ended in 1945 with the surrender of Japan, following the "
     "surrender of Germany earlier that year in May.",
     [("In what year did World War II end?", "1945")]),
    ("The novel 1984 was written by George Orwell and published in 1949 as a "
     "warning about totalitarianism.",
     [("Who wrote the novel 1984?", "George Orwell")]),
    ("Pride and Prejudice is a classic novel by Jane Austen, first published "
     "in 1813.",
     [("Who wrote Pride and Prejudice?", "Jane Austen")]),
    ("The chemical symbol for gold is Au, which comes from the Latin word aurum.",
     [("What is the chemical symbol for gold?", "Au")]),
    ("The chemical symbol for sodium is Na, derived from the Latin word natrium.",
     [("What is the chemical symbol for sodium?", "Na")]),
    ("Oxygen has the atomic number 8, meaning each oxygen atom has 8 protons "
     "in its nucleus.",
     [("What is the atomic number of oxygen?", "8")]),
    ("The Pythagorean theorem states that in a right triangle, the square of the "
     "hypotenuse equals the sum of the squares of the other two sides, often "
     "written as a squared plus b squared equals c squared.",
     [("In the Pythagorean theorem, what equals a squared plus b squared?", "c squared")]),
    ("The mathematical constant pi is approximately 3.14, representing the ratio "
     "of a circle's circumference to its diameter.",
     [("What is the approximate value of pi?", "3.14")]),
    ("The telephone was invented by Alexander Graham Bell in 1876.",
     [("Who invented the telephone?", "Alexander Graham Bell")]),
    ("The World Wide Web was invented by Tim Berners-Lee in 1989 at CERN.",
     [("Who invented the World Wide Web?", "Tim Berners-Lee")]),
    ("The Mona Lisa is a famous portrait painted by Leonardo da Vinci during "
     "the Italian Renaissance.",
     [("Who painted the Mona Lisa?", "Leonardo da Vinci")]),
    ("Ludwig van Beethoven was a German composer who wrote nine symphonies, "
     "despite becoming deaf later in life.",
     [("How many symphonies did Beethoven write?", "nine")]),
    ("The currency of the United Kingdom is the pound sterling, often symbolized "
     "with the pound sign.",
     [("What is the currency of the United Kingdom?", "pound sterling")]),
    ("Mandarin Chinese is the most spoken native language in the world by number "
     "of speakers.",
     [("What is the most spoken native language in the world?", "Mandarin Chinese")]),
    ("The Olympic flag features five interlocking rings, representing the five "
     "continents of the world.",
     [("How many rings are on the Olympic flag?", "five")]),
    ("The Great Wall of China was built over many centuries to protect against "
     "invasions from the north.",
     [("What was the Great Wall of China built to protect against?", "invasions")]),
    ("The Colosseum is a large amphitheater in Rome, Italy, built in the Roman "
     "Empire for public spectacles.",
     [("In which city is the Colosseum located?", "Rome")]),
    ("The Statue of Liberty was a gift from France to the United States, located "
     "in New York Harbor.",
     [("What country gifted the Statue of Liberty?", "France")]),
    ("The Pacific Ocean is the largest and deepest ocean on Earth, covering more "
     "area than all the landmasses combined.",
     [("What is the largest ocean on Earth?", "Pacific")]),
    ("Mercury is the closest planet to the Sun and has no atmosphere to retain "
     "heat, causing extreme temperature swings.",
     [("Which planet is closest to the Sun?", "Mercury")]),
    ("A caterpillar transforms into a butterfly through a process called "
     "metamorphosis.",
     [("What does a caterpillar transform into?", "butterfly")]),
    ("The currency used in Japan is the yen, which is issued by the Bank of Japan.",
     [("What is the currency of Japan?", "yen")]),
    ("Albert Einstein developed the theory of relativity, which includes the "
     "famous equation energy equals mass times the speed of light squared.",
     [("Who developed the theory of relativity?", "Albert Einstein")]),
    ("The novel The Hobbit was written by J.R.R. Tolkien and published in 1937 "
     "as a prelude to The Lord of the Rings.",
     [("Who wrote The Hobbit?", "J.R.R. Tolkien")]),
]


def build_records() -> list[dict]:
    """Flatten CORE into a list of {passage, question, answer} records."""
    records = []
    for passage, qas in CORE:
        for question, answer in qas:
            records.append({"passage": passage, "question": question, "answer": answer})
    return records


def validate(records: list[dict]) -> None:
    """Assert the two invariants: answer is a verbatim span of passage; no eval leakage."""
    errors = []
    for i, r in enumerate(records):
        if r["answer"].lower() not in r["passage"].lower():
            errors.append(f"  record {i}: answer {r['answer']!r} NOT in passage")
        if r["question"].lower().strip() in EVAL_QUESTIONS:
            errors.append(f"  record {i}: question collides with an eval item: {r['question']!r}")
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(e)
        sys.exit(1)
    print(f"VALIDATION OK: {len(records)} records, all answers are verbatim spans, "
          f"no eval leakage.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/grounded_qa_train.jsonl",
                    help="output JSONL path (default: data/grounded_qa_train.jsonl)")
    ap.add_argument("--validate", action="store_true",
                    help="assert answers are verbatim spans + no eval leakage, then exit")
    args = ap.parse_args()

    records = build_records()
    if args.validate:
        validate(records)
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_qa = len(records)
    n_passages = len(CORE)
    print(f"Wrote {n_qa} grounded-QA records ({n_passages} passages) -> {args.out}")
    print("Re-run with --validate to check the verbatim-span + no-leakage invariants.")


if __name__ == "__main__":
    main()