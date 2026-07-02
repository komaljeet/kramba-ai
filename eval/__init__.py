"""Evaluation harnesses for the Hamiltonian Vortex Network (HVN).

This package provides ready-to-run evaluation scripts that benchmark HVN
against Transformers and SSMs on real datasets:

  - ``eval/run_wikitext.py``   — WikiText-103 subset evaluation
  - ``eval/run_shakespeare.py`` — Tiny Shakespeare evaluation
  - ``eval/run_code.py``       — Python code corpus evaluation
  - ``eval/long_context.py``    — Long-context retrieval and scaling tests

Each script downloads (or accepts a local path to) the dataset, trains all
three architectures with the same hyperparameters, and reports perplexity,
training time, and parameter efficiency.
"""

from .datasets import (
    load_wikitext,
    load_shakespeare,
    load_code_corpus,
    DATASET_REGISTRY,
)

__all__ = [
    "load_wikitext",
    "load_shakespeare",
    "load_code_corpus",
    "DATASET_REGISTRY",
]