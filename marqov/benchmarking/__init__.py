"""Quantum benchmarking tools.

Currently provides SPAM (State Preparation And Measurement) benchmarking.
Future: refactor into BenchmarkSuite with .spam(), .qst(), .qpt(), .fidelity().
"""

from marqov.benchmarking.spam import (
    ConfusionMatrix,
    SPAMResult,
    apply_spam_correction,
    build_correction_matrix,
    spam_benchmark,
)

__all__ = [
    "ConfusionMatrix",
    "SPAMResult",
    "apply_spam_correction",
    "build_correction_matrix",
    "spam_benchmark",
]
