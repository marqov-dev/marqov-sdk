"""Quantum benchmarking tools.

Provides SPAM (State Preparation And Measurement) benchmarking and
application-level distribution-fidelity metrics (Lubinski/QED-C normalized
fidelity).
Future: refactor into BenchmarkSuite with .spam(), .qst(), .qpt(), .fidelity().
"""

from marqov.benchmarking.fidelity import (
    classical_fidelity,
    fidelity_with_uniform,
    normalized_fidelity,
)
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
    "classical_fidelity",
    "fidelity_with_uniform",
    "normalized_fidelity",
    "spam_benchmark",
]
