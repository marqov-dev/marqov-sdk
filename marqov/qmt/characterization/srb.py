"""Simultaneous Randomized Benchmarking (SRB) experiment runner.

Runs isolated and simultaneous single-qubit RB using the Clifford group,
with per-length checkpointing to resume interrupted experiments.

The key comparison: isolated RB measures intrinsic gate error on target
qubits, while simultaneous RB measures gate error when neighbor qubits
are being driven concurrently — the difference reveals cross-talk.

References:
    Gambetta et al. 2012, arXiv:1204.6308
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from marqov.circuits import Circuit
from marqov.executors.base import BaseExecutor
from marqov.qmt.characterization.clifford import (
    clifford_to_circuit,
    generate_rb_sequence,
)

logger = logging.getLogger(__name__)

_DEFAULT_SEQUENCE_LENGTHS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


@dataclass
class SRBConfig:
    """Configuration for an SRB experiment.

    Attributes:
        target_qubits: Qubits under test.
        neighbor_qubits: Qubits driven simultaneously (empty for isolated-only).
        sequence_lengths: Clifford sequence lengths to sweep.
        num_sequences_short: Number of random sequences for short lengths.
        num_sequences_long: Number of random sequences for long lengths.
        length_threshold: Lengths <= this use num_sequences_short.
        shots: Measurement shots per circuit.
        seed: Base random seed for reproducibility.
    """

    target_qubits: list[int]
    neighbor_qubits: list[int]
    sequence_lengths: list[int] = field(
        default_factory=lambda: list(_DEFAULT_SEQUENCE_LENGTHS)
    )
    num_sequences_short: int = 20
    num_sequences_long: int = 10
    length_threshold: int = 32
    shots: int = 100
    seed: int = 42

    def num_sequences_for_length(self, length: int) -> int:
        """Return the number of random sequences to use for a given length."""
        if length <= self.length_threshold:
            return self.num_sequences_short
        return self.num_sequences_long


@dataclass
class SRBResult:
    """Results from an SRB experiment.

    Attributes:
        config: The experiment configuration.
        isolated_survival: Average survival probability per sequence length (isolated).
        simultaneous_survival: Average survival probability per sequence length (simultaneous).
        isolated_raw: Per-sequence survival probabilities per length (isolated).
        simultaneous_raw: Per-sequence survival probabilities per length (simultaneous).
    """

    config: SRBConfig
    isolated_survival: dict[int, float]
    simultaneous_survival: dict[int, float]
    isolated_raw: dict[int, list[float]]
    simultaneous_raw: dict[int, list[float]]


def _merge_circuits(a: Circuit, b: Circuit) -> Circuit:
    """Merge two circuits by replaying operations from both.

    Gates from circuit *a* are applied first, then gates from *b*.
    The resulting circuit spans the union of qubits used by both.
    """
    merged = Circuit()
    for op in a._qf._elements:
        merged._qf += op
    for op in b._qf._elements:
        merged._qf += op
    return merged


def _measure_survival(
    counts: dict[str, int],
    target_qubits: list[int],
    total_qubits: int,
) -> float:
    """Compute survival probability from measurement counts.

    Survival = fraction of shots where all target qubits measure |0>.
    Bitstrings are big-endian: position 0 is the leftmost bit (highest qubit).

    Args:
        counts: Bitstring -> count mapping from execution.
        target_qubits: Physical qubit indices to check.
        total_qubits: Total number of qubits in the circuit.

    Returns:
        Survival probability in [0, 1].
    """
    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0

    surviving = 0
    for bitstring, count in counts.items():
        # Pad short bitstrings (simulator may omit leading zeros)
        padded = bitstring.zfill(total_qubits)
        # Big-endian: qubit i is at position (total_qubits - 1 - i)
        all_zero = all(
            padded[total_qubits - 1 - q] == "0" for q in target_qubits
        )
        if all_zero:
            surviving += count

    return surviving / total_shots


def _save_checkpoint(
    config: SRBConfig,
    isolated: dict[int, list[float]],
    simultaneous: dict[int, list[float]],
    output_dir: str | None,
) -> None:
    """Save intermediate results to a JSON checkpoint file."""
    if output_dir is None:
        return

    # Use target+neighbor qubits in filename to avoid overwriting across buffer distances
    target_str = "_".join(str(q) for q in config.target_qubits)
    neighbor_str = "_".join(str(q) for q in config.neighbor_qubits) if config.neighbor_qubits else "none"
    path = Path(output_dir) / f"srb_checkpoint_t{target_str}_n{neighbor_str}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "target_qubits": config.target_qubits,
        "neighbor_qubits": config.neighbor_qubits,
        "isolated_raw": {str(k): v for k, v in isolated.items()},
        "simultaneous_raw": {str(k): v for k, v in simultaneous.items()},
    }
    path.write_text(json.dumps(data, indent=2))
    logger.debug("Checkpoint saved to %s", path)


async def run_srb(
    config: SRBConfig,
    executor: BaseExecutor,
    output_dir: str | None = None,
) -> SRBResult:
    """Run a Simultaneous Randomized Benchmarking experiment.

    For each sequence length, runs isolated RB (target qubits only) and
    simultaneous RB (target + neighbor qubits driven concurrently). Uses
    the same random seeds for target circuits in both modes so the only
    variable is neighbor activity.

    Args:
        config: Experiment configuration.
        executor: Backend executor for circuit execution.
        output_dir: Directory for checkpoint files (optional).

    Returns:
        SRBResult with survival probabilities for both modes.
    """
    isolated_raw: dict[int, list[float]] = {}
    simultaneous_raw: dict[int, list[float]] = {}

    all_qubits = sorted(set(config.target_qubits) | set(config.neighbor_qubits))
    total_qubits = max(all_qubits) + 1 if all_qubits else 1
    has_neighbors = len(config.neighbor_qubits) > 0

    for length in config.sequence_lengths:
        num_sequences = config.num_sequences_for_length(length)
        iso_survivals: list[float] = []
        sim_survivals: list[float] = []

        logger.info(
            "SRB length=%d, sequences=%d, isolated+simultaneous=%s",
            length,
            num_sequences,
            has_neighbors,
        )

        for seq_idx in range(num_sequences):
            seq_seed = config.seed + length * 1000 + seq_idx

            # --- Build target circuit (same seed for both modes) ---
            target_circuits: list[Circuit] = []
            for qubit in config.target_qubits:
                rb_seq = generate_rb_sequence(length, seed=seq_seed + qubit)
                target_circuits.append(clifford_to_circuit(rb_seq, qubit))

            # Merge all target qubit circuits into one
            target_circuit = target_circuits[0]
            for tc in target_circuits[1:]:
                target_circuit = _merge_circuits(target_circuit, tc)

            # --- Isolated RB ---
            try:
                result = await executor.execute(
                    target_circuit, shots=config.shots
                )
                survival = _measure_survival(
                    result.counts, config.target_qubits, total_qubits
                )
                iso_survivals.append(survival)
            except Exception:
                logger.warning(
                    "Isolated RB failed: length=%d, seq=%d", length, seq_idx,
                    exc_info=True,
                )

            # --- Simultaneous RB ---
            if has_neighbors:
                try:
                    neighbor_seed = seq_seed + 50000
                    neighbor_circuits: list[Circuit] = []
                    for qubit in config.neighbor_qubits:
                        nb_seq = generate_rb_sequence(
                            length, seed=neighbor_seed + qubit
                        )
                        neighbor_circuits.append(
                            clifford_to_circuit(nb_seq, qubit)
                        )

                    # Merge neighbor circuits together
                    neighbor_circuit = neighbor_circuits[0]
                    for nc in neighbor_circuits[1:]:
                        neighbor_circuit = _merge_circuits(
                            neighbor_circuit, nc
                        )

                    # Merge target + neighbor
                    combined = _merge_circuits(target_circuit, neighbor_circuit)

                    result = await executor.execute(
                        combined, shots=config.shots
                    )
                    survival = _measure_survival(
                        result.counts, config.target_qubits, total_qubits
                    )
                    sim_survivals.append(survival)
                except Exception:
                    logger.warning(
                        "Simultaneous RB failed: length=%d, seq=%d",
                        length,
                        seq_idx,
                        exc_info=True,
                    )

        isolated_raw[length] = iso_survivals
        if has_neighbors:
            simultaneous_raw[length] = sim_survivals

        _save_checkpoint(config, isolated_raw, simultaneous_raw, output_dir)

    # Compute averages
    isolated_survival = {
        length: (sum(vals) / len(vals) if vals else 0.0)
        for length, vals in isolated_raw.items()
    }
    simultaneous_survival = {
        length: (sum(vals) / len(vals) if vals else 0.0)
        for length, vals in simultaneous_raw.items()
    }

    return SRBResult(
        config=config,
        isolated_survival=isolated_survival,
        simultaneous_survival=simultaneous_survival,
        isolated_raw=isolated_raw,
        simultaneous_raw=simultaneous_raw,
    )
