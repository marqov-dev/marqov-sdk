"""Circuit format conversion between Marqov and C++ simulation engine."""

import re


def count_qubits(qasm_str: str) -> int:
    """Extract qubit count from OpenQASM 2.0 string.

    Parses 'qreg q[N];' declarations. Sums across multiple qreg
    declarations if present.

    Raises:
        ValueError: If no qreg declaration is found.
    """
    matches = re.findall(r"qreg\s+\w+\[(\d+)\]", qasm_str)
    if not matches:
        raise ValueError("No qreg declaration found in OpenQASM string")
    return sum(int(m) for m in matches)


def ensure_measurements(qasm_str: str) -> str:
    """Append measurement instructions if the OpenQASM string lacks them.

    Marqov's Circuit class does not include measurement gates. The C++
    simulation engine requires explicit measurement instructions to
    produce counts.

    If no 'measure' statements are found, appends creg declarations
    and measure instructions for all qubits.
    """
    if re.search(r"\bmeasure\b", qasm_str):
        return qasm_str

    qreg_matches = re.findall(r"qreg\s+(\w+)\[(\d+)\]", qasm_str)
    if not qreg_matches:
        return qasm_str

    measure_lines = []
    for reg_name, reg_size in qreg_matches:
        n = int(reg_size)
        creg_name = f"c_{reg_name}"
        measure_lines.append(f"creg {creg_name}[{n}];")
        for i in range(n):
            measure_lines.append(f"measure {reg_name}[{i}] -> {creg_name}[{i}];")

    suffix = "\n".join(measure_lines)
    return qasm_str.rstrip() + "\n" + suffix + "\n"


def convert_counts(qristal_results) -> dict[str, int]:
    """Convert C++ simulation result format to Marqov counts dict.

    The C++ engine returns results as a pybind11-wrapped ``MapVectorBoolInt``
    where keys are ``VectorBool`` sequences and values are integer counts.
    Unlike a Python dict, this type has **no** ``.items()`` method —
    iteration yields keys only, and values are accessed via ``map[key]``.

    This function uses key-iteration with ``[]`` lookup, which works for
    both Python dicts and the pybind11 opaque map type.

    Returns:
        Dictionary mapping bitstrings ("00", "11") to shot counts.
    """
    counts: dict[str, int] = {}

    for key in qristal_results:
        count = qristal_results[key]
        bitstring = "".join("1" if b else "0" for b in key)
        counts[bitstring] = count

    return counts
