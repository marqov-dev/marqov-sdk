"""Run locally after installing marqov[qutip]; print recorded solver observables.

This example does not submit a cloud job or require an account.
"""

import numpy as np
from qutip import basis, mesolve, sigmam, sigmaz

from marqov.qutip import record


def main():
    times = np.linspace(0, 5, 11)
    # QuTiP's basis(2, 0) has sigma-z = +1; relaxation drives it toward -1.
    result = mesolve(
        0.5 * sigmaz(),
        basis(2, 0),
        times,
        c_ops=[np.sqrt(0.4) * sigmam()],
        e_ops=[sigmaz()],
        options={"store_states": False, "progress_bar": ""},
    )
    record(result, observable_names=["sigma_z"])


if __name__ == "__main__":
    main()
