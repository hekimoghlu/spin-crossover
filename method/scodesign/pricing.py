"""Price models (manuscript appendix C).  One histogram is one task / one circuit.

Assumption (documented): N_1q in the HQC formula counts the twelve ansatz Ry rotations;
basis-change rotations and state preparation are not counted; N_m = 5 measured qubits.
"""
from . import constants as C
from . import ansatz as A

ARM_HISTOGRAMS = {"raw": [1.0] * 3, "mit": [1.0, 1.5, 2.0] * 3,
                  "rsc": [1.0] * 6, "pmit": [1.0, 1.5, 2.0] * 6}


def histogram_cost(platform, shots, scale=1.0):
    p = C.PLATFORMS[platform]
    if platform == "Cepheus":
        return C.BRAKET_TASK + C.RIGETTI_SHOT * shots
    if platform == "Forte":
        return C.BRAKET_TASK + C.IONQ_SHOT * shots
    if platform == "Heron":
        return C.IBM_PER_MIN / 60.0 * C.IBM_SHOT_TIME * shots
    if platform == "Helios":
        n2, n1 = A.gate_counts(routed=p["routed"], scale=scale)
        return 5.0 + shots * (n1 + 10 * n2 + 5 * A.NQ) / 5000.0
    return None  # Willow: not priced


def histogram_cost_generic(platform, shots, scale, routed):
    if platform != "Helios":
        return histogram_cost(platform, shots, scale)
    n2, n1 = A.gate_counts(routed=routed, scale=scale)
    return 5.0 + shots * (n1 + 10 * n2 + 5 * A.NQ) / 5000.0


def eval_cost(platform, arm, shots):
    """Cost of one evaluation of one complex with the given arm."""
    if C.PLATFORMS[platform]["currency"] is None:
        return None
    return sum(histogram_cost(platform, shots, s) for s in ARM_HISTOGRAMS[arm])


def currency(platform):
    return C.PLATFORMS[platform]["currency"]
