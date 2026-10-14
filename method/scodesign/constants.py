"""Constants taken from the manuscript's method description.

Every number here is an *input* stated in the manuscript (ligand table,
platform table, price appendix, design condition).  No reported *result*
of the manuscript is stored in this package.
"""
from collections import OrderedDict

# ---------------------------------------------------------------- units
CM_TO_JMOL = 6.62607015e-34 * 2.99792458e10 * 6.02214076e23  # 1 cm^-1 in J/mol (11.96266)
HARTREE_CM = 219474.6313705
R_GAS = 8.314462618

# ---------------------------------------------------------------- design condition
T_BODY = 310.0     # K
MARGIN = 20.0      # K
DS_DEFAULT = 60.0  # J mol^-1 K^-1
T_LO = T_BODY - MARGIN
T_HI = T_BODY + MARGIN

# ---------------------------------------------------------------- ligand field (Table 1)
G_FE2 = 10.0e3     # cm^-1
B0_FE2 = 1058.0    # cm^-1
K_FE2 = 0.24       # assumed (flagged in paper)
C_OVER_B = 4.2

SCAFFOLD_LIGANDS = OrderedDict([
    ("H2O", (1.00, 1.0)),
    ("ox", (0.99, 1.5)),
    ("gly", (1.18, 1.5)),
    ("py", (1.23, 1.5)),
    ("NH3", (1.25, 1.4)),
    ("en", (1.28, 1.5)),
    ("bipy", (1.33, 1.5)),
])
RESTING = "H2O"
ANALYTES = OrderedDict([
    ("Cl", (0.78, 2.0)),
    ("N3", (0.83, 2.4)),
    ("NCS", (1.02, 1.6)),
    ("CN", (1.70, 2.0)),
])
SIXTH_SITE = ["H2O"] + list(ANALYTES.keys())
ALL_LIGANDS = OrderedDict(list(SCAFFOLD_LIGANDS.items()) + list(ANALYTES.items()))

N_SCAFFOLD_SITES = 5
MAX_TYPES = 3

# calibration compound: proxy for [Fe(phen)2(NCS)2] = 4 bipy + 2 NCS sites
CAL_COMPOUND = ["bipy"] * 4 + ["NCS"] * 2

# ---------------------------------------------------------------- platforms (Table 3)
# r2q, r1q, readout, routed-on-linear-chain?, currency
PLATFORMS = OrderedDict([
    ("Cepheus", dict(label="Rigetti Cepheus-1", r2q=9.0e-3, r1q=2.0e-3, ro=2.0e-2, routed=True, currency="USD")),
    ("Heron", dict(label="IBM Heron r3", r2q=1.52e-3, r1q=2.5e-4, ro=1.3e-2, routed=True, currency="USD")),
    ("Willow", dict(label="Google Willow", r2q=3.3e-3, r1q=3.5e-4, ro=7.7e-3, routed=True, currency=None)),
    ("Helios", dict(label="Quantinuum Helios", r2q=7.9e-4, r1q=2.5e-5, ro=3.3e-4, routed=False, currency="HQC")),
    ("Forte", dict(label="IonQ Forte", r2q=4.0e-3, r1q=2.0e-4, ro=5.0e-3, routed=False, currency="USD")),
])
SHOTS = [1000, 4000, 16000, 64000]
SHOT_TAGS = ["Lo", "Mid", "Hi", "Xhi"]
ZNE_SCALES = [1.0, 1.5, 2.0]
DRIFT_SIGMA = 0.5
N_EPOCHS = 8
N_EPOCHS_CALCOMP = 16   # "twice the usual number of drift epochs"

# ---------------------------------------------------------------- loop
N_CAL = 150
S_TARGET = 20.0   # K
TAU_FLOOR = 10.0  # K
BAND_FACTOR = 3.0

# ---------------------------------------------------------------- prices (appendix C)
BRAKET_TASK = 0.30
RIGETTI_SHOT = 0.000425
IONQ_SHOT = 0.08
IBM_PER_MIN = 96.0
IBM_SHOT_TIME = 250e-6

# ---------------------------------------------------------------- eps sweep (Table 6)
EPS_R2Q = [1e-4, 2e-4, 4e-4, 8e-4, 16e-4, 32e-4]
EPS_R1Q = 2.5e-5
EPS_RO = 3.3e-4
EPS_SHOTS = [16000, 64000]
EPS_EPOCHS = 6

SEED = 20260905
