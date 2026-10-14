"""Radiation budget: ionisation per particle per gray, Bethe proton stopping
power, and Rutherford/NRT displacement count."""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad

N_A = 6.02214076e23
E_CH = 1.602176634e-19
ME_C2_MEV = 0.51099895
MP_C2_MEV = 938.27208816
K_BETHE = 0.307075          # MeV cm^2 / mol
E2_EVCM = 1.439964547e-7    # e^2/(4 pi eps0) in eV cm

# element data: Z, A (g/mol), mean excitation energy I (eV; ICRU 37 / PDG values)
ELEMENTS = {
    "H": (1, 1.008, 19.2),
    "C": (6, 12.011, 78.0),
    "N": (7, 14.007, 82.0),
    "O": (8, 15.999, 95.0),
    "Fe": (26, 55.845, 286.0),
    "Pt": (78, 195.084, 790.0),
}
WATER = {"H": 2, "O": 1}
WATER_I_EV = 75.0
SCO_FRAMEWORK = {"Fe": 1, "Pt": 1, "C": 8, "H": 4, "N": 6}   # Fe(pz)[Pt(CN)4]


def molar_mass(comp):
    return sum(n * ELEMENTS[e][1] for e, n in comp.items())


def z_over_a_and_I(comp, I_override=None):
    num = sum(n * ELEMENTS[e][0] for e, n in comp.items())
    mass = molar_mass(comp)
    lnI = sum(n * ELEMENTS[e][0] * np.log(ELEMENTS[e][2]) for e, n in comp.items()) / num
    I = np.exp(lnI) if I_override is None else I_override
    return num / mass, I


def bethe_mass_stopping(T_MeV, comp, I_override=None, M_MeV=MP_C2_MEV, z=1):
    """Mass stopping power (MeV cm^2/g) from the Bethe formula with T_max,
    no shell or density-effect corrections."""
    ZA, I = z_over_a_and_I(comp, I_override)
    gamma = 1 + T_MeV / M_MeV
    beta2 = 1 - 1 / gamma ** 2
    bg2 = beta2 * gamma ** 2
    Tmax = 2 * ME_C2_MEV * bg2 / (1 + 2 * gamma * ME_C2_MEV / M_MeV + (ME_C2_MEV / M_MeV) ** 2)
    I_MeV = I * 1e-6
    L = 0.5 * np.log(2 * ME_C2_MEV * bg2 * Tmax / I_MeV ** 2) - beta2
    return K_BETHE * z * z * ZA / beta2 * L


def particle_mass_kg(d_nm, rho_gcm3=1.6):
    V = np.pi / 6 * (np.asarray(d_nm, float) * 1e-9) ** 3
    return V * rho_gcm3 * 1e3


def ionisations_per_gray(d_nm, rho_gcm3=1.6, W_eV=30.0):
    return particle_mass_kg(d_nm, rho_gcm3) / (W_eV * E_CH)


def robinson_damage_energy(T_eV, Z, A):
    """Lindhard-Robinson damage energy for a self-recoil (Z1=Z2=Z, A1=A2=A)."""
    EL = 30.724 * Z * Z * np.sqrt(2 * Z ** (2 / 3)) * 2.0
    eps = T_eV / EL
    k = 0.1337 * Z ** (1 / 6) * np.sqrt(Z / A)
    g = 3.4008 * eps ** (1 / 6) + 0.40244 * eps ** 0.75 + eps
    return T_eV / (1 + k * g)


def nrt(T_eV, Ed, Z, A, partition=True):
    Td = robinson_damage_energy(T_eV, Z, A) if partition else T_eV
    if Td < Ed:
        return 0.0
    if Td < 2 * Ed / 0.8:
        return 1.0
    return 0.8 * Td / (2 * Ed)


def displacement_cross_section(E_MeV, Z2, A2, Ed=25.0, Z1=1, A1=1.008, partition=True):
    """sigma_d = int_{Ed}^{Tmax} dsigma/dT nu_NRT(T) dT  (cm^2), Rutherford, non-relativistic."""
    E = E_MeV * 1e6
    Tmax = 4 * A1 * A2 / (A1 + A2) ** 2 * E
    C = np.pi * Z1 ** 2 * Z2 ** 2 * E2_EVCM ** 2 * A1 / (A2 * E)
    if Tmax <= Ed:
        return 0.0
    f = lambda lnT: C / np.exp(lnT) * nrt(np.exp(lnT), Ed, Z2, A2, partition)  # dT = T dlnT
    val, _ = quad(f, np.log(Ed), np.log(Tmax), limit=400, points=[np.log(2.5 * Ed)])
    return val


def displacements_per_particle(d_nm, fluence_cm2, E_MeV=10.0, rho_gcm3=1.6, comp=SCO_FRAMEWORK,
                               Ed=25.0, partition=True):
    n_fu = particle_mass_kg(d_nm, rho_gcm3) * 1e3 / molar_mass(comp) * N_A
    tot = 0.0
    for el, n in comp.items():
        Z, A, _ = ELEMENTS[el]
        tot += n * n_fu * displacement_cross_section(E_MeV, Z, A, Ed, partition=partition)
    return tot * fluence_cm2


def budget():
    ipg50 = float(ionisations_per_gray(50))
    S_w = float(bethe_mass_stopping(10.0, WATER, I_override=WATER_I_EV))
    S_s = float(bethe_mass_stopping(10.0, SCO_FRAMEWORK))
    phi = 1e10
    gy = phi * S_s * 1.602176634e-10  # MeV/g -> J/kg
    disp = displacements_per_particle(50, phi)
    disp_nopart = displacements_per_particle(50, phi, partition=False)
    ion = gy * ipg50
    return dict(
        ion_per_gy_50nm=ipg50,
        gy_per_ion_50nm=1 / ipg50,
        ion_per_gy_200nm=float(ionisations_per_gray(200)),
        particles_per_radiograph=1 / (0.010 * ipg50),
        frac_per_2gy_fraction_pct=100 * 2.0 * ipg50,
        ion_per_60gy_course=60.0 * ipg50,
        stopping_water_10MeV=S_w,
        stopping_sco_10MeV=S_s,
        I_sco_eV=float(z_over_a_and_I(SCO_FRAMEWORK)[1]),
        gy_per_1e10_fluence_sco=gy,
        disp_50nm_10MeV_1e10=disp,
        disp_50nm_10MeV_1e10_no_partition=disp_nopart,
        ion_50nm_10MeV_1e10=ion,
        ratio_ion_disp=ion / disp,
        ratio_ion_disp_no_partition=ion / disp_nopart,
    )
