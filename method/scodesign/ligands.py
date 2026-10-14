"""Ligand-space enumeration and Joergensen ligand field (manuscript sec. 2.2, 3.1)."""
import itertools
import numpy as np

from . import constants as C


def enumerate_scaffolds(max_types=C.MAX_TYPES, n_sites=C.N_SCAFFOLD_SITES, ligands=None):
    """Multisets of `n_sites` scaffold ligands with at most `max_types` distinct types.

    Order: itertools.combinations_with_replacement over the ligand table order
    (the Rust component reproduces exactly this order).
    """
    ligs = list(C.SCAFFOLD_LIGANDS.keys()) if ligands is None else list(ligands)
    out = []
    for combo in itertools.combinations_with_replacement(range(len(ligs)), n_sites):
        if len(set(combo)) <= max_types:
            out.append(tuple(ligs[i] for i in combo))
    return out


def enumerate_extended_scaffolds(min_types=C.MAX_TYPES + 1, n_sites=C.N_SCAFFOLD_SITES):
    ligs = list(C.SCAFFOLD_LIGANDS.keys())
    out = []
    for combo in itertools.combinations_with_replacement(range(len(ligs)), n_sites):
        if len(set(combo)) >= min_types:
            out.append(tuple(ligs[i] for i in combo))
    return out


def ligand_field(sites, k=C.K_FE2, g=C.G_FE2, B0=C.B0_FE2):
    """Delta = g * mean f, B = B0 (1 - k * mean h) over the six sites."""
    f = np.mean([C.ALL_LIGANDS[l][0] for l in sites])
    h = np.mean([C.ALL_LIGANDS[l][1] for l in sites])
    return g * f, B0 * (1.0 - k * h)


def pairing_energy(B, c_over_b=C.C_OVER_B):
    """P = 2.5 B + 4 C with C = 4.2 B  ->  P = 19.3 B."""
    return 2.5 * B + 4.0 * c_over_b * B


class Space:
    """The enumerated design space: scaffolds x sixth-site occupants."""

    def __init__(self, k=C.K_FE2, scaffolds=None):
        self.k = k
        self.scaffolds = enumerate_scaffolds() if scaffolds is None else scaffolds
        self.sixth = list(C.SIXTH_SITE)
        rows = []
        for si, sc in enumerate(self.scaffolds):
            for xi, x in enumerate(self.sixth):
                rows.append((si, xi, list(sc) + [x]))
        self.scaffold_idx = np.array([r[0] for r in rows])
        self.sixth_idx = np.array([r[1] for r in rows])
        self.sites = [r[2] for r in rows]
        lf = np.array([ligand_field(s, k=k) for s in self.sites])
        self.Delta = lf[:, 0]
        self.B = lf[:, 1]
        self.P = pairing_energy(self.B)
        self.J = self.P / 4.0
        # complex index lookup: (scaffold, sixth) -> row
        self.index = {(r[0], r[1]): i for i, r in enumerate(rows)}
        # design pairs: (scaffold, analyte) -> (resting complex row, analyte complex row)
        self.pairs = []
        for si in range(len(self.scaffolds)):
            for xi in range(1, len(self.sixth)):
                self.pairs.append((self.index[(si, 0)], self.index[(si, xi)], si, xi))
        self.pairs = np.array(self.pairs, dtype=int)

    @property
    def n_complexes(self):
        return len(self.sites)

    @property
    def n_pairs(self):
        return len(self.pairs)

    def name(self, i):
        sc = self.scaffolds[self.scaffold_idx[i]]
        parts = []
        for l in C.SCAFFOLD_LIGANDS:
            n = sc.count(l)
            if n:
                parts.append(f"{n}{l}")
        return "+".join(parts) + "|" + self.sixth[self.sixth_idx[i]]
