// Batched noisy density-matrix simulator for the paired-excitation ansatz.
//
// Independent reimplementation (C++17 + OpenMP).  Every task runs the same
// gate program (op list) with its own parameter vector:
//   code[k] = {opcode, q0, q1, pidx}, coef[k]
//   opcode 0 RY   : Ry(coef * params[pidx]) on wire q0
//   opcode 1 CNOT : control q0, target q1
//   opcode 2 DEP1 : single-qubit depolarising on q0, lambda = params[pidx]
//   opcode 3 DEP2 : two-qubit depolarising on (q0,q1), lambda = params[pidx]
// The state stays real because all gates are real (Ry, CNOT) and the initial
// state is a computational basis state.  Wire q is bit (nq-1-q) of an index.
#include <cmath>
#include <cstring>
#include <vector>
#include <omp.h>

namespace {

inline void apply_ry(double* r, int D, int b, double phi) {
    const double c = std::cos(0.5 * phi), s = std::sin(0.5 * phi);
    // left multiply (rows)
    for (int i0 = 0; i0 < D; ++i0) {
        if (i0 & b) continue;
        double* r0 = r + (size_t)i0 * D;
        double* r1 = r + (size_t)(i0 | b) * D;
        for (int j = 0; j < D; ++j) {
            const double a = r0[j], d = r1[j];
            r0[j] = c * a - s * d;
            r1[j] = s * a + c * d;
        }
    }
    // right multiply by U^T (columns)
    for (int i = 0; i < D; ++i) {
        double* ri = r + (size_t)i * D;
        for (int j0 = 0; j0 < D; ++j0) {
            if (j0 & b) continue;
            const double a = ri[j0], d = ri[j0 | b];
            ri[j0] = c * a - s * d;
            ri[j0 | b] = s * a + c * d;
        }
    }
}

inline void apply_cnot(double* r, int D, int bc, int bt) {
    for (int i = 0; i < D; ++i) {
        if ((i & bc) && !(i & bt)) {
            double* a = r + (size_t)i * D;
            double* b = r + (size_t)(i | bt) * D;
            for (int j = 0; j < D; ++j) { double t = a[j]; a[j] = b[j]; b[j] = t; }
        }
    }
    for (int i = 0; i < D; ++i) {
        double* ri = r + (size_t)i * D;
        for (int j = 0; j < D; ++j) {
            if ((j & bc) && !(j & bt)) { double t = ri[j]; ri[j] = ri[j | bt]; ri[j | bt] = t; }
        }
    }
}

inline void apply_dep1(double* r, int D, int b, double lam) {
    const double keep = 1.0 - lam;
    for (int i0 = 0; i0 < D; ++i0) {
        if (i0 & b) continue;
        const int i1 = i0 | b;
        for (int j0 = 0; j0 < D; ++j0) {
            if (j0 & b) continue;
            const int j1 = j0 | b;
            double& a = r[(size_t)i0 * D + j0];
            double& d = r[(size_t)i1 * D + j1];
            const double tr = a + d;
            a = keep * a + 0.5 * lam * tr;
            d = keep * d + 0.5 * lam * tr;
            r[(size_t)i0 * D + j1] *= keep;
            r[(size_t)i1 * D + j0] *= keep;
        }
    }
}

inline void apply_dep2(double* r, int D, int b0, int b1, double lam) {
    const double keep = 1.0 - lam;
    const int m = b0 | b1;
    const int off[4] = {0, b1, b0, b0 | b1};
    for (int i0 = 0; i0 < D; ++i0) {
        if (i0 & m) continue;
        for (int j0 = 0; j0 < D; ++j0) {
            if (j0 & m) continue;
            double tr = 0.0;
            for (int k = 0; k < 4; ++k) tr += r[(size_t)(i0 | off[k]) * D + (j0 | off[k])];
            for (int k = 0; k < 4; ++k)
                for (int l = 0; l < 4; ++l)
                    r[(size_t)(i0 | off[k]) * D + (j0 | off[l])] *= keep;
            for (int k = 0; k < 4; ++k) r[(size_t)(i0 | off[k]) * D + (j0 | off[k])] += 0.25 * lam * tr;
        }
    }
}

}  // namespace

extern "C" {

int dmsim_version() { return 1; }

int dmsim_max_threads() { return omp_get_max_threads(); }

// out: ntask * D * D doubles (row-major density matrices)
int dmsim_run_batch(int nq, int nops, const int* code, const double* coef,
                    int ntask, int nparam, const double* params, int init, double* out) {
    if (nq < 1 || nq > 10) return -1;
    const int D = 1 << nq;
#pragma omp parallel for schedule(dynamic, 4)
    for (int t = 0; t < ntask; ++t) {
        double* r = out + (size_t)t * D * D;
        std::memset(r, 0, sizeof(double) * D * D);
        r[(size_t)init * D + init] = 1.0;
        const double* p = params + (size_t)t * nparam;
        for (int k = 0; k < nops; ++k) {
            const int op = code[4 * k], q0 = code[4 * k + 1], q1 = code[4 * k + 2], pi = code[4 * k + 3];
            const int b0 = 1 << (nq - 1 - q0);
            switch (op) {
                case 0: apply_ry(r, D, b0, coef[k] * p[pi]); break;
                case 1: apply_cnot(r, D, b0, 1 << (nq - 1 - q1)); break;
                case 2: apply_dep1(r, D, b0, p[pi]); break;
                case 3: apply_dep2(r, D, b0, 1 << (nq - 1 - q1), p[pi]); break;
                default: break;
            }
        }
    }
    return 0;
}

}  // extern "C"
