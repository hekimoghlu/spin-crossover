// scmk.cpp -- OpenMP kernels for the scmaterials package (independent reimplementation).
//
//  sf_sum        : kinematic structure factor F(g) = sum_j f_{s_j}(g) exp(2 pi i g . r_j)
//  synth_image   : real-space synthesis  V(r) = Re sum_g F(g) exp(2 pi i g . r)
//  render_spheres: projected thickness map of an ensemble of spheres
//
// Plain C ABI for ctypes.  Accumulation in double precision.
#include <cmath>
#include <cstdint>
#ifdef _OPENMP
#include <omp.h>
#endif

static const double TWO_PI = 6.283185307179586476925286766559;

extern "C" {

int scmk_num_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// pos: n_atoms x 3 (Angstrom), species: n_atoms (index into ftab rows),
// g: n_g x 3 (1/Angstrom), ftab: n_species x n_g (scattering factor of each
// species evaluated at each g, Debye-Waller included by the caller).
void sf_sum(int64_t n_atoms, const double* pos, const int32_t* species,
            int64_t n_g, const double* g, const double* ftab,
            double* out_re, double* out_im) {
    #pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < n_g; ++k) {
        const double gx = g[3 * k], gy = g[3 * k + 1], gz = g[3 * k + 2];
        double re = 0.0, im = 0.0;
        for (int64_t j = 0; j < n_atoms; ++j) {
            const double ph = TWO_PI * (gx * pos[3 * j] + gy * pos[3 * j + 1] + gz * pos[3 * j + 2]);
            const double f = ftab[(int64_t)species[j] * n_g + k];
            re += f * std::cos(ph);
            im += f * std::sin(ph);
        }
        out_re[k] = re;
        out_im[k] = im;
    }
}

// xy: n_pix x 2, g2: n_g x 2, F: n_g complex (split re/im). out: n_pix.
void synth_image(int64_t n_pix, const double* xy, int64_t n_g, const double* g2,
                 const double* F_re, const double* F_im, double* out) {
    #pragma omp parallel for schedule(static)
    for (int64_t p = 0; p < n_pix; ++p) {
        const double x = xy[2 * p], y = xy[2 * p + 1];
        double acc = 0.0;
        for (int64_t k = 0; k < n_g; ++k) {
            const double ph = TWO_PI * (g2[2 * k] * x + g2[2 * k + 1] * y);
            acc += F_re[k] * std::cos(ph) - F_im[k] * std::sin(ph);
        }
        out[p] = acc;
    }
}

// Projected thickness t(x,y) = sum_i 2 sqrt(R_i^2 - rho_i^2) on an ny x nx grid
// with pixel size px (same length unit as cx, cy, rad); pixel centres at (ix+0.5)*px.
void render_spheres(int64_t nx, int64_t ny, double px, int64_t n_p,
                    const double* cx, const double* cy, const double* rad, double* out) {
    #pragma omp parallel for schedule(static)
    for (int64_t iy = 0; iy < ny; ++iy) {
        const double y = (iy + 0.5) * px;
        for (int64_t ix = 0; ix < nx; ++ix) {
            const double x = (ix + 0.5) * px;
            double t = 0.0;
            for (int64_t i = 0; i < n_p; ++i) {
                const double dx = x - cx[i], dy = y - cy[i];
                const double r2 = rad[i] * rad[i] - dx * dx - dy * dy;
                if (r2 > 0.0) t += 2.0 * std::sqrt(r2);
            }
            out[iy * nx + ix] = t;
        }
    }
}

}  // extern "C"
