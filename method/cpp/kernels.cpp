// CPU compute kernels for the implemented stack (C++17 + OpenMP), bound through ctypes.
//   qaoa_expect / qaoa_batch : 12-qubit (general n) QAOA statevector simulation with a diagonal cost
//   rdm_forward              : forward pass of the recurrent-depth network, all iterates
// Wire q is bit (n-1-q) of a basis index (same convention as the Python code).
#include <cmath>
#include <complex>
#include <vector>
#include <omp.h>

using cd = std::complex<double>;

namespace {
double qaoa_one(int n, const double* f, int p, const double* gam, const double* bet, double* probs) {
    const size_t N = size_t(1) << n;
    std::vector<cd> psi(N, cd(1.0 / std::sqrt(double(N)), 0.0));
    for (int l = 0; l < p; ++l) {
        for (size_t z = 0; z < N; ++z) psi[z] *= std::polar(1.0, -gam[l] * f[z]);
        const double c = std::cos(bet[l]), s = std::sin(bet[l]);
        for (int q = 0; q < n; ++q) {
            const size_t b = size_t(1) << (n - 1 - q);
            for (size_t z = 0; z < N; ++z) {
                if (z & b) continue;
                const cd a0 = psi[z], a1 = psi[z | b];
                psi[z] = c * a0 + cd(0, -s) * a1;          // exp(-i beta X)
                psi[z | b] = cd(0, -s) * a0 + c * a1;
            }
        }
    }
    double e = 0.0;
    for (size_t z = 0; z < N; ++z) {
        const double pr = std::norm(psi[z]);
        e += pr * f[z];
        if (probs) probs[z] = pr;
    }
    return e;
}
}  // namespace

extern "C" {

double qaoa_expect(int n, const double* f, int p, const double* gammas, const double* betas, double* probs) {
    return qaoa_one(n, f, p, gammas, betas, probs);
}

// params: nsets x (2p) = [gamma_1..gamma_p, beta_1..beta_p]
void qaoa_batch(int n, const double* f, int p, int nsets, const double* params, double* out) {
#pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < nsets; ++k) out[k] = qaoa_one(n, f, p, params + size_t(k) * 2 * p, params + size_t(k) * 2 * p + p, nullptr);
}

// X: N x nin (already normalised); Y: N x T (normalised outputs after each iteration)
void rdm_forward(int N, int nin, int H, const double* X, const double* W_in, const double* b_in,
                 const double* W_h, const double* W_x, const double* b_h, const double* w_o, double b_o,
                 double beta, int T, double* Y) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        std::vector<double> e(H), h(H), ex(H), nh(H);
        for (int j = 0; j < H; ++j) {
            double a = b_in[j];
            for (int k = 0; k < nin; ++k) a += X[size_t(i) * nin + k] * W_in[k * H + j];
            e[j] = std::tanh(a); h[j] = e[j];
        }
        for (int j = 0; j < H; ++j) {
            double a = b_h[j];
            for (int k = 0; k < H; ++k) a += e[k] * W_x[k * H + j];
            ex[j] = a;
        }
        for (int t = 0; t < T; ++t) {
            for (int j = 0; j < H; ++j) {
                double a = ex[j];
                for (int k = 0; k < H; ++k) a += h[k] * W_h[k * H + j];
                nh[j] = (1.0 - beta) * h[j] + beta * std::tanh(a);
            }
            h.swap(nh);
            double y = b_o;
            for (int j = 0; j < H; ++j) y += h[j] * w_o[j];
            Y[size_t(i) * T + t] = y;
        }
    }
}

}  // extern "C"
