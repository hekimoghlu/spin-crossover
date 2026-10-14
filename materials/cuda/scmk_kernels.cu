// scmk_kernels.cu -- CUDA kernels for the structure-factor sum and image synthesis.
//
// It was written in an environment without a GPU, nvcc or CuPy.  It mirrors
// cpp/scmk.cpp (one thread per reciprocal-lattice vector / per pixel) and is
// loaded through cupy.RawModule by scmaterials/cuda_kernels.py.  No result in
// this repository was produced with it.

extern "C" __global__
void sf_sum_kernel(long long n_atoms, const double* __restrict__ pos,
                   const int* __restrict__ species, long long n_g,
                   const double* __restrict__ g, const double* __restrict__ ftab,
                   double* __restrict__ out_re, double* __restrict__ out_im)
{
    const double TWO_PI = 6.283185307179586476925286766559;
    long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_g) return;
    const double gx = g[3 * k], gy = g[3 * k + 1], gz = g[3 * k + 2];
    double re = 0.0, im = 0.0;
    for (long long j = 0; j < n_atoms; ++j) {
        double s, c;
        const double ph = TWO_PI * (gx * pos[3 * j] + gy * pos[3 * j + 1] + gz * pos[3 * j + 2]);
        sincos(ph, &s, &c);
        const double f = ftab[(long long)species[j] * n_g + k];
        re += f * c;
        im += f * s;
    }
    out_re[k] = re;
    out_im[k] = im;
}

extern "C" __global__
void synth_image_kernel(long long n_pix, const double* __restrict__ xy,
                        long long n_g, const double* __restrict__ g2,
                        const double* __restrict__ F_re, const double* __restrict__ F_im,
                        double* __restrict__ out)
{
    const double TWO_PI = 6.283185307179586476925286766559;
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pix) return;
    const double x = xy[2 * p], y = xy[2 * p + 1];
    double acc = 0.0;
    for (long long k = 0; k < n_g; ++k) {
        double s, c;
        sincos(TWO_PI * (g2[2 * k] * x + g2[2 * k + 1] * y), &s, &c);
        acc += F_re[k] * c - F_im[k] * s;
    }
    out[p] = acc;
}
