// UNTESTED: this CUDA kernel has never been compiled or run.  The development
// machine has no GPU and no nvcc.  It mirrors cpp/dmsim.cpp (same op codes)
// and is loaded by scodesign/cuda_backend.py through a CuPy RawKernel; when
// CuPy or a GPU is absent the backend falls back to NumPy.
//
// One thread block per task; the 32x32 (D x D) real density matrix lives in
// shared memory; threads cooperate over matrix elements for each op.
//   opcode 0 RY(q0, coef*params[pidx]); 1 CNOT(q0->q1);
//   opcode 2 DEP1(q0, params[pidx]);     3 DEP2(q0,q1, params[pidx])
extern "C" __global__
void dmsim_batch(int nq, int nops, const int* __restrict__ code, const double* __restrict__ coef,
                 int nparam, const double* __restrict__ params, int init, double* __restrict__ out)
{
    extern __shared__ double r[];           // D*D doubles
    const int D = 1 << nq;
    const int t = blockIdx.x;
    const double* p = params + (size_t)t * nparam;
    for (int e = threadIdx.x; e < D * D; e += blockDim.x) r[e] = 0.0;
    __syncthreads();
    if (threadIdx.x == 0) r[init * D + init] = 1.0;
    __syncthreads();
    for (int k = 0; k < nops; ++k) {
        const int op = code[4 * k], q0 = code[4 * k + 1], q1 = code[4 * k + 2], pi = code[4 * k + 3];
        const int b0 = 1 << (nq - 1 - q0);
        if (op == 0) {
            const double phi = coef[k] * p[pi];
            const double c = cos(0.5 * phi), s = sin(0.5 * phi);
            // rows
            for (int e = threadIdx.x; e < D * D; e += blockDim.x) {
                int i = e / D, j = e % D;
                if (i & b0) continue;
                double a = r[i * D + j], d = r[(i | b0) * D + j];
                r[i * D + j] = c * a - s * d; r[(i | b0) * D + j] = s * a + c * d;
            }
            __syncthreads();
            for (int e = threadIdx.x; e < D * D; e += blockDim.x) {
                int i = e / D, j = e % D;
                if (j & b0) continue;
                double a = r[i * D + j], d = r[i * D + (j | b0)];
                r[i * D + j] = c * a - s * d; r[i * D + (j | b0)] = s * a + c * d;
            }
        } else if (op == 1) {
            const int bt = 1 << (nq - 1 - q1);
            for (int e = threadIdx.x; e < D * D; e += blockDim.x) {
                int i = e / D, j = e % D;
                if ((i & b0) && !(i & bt)) { double tmp = r[i * D + j]; r[i * D + j] = r[(i | bt) * D + j]; r[(i | bt) * D + j] = tmp; }
            }
            __syncthreads();
            for (int e = threadIdx.x; e < D * D; e += blockDim.x) {
                int i = e / D, j = e % D;
                if ((j & b0) && !(j & bt)) { double tmp = r[i * D + j]; r[i * D + j] = r[i * D + (j | bt)]; r[i * D + (j | bt)] = tmp; }
            }
        } else if (op == 2) {
            const double lam = p[pi], keep = 1.0 - lam;
            for (int e = threadIdx.x; e < D * D; e += blockDim.x) {
                int i0 = e / D, j0 = e % D;
                if ((i0 & b0) || (j0 & b0)) continue;
                int i1 = i0 | b0, j1 = j0 | b0;
                double a = r[i0 * D + j0], d = r[i1 * D + j1], tr = a + d;
                r[i0 * D + j0] = keep * a + 0.5 * lam * tr;
                r[i1 * D + j1] = keep * d + 0.5 * lam * tr;
                r[i0 * D + j1] *= keep; r[i1 * D + j0] *= keep;
            }
        } else if (op == 3) {
            const int b1 = 1 << (nq - 1 - q1), m = b0 | b1;
            const int off[4] = {0, b1, b0, b0 | b1};
            const double lam = p[pi], keep = 1.0 - lam;
            for (int e = threadIdx.x; e < D * D; e += blockDim.x) {
                int i0 = e / D, j0 = e % D;
                if ((i0 & m) || (j0 & m)) continue;
                double tr = 0.0;
                for (int a = 0; a < 4; ++a) tr += r[(i0 | off[a]) * D + (j0 | off[a])];
                for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b) r[(i0 | off[a]) * D + (j0 | off[b])] *= keep;
                for (int a = 0; a < 4; ++a) r[(i0 | off[a]) * D + (j0 | off[a])] += 0.25 * lam * tr;
            }
        }
        __syncthreads();
    }
    for (int e = threadIdx.x; e < D * D; e += blockDim.x) out[(size_t)t * D * D + e] = r[e];
}

// ---------------------------------------------------------------------------
// UNTESTED (no GPU / nvcc here).  Batched QAOA statevector evolution: one block per
// parameter set; the 2^n complex amplitudes live in global scratch `work`
// (nsets * 2 * 2^n doubles).  out[k] = <f> for parameter set k.
extern "C" __global__
void qaoa_batch(int n, const double* __restrict__ f, int p, const double* __restrict__ params,
                double* __restrict__ work, double* __restrict__ out)
{
    const int k = blockIdx.x;
    const int N = 1 << n;
    double* re = work + (size_t)k * 2 * N;
    double* im = re + N;
    const double* g = params + (size_t)k * 2 * p;
    const double* b = g + p;
    for (int z = threadIdx.x; z < N; z += blockDim.x) { re[z] = rsqrt((double)N); im[z] = 0.0; }
    __syncthreads();
    for (int l = 0; l < p; ++l) {
        for (int z = threadIdx.x; z < N; z += blockDim.x) {
            double c = cos(g[l] * f[z]), s = -sin(g[l] * f[z]);
            double a = re[z], d = im[z];
            re[z] = a * c - d * s; im[z] = a * s + d * c;
        }
        __syncthreads();
        const double c = cos(b[l]), s = sin(b[l]);
        for (int q = 0; q < n; ++q) {
            const int bit = 1 << (n - 1 - q);
            for (int z = threadIdx.x; z < N; z += blockDim.x) {
                if (z & bit) continue;
                double r0 = re[z], i0 = im[z], r1 = re[z | bit], i1 = im[z | bit];
                // [c, -i s; -i s, c]
                re[z] = c * r0 + s * i1;       im[z] = c * i0 - s * r1;
                re[z | bit] = s * i0 + c * r1; im[z | bit] = -s * r0 + c * i1;
            }
            __syncthreads();
        }
    }
    __shared__ double acc[256];
    double e = 0.0;
    for (int z = threadIdx.x; z < N; z += blockDim.x) e += (re[z] * re[z] + im[z] * im[z]) * f[z];
    acc[threadIdx.x] = e;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) acc[threadIdx.x] += acc[threadIdx.x + s]; __syncthreads(); }
    if (threadIdx.x == 0) out[k] = acc[0];
}

// UNTESTED.  Batched recurrent-depth inference: one thread per input row, H <= 64.
extern "C" __global__
void rdm_forward(int N, int nin, int H, const double* __restrict__ X, const double* __restrict__ W_in,
                 const double* __restrict__ b_in, const double* __restrict__ W_h, const double* __restrict__ W_x,
                 const double* __restrict__ b_h, const double* __restrict__ w_o, double b_o, double beta, int T,
                 double* __restrict__ Y)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double e[64], h[64], ex[64], nh[64];
    for (int j = 0; j < H; ++j) {
        double a = b_in[j];
        for (int k = 0; k < nin; ++k) a += X[(size_t)i * nin + k] * W_in[k * H + j];
        e[j] = tanh(a); h[j] = e[j];
    }
    for (int j = 0; j < H; ++j) { double a = b_h[j]; for (int k = 0; k < H; ++k) a += e[k] * W_x[k * H + j]; ex[j] = a; }
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < H; ++j) {
            double a = ex[j];
            for (int k = 0; k < H; ++k) a += h[k] * W_h[k * H + j];
            nh[j] = (1.0 - beta) * h[j] + beta * tanh(a);
        }
        double y = b_o;
        for (int j = 0; j < H; ++j) { h[j] = nh[j]; y += h[j] * w_o[j]; }
        Y[(size_t)i * T + t] = y;
    }
}
