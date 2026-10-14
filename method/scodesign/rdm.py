"""Recurrent-depth models in plain NumPy (manuscript sec. 6.3-6.4).

Architecture (ASSUMPTIONS.md section 28)
    e   = tanh(X W_in + b_in)                       input encoder
    h_0 = e
    h_t = (1 - beta) h_{t-1} + beta tanh(h_{t-1} W_h + e W_x + b_h)   shared recurrent block
    y_t = h_t w_o + b_o                             readout after every iteration
Training minimises the mean over iterations t = 1..T_train of the squared error of y_t, so that
every iterate is a valid estimate (hand-written backpropagation through time, Adam).
At test time the block is iterated until |y_t - y_{t-1}| < tol (adaptive depth / halting) or
T_max is reached; the number of iterations used is reported per input.
"""
import numpy as np


class RecurrentDepth:
    def __init__(self, n_in, n_hidden=16, beta=0.5, seed=0, scale=0.5):
        rng = np.random.default_rng(seed)
        H = n_hidden
        self.beta = beta
        self.p = dict(
            W_in=rng.normal(0, scale / np.sqrt(n_in), (n_in, H)), b_in=np.zeros(H),
            W_h=rng.normal(0, scale / np.sqrt(H), (H, H)), W_x=rng.normal(0, scale / np.sqrt(H), (H, H)),
            b_h=np.zeros(H), w_o=rng.normal(0, 1 / np.sqrt(H), H), b_o=np.zeros(1))
        self.xm = self.xs = None
        self.ym, self.ys = 0.0, 1.0

    # ------------------------------------------------------------ core passes
    def _forward(self, X, T):
        p, b = self.p, self.beta
        e = np.tanh(X @ p["W_in"] + p["b_in"])
        hs, zs, ys = [e], [], []
        h = e
        ex = e @ p["W_x"] + p["b_h"]
        for _ in range(T):
            z = np.tanh(h @ p["W_h"] + ex)
            h = (1 - b) * h + b * z
            hs.append(h); zs.append(z); ys.append(h @ p["w_o"] + p["b_o"][0])
        return e, hs, zs, ys

    def loss_and_grad(self, X, y, T, l2=0.0):
        p, b = self.p, self.beta
        e, hs, zs, ys = self._forward(X, T)
        N = X.shape[0]
        g = {k: np.zeros_like(v) for k, v in p.items()}
        loss = 0.0
        dh = np.zeros_like(e)
        de = np.zeros_like(e)
        for t in range(T, 0, -1):
            r = ys[t - 1] - y
            loss += np.mean(r ** 2) / T
            dy = 2 * r / (N * T)
            g["w_o"] += hs[t].T @ dy
            g["b_o"] += dy.sum()
            dh = dh + np.outer(dy, p["w_o"])
            da = dh * b * (1 - zs[t - 1] ** 2)
            g["W_h"] += hs[t - 1].T @ da
            g["W_x"] += e.T @ da
            g["b_h"] += da.sum(0)
            de += da @ p["W_x"].T
            dh = dh * (1 - b) + da @ p["W_h"].T
        de += dh
        dpre = de * (1 - e ** 2)
        g["W_in"] += X.T @ dpre
        g["b_in"] += dpre.sum(0)
        if l2:
            for k in ("W_in", "W_h", "W_x", "w_o"):
                loss += l2 * np.sum(p[k] ** 2)
                g[k] += 2 * l2 * p[k]
        return loss, g

    # ------------------------------------------------------------ training / inference
    def fit(self, X, y, T_train=10, steps=3000, lr=3e-3, batch=512, l2=1e-5, seed=0, val_frac=0.0):
        """Adam on the all-iterations loss.  With val_frac > 0 a random validation split is held out
        and the weights with the lowest validation loss (checked every 50 steps) are kept (early stopping)."""
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-12
        self.ym, self.ys = y.mean(), y.std() + 1e-12
        Xn, yn = (X - self.xm) / self.xs, (y - self.ym) / self.ys
        rng = np.random.default_rng(seed)
        Xv = yv = None
        if val_frac > 0:
            perm = rng.permutation(len(Xn)); nv = max(1, int(round(val_frac * len(Xn))))
            Xv, yv = Xn[perm[:nv]], yn[perm[:nv]]
            Xn, yn = Xn[perm[nv:]], yn[perm[nv:]]
        best = (np.inf, None)
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v = {k: np.zeros_like(v_) for k, v_ in self.p.items()}
        b1, b2 = 0.9, 0.999
        hist = []
        for it in range(1, steps + 1):
            idx = rng.choice(len(Xn), min(batch, len(Xn)), replace=False) if batch < len(Xn) else np.arange(len(Xn))
            loss, g = self.loss_and_grad(Xn[idx], yn[idx], T_train, l2)
            for k in self.p:
                m[k] = b1 * m[k] + (1 - b1) * g[k]
                v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
                self.p[k] -= lr * (m[k] / (1 - b1 ** it)) / (np.sqrt(v[k] / (1 - b2 ** it)) + 1e-8)
            if it % 100 == 0:
                hist.append(float(loss))
            if Xv is not None and it % 50 == 0:
                vl, _ = self.loss_and_grad(Xv, yv, T_train)
                if vl < best[0]:
                    best = (vl, {k: v_.copy() for k, v_ in self.p.items()})
        if best[1] is not None:
            self.p = best[1]
        return hist

    def iterates(self, Xn, T, backend="auto"):
        """Normalised outputs after each of T iterations; C++ kernel (OpenMP) or NumPy."""
        if backend in ("auto", "cpp"):
            from . import native
            Y = native.rdm_forward_native(Xn, self.p, self.beta, T)
            if Y is not None:
                return Y
            if backend == "cpp":
                raise RuntimeError("C++ kernels not built")
        if backend == "cuda":
            from . import cuda_backend
            return cuda_backend.rdm_forward(Xn, self.p, self.beta, T)
        _, _, _, ys = self._forward(Xn, T)
        return np.stack(ys, 1)

    def predict(self, X, tol=1e-3, T_max=64, T_min=1, backend="auto"):
        """Adaptive depth: iterate each input until its (normalised) output changes by < tol."""
        Xn = (np.asarray(X, float) - self.xm) / self.xs
        Y = self.iterates(Xn, T_max, backend)                 # (N, T_max)
        diff = np.abs(np.diff(Y, axis=1))                     # change between iterate t and t+1
        depth = np.full(len(Xn), T_max)
        for i in range(len(Xn)):
            ok = np.where(diff[i] < tol)[0]
            ok = ok[ok + 2 >= T_min]
            if len(ok):
                depth[i] = ok[0] + 2                          # iterations actually run
        out = Y[np.arange(len(Xn)), depth - 1]
        return out * self.ys + self.ym, depth

    def predict_fixed(self, X, T):
        Xn = (np.asarray(X, float) - self.xm) / self.xs
        _, _, _, ys = self._forward(Xn, T)
        return np.stack(ys, 1) * self.ys + self.ym            # (N, T)


# ------------------------------------------------------------------ decoder features
def decoder_features(Ec_scales, Er_scales, Eref_r, n_cnot, shots):
    """Scale-free inputs: folded-energy ratios of the circuit and of its reference twin, the
    reference twin's known attenuation, CNOT count and shot count."""
    Ec_scales = np.asarray(Ec_scales); Er_scales = np.asarray(Er_scales)
    f = [Ec_scales[:, 1] / Ec_scales[:, 0], Ec_scales[:, 2] / Ec_scales[:, 0],
         Er_scales[:, 1] / Er_scales[:, 0], Er_scales[:, 2] / Er_scales[:, 0],
         Er_scales[:, 0] / Eref_r,
         np.full(len(Ec_scales), n_cnot / 48.0), np.full(len(Ec_scales), np.log10(shots) - 3.0)]
    return np.column_stack(f)
