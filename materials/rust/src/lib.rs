//! scmc: hysteresis of spin-crossover particles versus size (std only).
//!
//! * `mf`: Slichter-Drickamer mean-field loop width with an inactive surface
//!   layer, Gamma(d) = Gamma_bulk (1 - 2 delta / d)^3 (same model as the
//!   Python `scmaterials.sizemap`).
//! * `mc`: Ising-like (Wajnflasz-Pick) Metropolis Monte Carlo on a simple cubic
//!   lattice cut to a sphere, surface layer removed, temperature-dependent
//!   field h(T) = (dH - T dS)/2 and coupling J = Gamma_bulk / (2 z), z = 6, so
//!   that its mean-field limit is the Slichter-Drickamer model.
//! * `rng`: SplitMix64 generator, reproduced bit-for-bit in the Python tests.

pub const R_GAS: f64 = 8.314462618;

pub mod rng {
    #[derive(Clone, Debug)]
    pub struct SplitMix64 {
        pub state: u64,
    }
    impl SplitMix64 {
        pub fn new(seed: u64) -> Self {
            SplitMix64 { state: seed }
        }
        pub fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        /// uniform in [0, 1) with 53 random bits
        pub fn next_f64(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
        }
    }
}

pub mod mf {
    use super::R_GAS;

    pub fn gamma_of_d(d_nm: f64, gamma_bulk: f64, delta_nm: f64) -> f64 {
        let f = (1.0 - 2.0 * delta_nm / d_nm).max(0.0);
        gamma_bulk * f * f * f
    }

    pub fn d_critical(gamma_bulk: f64, t_half: f64, delta_nm: f64) -> f64 {
        let r = 2.0 * R_GAS * t_half / gamma_bulk;
        if r >= 1.0 {
            return f64::INFINITY;
        }
        2.0 * delta_nm / (1.0 - r.powf(1.0 / 3.0))
    }

    pub fn t_of_x(x: f64, gamma: f64, t_half: f64, ds: f64) -> f64 {
        let dh = t_half * ds;
        (dh + gamma * (1.0 - 2.0 * x)) / (ds - R_GAS * (x / (1.0 - x)).ln())
    }

    /// golden-section search for the maximum of f on [a, b]
    fn golden_max<F: Fn(f64) -> f64>(f: F, mut a: f64, mut b: f64, tol: f64) -> f64 {
        let gr = (5f64.sqrt() - 1.0) / 2.0;
        let mut c = b - gr * (b - a);
        let mut d = a + gr * (b - a);
        let (mut fc, mut fd) = (f(c), f(d));
        while (b - a).abs() > tol {
            if fc > fd {
                b = d;
                d = c;
                fd = fc;
                c = b - gr * (b - a);
                fc = f(c);
            } else {
                a = c;
                c = d;
                fc = fd;
                d = a + gr * (b - a);
                fd = f(d);
            }
        }
        f(0.5 * (a + b))
    }

    /// Mean-field loop width (K): turning points of T(x).
    pub fn loop_width(gamma: f64, t_half: f64, ds: f64) -> f64 {
        if gamma <= 2.0 * R_GAS * t_half {
            return 0.0;
        }
        let eps = 1e-12;
        let t_up = golden_max(|x| t_of_x(x, gamma, t_half, ds), eps, 0.5, 1e-13);
        let t_dn = -golden_max(|x| -t_of_x(x, gamma, t_half, ds), 0.5, 1.0 - eps, 1e-13);
        (t_up - t_dn).max(0.0)
    }
}

pub mod mc {
    use super::rng::SplitMix64;
    use super::R_GAS;

    /// Spherical particle on a simple cubic lattice; only active sites kept.
    pub struct Particle {
        pub n: usize,
        pub nbr: Vec<[i64; 6]>, // neighbour indices into active list, -1 if none
    }

    pub fn build_particle(d_nm: f64, a_nm: f64, delta_nm: f64) -> Particle {
        let r_act = d_nm / 2.0 - delta_nm;
        let m = (d_nm / 2.0 / a_nm).ceil() as i64 + 1;
        let side = (2 * m + 1) as usize;
        let mut map = vec![-1i64; side * side * side];
        let mut coords = Vec::new();
        let id = |i: i64, j: i64, k: i64| -> usize {
            (((i + m) as usize) * side + (j + m) as usize) * side + (k + m) as usize
        };
        if r_act > 0.0 {
            for i in -m..=m {
                for j in -m..=m {
                    for k in -m..=m {
                        let r = a_nm * ((i * i + j * j + k * k) as f64).sqrt();
                        if r <= r_act {
                            map[id(i, j, k)] = coords.len() as i64;
                            coords.push((i, j, k));
                        }
                    }
                }
            }
        }
        let mut nbr = Vec::with_capacity(coords.len());
        for &(i, j, k) in &coords {
            let mut nb = [-1i64; 6];
            let d = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)];
            for (t, (di, dj, dk)) in d.iter().enumerate() {
                let (a, b, c) = (i + di, j + dj, k + dk);
                if a.abs() <= m && b.abs() <= m && c.abs() <= m {
                    nb[t] = map[id(a, b, c)];
                }
            }
            nbr.push(nb);
        }
        Particle { n: coords.len(), nbr }
    }

    pub struct McParams {
        pub gamma_bulk: f64, // J/mol
        pub t_half: f64,
        pub ds: f64,
        pub t_min: f64,
        pub t_max: f64,
        pub n_t: usize,
        pub sweeps_per_t: usize,
        pub seed: u64,
    }

    /// One Metropolis sweep in site order. s[i] = +1 (HS) or -1 (LS).
    pub fn sweep(p: &Particle, s: &mut [i8], jc: f64, h: f64, t: f64, rng: &mut SplitMix64) {
        let rt = R_GAS * t;
        for i in 0..p.n {
            let mut sum = 0.0;
            for &j in &p.nbr[i] {
                if j >= 0 {
                    sum += s[j as usize] as f64;
                }
            }
            let si = s[i] as f64;
            // E = -J sum_<ij> s_i s_j + h sum_i s_i ; flip s_i -> -s_i
            let de = 2.0 * jc * si * sum - 2.0 * h * si;
            let u = rng.next_f64();
            if de <= 0.0 || u < (-de / rt).exp() {
                s[i] = -s[i];
            }
        }
    }

    /// Heating then cooling sweep. Returns (T grid, x_HS heating, x_HS cooling).
    pub fn hysteresis(p: &Particle, prm: &McParams) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let jc = prm.gamma_bulk / (2.0 * 6.0);
        let dh = prm.t_half * prm.ds;
        let mut rng = SplitMix64::new(prm.seed);
        let ts: Vec<f64> = (0..prm.n_t)
            .map(|k| prm.t_min + (prm.t_max - prm.t_min) * k as f64 / (prm.n_t - 1) as f64)
            .collect();
        let mut s = vec![-1i8; p.n];
        let frac = |s: &[i8]| -> f64 {
            if s.is_empty() {
                return f64::NAN;
            }
            s.iter().filter(|&&v| v > 0).count() as f64 / s.len() as f64
        };
        let mut up = Vec::with_capacity(prm.n_t);
        for &t in &ts {
            let h = 0.5 * (dh - t * prm.ds);
            for _ in 0..prm.sweeps_per_t {
                sweep(p, &mut s, jc, h, t, &mut rng);
            }
            up.push(frac(&s));
        }
        let mut down = vec![0.0; prm.n_t];
        for k in (0..prm.n_t).rev() {
            let t = ts[k];
            let h = 0.5 * (dh - t * prm.ds);
            for _ in 0..prm.sweeps_per_t {
                sweep(p, &mut s, jc, h, t, &mut rng);
            }
            down[k] = frac(&s);
        }
        (ts, up, down)
    }

    /// temperature at which a monotone-ish branch crosses x = 1/2 (linear interp.)
    pub fn crossing(ts: &[f64], x: &[f64]) -> f64 {
        for k in 1..ts.len() {
            let (a, b) = (x[k - 1] - 0.5, x[k] - 0.5);
            if a.is_nan() || b.is_nan() {
                return f64::NAN;
            }
            if a == 0.0 {
                return ts[k - 1];
            }
            if a * b < 0.0 || b == 0.0 {
                return ts[k - 1] + (ts[k] - ts[k - 1]) * a / (a - b);
            }
        }
        f64::NAN
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_reference() {
        // reference values of SplitMix64 seeded with 1234567
        let mut r = rng::SplitMix64::new(1234567);
        assert_eq!(r.next_u64(), 6457827717110365317);
        assert_eq!(r.next_u64(), 3203168211198807973);
    }

    #[test]
    fn critical_diameters() {
        assert!((mf::d_critical(8e3, 310.0, 1.0) - 14.6769).abs() < 1e-3);
        assert!(mf::loop_width(mf::gamma_of_d(10.0, 8e3, 1.0), 310.0, 60.0) == 0.0);
    }

    #[test]
    fn loop_width_bulk() {
        let w = mf::loop_width(8e3, 310.0, 60.0);
        assert!((w - 42.8365).abs() < 1e-3, "{}", w);
    }

    #[test]
    fn mc_small_particle_has_no_loop() {
        let p = mc::build_particle(6.0, 0.73, 1.0);
        let prm = mc::McParams { gamma_bulk: 8e3, t_half: 310.0, ds: 60.0, t_min: 250.0,
            t_max: 370.0, n_t: 61, sweeps_per_t: 200, seed: 7 };
        let (ts, up, dn) = mc::hysteresis(&p, &prm);
        let w = mc::crossing(&ts, &up) - mc::crossing(&ts, &dn);
        assert!(w.abs() < 6.0, "{}", w);
    }
}
