//! scmc CLI.
//!   scmc mf  <out.csv>                      mean-field loop width vs diameter
//!   scmc mc  <out.csv> [sweeps] [threads]   Monte Carlo loops vs diameter (std::thread)
//!   scmc mcloop <out.csv> d gamma a delta tmin tmax nt sweeps seed   one MC loop (test hook)
use scmc::{mc, mf};
use std::env;
use std::fs::File;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::thread;

fn diameters() -> Vec<f64> {
    let mut d = Vec::new();
    let mut x = 4.0f64;
    while x <= 300.0 {
        d.push((x * 1000.0).round() / 1000.0);
        x *= 1.08;
    }
    d
}

fn run_mf(path: &str) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    writeln!(f, "gamma_bulk_kJmol,d_nm,gamma_J_mol,loop_width_K,d_c_nm")?;
    for &g in &[6e3, 8e3, 10e3, 12e3] {
        let dc = mf::d_critical(g, 310.0, 1.0);
        for d in diameters().into_iter().chain([10.0, 20.0, 50.0, 100.0, 200.0]) {
            let gd = mf::gamma_of_d(d, g, 1.0);
            writeln!(f, "{},{},{:.12},{:.12},{:.12}", g / 1e3, d, gd, mf::loop_width(gd, 310.0, 60.0), dc)?;
        }
        writeln!(f, "{},inf,{:.12},{:.12},{:.12}", g / 1e3, g, mf::loop_width(g, 310.0, 60.0), dc)?;
    }
    Ok(())
}

fn run_mc(path: &str, sweeps: usize, nthreads: usize) -> std::io::Result<()> {
    let ds_list = [4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0];
    let gammas = [8e3, 10e3, 12e3];
    let mut jobs = Vec::new();
    for (gi, &g) in gammas.iter().enumerate() {
        for (di, &d) in ds_list.iter().enumerate() {
            jobs.push((gi, di, g, d));
        }
    }
    let jobs = Arc::new(Mutex::new(jobs));
    let results = Arc::new(Mutex::new(Vec::new()));
    let mut hs = Vec::new();
    for _ in 0..nthreads {
        let jobs = Arc::clone(&jobs);
        let results = Arc::clone(&results);
        hs.push(thread::spawn(move || loop {
            let job = { jobs.lock().unwrap().pop() };
            let Some((gi, di, g, d)) = job else { break };
            let p = mc::build_particle(d, 0.73, 1.0);
            let prm = mc::McParams { gamma_bulk: g, t_half: 310.0, ds: 60.0, t_min: 230.0,
                t_max: 390.0, n_t: 81, sweeps_per_t: sweeps, seed: 1000 + (gi * 100 + di) as u64 };
            let (ts, up, dn) = mc::hysteresis(&p, &prm);
            let tu = mc::crossing(&ts, &up);
            let td = mc::crossing(&ts, &dn);
            results.lock().unwrap().push((g, d, p.n, tu, td, ts, up, dn));
        }));
    }
    for h in hs {
        h.join().unwrap();
    }
    let mut res = results.lock().unwrap().clone();
    res.sort_by(|a, b| (a.0, a.1).partial_cmp(&(b.0, b.1)).unwrap());
    let mut f = File::create(path)?;
    writeln!(f, "gamma_bulk_kJmol,d_nm,n_active,T_up,T_down,width_K")?;
    for r in &res {
        writeln!(f, "{},{},{},{:.6},{:.6},{:.6}", r.0 / 1e3, r.1, r.2, r.3, r.4, r.3 - r.4)?;
    }
    let mut g = File::create(format!("{}.loops.csv", path.trim_end_matches(".csv")))?;
    writeln!(g, "gamma_bulk_kJmol,d_nm,T,x_heat,x_cool")?;
    for r in &res {
        for k in 0..r.5.len() {
            writeln!(g, "{},{},{},{:.6},{:.6}", r.0 / 1e3, r.1, r.5[k], r.6[k], r.7[k])?;
        }
    }
    Ok(())
}

fn main() -> std::io::Result<()> {
    let a: Vec<String> = env::args().collect();
    if a.len() < 3 {
        eprintln!("usage: scmc mf|mc|mcloop <out.csv> ...");
        std::process::exit(2);
    }
    match a[1].as_str() {
        "mf" => run_mf(&a[2]),
        "mc" => {
            let sweeps = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(100);
            let nth = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(4);
            run_mc(&a[2], sweeps, nth)
        }
        "mcloop" => {
            let v: Vec<f64> = a[3..12].iter().map(|s| s.parse().unwrap()).collect();
            let p = mc::build_particle(v[0], v[2], v[3]);
            let prm = mc::McParams { gamma_bulk: v[1], t_half: 310.0, ds: 60.0, t_min: v[4],
                t_max: v[5], n_t: v[6] as usize, sweeps_per_t: v[7] as usize, seed: v[8] as u64 };
            let (ts, up, dn) = mc::hysteresis(&p, &prm);
            let mut f = File::create(&a[2])?;
            writeln!(f, "n_active,{}", p.n)?;
            writeln!(f, "T,x_heat,x_cool")?;
            for k in 0..ts.len() {
                writeln!(f, "{:.17e},{:.17e},{:.17e}", ts[k], up[k], dn[k])?;
            }
            Ok(())
        }
        _ => {
            eprintln!("unknown mode");
            std::process::exit(2)
        }
    }
}
