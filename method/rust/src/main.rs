//! scospace: std-only Rust sweep of the whole ligand space.
//!
//! Enumerates scaffolds (multisets of 5 scaffold ligands with <= 3 types, in
//! combinations_with_replacement order), attaches each sixth-site occupant,
//! computes the Joergensen ligand field, the mean-field gap, the PT2 pair
//! correlation, and the exact low-spin energy of the ten-dimensional
//! hard-core-boson block (cyclic Jacobi diagonalisation), in parallel with
//! std::thread, and writes a CSV.  Independent reimplementation.
//!
//! usage: scospace <out.csv> [k=0.24] [dS=60] [threads=4]
//!        scospace --extended <out.csv> [threads=4]
//!   generative-proposer candidate space: every scaffold with 4 or 5 ligand types, scored exactly
//!   (resting aqua and cyanide complexes, straddle condition) in parallel with std::thread.
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::thread;

const SCAFFOLD: [(&str, f64, f64); 7] = [
    ("H2O", 1.00, 1.0), ("ox", 0.99, 1.5), ("gly", 1.18, 1.5), ("py", 1.23, 1.5),
    ("NH3", 1.25, 1.4), ("en", 1.28, 1.5), ("bipy", 1.33, 1.5),
];
const SIXTH: [(&str, f64, f64); 5] = [
    ("H2O", 1.00, 1.0), ("Cl", 0.78, 2.0), ("N3", 0.83, 2.4), ("NCS", 1.02, 1.6), ("CN", 1.70, 2.0),
];
const G: f64 = 10.0e3;
const B0: f64 = 1058.0;
const C_OVER_B: f64 = 4.2;
const CM_TO_JMOL: f64 = 6.62607015e-34 * 2.99792458e10 * 6.02214076e23;

#[derive(Clone)]
struct Row { scaffold: usize, sixth: usize, label: String, delta: f64, b: f64, j: f64,
             e_ref: f64, e_hs: f64, e_ls: f64, e2: f64 }

fn combos_with_replacement(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut idx = vec![0usize; k];
    loop {
        out.push(idx.clone());
        // find rightmost index that can be incremented
        let mut i = k as isize - 1;
        while i >= 0 && idx[i as usize] == n - 1 { i -= 1; }
        if i < 0 { break; }
        let v = idx[i as usize] + 1;
        for t in (i as usize)..k { idx[t] = v; }
    }
    out
}

fn block_states() -> Vec<u32> { (0u32..32).filter(|x| x.count_ones() == 3).collect() }

fn bit(x: u32, q: u32) -> u32 { (x >> (4 - q)) & 1 }

fn hb_block(delta: f64, j: f64) -> [[f64; 10]; 10] {
    let eps = [-0.4 * delta, -0.4 * delta, -0.4 * delta, 0.6 * delta, 0.6 * delta];
    let u = 2.0 * j;
    let w = -2.0 * j; // 4U' - 2J with U' = 0
    let st = block_states();
    let mut h = [[0.0f64; 10]; 10];
    for a in 0..10 {
        let x = st[a];
        let mut d = 0.0;
        for m in 0..5u32 { if bit(x, m) == 1 { d += 2.0 * eps[m as usize] + u; } }
        for m in 0..5u32 { for mp in (m + 1)..5u32 { if bit(x, m) == 1 && bit(x, mp) == 1 { d += w; } } }
        h[a][a] = d;
        for b in 0..10 { if a != b && (st[a] ^ st[b]).count_ones() == 2 { h[a][b] = j; } }
    }
    h
}

/// Smallest eigenvalue of a symmetric 10x10 matrix by cyclic Jacobi rotations.
fn jacobi_min(mut a: [[f64; 10]; 10]) -> f64 {
    let n = 10;
    for _sweep in 0..100 {
        let mut off = 0.0;
        for p in 0..n { for q in (p + 1)..n { off += a[p][q] * a[p][q]; } }
        if off < 1e-22 { break; }
        for p in 0..n {
            for q in (p + 1)..n {
                if a[p][q].abs() < 1e-300 { continue; }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k][p]; let akq = a[k][q];
                    a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p][k]; let aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk;
                }
            }
        }
    }
    (0..n).map(|i| a[i][i]).fold(f64::INFINITY, f64::min)
}

fn pt2(h: &[[f64; 10]; 10]) -> f64 {
    let st = block_states();
    let r = st.iter().position(|&x| x == 0b11100).unwrap();
    let mut e = 0.0;
    for s in 0..10 { if s != r { e -= h[s][r] * h[s][r] / (h[s][s] - h[r][r]); } }
    e
}

fn compute(scaffold: usize, sixth: usize, sites: &[usize], x: usize, k: f64) -> Row {
    let mut fsum = 0.0; let mut hsum = 0.0;
    let mut label = String::new();
    for (li, l) in SCAFFOLD.iter().enumerate() {
        let n = sites.iter().filter(|&&s| s == li).count();
        if n > 0 { if !label.is_empty() { label.push('+'); } label.push_str(&format!("{}{}", n, l.0)); }
    }
    label.push('|'); label.push_str(SIXTH[x].0);
    for &s in sites { fsum += SCAFFOLD[s].1; hsum += SCAFFOLD[s].2; }
    fsum += SIXTH[x].1; hsum += SIXTH[x].2;
    let delta = G * fsum / 6.0;
    let b = B0 * (1.0 - k * hsum / 6.0);
    let p = 2.5 * b + 4.0 * C_OVER_B * b;
    let j = p / 4.0;
    let h = hb_block(delta, j);
    Row { scaffold, sixth, label, delta, b, j,
          e_ref: -2.4 * delta, e_hs: -0.4 * delta - 8.0 * j, e_ls: jacobi_min(h), e2: pt2(&h) }
}

fn extended(out: &str, nthreads: usize) {
    let cands: Vec<Vec<usize>> = combos_with_replacement(7, 5).into_iter()
        .filter(|c| { let mut u = c.clone(); u.dedup(); u.len() >= 4 }).collect();
    let chunk = (cands.len() + nthreads - 1) / nthreads;
    let mut handles = Vec::new();
    for (ci, part) in cands.chunks(chunk).enumerate() {
        let part = part.to_vec();
        handles.push(thread::spawn(move || {
            let conv = CM_TO_JMOL / 60.0;
            let rows: Vec<(String, f64, f64, bool)> = part.iter().map(|sc| {
                let r = compute(0, 0, sc, 0, 0.24);
                let a = compute(0, 4, sc, 4, 0.24);
                let tr = (r.e_hs - r.e_ls) * conv; let ta = (a.e_hs - a.e_ls) * conv;
                let ok = tr.min(ta) <= 290.0 && tr.max(ta) >= 330.0;
                (a.label.split('|').next().unwrap().to_string(), tr, ta, ok)
            }).collect();
            (ci, rows)
        }));
    }
    let mut parts: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    parts.sort_by_key(|p| p.0);
    let mut w = BufWriter::new(File::create(out).expect("cannot create output"));
    writeln!(w, "index,label,T_rest,T_CN,ok_CN").unwrap();
    let mut i = 0; let mut nok = 0;
    for (_, rows) in parts { for (l, tr, ta, ok) in rows {
        writeln!(w, "{},{},{:.12e},{:.12e},{}", i, l, tr, ta, ok as u8).unwrap(); i += 1; if ok { nok += 1; } } }
    eprintln!("scospace --extended: {} candidates, {} succeed for cyanide", i, nok);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 3 && args[1] == "--extended" {
        let nt: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(4);
        extended(&args[2], nt);
        return;
    }
    if args.len() < 2 { eprintln!("usage: scospace <out.csv> [k] [dS] [threads]"); std::process::exit(2); }
    let k: f64 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(0.24);
    let ds: f64 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(60.0);
    let nthreads: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(4);
    let scaffolds: Vec<Vec<usize>> = combos_with_replacement(7, 5).into_iter()
        .filter(|c| { let mut u = c.clone(); u.dedup(); u.len() <= 3 }).collect();
    let jobs: Vec<(usize, usize, Vec<usize>)> = scaffolds.iter().enumerate()
        .flat_map(|(si, sc)| (0..SIXTH.len()).map(move |x| (si, x, sc.clone()))).collect();
    let chunk = (jobs.len() + nthreads - 1) / nthreads;
    let mut handles = Vec::new();
    for (ci, part) in jobs.chunks(chunk).enumerate() {
        let part = part.to_vec();
        handles.push(thread::spawn(move || {
            (ci, part.iter().map(|(si, x, sc)| compute(*si, *x, sc, *x, k)).collect::<Vec<Row>>())
        }));
    }
    let mut parts: Vec<(usize, Vec<Row>)> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    parts.sort_by_key(|p| p.0);
    let rows: Vec<Row> = parts.into_iter().flat_map(|p| p.1).collect();
    let f = File::create(&args[1]).expect("cannot create output");
    let mut w = BufWriter::new(f);
    writeln!(w, "index,scaffold,sixth,label,Delta,B,J,E_ref,E_hs,E_ls,E2,dH_exact,dH_mf,dH_pt2,T_exact,T_mf,T_pt2").unwrap();
    let conv = CM_TO_JMOL / ds;
    for (i, r) in rows.iter().enumerate() {
        let dh_ex = r.e_hs - r.e_ls; let dh_mf = r.e_hs - r.e_ref; let dh_pt = dh_mf - r.e2;
        writeln!(w, "{},{},{},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
                 i, r.scaffold, r.sixth, r.label, r.delta, r.b, r.j, r.e_ref, r.e_hs, r.e_ls, r.e2,
                 dh_ex, dh_mf, dh_pt, dh_ex * conv, dh_mf * conv, dh_pt * conv).unwrap();
    }
    // summary on stderr: successful designs (resting = sixth 0)
    let n6 = SIXTH.len();
    let mut ok = 0;
    for s in 0..scaffolds.len() {
        let tr = (rows[s * n6].e_hs - rows[s * n6].e_ls) * conv;
        for x in 1..n6 {
            let ta = (rows[s * n6 + x].e_hs - rows[s * n6 + x].e_ls) * conv;
            if tr.min(ta) <= 290.0 && tr.max(ta) >= 330.0 { ok += 1; }
        }
    }
    eprintln!("scospace: {} scaffolds, {} complexes, {} pairs, {} successful designs (k={}, dS={}, threads={})",
              scaffolds.len(), rows.len(), scaffolds.len() * (n6 - 1), ok, k, ds, nthreads);
}
