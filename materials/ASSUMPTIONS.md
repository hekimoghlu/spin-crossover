# ASSUMPTIONS and underspecified choices

This document lists every choice that the manuscript (paper B) and the parts of the
companion method paper (paper A) we used leave open. Each choice was fixed before we
compared against the paper's numbers. None was adjusted afterwards to improve agreement.
Where a different choice would have reproduced a paper number, VALIDATION.md says so.

## Ligand-field and Kanamori model (from paper A, eqs. thalf, jorgensen, kanamori, hb)
- The parameters are as stated in the papers: g(Fe²⁺) = 10,000 cm⁻¹, B₀ = 1058 cm⁻¹,
  k = 0.24, C = 4.2B, P = 2.5B + 4C, J_H = P/4, U′ = 0, U = 2J_H,
  ε(t2g) = −0.4Δ, ε(eg) = +0.6Δ, ΔS = 60 J mol⁻¹ K⁻¹, T_b = 310 K and m = 20 K.
  1 cm⁻¹ = 11.9627 J mol⁻¹.
- E_LS is the lowest eigenvalue of the 10-dimensional seniority-zero (3 pairs in 5
  orbitals) block. E_HS = −0.4Δ − 8J_H is the closed-form M_S = 2 determinant.
  T½ = ΔH/ΔS. A complex with ΔH ≤ 0 is high spin at every temperature and is treated as
  T½ = 0 in the straddle condition. No complex in the space has ΔH ≤ 0 near the window.
- The scaffold space is every multiset of five scaffold ligands with at most three
  distinct types: 301 scaffolds, 1505 complexes and 1204 pairs, as in the paper. Each
  ligand counts as one of the five sites, so a bidentate ligand is not handled
  specially. This matches the paper's counts and its names (for example "5bipy").
- **Spin-flip sign.** As printed in eq. (kanamori), the term
  +J c†_{m↑}c_{m↓}c†_{m′↓}c_{m′↑} breaks SU(2) symmetry: the M_S = 0 component of the
  quintet does not come out degenerate with M_S = 2. Our full Fock-space check
  (`kanamori_full`) uses the rotationally invariant sign (−J for this operator order).
  The term vanishes on the seniority-zero sector and on the M_S = 2 determinant, so
  it does not affect any reported number. The test checks that the full-space lowest
  singlet and quintet equal the block and closed-form values to 1e-8 cm⁻¹.

## Interferent map
- The paper names the imidazole, amine, carboxylate and thiolate proxies. For
  **hydroxide and phosphate**, which have no tabulated factors in the paper, we use the
  oxalate (O-donor anion) factors f = 0.99, h = 1.5. The paper's statement that "their
  factors are below that of water" is consistent with this choice.
- **Thiolate** (S-bound thiocyanate) uses f = 0.73, Jørgensen's tabulated value. Its
  **h is not given**. We assume h = 2.0, the value the table gives Cl⁻ and CN⁻.
  The sensitivity scan is in results.json; the consequence is in VALIDATION.md.
- A false positive is a donor complex that lands on the analyte's side of the straddle
  condition: T ≥ T_b + m if the analyte complex is at or above T_b + m, otherwise
  T ≤ T_b − m.
- f\* is computed for each donor, using that donor's h, as the sixth-site f at which
  the resting complex crosses 310 K. The margin is f\* − f_donor (sign reversed for
  LS-resting designs, of which there are none). The f\* statistics reported are those
  for imidazole, the tightest donor in every design.
- Cross-analyte silence: the design's scaffold is evaluated with each of the other
  three analytes, using the same false-positive rule.

## Competition budget
- The concentrations and detection levels are taken from the paper's table, and the
  ratio is c_i/c_X.

## Size map (Slichter–Drickamer)
- The equations are as stated. ΔS = 60 J mol⁻¹ K⁻¹ (the model's ΔS; it sets the loop
  width) and T½ = 310 K, so ΔH = T½ΔS. The loop width is the difference between the
  turning points of T(x).

## Geometry
- Fe–N is 1.97 Å (LS) and 2.17 Å (HS), as in the paper. **Assumed standard values:**
  C≡N 1.15 Å and Pt–C 1.99 Å. Pyrazine is built from C–N 1.339 Å, C–C 1.397 Å and
  ∠CNC 115.6°, which gives an N···N span of 2.824 Å; C–H is 1.08 Å.
- The cell is tetragonal with Fe at the origin and Pt at (a/2, a/2, 0). The CN bridges
  lie along the in-plane diagonals. The pyrazine pillars lie along c, with the ring
  plane in the (1-10) diagonal plane (assumed). There is one Fe per cell.
- The coordination polyhedron is a regular octahedron with all six Fe–N bonds equal.

## Readout
- The Curie moment uses g = 2, S = 2, T = 310 K and B = 0.1 T. The particle is a point
  dipole on axis, with r = standoff + d/2. Fe density is 2.6 nm⁻³: the paper's
  "one per 0.38 nm³" is 1/2.6 rounded, and 2.6 reproduces its 170,170 centres.
- The time to SNR 3 is t = (3η/B)², with η = 1 µT Hz⁻¹ᐟ² (typical) or 10 nT Hz⁻¹ᐟ²
  (best).
- Array pitch: an infinite square array of identical z-dipoles, truncated at ±400
  sites, with the sensor at height h above the central particle. The critical pitch is
  the largest pitch at which the |neighbour sum|/self ratio still reaches 0.1.
- Graphene: the Helmholtz shift is σΔp/(ε₀ε_r). n = k_F²/π with k_F = eΔφ/(ħv_F) and
  v_F = 10⁶ m s⁻¹ (assumed).
- Response times are L²/2D and 1/k_ex, plus Eigen–Wilkins association
  τ = 1/(K_os k_ex c) with K_os = 1 M⁻¹, as in the paper.

## Radiation
- Ionisation per gray uses particle mass (1.6 g cm⁻³) / (30 eV), as in the paper.
- Bethe: PDG form with T_max, without shell or density corrections. I = 75 eV for
  water. For the framework, I comes from Bragg additivity over Fe(pz)[Pt(CN)₄]
  (FePtC₈H₄N₆) with the elemental I values H 19.2, C 78, N 82, Fe 286 and Pt 790 eV,
  giving 226 eV.
- Displacements: non-relativistic Rutherford recoil spectrum with E_d = 25 eV for
  every element. NRT count with the Lindhard–Robinson damage-energy partition
  (self-recoil parameters), as the default. The variant without partition is also
  reported. The atom count comes from the 1.6 g cm⁻³ particle mass and the formula
  mass of 435 g mol⁻¹.

## Simulated microscopy (all images are SIMULATED and labelled so)
- Electron scattering factors are the first-Born amplitude of a Wentzel (Yukawa)
  atom, R = 0.885 a₀ Z^(−1/3). The Debye–Waller factor B = 0.5 Å² is assumed.
- Projected potential: Fourier synthesis over all zone-axis reflections with
  |g| ≤ 1/(0.08 nm), a hard band limit at the paper's information limit, normalised
  to [0, 1].
- Radial diffraction profile: kinematic |F|² for all reflections, each broadened by a
  Gaussian of FWHM 0.9/d (Scherrer, nm⁻¹). There is no Lorentz or polarisation
  factor.
- Bright-field ensembles: I/I₀ = exp(−t/Λ) with Λ = 25 nm (as in the paper). Assumed:
  lognormal diameters with σ_ln = 0.25, random sequential addition with no overlap in
  projection, a 1-pixel Gaussian PSF, Poisson noise at 400 e⁻/pixel, and fixed seeds.
- Lattice image: band-limited [001] projected potential × normalised projected
  thickness of a 12 nm sphere (weak-phase-like contrast, assumed), plus noise. The
  FFT is shown as an inset.

## Rust Monte Carlo (extra, not used for any paper number)
- Ising-like (Wajnflasz–Pick) model on a simple cubic lattice with a = 0.73 nm (≈ the
  Fe–Fe distance), cut to a sphere with a 1 nm surface layer removed.
  J = Γ_bulk/(2z) with z = 6 (mean-field correspondence), and field
  h(T) = (ΔH − TΔS)/2. Metropolis sweeps in site order use a SplitMix64 RNG. Each
  temperature step is 2 K with 100 sweeps; the loop width is the difference of the
  x = ½ crossings. MC loops depend on the sweep rate and are narrower than mean
  field, because the exact Ising T_c is below the mean-field value.
