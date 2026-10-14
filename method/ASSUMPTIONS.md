# Assumptions and choices where the manuscript is underspecified

Everything below was decided from the method description alone. None of it was tuned to move
results towards the manuscript's reported numbers.

## Chemistry / model
1. **Enumeration.** "At most three ligand types on five sites" is implemented as multisets of size 5
   drawn from the seven scaffold ligands with at most three distinct types. Each site counts as
   one ligand slot, including for chelating ligands (ox, gly, en, bipy). This gives 301 scaffolds,
   1505 complexes and 1204 pairs, the manuscript's counts. Ordering is
   `itertools.combinations_with_replacement`, and the Rust component uses the same order.
2. **Ligand-field averages** are over all six sites: five scaffold sites plus the sixth.
3. **Calibration compound** ([Fe(phen)2(NCS)2] proxy) is 4 bipy + 2 NCS sites. It is outside the
   enumerated space.
4. **Kanamori spin-flip sign.** The paper's eq. (kanamori) is implemented as printed, with +J_H on
   the spin-flip term. The full-space check is also run with the standard rotationally invariant
   sign (-J_H). Neither sign affects the seniority-zero block or the M_S = 2 determinant. Both give
   the same lowest-singlet energy as the block (deviation about 1e-11 cm^-1). With the printed sign
   the M_S = 0 member of the quintet does not lie at the quintet energy. This does not affect anything used.
5. **Lowest quintet** is the lowest eigenvalue of the S_z = 2 sector. **Lowest singlet** is the
   lowest S_z = 0 eigenvector with <S^2> = 0.
6. **T1/2 for dH <= 0** is the negative value dH/dS. It is not clipped, and it counts as below
   T_b - m.
7. Unit conversion: 1 cm^-1 = h c N_A = 11.96266 J/mol.

## Ansatz and compilation
8. Ry(phi) = [[cos phi/2, -sin phi/2],[sin phi/2, cos phi/2]]. With this convention the compiled
   G_ia gives cos θ|10> + sin θ|01>, while the paper writes -sin θ. The sign of θ is absorbed by the optimiser.
9. Angles come from BFGS over the six angles: three starts, the zero vector plus two seeded random
   starts, with the seed set to the complex index.
10. The routed SWAP path on chain 0-3-1-4-2 moves the occupied qubit one position per SWAP until it
    is adjacent to the virtual qubit, then reverses the SWAPs. That is 2 SWAPs in and 2 out for
    G_23 and for G_04, giving 48 CNOTs in total.
11. Folding: a fixed seeded permutation of the CNOT list after SWAP expansion. Scale s folds the
    first round((s-1)/2 · N) CNOTs, replacing each with CNOT·CNOT·CNOT and a depolarising channel
    after each copy. One permutation is shared by all complexes, arms and platforms.

## Noise and estimation
12. Depolarising channels follow every CNOT (λ = 4r2q/3, two-qubit channel) and every Ry
    (λ = 2r1q). **State preparation (X gates) and the measurement-basis changes (H, S†) are
    noiseless.** Readout error is a symmetric, independent bit flip with the published rate on each qubit.
13. **Drift** acts on r2q only, with lognormal σ = 0.5. One independent draw is made per
    (platform, complex, epoch). r1q and readout do not drift.
14. **Raw and mitigated histograms are drawn independently** from the same noisy density matrices.
    The raw arm draws 3 histograms at scale 1. The mitigated arm draws 9. The twin circuit gets the
    same treatment.
15. **Readout correction** inverts the tensored 2x2 confusion matrix, assuming the true published
    flip rate. Corrected quasi-probabilities may be negative and are used as they are.
16. **ZNE**: c0 = 0. The exponential fit of ln|E(s)| is linear least squares over s = 1, 1.5, 2.
    "Monotone" means strictly increasing or strictly decreasing. Otherwise the fit is linear least
    squares in s, evaluated at s = 0.
17. The **reference-scaled** and **paired-mitigated** arms follow sec. 6.1. They use the twin's
    raw (uncorrected) scale-1 energy and its mitigated energy respectively.

## Loop
18. The calibration set is 150 complexes drawn uniformly at random with a fixed seed.
    **Calibration complexes are never dispatched.** Their final decisions use the depth-2 surrogate.
19. The deployed fit uses drift epoch 0 of the router-chosen arm. The SE band is 3 × the mean,
    over all 1505 complexes, of the standard error of the fitted mean,
    s·sqrt(1/n + (x - x̄)²/Sxx), converted to K.
20. Router predictions come from the calibration compound over 16 epochs (twice 8). They use the
    RMS of the temperature error, bias included. Only the raw and mitigated arms are routable.
    Willow is unpriced and not routable.
21. Router choice: the cheapest USD arm with predicted error ≤ τ (ties broken by accuracy). Failing
    that, the cheapest HQC arm. Failing that, the arm with the smallest predicted error among the
    priced arms.
22. Policy decisions on the dispatched pairs take the resting (aqua) complex's T from the depth-2
    surrogate. Classical rows use the same estimator for both complexes. "Hard" pairs are those whose
    exact analyte-complex T lies within 50 K of 290 K or 330 K.

## Prices (appendix C)
23. One histogram is one task. The Braket per-task fee ($0.30) applies to both Rigetti and IonQ.
    IBM charges $96/min × 250 µs per shot. The HQC formula counts N1q = 12 (the ansatz Ry gates;
    basis changes and state preparation are not counted), N2q = the CNOT count including SWAPs and
    folding, and N_m = 5.

## Extensions (reduced versions)
24. **Budget law**: for each platform, arm and shot level, subsets of n ∈ {5, 10, 20, 40, 75, 150}
    calibration points, 30 subsets each. Each subset uses one random drift epoch out of 0-5.
    σ²(n) = b² + s²/n is fitted by non-negative least squares to the mean held-out MSE.
25. **Roughness ablation**: the hidden term is F·w. F counts unordered ligand-type pairs over the
    6 sites (66 features) and w ~ N(0,1) is seeded. The term is projected orthogonal to the span
    of {1, Δ, B, E2, Δ², B², E2², ΔB, ΔE2, BE2} over the whole space, then scaled to the target
    RMS in K. Twelve levels run from 0 to 560 K, with 3 hidden-term seeds and 24 replicates.
    Device errors for calibration points are resampled per complex from that complex's own epochs
    0-5, which keeps its complex-specific bias. In the decision-level comparison, complexes with
    no device evaluation take errors from the pooled distribution.
26. **eps sweep**: the idealised all-to-all platform is simulated with drift. It uses the mitigated
    arm only, over the dispatched complexes and 6 epochs.

## Run defaults
27. The defaults are the manuscript's settings: 8 epochs, 16 for the calibration compound, shots
    1k/4k/16k/64k, n_cal = 150, 6 epochs for the reference-state tables, 24 roughness replicates
    and 3 seeds. Nothing is subsampled because the full run takes under a minute.
    `python3 -m scodesign run --quick` gives a reduced smoke-test configuration.

## Implemented stack (sec. 6.3-6.7)
28. **Recurrent-depth network** (`rdm.py`, NumPy only):
    - Structure: tanh encoder, hidden width 16, a shared damped recurrent block
      h_t = ½h_{t-1} + ½tanh(h_{t-1}W_h + eW_x + b_h) with the encoded input injected at every
      step, and a linear readout after every iteration.
    - Training: the loss is the mean squared error averaged over iterations 1..10, so every
      iterate is a valid estimate. Optimiser is Adam (lr 3e-3), with hand-written
      backpropagation through time (gradient-checked in the tests).
    - Halting: iterate until the normalised output changes by < 1e-3 between iterations, with a
      cap of T_max = 64. The reported depth is the number of iterations run.
29. **Decoder**:
    - Inputs: E(1.5)/E(1) and E(2)/E(1) for the circuit and for the reference twin (readout-corrected
      energies at the folding scales), the reference twin's attenuation E_ref,dev(1)/E_ref, CNOT/48,
      and log10(shots) - 3.
    - Output: the attenuation E(1)/E_exact of the circuit. The energy is the scale-1 corrected
      energy divided by the predicted attenuation.
    - Training uses twins only, from every evaluated complex (calibration + dispatched), epochs 0-5,
      all platforms and shot levels. Each twin is paired with 8 random other twins of the same
      platform, epoch and shot level, giving 194,880 examples. 3000 steps, batch 1024.
    - Test: each dispatched ansatz paired with its own twin, epochs 0-5. It consumes the same 18
      histograms as the paired-mitigated arm.
30. **Recurrent-depth surrogate**:
    - Inputs: the 11 ligand fractions over the six sites, Δ, B and E2. Target: E_corr.
    - Ensemble of five members, trained on all 150 calibration points (full batch, L2 1e-4,
      3000 steps).
    - Early stopping on a random 20 % validation split of the calibration points: this is our
      standard choice and the paper does not specify one.
    - Labels come from the deployed calibration arm at an **independent drift epoch (epoch 1)**:
      device-extrapolated, decoder-produced, paired-mitigated (line only) and exact.
    - Held-out RMS is computed over the non-calibration complexes. The depth correlation is with
      |E_corr| over all 1505 complexes.
31. **Generative proposer** (cyanide design):
    - Prior: five sites drawn independently from the ligand frequencies of the successful
      three-type scaffolds. Draws with fewer than four ligand types are rejected.
    - "Drawn to exhaustion" is taken to mean 20,000 accepted draws. Hit rate is reported over
      all draws and over the distinct proposals.
    - Scorer: L2-regularised logistic regression (λ = 1e-2, gradient descent) on the seven
      ligand fractions, fitted to all 301 enumerated scaffolds (success / failure for cyanide).
      It ranks the 161 extended scaffolds.
    - The 161 candidates are generated and exactly scored by the Rust binary
      (`scospace --extended`, std threads). Python checks the labels against its own solver.
32. **Q-learned escalation**:
    - Data: mitigated arm on the dispatched complexes; Helios and Heron, the two priced platforms
      the router uses. Train on epochs 0-3, test on epochs 4-7.
    - State: (shot level, distance of the current estimate to the nearer threshold divided by
      that level's RMS error on the training epochs), binned at 0.5/1/2/3.
    - Actions: stop (decide the three-way state against 290/330 K) or escalate to the next level.
      Every level visited is paid; the 1k level is always paid. Costs are normalised by the
      full-escalation cost.
    - Penalty λ per wrong state, six values 0.1, 0.3, 1, 3, 10, 30. These were chosen to span the
      cost frontier from the 1k policy to the 64k policy; the paper does not give its values.
    - Q-learning: α = 0.1, γ = 1, ε decaying from 1 to 0.05, 40,000 episodes. A fixed level-L
      policy pays for every level up to L.
    - "Top of frontier" means the learned policy at the largest λ compared with the fixed 64k policy.
33. **QAOA dispatch test**:
    - Candidates: the twelve dispatched complexes nearest a threshold.
    - Objective f(x) = Σ v_i x_i - P(Σ c_i x_i - B)² with v_i = 1 - d_i/band, c_i = the cost of
      the routed arm for that candidate (normalised to mean 1), B = 4 (a third of the
      candidates) and P = 1. The paper does not give these.
    - Ideal 12-qubit statevector simulation. Parameters: Nelder-Mead from 8 random starts plus,
      for p > 1, a warm start from the previous depth. The evaluation count includes all starts.
    - Approximation ratio = <f>/f_max. The min-max normalised ratio is also stored.
    - P(optimal) is the probability mass on all bitstrings tied at f_max.
    - CNOTs: each of the 66 ZZ couplings uses 2 CNOTs per layer.
    - Circuits for p = 1, 2, 3 are exported to `qasm/qaoa/` and verified by re-simulation.

## Language roles
34. Python: orchestration, training loops, analysis, figures. C++/OpenMP (`cpp/`): density-matrix
    kernel, QAOA statevector (`qaoa_expect`, `qaoa_batch`) and recurrent-depth forward pass
    (`rdm_forward`). CUDA (`cuda/`): untested GPU counterparts with CuPy wrappers that fall back to
    C++/NumPy. Rust: whole-space sweep and generative candidate generation and scoring. NASM: parity
    expectations, popcount enumeration, and the QAOA bitstring cost table (`weighted_bitsum`).
    OpenQASM 3: ansatz, twin and QAOA circuits.
