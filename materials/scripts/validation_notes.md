## Discussion of the mismatches and near-misses

- **`nbFpThiolate` (paper 2, here 0).** The paper does not give the nephelauxetic factor
  h of its thiolate proxy (S-bound thiocyanate). We fixed h = 2.0 in advance (the value the
  paper's table gives the monoanionic Cl⁻ and CN⁻) together with Jørgensen's tabulated
  f = 0.73. With that choice thiolate never switches a resting complex. The sensitivity
  scan in `results.json` (`interferents/thiolate_h_sensitivity`) shows that thiolate
  produces exactly 2 false positives (the two 5bipy designs) only for h ≈ 2.5, and more
  for h ≥ 2.6. The headline numbers (137 silent designs, 4 false-positive designs on the
  scaffolds 1en+4bipy and 5bipy, imidazole 4, amine 2) do not depend on this choice for
  h ≤ 2.6, and they match.
- **Array pitch (`nbPitchFifty` 162 vs 165, `nbPitchTwoHundred` 674 vs 661;
  `nbPitchHundred` 331 matches).** For identical point dipoles in an infinite square
  array, the neighbour/self field ratio depends only on pitch/height, so the critical
  pitch is exactly proportional to the sensor height (here 3.306 × height, checked by
  a test). The paper's three values are not proportional (3.24, 3.31, 3.37 × height),
  so its calculation includes something that is not stated (finite array extent,
  particle size, or a different reference point). We do not guess it.
- **Displacement count (`nbDispFiftyTenMeV` 1.8e-4 vs 9.6e-5; `nbRatioIonDisp` 5,700
  vs 11,000).** Our default is the NRT count with the Lindhard–Robinson damage-energy
  partition and the atom count from the stated 1.6 g cm⁻³. Without the partition
  (damage energy = recoil energy) we get 1.57e-4 and a ratio of 6,700. Counting atoms
  from the 2.6 Fe nm⁻³ framework density instead of 1.6 g cm⁻³ would raise this
  variant by a further ×1.17, to about 1.8e-4. That suggests the paper's numbers use
  no partition and the framework density, but the text does not say so, so the
  default is left as stated. Either way, ionisation exceeds displacement by thousands
  of times, which is the qualitative conclusion.
- **Coordination polyhedron volume (`mlDvolPoly` 30 % vs 34 %).** A regular octahedron
  whose six Fe–N bonds go from 1.97 to 2.17 Å expands by (2.17/1.97)³ − 1 = 33.7 %. The
  paper's 30 % implies a different (unstated) polyhedron construction.
- **c axis, cell volumes and Fe densities (near, < 1 %).** These come from the pyrazine
  N···N span. We build it from gas-phase ring geometry (C–N 1.339 Å, C–C 1.397 Å,
  ∠CNC 115.6°), which gives 2.824 Å. The paper's c values imply ≈ 2.79 Å. The a axis,
  every in-plane quantity and all the diffraction and fringe numbers (which depend
  only on a) match exactly.
