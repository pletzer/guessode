# guessode
Guess the ODE that produces data

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Lorenz worked example: parameters, form known

```
.venv/bin/python lorenz_example.py [--noise 0.5] [--tmax 20] [--seg-len 0.5] [--sigma 10 --rho 28 --beta 2.6667]
```

1. Generates noisy x, y, z data from the Lorenz system.
2. Plots the data (`figs/step3_data.png`).
3. Infers (sigma, rho, beta) with the model form known:
   - gradient matching (the Lorenz RHS is linear in its parameters, so this is
     linear least squares on smoothed derivatives) — fast initial guess;
   - multiple shooting (`scipy.optimize.least_squares` on short segments, all
     integrated together as one stacked system, sparse Jacobian) — accurate
     estimate with Gauss–Newton uncertainties;
   - naive single shooting, for comparison — fails because the system is chaotic.
4. Plots data vs the solution with the inferred parameters (`figs/step5_fit.png`).

## SINDy: discover the form too

```
.venv/bin/python sindy_example.py [--noise 0.5] [--degree 2] [--tol 0.05]
```

Only assumes `du/dt = Xi^T Theta(u)` with `Theta` = all monomials in x, y, z up
to `--degree` (30 candidate terms for degree 2), and looks for a sparse `Xi`.

- **strong-form SINDy** (pysindy): regress smoothed finite-difference
  derivatives on the library with sequentially thresholded least squares.
- **weak-form SINDy** (own implementation): integrate against compactly
  supported test functions, so the derivative moves onto the test function
  and the noisy data is never differentiated. (pysindy 2.1's
  `WeakPDELibrary` gave a wrong `dx/dt` equation even on noise-free data,
  hence the hand-written version.)
- **threshold selection**: sweep the threshold, score every distinct model by
  its RMS misfit on the last 30% of the record (segment initial states
  fitted, coefficients frozen), keep the sparsest one within `--tol` of the
  best. Fitting is done in `u/10` so all library columns are O(1).
- **refinement**: multiple shooting (from `lorenz_example.py`) on the
  discovered structure.

Figures: `figs/sindy_path.png` (misfit vs number of terms),
`figs/sindy_coefficients.png`, `figs/sindy_fit.png`.

Observed robustness (5 seeds each, degree-2 library, t in [0, 20]):

| noise std | weak SINDy structure | strong SINDy structure |
|---|---|---|
| 0.25 | 5/5 correct | 5/5 correct |
| 0.5  | 5/5 correct | 3/5 correct |
| 1.0  | 0/5: drops the small `-y` term in dy/dt | 0/5, same |

A degree-3 library (60 candidates) needs noise ≲ 0.25: on the Lorenz
attractor its columns are nearly collinear (condition number ~1e4), so
noise in the library matrix is amplified into spurious cubic terms, and a
longer record does not cure it.

## Double pendulum: masses and lengths from angle data

```
.venv/bin/python doublependulum_example.py [--noise 0.02] [--l1 1 --l2 0.7 --m1 1 --m2 0.5]
```

Uses the solver in `~/doublependulum` (override with `DOUBLEPENDULUM_DIR`) to
generate noisy `theta1(t)`, `theta2(t)`. Angular velocities are not observed.

- **Identifiability**: the dynamics depend only on `g/l1`, `l2/l1` and
  `m2/(m1+m2)`. With `g` known you can recover `l1`, `l2` and the ratio `m2/m1`,
  but not the absolute masses. The script fixes `m1 = 1` and checks that
  scaling both masses leaves the residuals unchanged (to ~1e-13).
- **Gradient matching**: the equations of motion are linear in
  `(g/l1, l2/l1, mu*l2/l1)`, so a linear least-squares fit on Savitzky–Golay
  first and second derivatives gives an initial guess.
- **Multiple shooting**: the angles are fitted, and continuity is enforced on
  the full 4-D state. The segment velocities are extra unknowns.

Default run (noise 0.02 rad, t in [0, 20] s, chaotic regime):
`l1 = 1.0013±0.0013`, `l2 = 0.7004±0.0010`, `m2/m1 = 0.5018±0.0025`
(true 1, 0.7, 0.5). The fit also converges from a poor guess (0.5, 0.5, 2).
With noise 0.1 rad: `1.006±0.007, 0.702±0.005, 0.509±0.013`.
Figures: `figs/dp_data.png`, `figs/dp_fit.png`.

### Same problem with the adjoint method

```
.venv/bin/python doublependulum_adjoint.py [--noise 0.02] [--seg-len 0.5]
```

Minimises the multiple-shooting cost J (angle misfit + continuity penalty)
with L-BFGS-B. The gradient with respect to all 163 unknowns comes from one
forward solve (all segments stacked, dense output kept) and one backward solve
of the adjoint equations, `dλ/dt = -(∂f/∂y)ᵀλ` and `dμ/dt = -(∂f/∂θ)ᵀλ`. The
backward solve is integrated between data times, with the data residuals added
as jumps in λ. Then `dJ/dθ = μ(0)` and `dJ/du0_k = λ_k(0)` (minus the
continuity term). The vector-Jacobian products use complex-step
differentiation of the solver's `rhs`, so no Jacobian is derived by hand.

- Adjoint gradient vs central finite differences: relative difference ~1e-8.
- Result: `l1 = 1.0013, l2 = 0.7004, m2/m1 = 0.5018`, the same as the
  Gauss–Newton multiple-shooting fit, with rms misfit 0.0198 rad (noise 0.02).
- Cost: about 640 L-BFGS-B iterations, about 85 s. The parameters settle
  after about 200 iterations. Gauss–Newton is much faster here (0.7 s)
  because the problem is small and a least-squares fit; the adjoint's
  advantage, a gradient whose cost doesn't depend on the number of unknowns,
  only pays off with many parameters or a large state.

Figures: `figs/dp_adjoint_history.png`, `figs/dp_adjoint_fit.png`.
