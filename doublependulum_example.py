#!/usr/bin/env python3
"""
Worked example: recover the masses and arm lengths of a double pendulum from
noisy measurements of its two angles.

The model is ~/doublependulum/doublependulum.py (point masses m1 at the joint
and m2 at the tip, massless arms l1, l2, angles from the downward vertical).
Only theta1(t), theta2(t) are observed; the angular velocities are not.

Identifiability
  Dividing the Lagrangian by (m1 + m2) l1^2 leaves only three combinations,
      g/l1,   l2/l1,   mu = m2 / (m1 + m2),
  so with g known the data determine l1, l2 and the mass *ratio* m2/m1, but
  not the overall mass scale: (m1, m2) and (c m1, c m2) give identical
  trajectories. We therefore fix m1 = 1 and infer (l1, l2, m2), i.e. m2/m1.

Steps
  1. Generate theta1, theta2 with the doublependulum solver, add noise.
  2. Plot the data.
  3. Gradient matching: the equations of motion,
         a1 + s (a2 cos d + w2^2 sin d) + p sin th1          = 0
         q a2 + a1 cos d - w1^2 sin d  + p sin th2           = 0
     (d = th1 - th2, p = g/l1, q = l2/l1, s = mu q) are linear in (p, q, s),
     so smoothed first and second derivatives give a linear least-squares
     problem -> cheap initial guess.
  4. Multiple shooting on (l1, l2, m2) with m1 = 1 (segment initial states,
     including the unobserved angular velocities, are extra unknowns).
  5. Plot data vs the solution with the inferred parameters.

Usage
  .venv/bin/python doublependulum_example.py [--noise 0.02] [--l1 1 --l2 0.7 --m1 1 --m2 0.5]
"""
import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.sparse import lil_matrix

from lorenz_example import covariance

DP_DIR = os.environ.get("DOUBLEPENDULUM_DIR", os.path.expanduser("~/doublependulum"))
sys.path.insert(0, DP_DIR)
import doublependulum as dp  # noqa: E402

PARAM_NAMES = ("l1", "l2", "m2/m1")
NDIM, NOBS = 4, 2          # state (th1, th2, w1, w2); observed (th1, th2)


def stacked_rhs(t, u, l1, l2, m1, m2, g):
    """dp.rhs for K states stacked as u of shape (4*K,)."""
    return np.concatenate(dp.rhs(t, u.reshape(NDIM, -1), l1, l2, m1, m2, g))


def integrate(phys, u0, t_eval, rtol=1e-9, atol=1e-9):
    """phys = (l1, l2, m1, m2, g); u0 shape (4,) or (4, K). Returns (4, K, nt)."""
    u0 = np.asarray(u0, dtype=float).reshape(NDIM, -1)
    l1, l2, m1, m2, _ = phys
    if min(l1, l2, m1) <= 0 or m2 < 0:
        return np.full((NDIM, u0.shape[1], len(t_eval)), 1e3)
    sol = solve_ivp(stacked_rhs, (t_eval[0], t_eval[-1]), u0.ravel(), t_eval=t_eval,
                    args=tuple(phys), method="DOP853", rtol=rtol, atol=atol)
    if not sol.success or sol.y.shape[1] != len(t_eval):
        return np.full((NDIM, u0.shape[1], len(t_eval)), 1e3)
    return sol.y.reshape(NDIM, u0.shape[1], len(t_eval))


# ---------------------------------------------------------------------------
# Step 1: data
# ---------------------------------------------------------------------------
def generate_data(phys, y0, tmax, dt, noise, rng):
    l1, l2, m1, m2, g = phys
    nt = int(round(tmax / dt)) + 1
    sol = dp.solve(y0, tmax, l1=l1, l2=l2, m1=m1, m2=m2, g=g, nt=nt)
    th_clean = sol.y[:NOBS]
    # a sensor reports angles modulo 2 pi; unwrap them as one would with real data
    th_meas = np.angle(np.exp(1j * (th_clean + noise * rng.standard_normal(th_clean.shape))))
    return sol.t, sol.y, np.unwrap(th_meas, axis=1)


# ---------------------------------------------------------------------------
# Step 3: gradient matching
# ---------------------------------------------------------------------------
def smooth_derivs(t, th, window, order=4):
    dt = t[1] - t[0]
    ths = savgol_filter(th, window, order, axis=1)
    w = savgol_filter(th, window, order, deriv=1, delta=dt, axis=1)
    a = savgol_filter(th, window, order, deriv=2, delta=dt, axis=1)
    return ths, w, a


def gradient_matching(t, th, g, window=41):
    (th1, th2), (w1, w2), (a1, a2) = smooth_derivs(t, th, window)
    s_ = slice(window, -window)
    d = th1 - th2
    sd, cd = np.sin(d), np.cos(d)
    z = np.zeros_like(th1)
    # unknowns (p, q, s) = (g/l1, l2/l1, mu l2/l1)
    A = np.vstack([np.column_stack([np.sin(th1), z, a2 * cd + w2**2 * sd])[s_],
                   np.column_stack([np.sin(th2), a2, z])[s_]])
    b = np.concatenate([-a1[s_], (-a1 * cd + w1**2 * sd)[s_]])
    p, q, s = np.linalg.lstsq(A, b, rcond=None)[0]
    l1 = g / p
    mu = s / q
    return np.array([l1, q * l1, mu / (1.0 - mu)])


# ---------------------------------------------------------------------------
# Step 4: multiple shooting
# ---------------------------------------------------------------------------
class MultipleShooting:
    """
    As in lorenz_example.py, but the state (th1, th2, w1, w2) is only partially
    observed: residuals compare the angles with the data, while continuity is
    enforced on the full state. Unknowns: theta = (l1, l2, m2) with m1 = 1, plus
    a 4-component initial state per segment.
    """

    def __init__(self, t, th, seg_len, g, continuity_weight=1.0, window=41):
        self.g = g
        dt = t[1] - t[0]
        self.n = int(round(seg_len / dt)) + 1
        self.K = (len(t) - 1) // (self.n - 1)
        self.tloc = t[: self.n] - t[0]
        self.idx = np.arange(self.K)[:, None] * (self.n - 1) + np.arange(self.n)
        self.data = th[:, self.idx]                          # (2, K, n)
        self.t0 = t[self.idx[:, 0]]
        self.w = continuity_weight
        # initial guess for the segment states: smoothed angles and their rates
        ths, w, _ = smooth_derivs(t, th, window)
        self.u0_guess = np.vstack([ths, w])[:, self.idx[:, 0]]   # (4, K)

    def phys(self, theta, m1=1.0):
        l1, l2, m2 = theta
        return (l1, l2, m1, m2, self.g)

    def unpack(self, p):
        return p[:3], p[3:].reshape(NDIM, self.K)

    def residuals(self, p, m1=1.0):
        theta, u0 = self.unpack(p)
        sim = integrate(self.phys(theta, m1), u0, self.tloc, rtol=1e-8, atol=1e-8)
        r_data = (sim[:NOBS] - self.data).ravel()
        r_cont = self.w * (sim[:, :-1, -1] - u0[:, 1:]).ravel()
        return np.concatenate([r_data, r_cont])

    def jac_sparsity(self):
        K, n, m = self.K, self.n, 3
        n_data, n_cont = NOBS * K * n, NDIM * (K - 1)
        S = lil_matrix((n_data + n_cont, m + NDIM * K), dtype=int)
        S[:, :m] = 1
        for c in range(NOBS):
            for k in range(K):
                rows = c * K * n + k * n + np.arange(n)
                for j in range(NDIM):
                    S[rows, m + j * K + k] = 1
        for c in range(NDIM):
            for k in range(K - 1):
                row = n_data + c * (K - 1) + k
                for j in range(NDIM):
                    S[row, m + j * K + k] = 1
                S[row, m + c * K + k + 1] = 1
        return S

    def fit(self, theta0, verbose=0):
        p0 = np.concatenate([theta0, self.u0_guess.ravel()])
        lb = np.full(p0.size, -np.inf)
        lb[:3] = 1e-3
        return least_squares(self.residuals, p0, jac_sparsity=self.jac_sparsity(),
                             bounds=(lb, np.inf), method="trf", x_scale="jac",
                             verbose=verbose)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_data(t, th, y_clean, fname):
    fig, axs = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    for i, ax in enumerate(axs):
        ax.plot(t, np.rad2deg(th[i]), ".", ms=1.5, color="C0", label="noisy data")
        ax.plot(t, np.rad2deg(y_clean[i]), "-", lw=0.6, color="k", alpha=0.6,
                label="noise-free")
        ax.set_ylabel(rf"$\theta_{i + 1}$ [deg]")
    axs[0].legend(loc="upper right", fontsize=8)
    axs[1].set_xlabel("t [s]")
    fig.suptitle("Step 2: double pendulum data (angles only)")
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)


def plot_fit(t, th, y_clean, ms, res, theta_true, fname):
    theta_fit, u0_seg = ms.unpack(res.x)
    phys = ms.phys(theta_fit)
    seg = integrate(phys, u0_seg, ms.tloc)
    u_free = integrate(phys, u0_seg[:, 0], t)[:, 0, :]

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i in range(2):
        ax = axs[i]
        ax.plot(t, np.rad2deg(th[i]), ".", ms=1.5, color="C0", label="data")
        for k in range(ms.K):
            ax.plot(ms.t0[k] + ms.tloc, np.rad2deg(seg[i, k]), "-", lw=1.0, color="C3",
                    label="multiple-shooting segments" if k == 0 else None)
        ax.plot(t, np.rad2deg(u_free[i]), "--", lw=0.9, color="k",
                label="free run, inferred params")
        ax.set_ylabel(rf"$\theta_{i + 1}$ [deg]")
    axs[0].legend(loc="upper right", fontsize=8, ncol=3)
    err = np.linalg.norm(u_free[:2] - y_clean[:2], axis=0)
    axs[2].semilogy(t, np.maximum(err, 1e-8), lw=0.8, color="k")
    axs[2].set_ylabel("|free run - truth| [rad]")
    axs[2].set_xlabel("t [s]")
    axs[2].set_title("free run eventually diverges: chaos, not a bad fit", fontsize=9)
    txt = "   ".join(f"{n}: true {tv:.4f}, fit {fv:.4f}"
                     for n, tv, fv in zip(PARAM_NAMES, theta_true, theta_fit))
    fig.suptitle("Step 5: data vs solution with inferred parameters\n" + txt, fontsize=10)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
def fmt(theta, std=None):
    if std is None:
        return "(" + ", ".join(f"{n}={v:.4f}" for n, v in zip(PARAM_NAMES, theta)) + ")"
    return "(" + ", ".join(f"{n}={v:.4f}±{s:.4f}"
                           for n, v, s in zip(PARAM_NAMES, theta, std)) + ")"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--l1", type=float, default=1.0)
    ap.add_argument("--l2", type=float, default=0.7)
    ap.add_argument("--m1", type=float, default=1.0)
    ap.add_argument("--m2", type=float, default=0.5)
    ap.add_argument("--g", type=float, default=9.81, help="gravity, assumed known")
    ap.add_argument("--theta1", type=float, default=120.0, help="initial theta1 [deg]")
    ap.add_argument("--theta2", type=float, default=-10.0, help="initial theta2 [deg]")
    ap.add_argument("--tmax", type=float, default=20.0)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--noise", type=float, default=0.02, help="angle noise std [rad]")
    ap.add_argument("--seg-len", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--outdir", default="figs")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    phys_true = (args.l1, args.l2, args.m1, args.m2, args.g)
    theta_true = np.array([args.l1, args.l2, args.m2 / args.m1])
    y0 = np.deg2rad([args.theta1, args.theta2, 0.0, 0.0])

    # Step 1
    t, y_clean, th = generate_data(phys_true, y0, args.tmax, args.dt, args.noise, rng)
    print(f"Step 1: {len(t)} angle samples on t in [0, {args.tmax}] s, "
          f"noise std = {args.noise} rad ({np.rad2deg(args.noise):.2f} deg)")
    print(f"        true: l1={args.l1}, l2={args.l2}, m1={args.m1}, m2={args.m2}  "
          f"-> {fmt(theta_true)}")

    # Step 2
    plot_data(t, th, y_clean, os.path.join(args.outdir, "dp_data.png"))

    # Step 3
    theta_gm = gradient_matching(t, th, args.g)
    print("Step 3: gradient matching   " + fmt(theta_gm))

    # Step 4
    ms = MultipleShooting(t, th, args.seg_len, args.g)
    theta_bad = np.array([0.5, 0.5, 2.0])
    for label, th0 in (("from poor guess", theta_bad), ("from grad-matching", theta_gm)):
        tic = time.time()
        res = ms.fit(th0)
        std = np.sqrt(np.diag(covariance(res))[:3])
        print(f"Step 4: multiple shooting {label:>18}: " + fmt(res.x[:3], std)
              + f"  ({ms.K} segments, {res.nfev} fevals, {time.time() - tic:.1f}s)")
    rms = np.sqrt(np.mean(res.fun[:ms.data.size] ** 2))
    print(f"        rms angle misfit {rms:.4f} rad (noise std {args.noise})")

    # only the ratio m2/m1 is identifiable: scaling both masses leaves the residuals unchanged
    l1f, l2f, m2f = res.x[:3]
    for c in (0.1, 10.0):
        p = res.x.copy()
        p[2] = c * m2f
        dr = np.max(np.abs(ms.residuals(p, m1=c) - res.fun))
        print(f"        (m1, m2) -> {c:g} x (m1, m2): max |residual change| = {dr:.1e}")
    print(f"        => masses known only up to a common factor: m1 = M, m2 = {m2f:.4f} M")

    # Step 5
    plot_fit(t, th, y_clean, ms, res, theta_true, os.path.join(args.outdir, "dp_fit.png"))
    print(f"Figures written to {args.outdir}/")


if __name__ == "__main__":
    main()
