#!/usr/bin/env python3
"""
Worked example: recover the parameters of the Lorenz system from noisy data.

    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z

Steps
  1. Generate synthetic x, y, z data for chosen (sigma, rho, beta), add noise.
  3. Plot the data.
  4. Infer (sigma, rho, beta) assuming the model form is known:
       a. gradient matching (smooth data, differentiate, linear least squares)
          -> cheap initial guess, no ODE integration needed;
       b. multiple shooting (nonlinear least squares on short segments)
          -> accurate estimate + uncertainties.
     For comparison, naive single shooting over the full window is also tried.
  5. Plot the data against the solution obtained with the inferred parameters.

Usage
  python lorenz_example.py [--noise 0.5] [--tmax 20] [--seed 1] [--outdir figs]
"""
import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.sparse import lil_matrix

PARAM_NAMES = ("sigma", "rho", "beta")


def lorenz_rhs(t, u, sigma, rho, beta):
    """Lorenz right-hand side. u has shape (3,) or (3*K,) for K stacked states."""
    x, y, z = u.reshape(3, -1)
    return np.concatenate([sigma * (y - x),
                           x * (rho - z) - y,
                           x * y - beta * z])


def integrate(theta, u0, t_eval, rtol=1e-9, atol=1e-9):
    """Integrate from u0 (shape (3,) or (3, K)) over t_eval. Returns (3, K, nt)."""
    u0 = np.asarray(u0, dtype=float).reshape(3, -1)
    sol = solve_ivp(lorenz_rhs, (t_eval[0], t_eval[-1]), u0.ravel(),
                    t_eval=t_eval, args=tuple(theta), method="DOP853",
                    rtol=rtol, atol=atol)
    if not sol.success or sol.y.shape[1] != len(t_eval):
        # blow-up for silly parameters: return something large but finite
        return np.full((3, u0.shape[1], len(t_eval)), 1e6)
    return sol.y.reshape(3, u0.shape[1], len(t_eval))


# ---------------------------------------------------------------------------
# Step 1: data
# ---------------------------------------------------------------------------
def generate_data(theta_true, u0, tmax, dt, noise, rng):
    t = np.arange(0.0, tmax + 0.5 * dt, dt)
    u_clean = integrate(theta_true, u0, t)[:, 0, :]          # (3, nt)
    u_noisy = u_clean + noise * rng.standard_normal(u_clean.shape)
    return t, u_clean, u_noisy


# ---------------------------------------------------------------------------
# Step 4a: gradient matching
# ---------------------------------------------------------------------------
def gradient_matching(t, u, window=21, order=3):
    """
    The Lorenz system is *linear in its parameters*:
        dx/dt           = sigma * (y - x)
        dy/dt + y + x z = rho * x
        dz/dt - x y     = -beta * z
    so after smoothing and differentiating the data, each parameter follows
    from a 1-D linear least-squares fit.
    """
    dt = t[1] - t[0]
    us = savgol_filter(u, window, order, axis=1)
    du = savgol_filter(u, window, order, deriv=1, delta=dt, axis=1)
    x, y, z = us
    dx, dy, dz = du
    # trim the edges where the filter is least accurate
    s = slice(window, -window)
    lsq = lambda a, b: float(np.dot(a[s], b[s]) / np.dot(a[s], a[s]))
    sigma = lsq(y - x, dx)
    rho = lsq(x, dy + y + x * z)
    beta = lsq(-z, dz - x * y)
    return np.array([sigma, rho, beta])


# ---------------------------------------------------------------------------
# Step 4b: multiple shooting
# ---------------------------------------------------------------------------
class MultipleShooting:
    """
    Split the record into K segments of n points each. The unknowns are the
    parameters theta plus an initial state for every segment. Each segment is
    integrated only over a short time (shorter than the ~1 time unit Lyapunov
    time), so the misfit stays smooth in theta even though the system is
    chaotic. A continuity penalty ties the end of segment k to the start of
    segment k+1.

    All K segments are integrated *simultaneously* as one 3K-dim system on
    the common local time grid, so each residual evaluation is a single
    solve_ivp call.
    """

    def __init__(self, t, u, seg_len, continuity_weight=1.0):
        dt = t[1] - t[0]
        self.n = int(round(seg_len / dt)) + 1              # points per segment
        self.K = (len(t) - 1) // (self.n - 1)              # number of segments
        self.tloc = t[: self.n] - t[0]
        idx = np.arange(self.K)[:, None] * (self.n - 1) + np.arange(self.n)
        self.idx = idx                                      # (K, n)
        self.data = u[:, idx]                               # (3, K, n)
        self.t0 = t[idx[:, 0]]
        self.w = continuity_weight

    def unpack(self, p):
        return p[:3], p[3:].reshape(3, self.K)

    def residuals(self, p):
        theta, u0 = self.unpack(p)
        sim = integrate(theta, u0, self.tloc, rtol=1e-8, atol=1e-8)
        r_data = (sim - self.data).ravel()
        # end of segment k should equal start of segment k+1
        r_cont = self.w * (sim[:, :-1, -1] - u0[:, 1:]).ravel()
        return np.concatenate([r_data, r_cont])

    def jac_sparsity(self):
        """Segment k's residuals depend only on theta and on u0 of segment k (and k+1)."""
        K, n = self.K, self.n
        n_data, n_cont = 3 * K * n, 3 * (K - 1)
        S = lil_matrix((n_data + n_cont, 3 + 3 * K), dtype=int)
        S[:, :3] = 1
        for c in range(3):                  # residual component
            for k in range(K):
                rows = c * K * n + k * n + np.arange(n)
                for j in range(3):          # u0 component
                    S[rows, 3 + j * K + k] = 1
            for k in range(K - 1):
                row = n_data + c * (K - 1) + k
                for j in range(3):
                    S[row, 3 + j * K + k] = 1
                S[row, 3 + c * K + k + 1] = 1
        return S

    def fit(self, theta0, verbose=0):
        # initialise segment states from the (noisy) data themselves
        p0 = np.concatenate([theta0, self.data[:, :, 0].ravel()])
        return least_squares(self.residuals, p0, jac_sparsity=self.jac_sparsity(),
                             method="trf", x_scale="jac", verbose=verbose)


def covariance(res):
    """Gauss-Newton covariance: s^2 (J^T J)^{-1}."""
    J = res.jac.toarray() if hasattr(res.jac, "toarray") else res.jac
    dof = max(len(res.fun) - len(res.x), 1)
    s2 = 2 * res.cost / dof
    return s2 * np.linalg.pinv(J.T @ J)


# ---------------------------------------------------------------------------
# For comparison: single shooting over the whole record
# ---------------------------------------------------------------------------
def single_shooting(t, u, theta0):
    def resid(p):
        return (integrate(p[:3], p[3:], t, rtol=1e-8, atol=1e-8)[:, 0, :] - u).ravel()
    p0 = np.concatenate([theta0, u[:, 0]])
    return least_squares(resid, p0, method="trf", x_scale="jac", max_nfev=200)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_data(t, u_noisy, u_clean, fname):
    fig = plt.figure(figsize=(13, 6))
    labels = ("x", "y", "z")
    for i in range(3):
        ax = fig.add_subplot(3, 2, 2 * i + 1)
        ax.plot(t, u_noisy[i], ".", ms=1.5, color="C0", label="noisy data")
        ax.plot(t, u_clean[i], "-", lw=0.6, color="k", alpha=0.6, label="noise-free")
        ax.set_ylabel(labels[i])
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
        if i == 2:
            ax.set_xlabel("t")
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.plot(*u_noisy, ".", ms=1.0, color="C0")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("data in phase space")
    fig.suptitle("Step 3: Lorenz data")
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)


def plot_fit(t, u_noisy, u_clean, ms, res, theta_true, fname):
    labels = ("x", "y", "z")
    theta_fit, u0_seg = ms.unpack(res.x)
    # (a) free-running forecast from the fitted initial state of segment 0
    u_free = integrate(theta_fit, u0_seg[:, 0], t)[:, 0, :]
    # (b) piecewise solution from the multiple-shooting fit
    seg = integrate(theta_fit, u0_seg, ms.tloc)            # (3, K, n)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(6, 2)
    for i in range(3):
        ax = fig.add_subplot(gs[2 * i:2 * i + 2, 0])
        ax.plot(t, u_noisy[i], ".", ms=1.5, color="C0", label="data")
        for k in range(ms.K):
            ax.plot(ms.t0[k] + ms.tloc, seg[i, k], "-", lw=1.0, color="C3",
                    label="multiple-shooting segments" if k == 0 else None)
        ax.plot(t, u_free[i], "--", lw=0.9, color="k",
                label="free run, inferred params" if i == 0 else None)
        ax.set_ylabel(labels[i])
        if i == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=3)
        if i == 2:
            ax.set_xlabel("t")

    ax = fig.add_subplot(gs[0:3, 1], projection="3d")
    ax.plot(*u_noisy, ".", ms=1.0, color="C0", alpha=0.5, label="data")
    # a long run with the inferred parameters: should land on the same attractor
    t_long = np.arange(0, 5 * t[-1], t[1] - t[0])
    u_long = integrate(theta_fit, u0_seg[:, 0], t_long)[:, 0, :]
    ax.plot(*u_long, "-", lw=0.3, color="C3", label="inferred model")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend(fontsize=8)
    ax.set_title("attractor: data vs inferred model")

    ax = fig.add_subplot(gs[3:6, 1])
    # compare against the noise-free truth (known here only because data are synthetic)
    err = np.linalg.norm(u_free - u_clean, axis=0)
    ax.semilogy(t, err, lw=0.8, color="k", label="|free run - noise-free truth|")
    lam = 0.906                                              # max Lyapunov exp.
    ax.semilogy(t, err[:50].mean() * np.exp(lam * t), ":", color="C1",
                label=r"$\propto e^{\lambda_1 t}$, $\lambda_1\approx0.91$")
    ax.set_ylim(1e-3, 1e2)
    ax.set_xlabel("t"); ax.set_ylabel("error")
    ax.set_title("free run diverges: chaos, not a bad fit")
    ax.legend(fontsize=8)

    txt = "   ".join(f"{n}: true {tv:.4f}, fit {fv:.4f}"
                     for n, tv, fv in zip(PARAM_NAMES, theta_true, theta_fit))
    fig.suptitle("Step 5: data vs solution with inferred parameters\n" + txt, fontsize=10)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sigma", type=float, default=10.0)
    ap.add_argument("--rho", type=float, default=28.0)
    ap.add_argument("--beta", type=float, default=8.0 / 3.0)
    ap.add_argument("--tmax", type=float, default=20.0)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--noise", type=float, default=0.5, help="std of additive Gaussian noise")
    ap.add_argument("--seg-len", type=float, default=0.5, help="multiple-shooting segment length")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--outdir", default="figs")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    theta_true = np.array([args.sigma, args.rho, args.beta])
    u0_true = np.array([-8.0, 7.0, 27.0])

    # Step 1
    t, u_clean, u = generate_data(theta_true, u0_true, args.tmax, args.dt, args.noise, rng)
    print(f"Step 1: {len(t)} samples on t in [0, {args.tmax}], noise std = {args.noise}")
    print("        true params  " + fmt(theta_true))

    # Step 3
    plot_data(t, u, u_clean, os.path.join(args.outdir, "step3_data.png"))

    # Step 4a
    theta_gm = gradient_matching(t, u)
    print("Step 4a: gradient matching   " + fmt(theta_gm))

    # For comparison: single shooting from a poor guess
    theta_bad = np.array([5.0, 15.0, 1.0])
    tic = time.time()
    res_ss = single_shooting(t, u, theta_bad)
    print(f"        single shooting from {fmt(theta_bad)} -> {fmt(res_ss.x[:3])}"
          f"  (rms misfit {np.sqrt(2 * res_ss.cost / res_ss.fun.size):.2f}, {time.time() - tic:.1f}s)")

    # Step 4b
    ms = MultipleShooting(t, u, args.seg_len)
    for label, th0 in (("from poor guess", theta_bad), ("from grad-matching", theta_gm)):
        tic = time.time()
        res = ms.fit(th0)
        cov = covariance(res)
        std = np.sqrt(np.diag(cov)[:3])
        print(f"Step 4b: multiple shooting {label:>18}: " + fmt(res.x[:3], std)
              + f"  ({ms.K} segments, {res.nfev} fevals, {time.time() - tic:.1f}s)")
    rms = np.sqrt(np.mean((integrate(res.x[:3], res.x[3:].reshape(3, -1), ms.tloc)
                           - ms.data) ** 2))
    print(f"        rms data misfit {rms:.3f} (noise std {args.noise})")

    # Step 5
    plot_fit(t, u, u_clean, ms, res, theta_true, os.path.join(args.outdir, "step5_fit.png"))
    print(f"Figures written to {args.outdir}/")


def fmt(theta, std=None):
    if std is None:
        return "(" + ", ".join(f"{n}={v:.4f}" for n, v in zip(PARAM_NAMES, theta)) + ")"
    return "(" + ", ".join(f"{n}={v:.4f}±{s:.4f}" for n, v, s in zip(PARAM_NAMES, theta, std)) + ")"


if __name__ == "__main__":
    main()
