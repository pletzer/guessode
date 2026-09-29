#!/usr/bin/env python3
"""
Worked example: discover the *form* of the Lorenz equations from noisy data
with SINDy (Sparse Identification of Nonlinear Dynamics), then refine the
coefficients of the discovered model by multiple shooting.

Nothing about the Lorenz structure is assumed. We only posit that
    du/dt = Xi^T Theta(u),   Theta(u) = [1, x, y, z, x^2, xy, xz, y^2, yz, z^2]
(all monomials up to --degree) and look for a *sparse* coefficient matrix Xi.

Steps
  1. Generate noisy x, y, z data (same set-up as lorenz_example.py).
  2. Build the candidate library Theta (pysindy.PolynomialLibrary).
  3. Sparse regression with sequentially thresholded least squares (STLSQ),
     in two flavours:
       strong form  du/dt ~ Theta(u) Xi, du/dt from smoothed finite differences
                    (pysindy.SINDy);
       weak form    integrate against compact test functions phi_k(t) so no
                    derivative of the noisy data is needed:
                        -int phi_k' u dt = int phi_k Theta(u) dt  Xi
  4. Choose the sparsity threshold automatically: sweep it, and for every
     distinct model on the path measure how well it explains held-out data
     (segment initial states fitted, coefficients frozen); keep the sparsest
     model within --tol of the best.
  5. Refine the nonzero coefficients of the selected weak-form model with
     multiple shooting, and compare everything with the true equations.

Usage
  python sindy_example.py [--noise 0.5] [--tmax 20] [--degree 2] [--outdir figs]
"""
import argparse
import os
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pysindy as ps

from lorenz_example import generate_data, integrate, MultipleShooting, covariance

VARS = ("x", "y", "z")
# dataviz reference palette, categorical slots in fixed order
C_STRONG, C_WEAK, C_REFINED = "#eb6834", "#1baf7a", "#4a3aa7"
C_INK, C_MUTED = "#222222", "#8a8a85"


# ---------------------------------------------------------------------------
# Library and polynomial model
# ---------------------------------------------------------------------------
def make_library(degree):
    lib = ps.PolynomialLibrary(degree=degree)
    lib.fit(np.zeros((2, 3)))
    return lib, lib.powers_.copy(), lib.get_feature_names(list(VARS))


def theta(u, powers):
    """Library evaluated at states u of shape (3, N) -> (m, N)."""
    return np.prod(u[None, :, :] ** powers[:, :, None], axis=1)


def poly_rhs_factory(powers, support):
    """
    RHS for integrate()/MultipleShooting: only the (term, equation) pairs in
    `support` (a boolean (m, 3) mask) are active; their coefficients are the
    parameters, in np.flatnonzero(support) order.
    """
    active = np.flatnonzero(support)

    def rhs(t, u, *coef):
        U = u.reshape(3, -1)
        Xi = np.zeros(support.size)
        Xi[active] = coef
        return (Xi.reshape(support.shape).T @ theta(U, powers)).ravel()
    return rhs


def rescale_coefficients(Xi_s, powers, s):
    """Coefficients found for u/s back to u:  c = c_s * s^(1 - degree)."""
    deg = powers.sum(axis=1)
    return Xi_s * (float(s) ** (1 - deg))[:, None]


def true_coefficients(powers, sigma, rho, beta):
    Xi = np.zeros((len(powers), 3))
    idx = {tuple(p): i for i, p in enumerate(powers)}
    X, Y, Z, XY, XZ = (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1)
    Xi[idx[X], 0], Xi[idx[Y], 0] = -sigma, sigma
    Xi[idx[X], 1], Xi[idx[Y], 1], Xi[idx[XZ], 1] = rho, -1.0, -1.0
    Xi[idx[Z], 2], Xi[idx[XY], 2] = -beta, 1.0
    return Xi


# ---------------------------------------------------------------------------
# Sparse regression
# ---------------------------------------------------------------------------
def stlsq(G, B, threshold, max_iter=20):
    """Sequentially thresholded least squares (Brunton et al. 2016)."""
    Xi = np.linalg.lstsq(G, B, rcond=None)[0]
    for _ in range(max_iter):
        small = np.abs(Xi) < threshold
        Xi[small] = 0.0
        for j in range(B.shape[1]):
            big = ~small[:, j]
            if big.any():
                Xi[big, j] = np.linalg.lstsq(G[:, big], B[:, j], rcond=None)[0]
    return Xi


def weak_system(t, u, powers, n_test=400, half_width=0.2, p=4):
    """
    Weak form with test functions phi_k(t) = (1 - ((t - c_k)/H)^2)^p on
    [c_k - H, c_k + H]. Because phi_k vanishes at both ends, integrating by
    parts moves the derivative from the noisy data onto phi_k:
        int phi_k u' dt = -int phi_k' u dt.
    Returns G (n_test, m) and B (n_test, 3) with G Xi ~ B.
    """
    dt = t[1] - t[0]
    Th = theta(u, powers).T                                  # (nt, m)
    G = np.zeros((n_test, Th.shape[1]))
    B = np.zeros((n_test, 3))
    for k, c in enumerate(np.linspace(t[0] + half_width, t[-1] - half_width, n_test)):
        sel = np.abs(t - c) <= half_width
        s = (t[sel] - c) / half_width
        phi = (1 - s ** 2) ** p
        dphi = -2 * p * s * (1 - s ** 2) ** (p - 1) / half_width
        w = np.full(sel.sum(), dt)
        w[[0, -1]] = dt / 2                                   # trapezoid rule
        norm = np.sum(w * phi)
        G[k] = (w * phi) @ Th[sel] / norm
        B[k] = -(w * dphi) @ u[:, sel].T / norm
    return G, B


def fit_weak(t, u, powers, threshold, **kw):
    G, B = weak_system(t, u, powers, **kw)
    return stlsq(G, B, threshold)


def fit_strong(t, u, degree, threshold):
    model = ps.SINDy(optimizer=ps.STLSQ(threshold=threshold),
                     feature_library=ps.PolynomialLibrary(degree=degree),
                     differentiation_method=ps.SmoothedFiniteDifference())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(u.T, t=t)
    return np.asarray(model.coefficients()).T                 # (m, 3)


# ---------------------------------------------------------------------------
# Model selection by misfit on held-out data
# ---------------------------------------------------------------------------
def validation_misfit(Xi, powers, t, u, seg_len=0.5):
    """
    How well can this model explain held-out data? Coefficients are frozen and
    only the initial state of each short segment is fitted (multiple shooting
    with no free parameters), so noise in the starting points does not count
    against the model. A correct model reaches the noise level.
    """
    support = Xi != 0
    if not support.any():
        return np.inf
    rhs_p = poly_rhs_factory(powers, support)
    coef = Xi[support]
    ms = MultipleShooting(t, u, seg_len, rhs=lambda tt, uu: rhs_p(tt, uu, *coef), nparam=0)
    res = ms.fit(np.array([]))
    n_data = ms.data.size
    return float(np.sqrt(np.mean(res.fun[:n_data] ** 2)))


def select_model(fit_fn, powers, t_val, u_val, thresholds, tol):
    """
    Sweep thresholds, score each distinct sparsity pattern by its misfit on
    validation data, return the sparsest one within (1+tol) of the best.
    """
    path = {}
    for lam in thresholds:
        Xi = fit_fn(lam)
        key = tuple(np.flatnonzero(Xi))
        if key and key not in path:
            path[key] = dict(lam=lam, Xi=Xi, nterms=len(key),
                             err=validation_misfit(Xi, powers, t_val, u_val))
    models = sorted(path.values(), key=lambda m: m["nterms"])
    best_err = min(m["err"] for m in models)
    chosen = next(m for m in models if m["err"] <= (1 + tol) * best_err)
    return chosen, models


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def equations(Xi, names, prec=3):
    lines = []
    for j, v in enumerate(VARS):
        terms = [f"{Xi[i, j]:+.{prec}f} {names[i]}".replace(" 1", "")
                 for i in range(len(names)) if Xi[i, j] != 0]
        lines.append(f"    d{v}/dt = " + (" ".join(terms) if terms else "0"))
    return "\n".join(lines)


def support_verdict(Xi, Xi_true):
    missing = int(np.sum((Xi == 0) & (Xi_true != 0)))
    extra = int(np.sum((Xi != 0) & (Xi_true == 0)))
    if missing == extra == 0:
        return "correct structure"
    return f"{missing} missing, {extra} spurious term(s)"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_MUTED)
    ax.tick_params(colors=C_INK, labelsize=8)
    ax.grid(True, color="#e6e6e3", lw=0.6)
    ax.set_axisbelow(True)


def plot_path(strong_models, strong_pick, weak_models, weak_pick, n_true, fname):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    style(ax)
    for models, pick, col, lab in ((strong_models, strong_pick, C_STRONG, "strong form"),
                                   (weak_models, weak_pick, C_WEAK, "weak form")):
        n = [m["nterms"] for m in models]
        e = [m["err"] for m in models]
        ax.semilogy(n, e, "-o", color=col, lw=2, ms=6, label=lab,
                    markeredgecolor="white", markeredgewidth=1.5)
        ax.semilogy(pick["nterms"], pick["err"], "o", ms=13, mfc="none",
                    mec=col, mew=2)
        ax.annotate("selected", (pick["nterms"], pick["err"]), xytext=(8, 8),
                    textcoords="offset points", fontsize=8, color=C_INK)
    ax.axvline(n_true, color=C_MUTED, ls=":", lw=1.2)
    ax.text(n_true, ax.get_ylim()[1], " true Lorenz (7 terms)", va="top",
            fontsize=8, color=C_MUTED)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([0.5, 1, 2, 5])
    ax.set_xlabel("number of terms in the model", color=C_INK)
    ax.set_ylabel("held-out RMS misfit", color=C_INK)
    ax.set_title("Sparsity path: one point per distinct model", color=C_INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)


def plot_coefficients(names, results, Xi_true, fname):
    """One panel per equation: coefficient of every library term, per method."""
    m = len(names)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    offsets = np.linspace(-0.27, 0.27, len(results))
    for j, ax in enumerate(axes):
        style(ax)
        ax.axhline(0, color=C_MUTED, lw=0.8)
        ax.bar(np.arange(m), Xi_true[:, j], width=0.8, color="#ededea",
               edgecolor=C_MUTED, lw=0.8, label="true", zorder=1)
        for (lab, Xi, col), off in zip(results, offsets):
            nz = Xi[:, j] != 0
            ax.plot(np.arange(m)[nz] + off, Xi[nz, j], "o", color=col, ms=7,
                    markeredgecolor="white", markeredgewidth=1.2, label=lab, zorder=3)
        ax.set_ylabel(f"d{VARS[j]}/dt", color=C_INK)
        if j == 0:
            ax.legend(frameon=False, fontsize=8, ncol=len(results) + 1, loc="upper right")
    axes[-1].set_xticks(np.arange(m))
    axes[-1].set_xticklabels(names)
    axes[-1].set_xlabel("library term", color=C_INK)
    fig.suptitle("Discovered coefficients (zero = term dropped)", color=C_INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)


def plot_trajectories(t, u, u_clean, Xi_ref, powers, u0, fname):
    rhs = poly_rhs_factory(powers, Xi_ref != 0)
    u_free = integrate(Xi_ref[Xi_ref != 0], u0, t, rhs=rhs)[:, 0, :]
    t_long = np.arange(0, 5 * t[-1], t[1] - t[0])
    u_long = integrate(Xi_ref[Xi_ref != 0], u0, t_long, rhs=rhs)[:, 0, :]

    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(3, 2, width_ratios=(1.6, 1))
    for i in range(3):
        ax = fig.add_subplot(gs[i, 0])
        style(ax)
        ax.plot(t, u[i], ".", ms=2, color=C_MUTED, label="data")
        ax.plot(t, u_free[i], "-", lw=1.4, color=C_REFINED, label="discovered model")
        ax.set_ylabel(VARS[i], color=C_INK)
        if i == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2, markerscale=5)
        if i == 2:
            ax.set_xlabel("t", color=C_INK)
    ax = fig.add_subplot(gs[:, 1], projection="3d")
    ax.plot(*u, ".", ms=1.5, color=C_MUTED, alpha=0.6, label="data")
    ax.plot(*u_long, "-", lw=0.3, color=C_REFINED, label="discovered model")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    leg = ax.legend(frameon=False, fontsize=8, markerscale=5)
    leg.get_lines()[1].set_linewidth(1.5)
    fig.suptitle("Data vs the discovered model (free run; diverges after a few "
                 "Lyapunov times because the system is chaotic)", color=C_INK, fontsize=10)
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
    ap.add_argument("--noise", type=float, default=0.5)
    ap.add_argument("--degree", type=int, default=2, help="max polynomial degree of the library")
    ap.add_argument("--scale", type=float, default=10.0,
                    help="fit in u/scale so all library columns are O(1)")
    ap.add_argument("--val-frac", type=float, default=0.3,
                    help="fraction of the record held out for model selection")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="accept the sparsest model within (1+tol) of the best held-out misfit")
    ap.add_argument("--seg-len", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--outdir", default="figs")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # 1. data
    theta_true = np.array([args.sigma, args.rho, args.beta])
    t, u_clean, u = generate_data(theta_true, np.array([-8.0, 7.0, 27.0]),
                                  args.tmax, args.dt, args.noise, rng)
    n_tr = int(round((1 - args.val_frac) * len(t)))
    t_tr, u_tr = t[:n_tr], u[:, :n_tr]
    t_va, u_va = t[n_tr:], u[:, n_tr:]
    print(f"Step 1: {len(t)} samples, noise std {args.noise}; "
          f"train t<{t[n_tr]:.1f}, validate t>={t[n_tr]:.1f}")

    # 2. library
    lib, powers, names = make_library(args.degree)
    Xi_true = true_coefficients(powers, *theta_true)
    print(f"Step 2: library of {len(names)} terms per equation "
          f"({3 * len(names)} candidates): {', '.join(names)}")
    print("        true model:\n" + equations(Xi_true, names))

    # 3-4. sparse regression + threshold selection, in scaled variables
    s = args.scale
    thresholds = np.logspace(-2, 1.5, 50)
    results = {}
    for label, fit_fn in (
        ("strong", lambda lam: fit_strong(t_tr, u_tr / s, args.degree, lam)),
        ("weak", lambda lam: fit_weak(t_tr, u_tr / s, powers, lam)),
    ):
        tic = time.time()
        # score models in physical units
        pick, models = select_model(lambda lam: rescale_coefficients(fit_fn(lam), powers, s),
                                    powers, t_va, u_va, thresholds, args.tol)
        results[label] = (pick, models)
        print(f"Step 3/4: {label}-form SINDy: {len(models)} distinct models on the path; "
              f"selected {pick['nterms']} terms (threshold {pick['lam']:.3g}, "
              f"held-out rms misfit {pick['err']:.3f}) -> {support_verdict(pick['Xi'], Xi_true)} "
              f"[{time.time() - tic:.1f}s]")
        print(equations(pick["Xi"], names))

    # 5. refine the weak-form model by multiple shooting on the whole record
    Xi_weak = results["weak"][0]["Xi"]
    support = Xi_weak != 0
    ms = MultipleShooting(t, u, args.seg_len, rhs=poly_rhs_factory(powers, support),
                          nparam=int(support.sum()))
    tic = time.time()
    res = ms.fit(Xi_weak[support])
    std = np.sqrt(np.diag(covariance(res))[: ms.nparam])
    Xi_ref = np.zeros_like(Xi_weak)
    Xi_ref[support] = res.x[: ms.nparam]
    Xi_std = np.zeros_like(Xi_weak)
    Xi_std[support] = std
    print(f"Step 5: multiple-shooting refinement ({res.nfev} fevals, {time.time() - tic:.1f}s):")
    for j, v in enumerate(VARS):
        terms = [f"{Xi_ref[i, j]:+.4f}(±{Xi_std[i, j]:.4f}) {names[i]}".replace(" 1", "")
                 for i in range(len(names)) if support[i, j]]
        print(f"    d{v}/dt = " + " ".join(terms))
    err = np.abs(Xi_ref - Xi_true)[support | (Xi_true != 0)]
    print(f"        max |coef - true| = {err.max():.4f}")

    # figures
    plot_path(results["strong"][1], results["strong"][0],
              results["weak"][1], results["weak"][0], int((Xi_true != 0).sum()),
              os.path.join(args.outdir, "sindy_path.png"))
    plot_coefficients(names, [("strong SINDy", results["strong"][0]["Xi"], C_STRONG),
                              ("weak SINDy", Xi_weak, C_WEAK),
                              ("weak + multiple shooting", Xi_ref, C_REFINED)],
                      Xi_true, os.path.join(args.outdir, "sindy_coefficients.png"))
    plot_trajectories(t, u, u_clean, Xi_ref, powers, res.x[ms.nparam:].reshape(3, -1)[:, 0],
                      os.path.join(args.outdir, "sindy_fit.png"))
    print(f"Figures written to {args.outdir}/")


if __name__ == "__main__":
    main()
