#!/usr/bin/env python3
"""
Infer l1, l2 and m2/m1 of a double pendulum from noisy angle data with the
adjoint method.

Same data and model as doublependulum_example.py (solver in ~/doublependulum,
only theta1, theta2 observed, m1 fixed to 1 because only m2/m1 is
identifiable), but the gradient of the cost is obtained from one forward and
one backward (adjoint) solve instead of a finite-difference Jacobian.

Problem
  Unknowns x = (theta, u0_1..u0_K), theta = (l1, l2, m2/m1), u0_k the 4-D
  initial state (th1, th2, w1, w2) of segment k. Each segment obeys
      dy_k/dt = f(y_k, theta),   y_k(0) = u0_k,   t in [0, T]
  and the cost is
      J = 1/2 sum_k sum_i |H y_k(t_i) - d_k,i|^2            (angles only, H = [I 0])
        + w^2/2 sum_k |y_k(T) - u0_{k+1}|^2                  (continuity)
  The segments are short (T = 0.5 s) because the system is chaotic: over the
  full 20 s record J would be far too rough to minimise.

Adjoint
  Backward in time, between data times, solve
      d lambda_k/dt = -(df/dy)^T lambda_k
      d mu/dt       = -sum_k (df/dtheta)^T lambda_k,         mu(T) = 0
  starting from lambda_k(T) = H^T r_k,n + w^2 c_k, and adding the jump H^T r_k,i
  at every data time t_i (r = residual, c = continuity mismatch). Then
      dJ/dtheta = mu(0),   dJ/du0_k = lambda_k(0) - w^2 c_{k-1}.
  The forward solution needed by the backward pass comes from solve_ivp's
  dense output. The vector-Jacobian products (df/dy)^T lambda and
  (df/dtheta)^T lambda are computed by complex-step differentiation of the
  solver's own rhs, so they are exact to rounding and no hand-derived
  Jacobian is needed.

The adjoint gradient is checked against central finite differences, then
L-BFGS-B minimises J starting from the gradient-matching guess.

Usage
  .venv/bin/python doublependulum_adjoint.py [--noise 0.02] [--seg-len 0.5]
"""
import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from doublependulum_example import (NDIM, NOBS, PARAM_NAMES, MultipleShooting, dp, fmt,
                                    generate_data, gradient_matching, plot_fit)

CSTEP = 1e-30          # complex-step size


class AdjointProblem(MultipleShooting):
    """Segments, data and initial-state guess from MultipleShooting; cost + adjoint gradient."""

    def __init__(self, t, th, seg_len, g, rtol=1e-9, atol=1e-9, **kw):
        super().__init__(t, th, seg_len, g, **kw)
        self.rtol, self.atol = rtol, atol
        self.nfwd = self.nadj = 0
        self.last = (None, None)

    # -- model and its vector-Jacobian products ------------------------------
    def f(self, y, theta):
        """rhs for states y of shape (4, K); theta = (l1, l2, m2/m1)."""
        l1, l2, r = theta
        return np.array(dp.rhs(0.0, y, l1, l2, 1.0, r, self.g))

    def vjp_y(self, y, theta, lam):
        """(df/dy)^T lam per segment, shape (4, K), by complex step on each state component."""
        out = np.empty_like(lam)
        for j in range(NDIM):
            yc = y.astype(complex)
            yc[j] += 1j * CSTEP
            out[j] = np.sum(lam * self.f(yc, theta).imag, axis=0) / CSTEP
        return out

    def vjp_theta(self, y, theta, lam):
        """sum_k (df/dtheta)^T lam_k, shape (3,)."""
        out = np.empty(3)
        for j in range(3):
            tc = np.asarray(theta, dtype=complex)
            tc[j] += 1j * CSTEP
            out[j] = np.sum(lam * self.f(y, tc).imag) / CSTEP
        return out

    # -- forward ---------------------------------------------------------------
    def forward(self, theta, u0):
        self.nfwd += 1
        rhs = lambda t, u: self.f(u.reshape(NDIM, -1), theta).ravel()
        sol = solve_ivp(rhs, (0.0, self.tloc[-1]), u0.ravel(), t_eval=self.tloc,
                        dense_output=True, method="DOP853", rtol=self.rtol, atol=self.atol)
        if not sol.success or sol.y.shape[1] != self.n:
            return None
        return sol, sol.y.reshape(NDIM, self.K, self.n)

    def cost(self, x):
        theta, u0 = self.unpack(x)
        fw = self.forward(theta, u0)
        if fw is None:
            return 1e10
        _, sim = fw
        r = sim[:NOBS] - self.data
        c = sim[:, :-1, -1] - u0[:, 1:]
        return 0.5 * np.sum(r**2) + 0.5 * self.w**2 * np.sum(c**2)

    # -- forward + backward ------------------------------------------------------
    def cost_and_grad(self, x):
        theta, u0 = self.unpack(x)
        fw = self.forward(theta, u0)
        if fw is None:
            return 1e10, np.zeros_like(x)
        sol, sim = fw
        r = sim[:NOBS] - self.data                       # (2, K, n)
        c = sim[:, :-1, -1] - u0[:, 1:]                  # (4, K-1)
        J = 0.5 * np.sum(r**2) + 0.5 * self.w**2 * np.sum(c**2)
        self.last = (x.copy(), J)

        K, nlam = self.K, NDIM * self.K

        def adjoint_rhs(t, z):
            lam = z[:nlam].reshape(NDIM, K)
            y = sol.sol(t).reshape(NDIM, K)
            return np.concatenate([-self.vjp_y(y, theta, lam).ravel(),
                                   -self.vjp_theta(y, theta, lam)])

        # terminal condition at t = T
        lam = np.zeros((NDIM, K))
        lam[:NOBS] += r[:, :, -1]
        lam[:, :-1] += self.w**2 * c
        mu = np.zeros(3)
        # integrate backward from data time to data time, adding the data jumps
        self.nadj += 1
        for i in range(self.n - 1, 0, -1):
            z = solve_ivp(adjoint_rhs, (self.tloc[i], self.tloc[i - 1]),
                          np.concatenate([lam.ravel(), mu]), method="DOP853",
                          rtol=self.rtol, atol=self.atol).y[:, -1]
            lam, mu = z[:nlam].reshape(NDIM, K).copy(), z[nlam:]
            lam[:NOBS] += r[:, :, i - 1]

        g_u0 = lam.copy()
        g_u0[:, 1:] -= self.w**2 * c
        return J, np.concatenate([mu, g_u0.ravel()])


def gradient_check(prob, x, rng, eps=1e-6):
    """Compare the adjoint gradient with central finite differences."""
    _, g = prob.cost_and_grad(x)
    print("        component         adjoint          finite diff      rel. diff")
    checks = [(f"dJ/d{n}", np.eye(x.size)[j]) for j, n in enumerate(("l1", "l2", "m2/m1"))]
    checks += [("dJ/du0[w1, seg 3]", np.eye(x.size)[3 + 2 * prob.K + 3]),
               ("random direction", rng.standard_normal(x.size))]
    for name, v in checks:
        fd = (prob.cost(x + eps * v) - prob.cost(x - eps * v)) / (2 * eps)
        ad = g @ v
        print(f"        {name:<18}{ad:>16.8e} {fd:>16.8e}   {abs(ad - fd) / max(abs(fd), 1e-300):.1e}")


def plot_history(hist, theta_true, fname):
    hist = np.array(hist)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].semilogy(hist[:, 0], color="k")
    axs[0].set_xlabel("L-BFGS-B iteration")
    axs[0].set_ylabel("cost J")
    axs[0].set_title("cost (forward + adjoint per gradient)")
    for j, n in enumerate(PARAM_NAMES):
        axs[1].plot(hist[:, 1 + j], color=f"C{j}", label=n)
        axs[1].axhline(theta_true[j], color=f"C{j}", ls=":", lw=0.8)
    axs[1].set_xlabel("L-BFGS-B iteration")
    axs[1].set_title("parameters (dotted: truth)")
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)


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
    ap.add_argument("--maxiter", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--outdir", default="figs")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    phys_true = (args.l1, args.l2, args.m1, args.m2, args.g)
    theta_true = np.array([args.l1, args.l2, args.m2 / args.m1])
    y0 = np.deg2rad([args.theta1, args.theta2, 0.0, 0.0])

    t, y_clean, th = generate_data(phys_true, y0, args.tmax, args.dt, args.noise, rng)
    print(f"Data: {len(t)} angle samples on t in [0, {args.tmax}] s, noise std = {args.noise} rad")
    print("      true                " + fmt(theta_true))

    theta_gm = gradient_matching(t, th, args.g)
    print("Initial guess (grad. matching) " + fmt(theta_gm))

    prob = AdjointProblem(t, th, args.seg_len, args.g)
    x0 = np.concatenate([theta_gm, prob.u0_guess.ravel()])
    print(f"Unknowns: 3 parameters + {NDIM * prob.K} segment states ({prob.K} segments)")

    print("Gradient check at the initial guess:")
    tic = time.time()
    gradient_check(prob, x0, rng)
    print(f"        ({time.time() - tic:.1f}s)")

    prob.nfwd = prob.nadj = 0
    hist = []

    def callback(xk):
        # L-BFGS-B's accepted iterate is normally the last point evaluated
        xl, Jl = prob.last
        J = Jl if xl is not None and np.array_equal(xk, xl) else prob.cost(xk)
        hist.append([J, *xk[:3]])

    lb = np.full(x0.size, -np.inf)
    lb[:3] = 1e-3
    callback(x0)
    tic = time.time()
    res = minimize(prob.cost_and_grad, x0, jac=True, method="L-BFGS-B",
                   bounds=list(zip(lb, np.full(x0.size, np.inf))), callback=callback,
                   options=dict(maxiter=args.maxiter, maxfun=5 * args.maxiter,
                                ftol=1e-15, gtol=1e-8, maxcor=30))
    elapsed = time.time() - tic
    print(f"Adjoint + L-BFGS-B: {fmt(res.x[:3])}")
    print(f"        {res.nit} iterations, {prob.nadj} adjoint solves, "
          f"{prob.nfwd} forward solves, {elapsed:.1f}s")
    print(f"        {res.message}")
    rms = np.sqrt(2 * prob.cost(res.x) / prob.data.size)
    print(f"        rms angle misfit {rms:.4f} rad (noise std {args.noise})")

    plot_history(hist, theta_true, os.path.join(args.outdir, "dp_adjoint_history.png"))
    plot_fit(t, th, y_clean, prob, res, theta_true, os.path.join(args.outdir, "dp_adjoint_fit.png"))
    print(f"Figures written to {args.outdir}/")


if __name__ == "__main__":
    main()
