#!/usr/bin/env python3
"""
Gradient of an ODE least-squares objective with respect to parameters, by the
adjoint method, for any right-hand side given as sympy expressions.

Problem
      dy/dt = f(t, y, p),   y(0) = y0,   t in [0, T]
      L(p)  = 1/2 int_0^T |y - ybar|^2 dt
  with ybar(t) the (given) observations.

Adjoint (Lagrangian L + int lambda^T (dy/dt - f) dt, integrate by parts)
      d lambda/dt = (y - ybar) - f_y^T lambda,   lambda(T) = 0     (solved backward)
      dL/dp       = -int_0^T lambda^T f_p dt  -  lambda(0)^T dy0/dp
  f_y = df/dy (n x n) and f_p = df/dp (n x m) are derived symbolically with
  sympy and evaluated along the forward solution. The last term vanishes when
  y0 does not depend on p.

Numerics
  forward and adjoint ODEs: scipy solve_ivp with dense output (DOP853 by
  default; pass method='Radau' for stiff problems -- avoid LSODA, whose
  dense output limits the gradient to ~1e-6 relative accuracy);
  the integrals in L and dL/dp: composite Simpson on a uniform grid of nquad
  points.

Usage
  .venv/bin/python adjoint_sympy.py          # Lotka-Volterra demo + gradient check

  from adjoint_sympy import AdjointODE
  t = sp.Symbol('t'); x, y = sp.symbols('x y'); a, b = sp.symbols('a b')
  prob = AdjointODE([a*x - x*y, -b*y + x*y], [x, y], [a, b], t)
  L, dLdp, info = prob.gradient(p, y0, T, ybar)
"""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp, simpson
from scipy.interpolate import CubicSpline


class AdjointODE:
    """dy/dt = f(t, y, p) with f a list of sympy expressions in t, y, p."""

    def __init__(self, f, y, p, t=None):
        self.t = t if t is not None else sp.Symbol('t')
        self.y = list(y)
        self.p = list(p)
        F = sp.Matrix(f)
        if F.shape != (len(self.y), 1):
            raise ValueError(f"f has {F.shape[0]} components but y has {len(self.y)}")
        self.F = F
        self.Fy = F.jacobian(self.y)       # n x n
        self.Fp = F.jacobian(self.p)       # n x m
        args = (self.t, self.y, self.p)
        self._f = sp.lambdify(args, list(F), modules='numpy', cse=True)
        self._fy = sp.lambdify(args, self.Fy.tolist(), modules='numpy', cse=True)
        self._fp = sp.lambdify(args, self.Fp.tolist(), modules='numpy', cse=True)

    @property
    def n(self):
        return len(self.y)

    @property
    def m(self):
        return len(self.p)

    # -- numerical evaluation of f, f_y, f_p --------------------------------
    def f(self, t, y, p):
        return np.array(self._f(t, y, p), dtype=float)

    def f_y(self, t, y, p):
        return np.array(self._fy(t, y, p), dtype=float).reshape(self.n, self.n)

    def f_p(self, t, y, p):
        return np.array(self._fp(t, y, p), dtype=float).reshape(self.n, self.m)

    # -- solves --------------------------------------------------------------
    def forward(self, p, y0, T, rtol=1e-10, atol=1e-12, method='DOP853'):
        """Solve dy/dt = f on [0, T]; returns the solve_ivp result with dense output."""
        p = np.asarray(p, dtype=float)
        sol = solve_ivp(lambda t, y: self.f(t, y, p), (0.0, T), np.asarray(y0, dtype=float),
                        method=method, dense_output=True, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"forward solve failed: {sol.message}")
        return sol

    def adjoint(self, p, ysol, ybar, T, rtol=1e-10, atol=1e-12, method='DOP853',
                include_fy=True):
        """Solve d lambda/dt = (y - ybar) - f_y^T lambda backward from lambda(T) = 0.

        ysol is the forward solution (dense output), ybar a callable t -> (n,).
        include_fy=False drops the -f_y^T lambda term (wrong unless f_y = 0;
        kept only to show its effect).
        """
        p = np.asarray(p, dtype=float)

        def rhs(t, lam):
            y = ysol.sol(t)
            r = y - ybar(t)
            if include_fy:
                r = r - self.f_y(t, y, p).T @ lam
            return r

        sol = solve_ivp(rhs, (T, 0.0), np.zeros(self.n), method=method,
                        dense_output=True, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"adjoint solve failed: {sol.message}")
        return sol

    # -- objective and gradient ---------------------------------------------
    @staticmethod
    def _grid(T, nquad):
        if nquad % 2 == 0:
            nquad += 1                      # odd number of points for Simpson
        return np.linspace(0.0, T, nquad)

    def objective(self, p, y0, T, ybar, nquad=2001, ysol=None, **solver_kw):
        """L = 1/2 int_0^T |y - ybar|^2 dt."""
        if ysol is None:
            ysol = self.forward(p, y0, T, **solver_kw)
        tq = self._grid(T, nquad)
        r = ysol.sol(tq) - np.column_stack([ybar(ti) for ti in tq])     # (n, nq)
        return 0.5 * simpson(np.sum(r * r, axis=0), x=tq)

    def gradient(self, p, y0, T, ybar, nquad=2001, dy0dp=None, include_fy=True,
                 **solver_kw):
        """Return (L, dL/dp, info).

        p      : (m,) parameter values
        y0     : (n,) initial condition
        T      : final time
        ybar   : callable t -> (n,) observations (see interpolate_obs)
        nquad  : number of quadrature points for the time integrals
        dy0dp  : (n, m) sensitivity of y0 to p, if y0 depends on p
        info   : dict with the forward ('y') and adjoint ('lam') solutions
        """
        p = np.asarray(p, dtype=float)
        ysol = self.forward(p, y0, T, **solver_kw)
        lsol = self.adjoint(p, ysol, ybar, T, include_fy=include_fy, **solver_kw)

        tq = self._grid(T, nquad)
        Y = ysol.sol(tq)
        Lam = lsol.sol(tq)
        # integrand lambda^T f_p at each quadrature point, shape (nq, m)
        g = np.array([Lam[:, k] @ self.f_p(tq[k], Y[:, k], p) for k in range(tq.size)])
        dLdp = -simpson(g, x=tq, axis=0)
        if dy0dp is not None:
            dLdp -= Lam[:, 0] @ np.asarray(dy0dp, dtype=float).reshape(self.n, self.m)

        L = self.objective(p, y0, T, ybar, nquad=nquad, ysol=ysol)
        return L, dLdp, {'y': ysol, 'lam': lsol}

    def fd_gradient(self, p, y0, T, ybar, h=1e-6, nquad=2001, **solver_kw):
        """Central finite-difference dL/dp, for checking."""
        p = np.asarray(p, dtype=float)
        g = np.empty(self.m)
        for j in range(self.m):
            dp = np.zeros(self.m)
            dp[j] = h * max(1.0, abs(p[j]))
            Lp = self.objective(p + dp, y0, T, ybar, nquad=nquad, **solver_kw)
            Lm = self.objective(p - dp, y0, T, ybar, nquad=nquad, **solver_kw)
            g[j] = (Lp - Lm) / (2 * dp[j])
        return g


def interpolate_obs(t_obs, y_obs):
    """Turn samples y_obs (len(t_obs), n) into a callable ybar(t) -> (n,) (cubic spline)."""
    spl = CubicSpline(np.asarray(t_obs), np.asarray(y_obs), axis=0)
    return lambda t: spl(t)


def main():
    import time

    # Lotka-Volterra: dx/dt = a x - b x y,  dy/dt = -c y + d x y
    t = sp.Symbol('t')
    x, y = sp.symbols('x y')
    a, b, c, d = sp.symbols('a b c d')
    prob = AdjointODE([a*x - b*x*y, -c*y + d*x*y], [x, y], [a, b, c, d], t)
    print("f   =", list(prob.F))
    print("f_y =", prob.Fy.tolist())
    print("f_p =", prob.Fp.tolist())

    # synthetic observations: true parameters, noisy samples, spline
    p_true = np.array([1.0, 0.4, 1.2, 0.3])
    y0 = np.array([5.0, 2.0])
    T = 15.0
    rng = np.random.default_rng(0)
    t_obs = np.linspace(0.0, T, 151)
    y_obs = prob.forward(p_true, y0, T).sol(t_obs).T
    y_obs += 0.05 * rng.standard_normal(y_obs.shape)
    ybar = interpolate_obs(t_obs, y_obs)

    p = np.array([0.9, 0.45, 1.1, 0.32])          # evaluate gradient away from the truth
    t0 = time.perf_counter()
    L, g_adj, _ = prob.gradient(p, y0, T, ybar)
    t_adj = time.perf_counter() - t0
    t0 = time.perf_counter()
    g_fd = prob.fd_gradient(p, y0, T, ybar)
    t_fd = time.perf_counter() - t0
    _, g_nofy, _ = prob.gradient(p, y0, T, ybar, include_fy=False)

    def rel(g):
        return np.linalg.norm(g - g_fd) / np.linalg.norm(g_fd)

    np.set_printoptions(precision=8, suppress=False)
    print(f"\np = {p},  L = {L:.10g}")
    print(f"dL/dp adjoint        = {g_adj}   ({t_adj:.2f} s)")
    print(f"dL/dp central FD     = {g_fd}   ({t_fd:.2f} s)")
    print(f"relative difference  = {rel(g_adj):.2e}")
    print(f"without -f_y^T lambda: {g_nofy}   relative error {rel(g_nofy):.2e}")


if __name__ == '__main__':
    main()
