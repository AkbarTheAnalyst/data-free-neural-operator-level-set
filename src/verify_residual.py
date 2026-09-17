"""Consistency check on the transport residual. Run before trusting any result.

Two things are checked:

  1. Consistency -- the exact solution should drive the residual toward zero
     under mesh refinement, at the second order the central differences give.
     (The field is C^1 away from the circle centre, where the distance function
     has a measure-zero kink, so convergence is not perfectly clean.)

  2. Sensitivity -- a field that does NOT satisfy transport, here the exact
     solution frozen at t=0, must give a residual orders of magnitude larger.
     A residual that is small for everything is worse than useless as a
     training signal, and this is what catches that.
"""

import torch
from ro_family import make_grid, velocity, exact
from residuals import strong_residual


if __name__ == "__main__":
    print(f"{'h':>10} {'exact':>14} {'frozen':>14} {'ratio':>10}")
    prev = None
    for n in (17, 33, 65, 129):
        X, Y, Tt, h = make_grid(n, n, n, dtype=torch.float64)
        u, v = velocity(X, Y)
        phi = exact(X, Y, Tt, rho=0.25, alpha=torch.pi / 2, R=0.15)[None]
        frozen = phi[..., :1].expand_as(phi).contiguous()
        e = float(strong_residual(phi, u, v, h).abs().mean())
        f = float(strong_residual(frozen, u, v, h).abs().mean())
        rate = "" if prev is None else f"  (x{prev / e:.1f})"
        print(f"{h[0]:10.4f} {e:14.3e} {f:14.3e} {f / e:10.1f}{rate}")
        prev = e
