"""
Reversed single vortex (RV) benchmark, lifted from a single instance to a
family over initial conditions.

Velocity field, from Khan & Raees:

    u = -sin^2(pi x) sin(2 pi y) cos(pi t / T)
    v =  sin^2(pi y) sin(2 pi x) cos(pi t / T)

with T = 2. The cosine factor reverses the flow at t = T/2, so the interface
stretches into a thin filament and then returns to its initial shape at t = T.
The final error is therefore magnified by the reversal rather than cancelling
with it, which is what makes this the hard benchmark.

Divergence-free by construction:

    d_x u = -pi sin(2 pi x) sin(2 pi y) f(t)
    d_y v =  pi sin(2 pi y) sin(2 pi x) f(t)

so the enclosed area is conserved exactly and mass_drift remains a valid
reference-free diagnostic, exactly as in the rotation case.

TWO THINGS DIFFER STRUCTURALLY FROM RO, AND BOTH MATTER

1. No closed form. The exact solution is obtained by backward characteristics:
   advect each evaluation point back from t to 0 along the flow and evaluate
   phi_0 there. That is an ODE integration per snapshot time, done here with
   RK45 at rtol 1e-8 / atol 1e-10 to match the published protocol.

   Because the velocity does not depend on the instance, the back-traced
   coordinates are the SAME for every family member. They are computed once
   and cached; each instance then just evaluates its own phi_0 at those
   coordinates. Without that reuse, reference generation would dominate the
   runtime.

2. The exact solution is NOT a signed distance function for t > 0. Under
   rotation it stays one, so the eikonal residual of the truth vanishes there.
   Here it does not, which is precisely the regime SAW's gate was designed for
   and which the rotation benchmark cannot test.
"""

import os

import numpy as np
import torch
from scipy.integrate import solve_ivp

T_FINAL = 2.0

# Velocity-family range. T is the deformation-severity axis: the flow reverses
# at t = T_PERIOD/2, so a smaller period means the interface completes its
# stretch-and-return sooner and is less deformed at any given time. The
# published PINN study reports RV at T = 2 and T = 8, so this is the axis a
# reader recognises. Amplitude scaling is close to redundant with it.
#
# The integration horizon T_FINAL stays fixed at 2 for every instance. Only the
# period varies, so an instance with T_PERIOD = 1.5 has already reversed and
# passed its return point by t = 2, while one with T_PERIOD = 3 is still
# mid-stretch. That is genuine variation in the flow, not a rescaling of time.
T_PERIOD_RANGE = (1.5, 3.0)

# Family ranges. Constraint: the circle must start inside the domain and the
# flow must not carry it across dOmega. The vortex vanishes on the boundary,
# so a circle starting clear of it stays clear.
CX_RANGE = (0.40, 0.60)
CY_RANGE = (0.68, 0.82)
R_RANGE = (0.10, 0.18)

# Velocity family. T is the deformation-severity axis: the flow reverses at
# t = T/2, so a larger T stretches the interface further before returning it.
# The published PINN study reports RV at T = 2 and T = 8, so this is the axis a
# reader recognises; scaling the vortex amplitude instead is close to redundant
# with it. The horizon stays fixed at T_FINAL, so T changes how far the flow
# gets through its own cycle rather than how long the operator must predict.
T_RANGE = (1.5, 3.0)

_CACHE = "rv_characteristics_{nx}x{ny}x{nt}.npz"
_CACHE_V = "rv_characteristics_{nx}x{ny}x{nt}_T{T:.4f}.npz"


def make_grid(nx, ny, nt, device="cpu", dtype=torch.float32):
    """Uniform tensor-product grid on [0,1]^2 x [0,T]."""
    x = torch.linspace(0.0, 1.0, nx, device=device, dtype=dtype)
    y = torch.linspace(0.0, 1.0, ny, device=device, dtype=dtype)
    t = torch.linspace(0.0, T_FINAL, nt, device=device, dtype=dtype)
    X, Y, Tt = torch.meshgrid(x, y, t, indexing="ij")
    return X, Y, Tt, (float(x[1] - x[0]), float(y[1] - y[0]), float(t[1] - t[0]))


def velocity(X, Y, Tt, period=T_FINAL):
    """Vortex field on the grid. Time-dependent, unlike the rotation case.

    period may be a scalar or a tensor of shape (B, 1, 1, 1) for one field per
    instance.
    """
    f = torch.cos(np.pi * Tt / period)
    u = -torch.sin(np.pi * X) ** 2 * torch.sin(2 * np.pi * Y) * f
    v = torch.sin(np.pi * Y) ** 2 * torch.sin(2 * np.pi * X) * f
    return u, v


def velocity_batch(X, Y, Tt, params, device="cpu"):
    """Per-instance velocity fields. Returns (B, nx, ny, nt) each.

    With a three-column params array the period is fixed and one field is
    broadcast over the batch. With four columns the fourth is the period and
    each instance gets its own field.
    """
    p = np.asarray(params)
    if p.shape[1] < 4:
        u, v = velocity(X[None], Y[None], Tt[None])
        return u.expand(len(p), *u.shape[1:]), v.expand(len(p), *v.shape[1:])
    per = torch.as_tensor(p[:, 3], dtype=torch.float32,
                          device=device).view(-1, 1, 1, 1)
    return velocity(X[None], Y[None], Tt[None], per)


def _characteristics(nx, ny, nt, period=T_FINAL, cache=True):
    """Back-traced coordinates (x0, y0) for every grid point and time.

    Returns two arrays of shape (nx, ny, nt): where each evaluation point came
    from at t = 0. Integrating backward from t to 0 has to be done separately
    per target time, so this is nt integrations of a 2*nx*ny dimensional
    system. Cached to disk -- it is identical for every instance and every
    run at this resolution, and recomputing it would dominate the wall clock.
    """
    path = (_CACHE.format(nx=nx, ny=ny, nt=nt) if period == T_FINAL
            else _CACHE_V.format(nx=nx, ny=ny, nt=nt, T=period))
    if cache and os.path.exists(path):
        z = np.load(path)
        return z["x0"], z["y0"]

    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    ts = np.linspace(0.0, T_FINAL, nt)
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
    flat = np.concatenate([Xg.ravel(), Yg.ravel()])
    N = Xg.size

    def rhs(t, s):
        x, y = s[:N], s[N:]
        f = np.cos(np.pi * t / period)
        return np.concatenate([
            -np.sin(np.pi * x) ** 2 * np.sin(2 * np.pi * y) * f,
            np.sin(np.pi * y) ** 2 * np.sin(2 * np.pi * x) * f])

    x0 = np.empty((nx, ny, nt))
    y0 = np.empty((nx, ny, nt))
    for k, tk in enumerate(ts):
        if np.isclose(tk, 0.0):
            x0[..., k], y0[..., k] = Xg, Yg
            continue
        sol = solve_ivp(rhs, [tk, 0.0], flat, method="RK45",
                        rtol=1e-8, atol=1e-10)
        x0[..., k] = sol.y[:N, -1].reshape(nx, ny)
        y0[..., k] = sol.y[N:, -1].reshape(nx, ny)


    if cache:
        np.savez_compressed(path, x0=x0, y0=y0)
    return x0, y0


def exact(x0, y0, cx, cy, R, device="cpu"):
    """Exact field from precomputed characteristics.

    phi(x, t) = phi_0(X(x, t; 0)) -- the solution of pure transport is the
    initial field evaluated at the foot of the characteristic. Since phi_0 is
    a circle here, that is one distance evaluation at the back-traced point.
    """
    x0 = torch.as_tensor(x0, dtype=torch.float32, device=device)
    y0 = torch.as_tensor(y0, dtype=torch.float32, device=device)
    return torch.sqrt((x0 - cx) ** 2 + (y0 - cy) ** 2) - R


def sample_family(n, seed, vary_v=False):
    """Latin hypercube over the family.

    Returns (n, 3) as (cx, cy, R) when vary_v is False, and (n, 4) as
    (cx, cy, R, period) when it is True.

    The first three columns are identical for the same seed and n either way,
    so a fixed-velocity checkpoint stays comparable with a varying-velocity one
    on the initial-condition axis.
    """
    rng = np.random.default_rng(seed)

    def strat(lo, hi):
        edges = (np.arange(n) + rng.random(n)) / n
        return lo + (hi - lo) * rng.permutation(edges)

    cols = [strat(*CX_RANGE), strat(*CY_RANGE), strat(*R_RANGE)]
    if vary_v:
        cols.append(strat(*T_PERIOD_RANGE))
    return np.stack(cols, axis=1)


def build_dataset(params, nx, ny, nt, device="cpu"):
    """phi0, exact spacetime fields, and radii for a set of family members.

    With a fixed period the characteristics are computed once and every
    instance reuses them. With the period varying each instance needs its own
    back-trace -- nt integrations apiece -- so those are cached per period.
    Distinct periods are what makes the velocity family expensive to set up;
    the cache means the cost is paid once per resolution, not once per run.
    """
    p = np.asarray(params)
    vary = p.shape[1] > 3

    if not vary:
        x0, y0 = _characteristics(nx, ny, nt)
        ex = torch.stack([exact(x0, y0, cx, cy, R, device)
                          for cx, cy, R in p])
    else:
        fields = []
        for i, (cx, cy, R, per) in enumerate(p):
            x0, y0 = _characteristics(nx, ny, nt, period=float(per))
            fields.append(exact(x0, y0, cx, cy, R, device))
            print(f"  characteristics {i + 1}/{len(p)}  T={per:.3f}", flush=True)
        ex = torch.stack(fields)

    return (ex[..., 0].contiguous(), ex,
            torch.tensor(p[:, 2], device=device, dtype=torch.float32))


# ----------------------------------------------------------------------------
# Metrics -- identical definitions to the rotation benchmark so numbers are
# comparable across the two.
# ----------------------------------------------------------------------------

def rel_l2(pred, ref, reduce_time=True):
    num = torch.sqrt(((pred - ref) ** 2).sum(dim=(1, 2)))
    den = torch.sqrt((ref ** 2).sum(dim=(1, 2)))
    e = 100.0 * num / den
    return e.mean(dim=1) if reduce_time else e


def abs_l2(pred, ref):
    return torch.sqrt(((pred - ref) ** 2).mean(dim=(1, 2))).mean(dim=1)


def mass_mape(pred, ref, hx, hy):
    """Enclosed-area error against the reference field, counted the same way
    on both sides so the discretisation of the boundary cancels."""
    a = (pred < 0).float().sum(dim=(1, 2)) * hx * hy
    b = (ref < 0).float().sum(dim=(1, 2)) * hx * hy
    return (100.0 * (a - b).abs() / b.clamp_min(1e-12)).mean(dim=1)


def mass_drift(pred, hx, hy):
    """Area drift from the prediction's own t=0 slice. Reference-free: the
    vortex is divergence-free, so the area must equal its initial value at
    every time. Note this certifies conservation, not correctness -- a field
    that never moves scores zero."""
    a = (pred < 0).float().sum(dim=(1, 2)) * hx * hy
    return (100.0 * (a[:, 1:] - a[:, :1]).abs()
            / a[:, :1].clamp_min(1e-12)).mean(dim=1)


def eikonal_dev(pred, hx, hy):
    gx = (pred[2:, 1:-1] - pred[:-2, 1:-1]) / (2 * hx)
    gy = (pred[1:-1, 2:] - pred[1:-1, :-2]) / (2 * hy)
    return (torch.sqrt(gx ** 2 + gy ** 2 + 1e-12) - 1.0).abs().mean()
