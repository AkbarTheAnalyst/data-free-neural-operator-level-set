"""
Solid-body rotation (RO) benchmark, lifted from a single instance to a family.

Velocity field (fixed, divergence-free, rigid):
    u = -(y - 0.5),   v = (x - 0.5),   T = 2*pi   (one full revolution)

A circle of radius R whose centre sits at orbit radius rho and phase alpha
about (0.5, 0.5) stays a circle for all time; its centre simply rotates.
The exact solution is therefore a signed distance function everywhere and for
all t, which is what makes RO the right first gate: no eikonal ambiguity, no
integrator, no discretisation error in the reference.

    xc(t) = 0.5 + (cx-0.5) cos t - (cy-0.5) sin t
    yc(t) = 0.5 + (cx-0.5) sin t + (cy-0.5) cos t
    phi_ref(x,y,t) = sqrt((x-xc)^2 + (y-yc)^2) - R

Setting (rho, alpha) = (0.25, pi/2), R = 0.15 recovers the RO benchmark of
Khan & Raees exactly (centre (0.5, 0.75), xc = 0.5 - 0.25 sin t,
yc = 0.5 + 0.25 cos t).

The family varies (rho, alpha, R). Note that alpha is a symmetry of the flow
(rotating the initial condition just shifts t), so it is a deliberately easy
axis; R is the axis that carries genuine geometric variation. This is a gate,
not a hard test -- if the operator cannot do this, nothing downstream matters.
"""

import numpy as np
import torch

T_FINAL = 2.0 * np.pi

# Family ranges. Constraint: |centre jitter| + rho + R < 0.5 keeps the circle
# clear of dOmega. At the extremes: 0.02 + 0.26 + 0.18 = 0.46.
RHO_RANGE = (0.18, 0.26)
R_RANGE = (0.10, 0.18)

# Velocity-field ranges. omega scales the angular velocity, so over the fixed
# horizon T = 2*pi an instance completes omega revolutions -- this is the axis
# that actually changes how far the interface travels. The rotation centre
# jitter is small by necessity (the orbit must stay inside the domain) but it
# makes the velocity field genuinely different pointwise, not just rescaled.
OMEGA_RANGE = (0.6, 1.4)
CENTRE_JITTER = 0.02


def make_grid(nx, ny, nt, device="cpu", dtype=torch.float32):
    """Uniform tensor-product grid on [0,1]^2 x [0,T]. Returns (X, Y, Tt) each
    of shape (nx, ny, nt), plus the spacings."""
    x = torch.linspace(0.0, 1.0, nx, device=device, dtype=dtype)
    y = torch.linspace(0.0, 1.0, ny, device=device, dtype=dtype)
    t = torch.linspace(0.0, T_FINAL, nt, device=device, dtype=dtype)
    X, Y, Tt = torch.meshgrid(x, y, t, indexing="ij")
    hx = float(x[1] - x[0])
    hy = float(y[1] - y[0])
    ht = float(t[1] - t[0])
    return X, Y, Tt, (hx, hy, ht)


def velocity(X, Y, omega=1.0, cx0=0.5, cy0=0.5):
    """Rotation field about (cx0, cy0) at angular velocity omega.

    Divergence-free and steady for any parameter choice, so the enclosed area
    is still conserved exactly and mass_drift remains a valid reference-free
    diagnostic.

    omega, cx0, cy0 may be scalars (one field) or tensors of shape (B, 1, 1, 1)
    (one field per instance, broadcast over the grid).
    """
    return -omega * (Y - cy0), omega * (X - cx0)


def velocity_batch(X, Y, Tt, params, device="cpu"):
    """Per-instance velocity fields for a family. Returns (B, nx, ny, nt) each.

    Tt is accepted but unused: the rotation field is steady. The argument
    exists so this and rv_family.velocity_batch have the same signature and
    the training script needs no branch on which benchmark is loaded.
    """
    p = torch.as_tensor(params, dtype=torch.float32, device=device)
    om = p[:, 3].view(-1, 1, 1, 1)
    cx0 = p[:, 4].view(-1, 1, 1, 1)
    cy0 = p[:, 5].view(-1, 1, 1, 1)
    return velocity(X[None], Y[None], om, cx0, cy0)


def exact(X, Y, Tt, rho, alpha, R, omega=1.0, cx0=0.5, cy0=0.5):
    """Exact level-set field for one family member, on the full spacetime grid.

    The circle orbits the rotation centre (cx0, cy0) at angular velocity omega,
    so the rotation angle is omega*t rather than t. Still closed-form: rigid
    rotation maps a circle to a circle for any omega.
    """
    cx = cx0 + rho * np.cos(alpha)
    cy = cy0 + rho * np.sin(alpha)
    dx0, dy0 = cx - cx0, cy - cy0
    th = omega * Tt
    xc = cx0 + dx0 * torch.cos(th) - dy0 * torch.sin(th)
    yc = cy0 + dx0 * torch.sin(th) + dy0 * torch.cos(th)
    return torch.sqrt((X - xc) ** 2 + (Y - yc) ** 2) - R


def sample_family(n, seed, vary_v=False, rho_range=RHO_RANGE, r_range=R_RANGE):
    """Latin hypercube sample of the family. Returns (n, 6):
    (rho, alpha, R, omega, cx0, cy0).

    vary_v=False fixes omega=1, centre=(0.5,0.5), reproducing the
    initial-condition-only family exactly -- the first three columns are
    IDENTICAL to what the earlier three-column version produced for the same
    seed and n, so previously trained checkpoints remain comparable.

    vary_v=True additionally samples the velocity field, making this an
    operator over (phi_0, v) rather than over phi_0 alone.
    """
    rng = np.random.default_rng(seed)

    def strat(lo, hi):
        edges = (np.arange(n) + rng.random(n)) / n
        return lo + (hi - lo) * rng.permutation(edges)

    # Drawn first and in this order so the IC columns match the old sampler.
    rho = strat(*rho_range)
    R = strat(*r_range)
    alpha = strat(0.0, 2.0 * np.pi)

    if vary_v:
        omega = strat(*OMEGA_RANGE)
        cx0 = strat(0.5 - CENTRE_JITTER, 0.5 + CENTRE_JITTER)
        cy0 = strat(0.5 - CENTRE_JITTER, 0.5 + CENTRE_JITTER)
    else:
        omega = np.ones(n)
        cx0 = np.full(n, 0.5)
        cy0 = np.full(n, 0.5)
    return np.stack([rho, alpha, R, omega, cx0, cy0], axis=1)


def build_dataset(params, nx, ny, nt, device="cpu"):
    """Assemble inputs and exact references for a set of family members.

    Returns
        phi0 : (B, nx, ny)          initial field
        phi_ex : (B, nx, ny, nt)    exact spacetime solution (evaluation only)
        R : (B,)                    radii, for the mass metric
    """
    X, Y, Tt, _ = make_grid(nx, ny, nt, device=device)
    phi_ex, phi0, Rs = [], [], []
    for rho, alpha, R, om, cx0, cy0 in params:
        pe = exact(X, Y, Tt, rho, alpha, R, om, cx0, cy0)
        phi_ex.append(pe)
        phi0.append(pe[..., 0])
        Rs.append(R)
    return (torch.stack(phi0), torch.stack(phi_ex),
            torch.tensor(Rs, device=device, dtype=torch.float32))


# ----------------------------------------------------------------------------
# Metrics. These follow Khan & Raees so the numbers stay comparable to the
# published PINN results.
# ----------------------------------------------------------------------------

def rel_l2(pred, ref, reduce_time=True):
    """Relative L2 error in percent, per time slice. pred/ref: (B, nx, ny, nt)."""
    num = torch.sqrt(((pred - ref) ** 2).sum(dim=(1, 2)))
    den = torch.sqrt((ref ** 2).sum(dim=(1, 2)))
    e = 100.0 * num / den                      # (B, nt)
    return e.mean(dim=1) if reduce_time else e


def abs_l2(pred, ref):
    """RMS field error, averaged over time. (B,)"""
    return torch.sqrt(((pred - ref) ** 2).mean(dim=(1, 2))).mean(dim=1)


def mass_mape(pred, ref, hx, hy):
    """Enclosed-area error in percent against the reference field, averaged
    over time.

    Counted the same way on both sides -- cells with phi < 0 on the evaluation
    grid -- so the discretisation of the boundary cancels. Comparing a cell
    count against the analytic pi*R^2 does NOT cancel it: at R ~ 0.12 on a 64^2
    grid the circle is ~185 cells, the boundary layer is a large fraction of
    that, and the reference field itself fails the test. Any version of this
    metric must return ~0 for the exact solution; check that before trusting a
    number it produces.
    """
    a_pred = (pred < 0).float().sum(dim=(1, 2)) * hx * hy
    a_ref = (ref < 0).float().sum(dim=(1, 2)) * hx * hy
    return (100.0 * (a_pred - a_ref).abs() / a_ref.clamp_min(1e-12)).mean(dim=1)


def mass_drift(pred, hx, hy):
    """Area drift from the prediction's own t=0 slice, in percent.

    Reference-free: the velocity field is divergence-free, so the enclosed area
    must equal its own initial value at every time, and the hard IC makes the
    t=0 slice exactly phi0. Computable at inference on an unseen instance.

    Note the failure mode this metric has by construction -- a field that never
    moves scores a perfect zero. It certifies conservation, not correctness,
    and cannot on its own tell a good solution from a frozen one.
    """
    a = (pred < 0).float().sum(dim=(1, 2)) * hx * hy          # (B, nt)
    a0 = a[:, :1]
    return (100.0 * (a[:, 1:] - a0).abs() / a0.clamp_min(1e-12)).mean(dim=1)


def eikonal_dev(pred, hx, hy):
    """Mean | |grad phi| - 1 | over the interior. Also reference-free."""
    gx = (pred[2:, 1:-1] - pred[:-2, 1:-1]) / (2 * hx)
    gy = (pred[1:-1, 2:] - pred[1:-1, :-2]) / (2 * hy)
    return (torch.sqrt(gx ** 2 + gy ** 2 + 1e-12) - 1.0).abs().mean()
