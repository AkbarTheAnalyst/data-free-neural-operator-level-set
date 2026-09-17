r"""
Residual operators for level-set advection.

The transport residual is evaluated by second-order central differences on the
spacetime grid, and the eikonal constraint pointwise from the same stencils.
"""

import torch
import torch.nn.functional as F

def strong_residual(phi, u, v, h):
    """Second-order finite-difference collocation residual. Cross-check only."""
    hx, hy, ht = h
    p = phi[:, 1:-1, 1:-1, 1:-1]
    dt = (phi[:, 1:-1, 1:-1, 2:] - phi[:, 1:-1, 1:-1, :-2]) / (2 * ht)
    dx = (phi[:, 2:, 1:-1, 1:-1] - phi[:, :-2, 1:-1, 1:-1]) / (2 * hx)
    dy = (phi[:, 1:-1, 2:, 1:-1] - phi[:, 1:-1, :-2, 1:-1]) / (2 * hy)
    def _interior(w):
        if w.dim() == 4:                      # (B, nx, ny, nt), per instance
            return w[:, 1:-1, 1:-1, 1:-1]
        if w.dim() == 3:                      # (nx, ny, nt), shared
            return w[1:-1, 1:-1, 1:-1]
        return w
    ui, vi = _interior(u), _interior(v)
    del p
    return dt + ui * dx + vi * dy


def strong_loss(phi, u, v, h, causal_eps=0.0):
    """Collocation residual, optionally weighted along the time axis.

    causal_eps > 0 is standard causal weighting: a slab is down-weighted until
    the slabs before it are satisfied. causal_eps < 0 inverts it, up-weighting
    LATE slabs instead.

    The inverted sign is the one worth trying here. Causal weighting fixes a
    network that fits late times while early times are still wrong. That is not
    the observed failure -- error at t=1.42 is near zero and grows monotonically
    to t=6.28, so the ordering is already correct and the hard IC pins t=0
    exactly. Down-weighting late slabs would down-weight precisely where the
    error lives.

    Neither sign will fix truncation error in d_t if that is the cause, since
    reweighting cannot recover information the discretisation never had. Both
    are here to test that, cheaply, rather than to assume it.
    """
    r2 = strong_residual(phi, u, v, h) ** 2
    if causal_eps == 0.0:
        return r2.mean()
    slab = r2.mean(dim=(0, 1, 2))                          # (nt-2,)
    if causal_eps > 0:
        cum = torch.cat([torch.zeros(1, device=slab.device, dtype=slab.dtype),
                         torch.cumsum(slab, 0)[:-1]])
        w = torch.exp(-causal_eps * cum).detach()
    else:
        t = torch.linspace(0, 1, slab.shape[0], device=slab.device,
                           dtype=slab.dtype)
        w = torch.exp(-causal_eps * t).detach()            # eps<0 -> grows in t
        w = w / w.mean()
    return (w * slab).mean()


def eikonal_loss(phi, h, band=0.0):
    """(|grad phi| - 1)^2, optionally restricted to a band around the interface.

    band = 0 gives the global term: the constraint is enforced with equal
    weight over the whole domain. Every configuration tried so far fails with
    it. Fixed w_eik = 1e-1 froze training at the trivial solution; w_eik = 0
    and the SAW ratio (which drove the weight to 1e-10) both converged to a
    field with 2.6% L2 error and 261% mass error. A supervised L2 loss on 470
    samples produced 95-98% area error. The interface dies in all three.

    The global term is doing two unrelated jobs at once. Away from the
    interface it asks phi to stay a distance function, which is cosmetic --
    nothing downstream reads the far field. Near the interface it is the only
    thing preventing |grad phi| -> 0, which is what destroys the zero contour.
    A single scalar weight cannot separate those, and that is why no value of
    it works: large enough to protect the interface is large enough to freeze
    the far field, and small enough to free the far field abandons the
    interface.

    Restricting to a band separates them. The weight

        w = exp(-(phi/band)^2)

    concentrates the constraint where the level set lives and leaves the far
    field free to deform, which is what transport needs.

    Two details that matter:

    - w is detached. Left attached, the network can reduce the loss by moving
      the band away from where the constraint is violated rather than by
      satisfying it -- optimising the mask instead of the physics.
    - The loss is a weighted mean, sum(w r) / sum(w), not mean(w r). Otherwise
      narrowing the band shrinks the loss on its own and the band width
      silently becomes a second weight.
    """
    hx, hy, _ = h
    gx = (phi[:, 2:, 1:-1, :] - phi[:, :-2, 1:-1, :]) / (2 * hx)
    gy = (phi[:, 1:-1, 2:, :] - phi[:, 1:-1, :-2, :]) / (2 * hy)
    r = (torch.sqrt(gx ** 2 + gy ** 2 + 1e-12) - 1.0) ** 2
    if band <= 0.0:
        return r.mean()
    w = torch.exp(-(phi[:, 1:-1, 1:-1, :] / band) ** 2).detach()
    return (w * r).sum() / (w.sum() + 1e-12)


def band_eik_dev(phi, h, band):
    """Mean | |grad phi| - 1 | inside the band. Reference-free diagnostic:
    this is the quantity that tracks whether the level set is surviving."""
    hx, hy, _ = h
    gx = (phi[:, 2:, 1:-1, :] - phi[:, :-2, 1:-1, :]) / (2 * hx)
    gy = (phi[:, 1:-1, 2:, :] - phi[:, 1:-1, :-2, :]) / (2 * hy)
    d = (torch.sqrt(gx ** 2 + gy ** 2 + 1e-12) - 1.0).abs()
    w = torch.exp(-(phi[:, 1:-1, 1:-1, :] / max(band, 1e-6)) ** 2)
    return float((w * d).sum() / (w.sum() + 1e-12))


def eikonal_pointwise(phi, h):
    """Pointwise squared eikonal residual (|grad phi| - 1)^2, interior only.

    Returned as a field rather than a scalar because SAW's gate needs the
    per-point distribution to take a quantile of.
    """
    hx, hy, _ = h
    gx = (phi[:, 2:, 1:-1, :] - phi[:, :-2, 1:-1, :]) / (2 * hx)
    gy = (phi[:, 1:-1, 2:, :] - phi[:, 1:-1, :-2, :]) / (2 * hy)
    return (torch.sqrt(gx ** 2 + gy ** 2 + 1e-12) - 1.0) ** 2


class SAW:
    """SDF-Aware Weighting, ported faithfully from ro3d_core.SAW.

    Gate and ratio, in that order, both required. An earlier version here kept
    only the ratio, on the reasoning that the published negative controls show
    the gate hurting on rigid rotation. That was wrong: those controls are
    about final accuracy, not about whether the ratio can find a usable weight
    without the gate. The RO3D sweep settles it -- gate off pins w_eik at
    1.000 (the clamp) for every seed; gate on settles at 0.179.

    The mechanism is indirect, which is what makes it easy to miss. gn_eik is
    computed from the UNGATED mean, so at any single step the gate does not
    enter the ratio at all. It acts through the training dynamics: with the
    gate on, the tail points are never optimised, their residuals stay large,
    mean(res_eik) stays large, gn_eik stays large, and the ratio stays low.
    With the gate off the network crushes every eikonal residual including the
    tail, gn_eik collapses, and the ratio saturates at its clamp.

    That is why the applied loss uses the gated mean while the ratio uses the
    ungated one. Computing both from the gated term would break the feedback.

    One forced deviation from the reference: the numerator there is
    gn_pde + gn_ic. Here the initial condition is hard-constrained, so there is
    no L_ic and no gn_ic, and the numerator is gn_pde alone. This matters --
    with w_ic = 10 the reference numerator is dominated by the IC term, so the
    operator's ratio is not on the same scale as the PINN's, and w_eik values
    are not directly comparable between them.
    """

    def __init__(self, gn_params, beta=0.999, q=0.95, w_pde=1.0,
                 thresh_beta=0.99, eps=1e-8, init="ratio"):
        """init selects how the weight EMA is seeded.

        "ratio" is the published behaviour: the EMA starts at the first raw
        ratio. That assumes a soft initial condition, where the network is
        randomly initialised and phi is nothing like a distance function, so
        g_eik is large from the first step and the ratio is informative.

        "zero" starts the EMA at 0 and lets the weight climb. Under a HARD
        initial condition the field starts exactly at phi_0, which IS a
        distance function -- its eikonal residual is ~1e-4 -- so g_eik starts
        near zero, the raw ratio saturates, and "ratio" seeds the EMA at the
        clamp. It then needs O(1/(1-beta)) steps to descend, and on a
        deforming flow the eikonal term dominates throughout that window and
        drives the field toward a distance function the truth is not.

        Seeding at zero inverts the failure mode into a safe one: the weight
        is too small early, when the field has barely moved and the eikonal
        term carries little information anyway, and rises as the field
        genuinely departs from phi_0. The asymmetry is the point -- starting
        low costs a slow start, starting high costs the solution.
        """
        self.gn_params, self.beta, self.q = gn_params, beta, q
        self.w_pde, self.tb, self.eps = w_pde, thresh_beta, eps
        self.init = init
        self.thresh, self.ema_w, self.w_hist = None, None, []

    def _gn(self, loss):
        g = torch.autograd.grad(loss, self.gn_params, retain_graph=True,
                                allow_unused=True)
        return sum(x.norm() ** 2 for x in g if x is not None).sqrt()

    def __call__(self, l_pde, res_eik, l_ic_weighted=None):
        """Returns (weighted gated eikonal term, w_eik, raw ratio).

        l_ic_weighted is w_ic * L_ic, matching the reference, which takes the
        gradient norm of the ALREADY-WEIGHTED IC loss. Pass None under a hard
        initial condition; the numerator is then g_pde alone.
        """
        gn_pde = self._gn(l_pde).detach()
        gn_eik = self._gn(res_eik.mean()).detach()        # ungated, deliberately
        gn_ic = (self._gn(l_ic_weighted).detach()
                 if l_ic_weighted is not None else 0.0)

        with torch.no_grad():
            raw = torch.clamp((gn_pde + gn_ic) / (gn_eik + self.eps),
                              0.0, self.w_pde)
            if self.ema_w is None:
                self.ema_w = torch.zeros_like(raw) if self.init == "zero" else raw
            self.ema_w = self.beta * self.ema_w + (1 - self.beta) * raw
            w = float(self.ema_w)

            cq = torch.quantile(res_eik.detach().flatten().float(), self.q)
            self.thresh = cq if self.thresh is None else \
                self.tb * self.thresh + (1 - self.tb) * cq
            mask = (res_eik.detach() <= self.thresh).float()
            self.w_hist.append(w)

        return w * (mask * res_eik).mean(), w, float(raw)
