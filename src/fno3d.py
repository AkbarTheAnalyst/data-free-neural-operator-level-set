"""
Spacetime FNO for level-set advection.

Two design choices worth stating, because both are load-bearing.

SPACETIME, NOT AUTOREGRESSIVE. The operator emits the entire trajectory in one
forward pass, so the residual is evaluated over Omega x [0,T] at once. An
autoregressive stepper would have to backpropagate through the whole rollout to
do the same, which is expensive and unstable, and it accumulates error across
steps.

HARD INITIAL CONDITION. The output is

    phi = phi0 + t_norm * N_theta[phi0, v]

so phi(.,0) = phi0 exactly, by construction. In a data-free method the initial
condition is the only anchor to the truth that exists. As a soft penalty it
competes with the residual, and the optimiser can buy residual reduction by
drifting off the initial data -- the transport equation is perfectly happy with
a solution that starts somewhere else. Hard-constraining it removes that
failure mode entirely and leaves the residual with exactly one job.
"""

import torch
import torch.nn as nn


class SpectralConv3d(nn.Module):
    def __init__(self, cin, cout, m1, m2, m3):
        super().__init__()
        self.m1, self.m2, self.m3 = m1, m2, m3
        scale = 1.0 / (cin * cout)
        # Four corner blocks: rfftn halves the last axis, so (+/-m1, +/-m2, +m3).
        self.w = nn.ParameterList([
            nn.Parameter(scale * torch.rand(cin, cout, m1, m2, m3, dtype=torch.cfloat))
            for _ in range(4)])

    @staticmethod
    def _mul(a, b):
        return torch.einsum("bixyz,ioxyz->boxyz", a, b)

    def forward(self, x):
        B, C, nx, ny, nt = x.shape
        xf = torch.fft.rfftn(x, dim=(-3, -2, -1))
        out = torch.zeros(B, self.w[0].shape[1], nx, ny, nt // 2 + 1,
                          dtype=torch.cfloat, device=x.device)
        m1, m2, m3 = self.m1, self.m2, self.m3
        out[:, :, :m1, :m2, :m3] = self._mul(xf[:, :, :m1, :m2, :m3], self.w[0])
        out[:, :, -m1:, :m2, :m3] = self._mul(xf[:, :, -m1:, :m2, :m3], self.w[1])
        out[:, :, :m1, -m2:, :m3] = self._mul(xf[:, :, :m1, -m2:, :m3], self.w[2])
        out[:, :, -m1:, -m2:, :m3] = self._mul(xf[:, :, -m1:, -m2:, :m3], self.w[3])
        return torch.fft.irfftn(out, s=(nx, ny, nt), dim=(-3, -2, -1))


class FNO3d(nn.Module):
    """Input channels: phi0 (broadcast in t), u, v, x, y, t_norm."""

    def __init__(self, width=20, modes=(12, 12, 8), n_layers=4, cin=6):
        super().__init__()
        self.lift = nn.Linear(cin, width)
        self.spec = nn.ModuleList(
            [SpectralConv3d(width, width, *modes) for _ in range(n_layers)])
        self.pw = nn.ModuleList(
            [nn.Conv3d(width, width, 1) for _ in range(n_layers)])
        self.proj = nn.Sequential(nn.Linear(width, 128), nn.GELU(),
                                  nn.Linear(128, 1))

    def forward(self, feats):
        # feats: (B, nx, ny, nt, cin)
        x = self.lift(feats).permute(0, 4, 1, 2, 3)
        for i, (s, p) in enumerate(zip(self.spec, self.pw)):
            z = s(x) + p(x)
            x = torch.nn.functional.gelu(z) if i < len(self.spec) - 1 else z
        return self.proj(x.permute(0, 2, 3, 4, 1)).squeeze(-1)


class LevelSetOperator(nn.Module):
    """Wraps FNO3d with the hard initial-condition constraint."""

    def __init__(self, hard_ic=True, **kw):
        super().__init__()
        self.net = FNO3d(**kw)
        self.hard_ic = hard_ic

    def forward(self, phi0, feats, t_norm):
        # phi0: (B, nx, ny); feats: (B, nx, ny, nt, cin); t_norm: (nt,) in [0,1]
        out = self.net(feats)
        if not self.hard_ic:
            # Soft IC: the network is free at t=0 and an L_ic penalty supplies
            # the anchor. Needed to test SAW with its full numerator, since
            # g_ic only exists if there is an L_ic to differentiate.
            return out
        return phi0[..., None] + t_norm.view(1, 1, 1, -1) * out


def build_features(phi0, u, v, X, Y, Tt):
    """Assemble the input channel stack.

    X, Y, Tt are (nx, ny, nt). u and v may be either (nx, ny, nt) -- one shared
    velocity field -- or (B, nx, ny, nt) -- one field per instance, which is
    what makes this an operator over (phi_0, v) rather than over phi_0 alone.
    """
    B, nt = phi0.shape[0], Tt.shape[-1]
    coords = torch.stack([X, Y, Tt / Tt.max()], dim=-1)           # (nx,ny,nt,3)
    coords = coords[None].expand(B, *coords.shape)
    if u.dim() == 3:
        u = u[None].expand(B, *u.shape)
        v = v[None].expand(B, *v.shape)
    p0 = phi0[..., None, None].expand(-1, -1, -1, nt, 1)
    return torch.cat([p0, u[..., None], v[..., None], coords], dim=-1)
