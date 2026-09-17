"""
Neural operator for level-set advection on the RO family.

The question this answers, and the only one it answers:

    Trained without a single reference solution, does the operator produce
    usable predictions on initial conditions it has never seen?

Nothing here is a contribution. RO is rigid rotation -- the easiest case in
the whole benchmark suite, with an exact SDF solution for all time and no
deformation. If the answer is no, the deforming-flow version is not worth
attempting and eight months have been saved.

Usage
    python train_operator.py --steps 8000
    python train_operator.py --loss strong        # cross-check
    python train_operator.py --nx 32 --nt 16 --steps 200   # smoke test
"""

import argparse, json, time
import numpy as np
import torch

# The benchmark module is imported inside main(), after argparse has run.
# Importing it at module level would mean naming a fallback benchmark that is
# loaded on every run whether or not it is used -- which forces every drive to
# carry every family file, and fails with ModuleNotFoundError when one is
# absent.
import importlib
from residuals import (strong_loss, eikonal_loss, band_eik_dev,
                       eikonal_pointwise, SAW)
from fno3d import LevelSetOperator, build_features


def evaluate(model, phi0, feats, t_norm, phi_ex, h, fam, band=0.0, chunk=8):
    """Chunked so the test set can be large.

    The projection layer lifts to 128 channels, so one forward pass over B
    instances allocates B*nx*ny*nt*128 floats -- 6.25 GiB at B=100 on a
    64^2 x 32 grid, which OOMs a T4. Evaluation needs no graph, so splitting
    it costs nothing.
    """
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(phi0[i:i + chunk], feats[i:i + chunk], t_norm)
                          for i in range(0, phi0.shape[0], chunk)])
    model.train()
    return {
        "rel_l2_pct": fam.rel_l2(pred, phi_ex).cpu().numpy(),
        "abs_l2": fam.abs_l2(pred, phi_ex).cpu().numpy(),
        "mass_mape_pct": fam.mass_mape(pred, phi_ex, h[0], h[1]).cpu().numpy(),
        "drift_pct": fam.mass_drift(pred, h[0], h[1]).cpu().numpy(),
        "eik_dev": float(fam.eikonal_dev(pred[0], h[0], h[1])),
        "band_eik": band_eik_dev(pred, h, band) if band > 0 else float("nan"),
    }


def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fam = importlib.import_module(f"{a.benchmark}_family")
    make_grid, velocity_batch = fam.make_grid, fam.velocity_batch
    sample_family, build_dataset = fam.sample_family, fam.build_dataset
    torch.manual_seed(a.seed)
    print(f"device={dev}  grid={a.nx}x{a.ny}x{a.nt}  loss={a.loss}")

    X, Y, Tt, h = make_grid(a.nx, a.ny, a.nt, device=dev)
    t_norm = torch.linspace(0, 1, a.nt, device=dev)

    tr_p = sample_family(a.n_train, seed=a.seed, vary_v=a.vary_v)
    te_p = sample_family(a.n_test, seed=a.seed + 999, vary_v=a.vary_v)
    phi0_tr, ex_tr, _ = build_dataset(tr_p, a.nx, a.ny, a.nt, dev)
    phi0_te, ex_te, _ = build_dataset(te_p, a.nx, a.ny, a.nt, dev)
    # Per-instance velocity when --vary_v, one shared field otherwise. The
    # residual and the input channels must use the SAME field per instance or
    # the network is asked to satisfy a PDE it was not shown.
    u_tr, v_tr = velocity_batch(X, Y, Tt, tr_p, dev)
    u_te, v_te = velocity_batch(X, Y, Tt, te_p, dev)
    f_tr = build_features(phi0_tr, u_tr, v_tr, X, Y, Tt)
    f_te = build_features(phi0_te, u_te, v_te, X, Y, Tt)
    print(f"benchmark: {a.benchmark.upper()}"
          f"{'  (velocity varies per instance)' if a.vary_v else '  (velocity fixed)'}")
    if a.saw:
        print(f"SAW: q={a.saw_q}, beta={a.saw_beta}, init={a.saw_init}")
    if a.loss != "supervised":
        print(f"labels: {a.n_labelled}/{a.n_train} instances"
              f"{' (data-free)' if a.n_labelled == 0 else ''}"
              f"  lambda_data={a.lambda_data:g}")
    # The exact fields are held out of training entirely. They exist in this
    # script for one purpose: measuring whether the data-free run worked.

    model = LevelSetOperator(hard_ic=not a.soft_ic, width=a.width,
                             modes=(a.modes, a.modes, a.modes_t)).to(dev)
    print(f"IC: {'soft (w_ic=%g)' % a.w_ic if a.soft_ic else 'hard (structural)'}")
    n_par = sum(p.numel() for p in model.parameters())
    print(f"params={n_par/1e6:.2f}M  train={a.n_train}  test={a.n_test}")

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=1e-5)

    saw = SAW(list(model.parameters())[-2:], beta=a.saw_beta, q=a.saw_q,
              w_pde=a.kappa, init=a.saw_init) if a.saw else None
    saw_state, saw_raw = a.w_eik, float("nan")
    hist, t0 = [], time.time()
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, a.n_train, (a.batch,), device=dev)
        pred = model(phi0_tr[idx], f_tr[idx], t_norm)
        ub, vb = u_tr[idx], v_tr[idx]

        l_ic = None
        # Hybrid: physics on every instance, data on the labelled subset.
        # n_labelled = 0 reduces to data-free, n_labelled = n_train to
        # supervised-plus-physics. The subset is the FIRST n_labelled
        # instances, so the label sets are nested as n_labelled grows and the
        # sweep is a genuine curve rather than a set of unrelated runs.
        if a.loss == "supervised":
            # Control, not a method. Same architecture, same family, same
            # budget -- but trained on exact labels. This is the upper bound
            # the data-free run is being asked to approach, and the gap
            # between the two IS the price of refusing data. If supervised
            # reaches 1% and data-free stalls at 40%, the loss is the
            # problem, not the architecture, the capacity or the family.
            # Report both numbers in the paper whatever happens.
            l_pde = ((pred - ex_tr[idx]) ** 2).mean()
            l_eik = eikonal_loss(pred, h, band=a.band)
            loss = l_pde
        else:
            l_pde = strong_loss(pred, ub, vb, h, causal_eps=a.causal)
            l_data = None
            if a.n_labelled > 0:
                m = idx < a.n_labelled
                if m.any():
                    l_data = ((pred[m] - ex_tr[idx[m]]) ** 2).mean()
            if a.soft_ic:
                l_ic = ((pred[..., 0] - phi0_tr[idx]) ** 2).mean()
            if a.saw:
                res_eik = eikonal_pointwise(pred, h)
                # SAW's numerator is every term OTHER than the eikonal one --
                # gn_pde + gn_ic in the reference. With a data term present it
                # belongs there too, otherwise the eikonal weight is balanced
                # against only part of what is driving the update.
                other = None
                if l_ic is not None:
                    other = a.w_ic * l_ic
                if l_data is not None:
                    other = (a.lambda_data * l_data if other is None
                             else other + a.lambda_data * l_data)
                eik_term, saw_state, saw_raw = saw(l_pde, res_eik, other)
                l_eik = res_eik.mean().detach()
                loss = l_pde + eik_term
            else:
                l_eik = eikonal_loss(pred, h, band=a.band)
                loss = l_pde + a.w_eik * l_eik
            if l_ic is not None:
                loss = loss + a.w_ic * l_ic
            if l_data is not None:
                loss = loss + a.lambda_data * l_data
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % a.eval_every == 0 or step == a.steps:
            m_tr = evaluate(model, phi0_tr, f_tr, t_norm, ex_tr, h, fam, a.band)
            m_te = evaluate(model, phi0_te, f_te, t_norm, ex_te, h, fam, a.band)
            rec = {"step": step, "loss": float(loss.detach()), "pde": float(l_pde.detach()),
                   "eik": float(l_eik.detach()),
                   "train_rel_l2": float(m_tr["rel_l2_pct"].mean()),
                   "test_rel_l2": float(m_te["rel_l2_pct"].mean()),
                   "test_rel_l2_max": float(m_te["rel_l2_pct"].max()),
                   "test_mass_mape": float(m_te["mass_mape_pct"].mean()),
                   "test_drift": float(m_te["drift_pct"].mean()),
                   "band_eik": m_te["band_eik"],
                   "w_eik": float(saw_state if a.saw else a.w_eik),
                   "w_eik_raw": saw_raw,
                   "l_ic": float(l_ic.detach()) if l_ic is not None else float("nan"),
                   "l_data": (float(l_data.detach())
                              if a.loss != "supervised" and a.n_labelled > 0
                              and l_data is not None else float("nan")),
                   "test_eik_dev": m_te["eik_dev"],
                   "min": (time.time() - t0) / 60}
            hist.append(rec)
            wtag = f"| w_eik {rec['w_eik']:.3e} " if a.saw else ""
            btag = f"| bandeik {rec['band_eik']:.3f} " if a.band > 0 else ""
            btag += f"| ic {rec['l_ic']:.2e} " if a.soft_ic else ""
            btag += f"| data {rec['l_data']:.2e} " if a.n_labelled > 0 else ""
            print(f"{step:6d} loss {rec['loss']:.3e} | train {rec['train_rel_l2']:7.3f}% "
                  f"| test {rec['test_rel_l2']:7.3f}% | mass {rec['test_mass_mape']:6.2f}% "
                  f"| drift {rec['test_drift']:6.2f}% {btag}{wtag}| {rec['min']:.1f}m")

    final = evaluate(model, phi0_te, f_te, t_norm, ex_te, h, fam, a.band)
    print("\nheld-out, per instance (rel L2 %):",
          np.array2string(final["rel_l2_pct"], precision=3))
    print(f"mean {final['rel_l2_pct'].mean():.3f}%   "
          f"median {np.median(final['rel_l2_pct']):.3f}%   "
          f"worst {final['rel_l2_pct'].max():.3f}%")
    print(f"mass MAPE mean {final['mass_mape_pct'].mean():.3f}%   "
          f"drift mean {final['drift_pct'].mean():.3f}%")
    if a.saw:
        print(f"final w_eik {saw_state:.4e}  (clamp {a.kappa:.3g}, q={a.saw_q})")
        np.save(f"saw_whist_q{a.saw_q:g}_s{a.seed}.npy",
                np.asarray(saw.w_hist, dtype=np.float32))

    tag = (f"{'rv_' if a.benchmark == 'rv' else ''}{a.loss}{'_saw' if a.saw else ''}{'_varyv' if a.vary_v else ''}"
           f"{'_lab%d' % a.n_labelled if a.n_labelled > 0 else ''}"
           f"{'_saw0' if a.saw and a.saw_init == 'zero' else ''}"
           f"_n{a.n_train}_{a.steps // 1000}k"
           f"{'_softic' if a.soft_ic else ''}"
           f"{'_band' if a.band>0 else ''}_{a.nx}x{a.nt}_s{a.seed}")
    json.dump({"args": vars(a), "history": hist,
               "final_test_rel_l2": final["rel_l2_pct"].tolist(),
               "final_test_mass_mape": final["mass_mape_pct"].tolist()},
              open(f"operator_{tag}.json", "w"), indent=2)
    torch.save(model.state_dict(), f"operator_{tag}.pt")
    print(f"\nwrote operator_{tag}.json / .pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--nt", type=int, default=32)
    p.add_argument("--n_train", type=int, default=16)
    p.add_argument("--n_test", type=int, default=8)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=20)
    p.add_argument("--modes", type=int, default=12)
    p.add_argument("--modes_t", type=int, default=8)
    p.add_argument("--w_eik", type=float, default=1e-1)
    p.add_argument("--causal", type=float, default=0.0,
                   help="time weighting: >0 causal (early first), "
                        "<0 inverted (up-weight late slabs), 0 off")
    p.add_argument("--n_labelled", type=int, default=0,
                   help="how many training instances have reference solutions. "
                        "0 = data-free, n_train = supervised + physics, "
                        "anything between = hybrid")
    p.add_argument("--lambda_data", type=float, default=1.0)
    p.add_argument("--benchmark", default="ro",
                   help="name of the family module without the _family suffix; "
                        "a new benchmark needs only a new <name>_family.py")
    p.add_argument("--vary_v", action="store_true",
                   help="sample the velocity field per instance, making this "
                        "an operator over (phi_0, v) not phi_0 alone")
    p.add_argument("--soft_ic", action="store_true",
                   help="penalise the IC instead of enforcing it structurally")
    p.add_argument("--w_ic", type=float, default=10.0)
    p.add_argument("--band", type=float, default=0.0,
                   help="eikonal band half-width in phi units; 0 = global")
    p.add_argument("--saw", action="store_true",
                   help="set w_eik by the SAW gradient-norm ratio")
    p.add_argument("--kappa", type=float, default=1.0,
                   help="clamp on the raw ratio; w_pde in the reference")
    p.add_argument("--saw_q", type=float, default=0.95,
                   help="residual-quantile gate; 1.0 disables it (ablation)")
    p.add_argument("--saw_beta", type=float, default=0.999)
    p.add_argument("--saw_init", choices=["ratio", "zero"], default="ratio",
                   help="how the eikonal-weight EMA is seeded. 'ratio' is the "
                        "published behaviour and assumes a soft IC; 'zero' "
                        "lets the weight climb from 0, which is what a hard IC "
                        "needs because the field starts as a distance function")
    p.add_argument("--loss", choices=["strong", "supervised"], default="strong")
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
