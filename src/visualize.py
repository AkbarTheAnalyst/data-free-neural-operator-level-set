"""
Visual comparison of trained checkpoints.

Produces three figures:

  1. interfaces_<tag>.png  -- predicted vs exact zero contour at five times,
     one row per model. This is the plot that matters: relative L2 and mass
     MAPE are summaries, and a 5% field error can mean a slightly displaced
     interface or a smeared one. Only the contour tells you which.

  2. fields_<tag>.png -- phi field and signed error at the final time, so
     you can see WHERE the error lives (interface band vs far field).

  3. spread_<tag>.png -- per-instance error over the whole test set, sorted.
     A mean of 5.4% hides whether every instance is at 5% or half are at 3%
     and half at 8%.

IMPORTANT -- comparing two models fairly. sample_family(n, seed) is not
nested: the first 8 draws of sample_family(100, ...) are NOT sample_family(8,
...). A model trained with --n_test 8 and one with --n_test 100 therefore have
completely different test sets. Pass --n_test explicitly here so both models
are evaluated on the same instances, whatever they were trained with.

Usage
    python visualize.py strong_saw_64x32_s0 --n_test 100
    python visualize.py strong_saw_64x32_s0 supervised_64x32_s0 --n_test 100
"""

import argparse, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import importlib
_fam = None  # bound in main() from the checkpoint's own config
from fno3d import LevelSetOperator, build_features

def label(cfg):
    """Arm name from the config. Hybrid runs use loss='strong' too, so keying
    on the loss alone would label them as physics-only."""
    if cfg["loss"] == "supervised":
        return "supervised"
    n = cfg.get("n_labelled", 0)
    return "physics only" if n == 0 else f"hybrid ({n} labels)"


DEFAULTS = dict(nx=64, ny=64, nt=32, n_train=16, n_test=8, width=20,
                modes=12, modes_t=8, seed=0, soft_ic=False, band=0.0)


def read_args(tag):
    """Config for a run, from its JSON if present.

    The JSON is written immediately before the .pt, so a checkpoint can
    outlive its config if the filesystem drops between the two writes --
    which Colab's Drive mount does. Fall back to the training defaults and
    infer the loss from the tag, then say so loudly: if the run used
    non-default width or modes, load_state_dict will fail with a shape
    mismatch rather than silently loading the wrong model.
    """
    try:
        return json.load(open(f"operator_{tag}.json"))["args"]
    except FileNotFoundError:
        a = dict(DEFAULTS)
        a["loss"] = ("supervised" if "supervised" in tag
                     else "strong")
        a["soft_ic"] = "softic" in tag
        for part in tag.split("_"):
            if "x" in part and part.replace("x", "").isdigit():
                nx, nt = part.split("x")
                a["nx"] = a["ny"] = int(nx); a["nt"] = int(nt)
            if part.startswith("s") and part[1:].isdigit():
                a["seed"] = int(part[1:])
        print(f"  ! operator_{tag}.json missing; assuming training defaults "
              f"(nx={a['nx']}, nt={a['nt']}, width={a['width']}, "
              f"modes={a['modes']}, seed={a['seed']})")
        return a


def load(tag, dev):
    a = read_args(tag)
    m = LevelSetOperator(hard_ic=not a.get("soft_ic", False), width=a["width"],
                         modes=(a["modes"], a["modes"], a["modes_t"])).to(dev)
    m.load_state_dict(torch.load(f"operator_{tag}.pt", map_location=dev))
    m.eval()
    return m, a


def predict(model, phi0, feats, t_norm, chunk=8):
    with torch.no_grad():
        return torch.cat([model(phi0[i:i + chunk], feats[i:i + chunk], t_norm)
                          for i in range(0, phi0.shape[0], chunk)])


def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [load(t, dev) for t in a.tags]
    cfg = models[0][1]
    fam = importlib.import_module(f'{cfg.get("benchmark", "ro")}_family')
    make_grid, velocity_batch = fam.make_grid, fam.velocity_batch
    sample_family, build_dataset = fam.sample_family, fam.build_dataset
    rel_l2, T_FINAL = fam.rel_l2, fam.T_FINAL

    # All models must share the evaluation grid. An FNO will happily run on a
    # grid it was not trained on -- rfftn just returns more modes and the
    # slicing still succeeds -- so a mismatch produces plausible-looking
    # numbers that mean nothing. Refuse instead.
    for t, (_, c) in zip(a.tags, models):
        if (c["nx"], c["ny"], c["nt"]) != (cfg["nx"], cfg["ny"], cfg["nt"]):
            raise SystemExit(
                f"grid mismatch: {a.tags[0]} is "
                f"{cfg['nx']}x{cfg['ny']}x{cfg['nt']} but {t} is "
                f"{c['nx']}x{c['ny']}x{c['nt']}.\nTrain both at the same "
                f"resolution before comparing, or plot them separately.")
    nx, ny, nt = cfg["nx"], cfg["ny"], cfg["nt"]

    X, Y, Tt, h = make_grid(nx, ny, nt, device=dev)
    t_norm = torch.linspace(0, 1, nt, device=dev)

    p = sample_family(a.n_test, seed=cfg["seed"] + 999,
                      vary_v=cfg.get("vary_v", False))
    phi0, ex, R = build_dataset(p, nx, ny, nt, dev)
    uu, vv = velocity_batch(X, Y, Tt, p, dev)
    feats = build_features(phi0, uu, vv, X, Y, Tt)

    preds, errs = [], []
    for m, _ in models:
        pr = predict(m, phi0, feats, t_norm)
        preds.append(pr)
        errs.append(rel_l2(pr, ex, reduce_time=False).mean(dim=1).cpu().numpy())

    # Instance to plot: the median performer of the first model, unless told.
    idx = a.instance if a.instance >= 0 else int(np.argsort(errs[0])[len(errs[0]) // 2])
    print(f"plotting instance {idx}: p={np.round(p[idx], 3)}"
          + (f" omega={p[idx,3]:.3f}" if p.shape[1] > 3 else ""))
    for (m, c), e in zip(models, errs):
        print(f"  {label(c):<22} "
              f"mean {e.mean():.3f}%  this instance {e[idx]:.3f}%")

    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    ti = np.linspace(0, nt - 1, 5).astype(int)
    exi = ex[idx].cpu().numpy()

    # --- 1. interface contours ------------------------------------------
    fig, axes = plt.subplots(len(models), 5, figsize=(16, 3.3 * len(models)),
                             squeeze=False)
    for r, ((m, c), pr) in enumerate(zip(models, preds)):
        pi = pr[idx].cpu().numpy()
        for k, tk in enumerate(ti):
            ax = axes[r][k]
            ax.contour(xs, ys, exi[:, :, tk].T, [0.0], colors="k",
                       linewidths=2.2, linestyles="--")
            ax.contour(xs, ys, pi[:, :, tk].T, [0.0], colors="crimson",
                       linewidths=1.8)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t = {tk / (nt - 1) * T_FINAL:.2f}", fontsize=11)
            if k == 0:
                ax.set_ylabel(label(c), fontsize=11)
    fig.suptitle("zero level set:  exact (black dashed)  vs  predicted (red)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(f"interfaces_{a.tags[0]}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"interfaces_{a.tags[0]}.pdf", bbox_inches="tight")

    # --- 2. field and error at final time --------------------------------
    fig, axes = plt.subplots(len(models), 3, figsize=(13, 3.8 * len(models)),
                             squeeze=False)
    for r, ((m, c), pr) in enumerate(zip(models, preds)):
        pi = pr[idx].cpu().numpy()
        d = pi[:, :, -1] - exi[:, :, -1]
        lim = float(np.abs(d).max()) or 1e-6
        for k, (fld, ttl, cm, kw) in enumerate([
                (exi[:, :, -1], "exact $\\phi$", "RdBu_r", {}),
                (pi[:, :, -1], "predicted $\\phi$", "RdBu_r", {}),
                (d, "error", "coolwarm", dict(vmin=-lim, vmax=lim))]):
            ax = axes[r][k]
            im = ax.pcolormesh(xs, ys, fld.T, cmap=cm, shading="auto", **kw)
            ax.contour(xs, ys, exi[:, :, -1].T, [0.0], colors="k",
                       linewidths=1.4, linestyles="--")
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(ttl, fontsize=11)
            if k == 0:
                ax.set_ylabel(label(c), fontsize=11)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"final time, t = {T_FINAL:.2f}", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"fields_{a.tags[0]}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"fields_{a.tags[0]}.pdf", bbox_inches="tight")

    # --- 3. error spread across the test set -----------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for (m, c), e in zip(models, errs):
        lab = label(c)
        ax1.plot(np.sort(e), lw=1.8, label=f"{lab}  (mean {e.mean():.2f}%)")
        ax2.hist(e, bins=20, alpha=0.55, label=lab)
    ax1.set_xlabel("instance (sorted)"); ax1.set_ylabel("relative $L^2$ (%)")
    ax1.set_title(f"per-instance error, {a.n_test} held-out instances")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
    ax2.set_xlabel("relative $L^2$ (%)"); ax2.set_ylabel("count")
    ax2.set_title("distribution"); ax2.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"spread_{a.tags[0]}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"spread_{a.tags[0]}.pdf", bbox_inches="tight")

    print(f"\nwrote interfaces_{a.tags[0]}, fields_{a.tags[0]}, "
          f"spread_{a.tags[0]}  (.png and .pdf)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("tags", nargs="+", help="checkpoint tags, first is primary")
    p.add_argument("--n_test", type=int, default=100,
                   help="MUST match across models being compared")
    p.add_argument("--instance", type=int, default=-1,
                   help="which test instance to plot; default is the median")
    main(p.parse_args())
