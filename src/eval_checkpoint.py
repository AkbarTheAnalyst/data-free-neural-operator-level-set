"""
Full metrics for a saved run, on BOTH the training and held-out instances.

Why this exists: the training printout reported field error from the training
set and mass error from the test set, side by side, with no indication that
they came from different instances. At n_train=1 those are different circles
and the pairing is meaningless -- a 1.81% training solution was read as having
290% mass error, which was the test instance's number.

Also reports Khan & Raees's mass metric (cell count against the analytic
pi*R^2) alongside the reference-cell version, so numbers here can be compared
directly with the published PINN results. The two differ at coarse
resolution: on the exact solution the analytic version scores 0.910% at
n=64 and 0.136% at n=300, which is the discretisation floor, not error.

Usage
    python eval_checkpoint.py strong_saw_64x32_s0
    python eval_checkpoint.py supervised_n16_40k_64x32_s0
"""

import argparse, json
import numpy as np
import torch

import importlib
_fam = None  # bound in main() from the checkpoint's own config
from fno3d import LevelSetOperator, build_features


def analytic_mape(pred, R, hx, hy):
    """Khan & Raees: enclosed-area error against pi*R^2."""
    a = (pred < 0).float().sum(dim=(1, 2)) * hx * hy
    ex = np.pi * R[:, None] ** 2
    return (100.0 * (a - ex).abs() / ex).mean(dim=1)


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


def main(tag, n_test_override=None, chunk=8):
    a = read_args(tag)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fam = importlib.import_module(
        "rv_family" if a.get("benchmark", "ro") == "rv" else "ro_family")
    make_grid, velocity_batch = fam.make_grid, fam.velocity_batch
    sample_family, build_dataset = fam.sample_family, fam.build_dataset
    rel_l2, abs_l2 = fam.rel_l2, fam.abs_l2
    mass_mape, mass_drift = fam.mass_mape, fam.mass_drift

    X, Y, Tt, h = make_grid(a["nx"], a["ny"], a["nt"], device=dev)
    t_norm = torch.linspace(0, 1, a["nt"], device=dev)

    model = LevelSetOperator(hard_ic=not a.get("soft_ic", False),
                             width=a["width"],
                             modes=(a["modes"], a["modes"], a["modes_t"])).to(dev)
    model.load_state_dict(torch.load(f"operator_{tag}.pt", map_location=dev))
    model.eval()
    n_te = n_test_override if n_test_override is not None else a["n_test"]

    print(f"\n{tag}   loss={a['loss']}  "
          f"IC={'soft' if a.get('soft_ic') else 'hard'}  "
          f"n_train={a['n_train']}  grid={a['nx']}^2 x {a['nt']}  "
          f"bench={a.get('benchmark', 'ro').upper()}  "
          f"v={'varying' if a.get('vary_v') else 'fixed'}  n_test={n_te}")
    print(f"{'set':<8}{'rel L2':>10}{'abs L2':>11}{'mass(ref)':>12}"
          f"{'mass(pi R^2)':>14}{'drift':>10}")
    print("-" * 65)

    for name, seed, n in (("train", a["seed"], a["n_train"]),
                          ("test", a["seed"] + 999, n_te)):
        p = sample_family(n, seed=seed, vary_v=a.get("vary_v", False))
        phi0, ex, R = build_dataset(p, a["nx"], a["ny"], a["nt"], dev)
        uu, vv = velocity_batch(X, Y, Tt, p, dev)
        feats = build_features(phi0, uu, vv, X, Y, Tt)
        with torch.no_grad():
            # Chunked: the projection lifts to 128 channels, so one pass over
            # 100 instances allocates 6.25 GiB on a 64^2 x 32 grid.
            pred = torch.cat([model(phi0[i:i + chunk], feats[i:i + chunk],
                                    t_norm)
                              for i in range(0, phi0.shape[0], chunk)])
        print(f"{name:<8}{float(rel_l2(pred, ex).mean()):>9.3f}%"
              f"{float(abs_l2(pred, ex).mean()):>11.3e}"
              f"{float(mass_mape(pred, ex, h[0], h[1]).mean()):>11.2f}%"
              f"{float(analytic_mape(pred, R, h[0], h[1]).mean()):>13.2f}%"
              f"{float(mass_drift(pred, h[0], h[1]).mean()):>9.2f}%")

    # Discretisation floor: the same metrics on the exact solution.
    p = sample_family(a["n_train"], seed=a["seed"], vary_v=a.get("vary_v", False))
    _, ex, R = build_dataset(p, a["nx"], a["ny"], a["nt"], dev)
    print(f"{'(exact)':<8}{0.0:>9.3f}%{0.0:>11.3e}{0.0:>11.2f}%"
          f"{float(analytic_mape(ex, R, h[0], h[1]).mean()):>13.2f}%"
          f"{float(mass_drift(ex, h[0], h[1]).mean()):>9.2f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("tag")
    p.add_argument("--n_test", type=int, default=None,
                   help="override the checkpoint's n_test. IMPORTANT when "
                        "comparing models: sample_family(n, seed) is not "
                        "nested, so a model saved with n_test=8 and one with "
                        "n_test=100 were scored on completely different "
                        "circles. Pass the same value for every model in a "
                        "table.")
    p.add_argument("--chunk", type=int, default=8)
    args = p.parse_args()
    main(args.tag, args.n_test, args.chunk)
