"""Read every operator_*.json in the directory and print one results table.

Assembling fifteen-plus runs by hand is where transcription errors come from,
and the numbers in a paper should be generated from the artefacts rather than
retyped.

Usage
    python collect_results.py
    python collect_results.py --sort test
"""

import argparse, glob, json


def rows():
    out = []
    for f in sorted(glob.glob("operator_*.json")):
        try:
            b = json.load(open(f))
        except Exception:
            continue
        a, h = b.get("args", {}), b.get("history", [])
        if not a or not h:
            continue
        last = h[-1]
        arm = ("supervised" if a["loss"] == "supervised"
               else "data-free" if a.get("n_labelled", 0) == 0
               else f"hybrid-{a['n_labelled']}")
        out.append({
            "tag": f[9:-5],
            "arm": arm,
            "v": "vary" if a.get("vary_v") else "fixed",
            "n": a["n_train"],
            "grid": f"{a['nx']}^2x{a['nt']}",
            "steps": a["steps"],
            "seed": a["seed"],
            "train": last["train_rel_l2"],
            "test": last["test_rel_l2"],
            "mass": last["test_mass_mape"],
            "min": last["min"],
        })
    return out


def main(a):
    r = rows()
    if not r:
        print("no operator_*.json found in this directory")
        return
    if a.sort in ("test", "mass", "train"):
        r.sort(key=lambda x: x[a.sort])

    print(f"{'arm':<12}{'v':<6}{'n':>4}{'grid':>11}{'steps':>7}{'sd':>3}"
          f"{'train':>9}{'test':>9}{'mass':>9}{'min':>7}")
    print("-" * 77)
    for x in r:
        print(f"{x['arm']:<12}{x['v']:<6}{x['n']:>4}{x['grid']:>11}"
              f"{x['steps']:>7}{x['seed']:>3}{x['train']:>8.3f}%"
              f"{x['test']:>8.3f}%{x['mass']:>8.2f}%{x['min']:>7.1f}")

    # Seed statistics wherever a configuration has been repeated.
    grp = {}
    for x in r:
        grp.setdefault((x["arm"], x["v"], x["n"], x["grid"], x["steps"]),
                       []).append((x["test"], x["mass"]))
    rep = {k: v for k, v in grp.items() if len(v) > 1}
    if rep:
        def ms(vals):
            m = sum(vals) / len(vals)
            return m, (sum((z - m) ** 2 for z in vals) / (len(vals) - 1)) ** 0.5
        print(f"\n{'arm':<12}{'v':<6}{'n':>4}{'steps':>7}{'seeds':>6}"
              f"{'test mean':>11}{'std':>8}{'mass mean':>11}{'std':>8}")
        print("-" * 73)
        for k, v in sorted(rep.items()):
            tm, ts = ms([a for a, _ in v])
            mm, msd = ms([b for _, b in v])
            print(f"{k[0]:<12}{k[1]:<6}{k[2]:>4}{k[4]:>7}{len(v):>6}"
                  f"{tm:>10.3f}%{ts:>7.3f}%{mm:>10.2f}%{msd:>7.2f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sort", default="none", choices=["none", "test", "mass", "train"])
    main(p.parse_args())
