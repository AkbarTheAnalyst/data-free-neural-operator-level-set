"""
Plot the eikonal weight against training step for each configuration.

Section 6.3 argues from where the weight ends up, and a table of four final
values does not show how it got there. The trajectory does: under a structural
initial condition the field begins at an exact distance function, so the ratio
is uninformative at the first iteration and the published seeding holds the
weight near its clamp for thousands of steps. Zero seeding lets it rise from
below instead. Whether that matters depends on the benchmark, and the two
panels put that side by side.

The weight is read from the run records, so nothing is recomputed.

Run from the folder holding the run records, or from a folder containing the
two benchmark folders -- the script looks one level down as well.

    python plot_weights.py
    python plot_weights.py --out fig_weight.pdf
"""

import argparse
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# One explicit tag per curve. Matching on substrings picked up whichever
# budget sorted first, which put the 20,000-step rotation run next to the
# 30,000-step ones; the two are different experiments and must not share a
# panel. Edit these if a tag changes.
CURVES = [
    ("ro", "strong_saw_saw0_n16_30k_64x32_s{seed}",
     "zero seeding", "#1f77b4", "-"),
    ("ro", "strong_saw_n16_30k_64x32_s{seed}",
     "ratio seeding", "#d62728", "--"),
    ("ro", "strong_saw_n16_30k_softic_64x32_s{seed}",
     "soft initial condition", "#ff7f0e", ":"),
    ("rv", "rv_strong_saw_saw0_n16_20k_64x32_s{seed}",
     "zero seeding", "#1f77b4", "-"),
    ("rv", "rv_strong_saw_n16_20k_64x32_s{seed}",
     "ratio seeding", "#d62728", "--"),
    ("rv", "rv_strong_saw_n16_20k_softic_64x32_s{seed}",
     "soft initial condition", "#ff7f0e", ":"),
]


def history(tag):
    """The eikonal-weight trace of one run, or None if the record is absent."""
    try:
        b = json.load(open(f"operator_{tag}.json"))
    except FileNotFoundError:
        return None
    return [(r["step"], r["w_eik"]) for r in b.get("history", []) if "w_eik" in r]


def main(a):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    titles = {"ro": "solid-body rotation", "rv": "reversed vortex"}
    missing, drawn = [], 0

    for ax, bench in zip(axes, ("ro", "rv")):
        for b, pattern, label, colour, style in CURVES:
            if b != bench:
                continue
            tag = pattern.format(seed=a.seed)
            h = history(tag)
            if not h:
                missing.append(tag)
                continue
            ax.plot([s for s, _ in h], [w for _, w in h], style,
                    color=colour, lw=1.6, label=label)
            drawn += 1
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_title(titles[bench], fontsize=11)
        ax.grid(alpha=0.3, which="both", lw=0.4)

    axes[0].set_ylabel(r"eikonal weight  $w_{\mathrm{eik}}$")
    axes[0].legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"{drawn} of {len(CURVES)} trajectories -> {a.out} and .png")
    for t in missing:
        print(f"  not found: operator_{t}.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="fig_weight_trajectory.pdf")
    main(p.parse_args())
