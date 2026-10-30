#!/usr/bin/env python3
"""
Compare the raster_scan.py results across arrangements.

Reads <root>/<arrangement>/summary.csv (written by raster_scan.py / batch_render)
and makes two figures:

  trap_map.svg/.png   rows = arrangement, columns = height z.
                      Each cell is one (x, y) target point:
                        coloured  = bead held; colour = how far it settled
                                    from the commanded point (mm)
                        grey + x  = bead dropped (no trap there)
  held_vs_z.svg/.png  % of points held vs height, one line per arrangement

Run:
    python plot_raster_summary.py                 # reads ./sim_output
    python plot_raster_summary.py --root run_16V
"""

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ARRANGEMENTS = [("two_opposed_boards", "Two opposed boards"),
                ("single_board_reflector", "Single board + reflector"),
                ("single_board_free", "Single board, free field")]
LINE_COLORS = {"two_opposed_boards": "#2a6fdb", "single_board_reflector": "#e07b00",
               "single_board_free": "#3a9d5d"}


def load(root):
    data = {}
    for key, _ in ARRANGEMENTS:
        fn = os.path.join(root, key, "summary.csv")
        if not os.path.exists(fn):
            continue
        rows = []
        with open(fn) as f:
            for r in csv.DictReader(f):
                rows.append({k: float(v) for k, v in r.items()})
        data[key] = rows
    return data


def trap_map(data, out):
    keys = [k for k, _ in ARRANGEMENTS if k in data]
    names = dict(ARRANGEMENTS)
    zs = sorted({r["z_mm"] for k in keys for r in data[k]}, reverse=True)
    xs = sorted({r["x_mm"] for k in keys for r in data[k]})
    ys = sorted({r["y_mm"] for k in keys for r in data[k]})
    offs = [np.hypot(np.hypot(r["offset_x_mm"], r["offset_y_mm"]), r["offset_z_mm"])
            for k in keys for r in data[k] if r["held"] > 0.5]
    vmax = max(np.ceil(max(offs)) if offs else 1.0, 1.0)
    cmap = plt.get_cmap("viridis").copy()

    fig, axes = plt.subplots(len(keys), len(zs), figsize=(3.1 * len(zs) + 1.2, 3.0 * len(keys)),
                             squeeze=False, layout="constrained")
    im = None
    for i, k in enumerate(keys):
        for j, z in enumerate(zs):
            ax = axes[i, j]
            grid = np.full((len(ys), len(xs)), np.nan)
            dropped = np.zeros_like(grid, bool)
            for r in data[k]:
                if r["z_mm"] != z:
                    continue
                yi, xi = ys.index(r["y_mm"]), xs.index(r["x_mm"])
                if r["held"] > 0.5:
                    grid[yi, xi] = np.hypot(np.hypot(r["offset_x_mm"], r["offset_y_mm"]), r["offset_z_mm"])
                else:
                    dropped[yi, xi] = True
            # grey background for dropped cells
            ax.imshow(np.where(dropped, 1.0, np.nan), cmap=ListedColormap(["#d9d9d9"]),
                      origin="lower", vmin=0, vmax=1)
            im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=vmax, origin="lower")
            for yi in range(len(ys)):
                for xi in range(len(xs)):
                    if dropped[yi, xi]:
                        ax.text(xi, yi, "×", ha="center", va="center", color="#7a7a7a", fontsize=14)
                    elif not np.isnan(grid[yi, xi]):
                        v = grid[yi, xi]
                        ax.text(xi, yi, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                                color="white" if v < 0.55 * vmax else "black")
            n_held = int(np.sum(~np.isnan(grid)))
            n_tot = n_held + int(dropped.sum())
            ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs], fontsize=8)
            ax.set_yticks(range(len(ys)), [f"{y:g}" for y in ys], fontsize=8)
            ax.set_xticks(np.arange(-0.5, len(xs)), minor=True)
            ax.set_yticks(np.arange(-0.5, len(ys)), minor=True)
            ax.grid(which="minor", color="white", lw=1.5)
            ax.tick_params(which="minor", length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(f"z = {z:g} mm", fontsize=11, fontweight="bold")
            if i == len(keys) - 1:
                ax.set_xlabel("x (mm)", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{names[k]}\n\ny (mm)", fontsize=9)
            ax.text(0.98, 1.01, f"held {n_held}/{n_tot}", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=8, color="#555555")
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("settled offset from commanded point (mm)")
    fig.suptitle("Where each arrangement holds a bead  (× = dropped)", fontsize=12)
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(out, f"trap_map.{ext}"), dpi=150)
    plt.close(fig)


def held_vs_z(data, out):
    fig, ax = plt.subplots(figsize=(5.2, 3.6), layout="constrained")
    for k, name in ARRANGEMENTS:
        if k not in data:
            continue
        zs = sorted({r["z_mm"] for r in data[k]})
        frac = [100 * np.mean([r["held"] for r in data[k] if r["z_mm"] == z]) for z in zs]
        ax.plot(zs, frac, "-o", color=LINE_COLORS[k], lw=2, ms=6,
                zorder=3 if k == "two_opposed_boards" else 2)
        ax.annotate(name, (zs[-1], frac[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=LINE_COLORS[k])
    ax.set_xlabel("height z (mm)")
    ax.set_ylabel("points where the bead is held (%)")
    ax.set_ylim(-3, 103)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6")
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax + 0.45 * (xmax - xmin))     # room for the line labels
    ax.set_title("Trap coverage vs height", fontsize=11)
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(out, f"held_vs_z.{ext}"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="sim_output", help="folder raster_scan.py wrote into")
    ap.add_argument("--out", help="where to save the figures (default: the root folder)")
    a = ap.parse_args()
    data = load(a.root)
    if not data:
        raise SystemExit(f"no summary.csv files found under {a.root}/")
    out = a.out or a.root
    os.makedirs(out, exist_ok=True)
    trap_map(data, out)
    held_vs_z(data, out)
    print(f"saved trap_map.svg/png and held_vs_z.svg/png in {out}/")


if __name__ == "__main__":
    main()