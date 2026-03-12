"""
Step 1 — 3D Energy Deposition Visualizer
Reads the 4 ray CSVs and renders a 3D voxel heatmap for each one.
Saves PNGs to Steps/Step1_Rays/ and deletes the blank .wrl files.

Run from project root:
    python3 scripts/visualize_rays.py
"""

import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

GRID = 50
RAYS = {
    "Gamma":      {"csv": "gamma_edep.csv",   "color": "Blues",    "particle": "γ  (gamma)"},
    "Neutron":    {"csv": "neutron_edep.csv",  "color": "Greens",   "particle": "n  (neutron)"},
    "Carbon Ion": {"csv": "carbon_edep.csv",   "color": "Oranges",  "particle": "12C (carbon ion)"},
    "Alpha":      {"csv": "alpha_edep.csv",    "color": "Reds",     "particle": "α  (alpha)"},
}
OUT_DIR = "Steps/Step1_Rays"
CORE_MIN, CORE_MAX = 20, 30

def load_edep(csv_path):
    grid = np.zeros((GRID, GRID, GRID))
    if not os.path.exists(csv_path):
        print(f"  [MISSING] {csv_path}")
        return grid
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                ix, iy, iz, edep = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
                if edep > 0 and 0 <= ix < GRID and 0 <= iy < GRID and 0 <= iz < GRID:
                    grid[ix, iy, iz] = edep
            except ValueError:
                continue
    return grid

def plot_ray(name, cfg, out_dir):
    print(f"  Rendering {name}...")
    grid = load_edep(cfg["csv"])
    total = grid.sum()
    if total == 0:
        print(f"  [SKIP] No energy data in {cfg['csv']}")
        return

    threshold = grid.max() * 0.01
    xi, yi, zi = np.where(grid > threshold)
    edep_vals  = grid[xi, yi, zi]
    norm_vals  = (edep_vals - edep_vals.min()) / (edep_vals.max() - edep_vals.min() + 1e-12)

    fig = plt.figure(figsize=(14, 6), facecolor="#0d0d0d")
    fig.suptitle(
        f"Step 1 — {name}  ({cfg['particle']})   Total Energy: {total:.2f} MeV",
        color="white", fontsize=13, fontweight="bold", y=0.97
    )

    cmap = plt.get_cmap(cfg["color"])

    ax1 = fig.add_subplot(121, projection="3d", facecolor="#0d0d0d")
    sc = ax1.scatter(xi, yi, zi, c=edep_vals, cmap=cfg["color"],
                     s=norm_vals * 18 + 1, alpha=0.6, linewidths=0)

    for x in [CORE_MIN, CORE_MAX]:
        for y in [CORE_MIN, CORE_MAX]:
            ax1.plot([x,x],[y,y],[CORE_MIN,CORE_MAX],color="white",alpha=0.25,lw=0.5)
    for x in [CORE_MIN, CORE_MAX]:
        for z in [CORE_MIN, CORE_MAX]:
            ax1.plot([x,x],[CORE_MIN,CORE_MAX],[z,z],color="white",alpha=0.25,lw=0.5)
    for y in [CORE_MIN, CORE_MAX]:
        for z in [CORE_MIN, CORE_MAX]:
            ax1.plot([CORE_MIN,CORE_MAX],[y,y],[z,z],color="white",alpha=0.25,lw=0.5)

    ax1.set_xlabel("X voxel", color="white", fontsize=8)
    ax1.set_ylabel("Y voxel", color="white", fontsize=8)
    ax1.set_zlabel("Z voxel (depth)", color="white", fontsize=8)
    ax1.set_title("3D Energy Deposition\n(white box = prion core zone)", color="white", fontsize=9)
    ax1.tick_params(colors="white", labelsize=7)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.xaxis.pane.set_edgecolor("#333333")
    ax1.yaxis.pane.set_edgecolor("#333333")
    ax1.zaxis.pane.set_edgecolor("#333333")
    ax1.view_init(elev=25, azim=45)
    cb1 = fig.colorbar(sc, ax=ax1, pad=0.1, fraction=0.03)
    cb1.set_label("Energy Dep (MeV)", color="white", fontsize=7)
    cb1.ax.yaxis.set_tick_params(color="white", labelsize=6)
    plt.setp(cb1.ax.yaxis.get_ticklabels(), color="white")

    ax2 = fig.add_subplot(122, facecolor="#111111")
    depth_profile = grid.sum(axis=(0,1))
    z_vals = np.arange(GRID)
    colors = [cmap(v) for v in np.linspace(0.3, 1.0, GRID)]
    ax2.bar(z_vals, depth_profile, color=colors, width=1.0, edgecolor="none")
    ax2.axvspan(CORE_MIN, CORE_MAX, alpha=0.15, color="white", label="Core zone")
    ax2.set_xlabel("Z voxel (depth into brain)", color="white", fontsize=9)
    ax2.set_ylabel("Total Energy Dep (MeV)", color="white", fontsize=9)
    ax2.set_title("Depth-Energy Profile\n(shows Bragg peak / scatter pattern)", color="white", fontsize=9)
    ax2.tick_params(colors="white", labelsize=8)
    ax2.spines["bottom"].set_color("#444")
    ax2.spines["left"].set_color("#444")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(fontsize=8, facecolor="#222", labelcolor="white")

    core_energy = grid[CORE_MIN:CORE_MAX, CORE_MIN:CORE_MAX, CORE_MIN:CORE_MAX].sum()
    surf_energy = total - core_energy
    ratio = surf_energy / core_energy if core_energy > 0 else 0
    voxels_hit = int((grid > 0).sum())
    peak_z = int(np.argmax(depth_profile))
    stats_txt = (
        f"Voxels hit:    {voxels_hit:,}\n"
        f"Total energy:  {total:.2f} MeV\n"
        f"Core energy:   {core_energy:.2f} MeV\n"
        f"Surf/Core:     {ratio:.4f}\n"
        f"Peak depth:    Z={peak_z}"
    )
    ax2.text(0.97, 0.97, stats_txt, transform=ax2.transAxes,
             fontsize=8, color="white", va="top", ha="right",
             fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#222", alpha=0.7, edgecolor="#555"))

    plt.tight_layout(rect=[0,0,1,0.95])
    out_path = os.path.join(out_dir, f"vis_{name.replace(' ','_')}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"  Saved {out_path}")

def cleanup_wrl(out_dir):
    removed = 0
    for f in os.listdir(out_dir):
        if f.endswith(".wrl"):
            os.remove(os.path.join(out_dir, f))
            removed += 1
    if removed:
        print(f"  Removed {removed} blank .wrl file(s)")

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("\n  Step 1 — 3D Visualizer\n" + "=" * 50)
    cleanup_wrl(OUT_DIR)
    for name, cfg in RAYS.items():
        plot_ray(name, cfg, OUT_DIR)
    print(f"\n  Done! Run: open {OUT_DIR}/")
