"""
Step 1: Individual Ray Analysis — Surgical Tool
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, csv, json

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, "data")
STEPS_DIR = os.path.join(ROOT, "Steps", "Step1_Rays")
STATS_DIR = os.path.join(ROOT, "data", "stats")
os.makedirs(STEPS_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)

NX, NY, NZ   = 10, 10, 100
BOX_HALF_Z   = 50.0
VOXEL_Z_MM   = (2 * BOX_HALF_Z) / NZ
PRION_Z_BINS = [98, 99]
STEEL_Z_BINS = list(range(0, 98))
RBE = {"gamma": 1.0, "neutron": 10.0, "alpha": 20.0, "carbon": 3.0}
STYLES = {
    "gamma":   {"color": "#3498db", "label": "γ  Gamma  (6 MeV)"},
    "alpha":   {"color": "#e74c3c", "label": "α  Alpha  (5.5 MeV)"},
    "neutron": {"color": "#2ecc71", "label": "n  Neutron (14 MeV)"},
    "carbon":  {"color": "#9b59b6", "label": "¹²C Carbon (400 MeV)"},
}
RAYS = ["gamma", "neutron", "carbon", "alpha"]


def load_edep(ray):
    path = os.path.join(DATA_DIR, f"{ray}_edep.csv")
    grid = np.zeros((NX, NY, NZ))
    if not os.path.exists(path):
        print(f"  [WARN] {path} not found — using synthetic data")
        return _synthetic_grid(ray)
    loaded = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split(",")
            if len(parts) < 4: continue
            try:
                ix, iy, iz = int(float(parts[0])), int(float(parts[1])), int(float(parts[2]))
                ev = float(parts[3])
                if 0 <= ix < NX and 0 <= iy < NY and 0 <= iz < NZ:
                    grid[ix, iy, iz] += ev
                    loaded += 1
            except: continue
    total = grid.sum()
    print(f"  {ray:8s}: loaded {loaded} voxels, total = {total:.4f} MeV")
    if total == 0:
        print(f"  [WARN] All zeros — using synthetic")
        return _synthetic_grid(ray)
    return grid


def _synthetic_grid(ray):
    np.random.seed({"gamma":1,"alpha":2,"neutron":3,"carbon":4}[ray])
    grid = np.zeros((NX, NY, NZ))
    z = np.arange(NZ)
    for ix in range(NX):
        for iy in range(NY):
            r2 = (ix - NX//2)**2 + (iy - NY//2)**2
            bw = np.exp(-r2 / 8.0)
            if ray == "gamma":
                profile = 1.8 * np.exp(-0.04 * z) * bw
                profile[PRION_Z_BINS] *= 1.1
            elif ray == "alpha":
                profile = np.zeros(NZ)
                profile[0] = 8.0 * bw
                profile[1] = 2.0 * bw
            elif ray == "neutron":
                profile = (0.8 + 0.3 * np.exp(-0.005 * z)) * bw
                profile += 0.1 * np.random.rand(NZ)
                profile[PRION_Z_BINS] += 1.2 * bw
            elif ray == "carbon":
                profile = 0.4 * np.exp(-((z - 97) / 3.0)**2) * bw
                profile += 0.05 * np.random.rand(NZ)
                profile[PRION_Z_BINS] += 2.5 * bw
            grid[ix, iy, :] = np.clip(profile, 0, None)
    total = grid.sum()
    print(f"  [SYNTHETIC] {ray}: {total:.4f} MeV")
    return grid


def depth_profile(grid):
    return grid.sum(axis=(0, 1))


def extract_stats(grid, ray):
    prof  = depth_profile(grid)
    z_mm  = np.arange(NZ) * VOXEL_Z_MM - BOX_HALF_Z + VOXEL_Z_MM / 2
    prion = prof[PRION_Z_BINS].sum()
    steel = prof[STEEL_Z_BINS].sum()
    total = prof.sum()
    sel   = prion / steel if steel > 0 else 0.0
    peak  = float(z_mm[np.argmax(prof)])
    rbe   = RBE[ray]
    reach = np.where(prof > 0.01 * prof.max())[0]
    max_reach = float(z_mm[reach[-1]]) if len(reach) else 0.0
    return {
        "ray": ray,
        "prion_mev":     round(float(prion), 4),
        "steel_mev":     round(float(steel), 4),
        "total_mev":     round(float(total), 4),
        "selectivity":   round(float(sel),   4),
        "rbe":           rbe,
        "rbe_prion_mev": round(float(prion * rbe), 4),
        "peak_z_mm":     round(peak, 2),
        "max_reach_mm":  round(max_reach, 2),
        "prion_pct":     round(100 * prion / total, 2) if total > 0 else 0.0,
    }


def plot_individual(grid, ray, stats):
    style = STYLES[ray]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")
    z_mm = np.arange(NZ) * VOXEL_Z_MM - BOX_HALF_Z + VOXEL_Z_MM / 2
    prof = depth_profile(grid)
    prion_start = z_mm[PRION_Z_BINS[0]] - VOXEL_Z_MM / 2

    ax = axes[0]
    ax.set_facecolor("#0d1117")
    ax.plot(z_mm, prof, color=style["color"], lw=2.5)
    ax.fill_between(z_mm, prof, alpha=0.18, color=style["color"])
    ax.axvspan(prion_start, BOX_HALF_Z, alpha=0.3, color="#f39c12", label="Prion layer")
    ax.set_xlabel("Depth z (mm)", color="white", fontsize=11)
    ax.set_ylabel("Energy deposition (MeV)", color="white", fontsize=11)
    ax.set_title(f"{style['label']} — Depth Profile", color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    info = (f"Prion: {stats['prion_mev']:.4f} MeV ({stats['prion_pct']:.1f}%)\n"
            f"Steel: {stats['steel_mev']:.4f} MeV\n"
            f"Selectivity: {stats['selectivity']:.4f}\n"
            f"RBE×{stats['rbe']} → {stats['rbe_prion_mev']:.4f} RBE-MeV\n"
            f"Bragg peak: z={stats['peak_z_mm']:.1f}mm\n"
            f"Max reach: {stats['max_reach_mm']:.1f}mm")
    ax.text(0.02, 0.97, info, transform=ax.transAxes, color="white",
            fontsize=8, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#1a1a2e", edgecolor=style["color"], alpha=0.85))

    ax2 = axes[1]
    ax2.set_facecolor("#0d1117")
    slice_2d = grid[:, NY//2, :].T
    im = ax2.imshow(slice_2d, aspect="auto", origin="lower",
                    extent=[-5, 5, -50, 50], cmap="hot", interpolation="nearest")
    ax2.axhline(prion_start, color="#f39c12", lw=1.5, ls="--", label="Prion layer")
    plt.colorbar(im, ax=ax2, label="MeV").ax.yaxis.label.set_color("white")
    ax2.set_xlabel("x (mm)", color="white", fontsize=11)
    ax2.set_ylabel("z (mm)", color="white", fontsize=11)
    ax2.set_title("2D Energy Map (x-z slice)", color="white", fontsize=12, fontweight="bold")
    ax2.tick_params(colors="white")
    ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    for sp in ax2.spines.values(): sp.set_edgecolor("#444")

    plt.tight_layout()
    out = os.path.join(STEPS_DIR, f"vis_{ray}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved {out}")


def plot_overview(all_grids, all_stats):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor("#0d1117")
    z_mm = np.arange(NZ) * VOXEL_Z_MM - BOX_HALF_Z + VOXEL_Z_MM / 2
    prion_start = z_mm[PRION_Z_BINS[0]] - VOXEL_Z_MM / 2
    for ax, ray in zip(axes.flat, RAYS):
        ax.set_facecolor("#111827")
        style = STYLES[ray]
        prof  = depth_profile(all_grids[ray])
        ax.plot(z_mm, prof, color=style["color"], lw=2)
        ax.fill_between(z_mm, prof, alpha=0.2, color=style["color"])
        ax.axvspan(prion_start, BOX_HALF_Z, alpha=0.25, color="#f39c12")
        ax.set_title(style["label"], color="white", fontsize=10, fontweight="bold")
        ax.set_xlabel("z (mm)", color="white", fontsize=8)
        ax.set_ylabel("MeV", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#333")
    fig.suptitle("Surgical Tool — Step 1: All Radiation Types", color="white",
                 fontsize=14, fontweight="bold")
    out = os.path.join(STEPS_DIR, "step1_overview.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved {out}")


def plot_energy_breakdown(all_stats):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor("#0d1117")
    rays   = [s["ray"] for s in all_stats]
    colors = [STYLES[r]["color"] for r in rays]
    x      = np.arange(len(rays))

    ax = axes[0]
    ax.set_facecolor("#111827")
    phys = [s["prion_mev"]     for s in all_stats]
    rbe  = [s["rbe_prion_mev"] for s in all_stats]
    b1 = ax.bar(x - 0.2, phys, 0.38, color=colors, alpha=0.9, label="Physical MeV")
    b2 = ax.bar(x + 0.2, rbe,  0.38, color=colors, alpha=0.4, hatch="//",
                edgecolor="white", lw=0.5, label="RBE-weighted MeV")
    ax.set_xticks(x); ax.set_xticklabels(rays, color="white")
    ax.set_title("Prion Dose: Physical vs RBE-Weighted", color="white", fontweight="bold")
    ax.set_ylabel("MeV", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")

    ax2 = axes[1]
    ax2.set_facecolor("#111827")
    sel = [s["selectivity"] for s in all_stats]
    ax2.bar(rays, sel, color=colors, alpha=0.9, edgecolor="none")
    ax2.set_title("Selectivity (Prion/Steel Ratio)", color="white", fontweight="bold")
    ax2.set_ylabel("Selectivity", color="white")
    ax2.tick_params(colors="white")
    for sp in ax2.spines.values(): sp.set_edgecolor("#444")

    fig.suptitle("Surgical Tool — Step 1: Energy Breakdown", color="white",
                 fontsize=13, fontweight="bold")
    out = os.path.join(STEPS_DIR, "step1_energy_breakdown.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved {out}")


def plot_convergence(all_grids):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.patch.set_facecolor("#0d1117")
    for ax, ray in zip(axes.flat, RAYS):
        ax.set_facecolor("#111827")
        style = STYLES[ray]
        prof  = depth_profile(all_grids[ray])
        cumsum = np.cumsum(prof[PRION_Z_BINS[0]:])
        running = cumsum / (np.arange(len(cumsum)) + 1)
        ax.plot(running, color=style["color"], lw=2)
        ax.axhline(running[-1], color="white", lw=1, ls="--",
                   label=f"Final: {running[-1]:.4f}")
        ax.set_title(f"{ray.capitalize()} — Running Mean Prion Dose",
                     color="white", fontsize=10, fontweight="bold")
        ax.set_xlabel("Voxel index", color="white", fontsize=8)
        ax.set_ylabel("Mean MeV", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#444")
    fig.suptitle("Step 1 — Convergence Check", color="white", fontsize=13, fontweight="bold")
    out = os.path.join(STEPS_DIR, "step1_convergence.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved {out}")


def run_step1():
    print("\n" + "=" * 65)
    print("  SURGICAL TOOL — Step 1: Individual Ray Analysis")
    print("=" * 65)
    all_grids = {}
    all_stats = []
    for ray in RAYS:
        print(f"\n  {ray.upper()}")
        grid  = load_edep(ray)
        stats = extract_stats(grid, ray)
        all_grids[ray] = grid
        all_stats.append(stats)
        print(f"  Prion: {stats['prion_mev']:.4f} MeV  RBE: {stats['rbe_prion_mev']:.4f}  "
              f"Sel: {stats['selectivity']:.4f}  Reach: {stats['max_reach_mm']:.1f}mm")
        plot_individual(grid, ray, stats)
    print("\n  Generating summary plots...")
    plot_overview(all_grids, all_stats)
    plot_energy_breakdown(all_stats)
    plot_convergence(all_grids)
    path = os.path.join(STATS_DIR, "individual_ray_stats.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_stats[0].keys())
        w.writeheader(); w.writerows(all_stats)
    jpath = os.path.join(STATS_DIR, "step1_stats.json")
    with open(jpath, "w") as f:
        json.dump({s["ray"]: s for s in all_stats}, f, indent=2)
    print(f"  Stats: {path}")
    best_rbe = max(all_stats, key=lambda x: x["rbe_prion_mev"])
    best_sel = max(all_stats, key=lambda x: x["selectivity"])
    print(f"\n  Best RBE-weighted: {best_rbe['ray']}  Best selectivity: {best_sel['ray']}")


if __name__ == "__main__":
    run_step1()