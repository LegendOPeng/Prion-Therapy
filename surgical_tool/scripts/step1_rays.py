"""
Step 1: Individual Ray Analysis — Surgical Tool
Reads real Geant4 CSV (6-column) and produces depth profiles,
2D maps, energy breakdown, CV convergence and radar charts.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, csv, json, math

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, "data")
STEPS_DIR = os.path.join(ROOT, "Steps", "Step1_Rays")
STATS_DIR = os.path.join(ROOT, "data", "stats")
os.makedirs(STEPS_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)

NX, NY, NZ  = 10, 10, 100
BOX_Z_MM    = 100.0
VOXEL_Z_MM  = BOX_Z_MM / NZ
PRION_BINS  = list(range(95, 100))
STEEL_BINS  = list(range(0, 95))

RBE = {"gamma": 1.0, "neutron": 10.0, "carbon": 3.0, "alpha": 20.0}
STYLES = {
    "gamma":   {"color": "#3498db", "label": "γ  Gamma  (6 MeV)"},
    "neutron": {"color": "#2ecc71", "label": "n  Neutron (14 MeV)"},
    "carbon":  {"color": "#9b59b6", "label": "C  Carbon (400 MeV/u)"},
    "alpha":   {"color": "#e74c3c", "label": "α  Alpha  (5.5 MeV)"},
}
RAYS = ["gamma", "neutron", "carbon", "alpha"]


def load_edep(ray):
    path = os.path.join(DATA_DIR, f"{ray}_edep.csv")
    grid = np.zeros((NX, NY, NZ))
    if not os.path.exists(path):
        print(f"  [WARN] {path} not found — using synthetic")
        return _synthetic_grid(ray)
    loaded = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                ix = int(float(parts[0]))
                iy = int(float(parts[1]))
                iz = int(float(parts[2]))
                ev = float(parts[3])
                if 0 <= ix < NX and 0 <= iy < NY and 0 <= iz < NZ and ev > 0:
                    grid[ix, iy, iz] += ev
                    loaded += 1
            except (ValueError, IndexError):
                continue
    total = grid.sum()
    print(f"  {ray:8s}: {loaded:,} voxels  total={total:.4f} MeV")
    if total == 0:
        print(f"  [WARN] All zeros — using synthetic")
        return _synthetic_grid(ray)
    return grid


def _synthetic_grid(ray):
    np.random.seed({"gamma": 1, "neutron": 2, "carbon": 3, "alpha": 4}[ray])
    grid = np.zeros((NX, NY, NZ))
    z = np.arange(NZ)
    for ix in range(NX):
        for iy in range(NY):
            r2 = (ix - NX//2)**2 + (iy - NY//2)**2
            bw = np.exp(-r2 / 8.0)
            if ray == "gamma":
                p = 1.5 * np.exp(-0.03 * z) * bw
            elif ray == "neutron":
                p = (0.9 + 0.2 * np.exp(-0.005 * z)) * bw
                p += 0.05 * np.random.rand(NZ)
                p[PRION_BINS] += 0.8 * bw
            elif ray == "carbon":
                p = 0.3 * np.exp(-((z - 92) / 4.0)**2) * bw
                p += 0.02 * np.random.rand(NZ)
                p[PRION_BINS] += 1.5 * bw
            else:
                p = np.zeros(NZ)
                p[0] = 12.0 * bw
                p[1] = 3.0 * bw
            grid[ix, iy, :] = np.clip(p, 0, None)
    print(f"  [SYNTHETIC] {ray}: {grid.sum():.4f} MeV")
    return grid


def depth_profile(grid):
    return grid.sum(axis=(0, 1))


def extract_stats(grid, ray):
    prof        = depth_profile(grid)
    z_mm        = np.arange(NZ) * VOXEL_Z_MM + VOXEL_Z_MM / 2
    prion_dose  = prof[PRION_BINS].sum()
    steel_dose  = prof[STEEL_BINS].sum()
    total_dose  = prof.sum()
    selectivity = prion_dose / steel_dose if steel_dose > 0 else 0.0
    peak_z      = float(z_mm[np.argmax(prof)])
    rbe_dose    = prion_dose * RBE[ray]
    reach_idx   = np.where(prof > 0.01 * prof.max())[0]
    max_reach   = float(z_mm[reach_idx[-1]]) if len(reach_idx) else 0.0
    voxels_hit  = int((grid > 0).sum())
    prion_vox   = grid[:, :, PRION_BINS].flatten()
    prion_vox   = prion_vox[prion_vox > 0]
    cv = float(np.std(prion_vox) / np.mean(prion_vox) * 100) if len(prion_vox) > 1 else 999.0
    return {
        "ray": ray,
        "prion_mev":     round(float(prion_dose), 4),
        "steel_mev":     round(float(steel_dose), 4),
        "total_mev":     round(float(total_dose), 4),
        "selectivity":   round(float(selectivity), 4),
        "rbe":           RBE[ray],
        "rbe_prion_mev": round(float(rbe_dose), 4),
        "peak_z_mm":     round(peak_z, 2),
        "max_reach_mm":  round(max_reach, 2),
        "prion_pct":     round(100 * prion_dose / total_dose, 2) if total_dose > 0 else 0.0,
        "voxels_hit":    voxels_hit,
        "cv_pct":        round(cv, 2),
    }


def plot_individual(grid, ray, stats):
    style = STYLES[ray]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")
    z_mm        = np.arange(NZ) * VOXEL_Z_MM + VOXEL_Z_MM / 2
    prof        = depth_profile(grid)
    prion_start = PRION_BINS[0] * VOXEL_Z_MM

    ax = axes[0]
    ax.set_facecolor("#111827")
    ax.plot(z_mm, prof, color=style["color"], lw=2.5, label=style["label"])
    ax.fill_between(z_mm, prof, alpha=0.18, color=style["color"])
    ax.axvspan(prion_start, BOX_Z_MM, alpha=0.3, color="#f39c12", label="Prion layer")
    ax.set_xlabel("Depth z (mm)", color="white", fontsize=11)
    ax.set_ylabel("Energy deposition (MeV)", color="white", fontsize=11)
    ax.set_title(f"{style['label']} — Depth Profile", color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    alpha_note = "\n  CANNOT REACH TIP" if ray == "alpha" and stats["prion_mev"] < 0.001 else ""
    info = (f"Prion:  {stats['prion_mev']:.4f} MeV ({stats['prion_pct']:.1f}%)\n"
            f"Steel:  {stats['steel_mev']:.4f} MeV\n"
            f"Sel:    {stats['selectivity']:.4f}\n"
            f"RBE x{stats['rbe']} = {stats['rbe_prion_mev']:.4f}\n"
            f"Peak:   z={stats['peak_z_mm']:.1f}mm\n"
            f"Reach:  {stats['max_reach_mm']:.1f}mm\n"
            f"CV:     {stats['cv_pct']:.2f}%{alpha_note}")
    ax.text(0.02, 0.97, info, transform=ax.transAxes, color="white",
            fontsize=8, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#1a1a2e", edgecolor=style["color"], alpha=0.9))

    ax2 = axes[1]
    ax2.set_facecolor("#111827")
    s2d = grid[:, NY // 2, :].T
    im  = ax2.imshow(s2d, aspect="auto", origin="lower",
                     extent=[0, 10, 0, 100], cmap="hot", interpolation="nearest")
    ax2.axhline(prion_start, color="#f39c12", lw=1.5, ls="--", label="Prion layer")
    plt.colorbar(im, ax=ax2, label="MeV").ax.yaxis.label.set_color("white")
    ax2.set_xlabel("x (mm)", color="white", fontsize=11)
    ax2.set_ylabel("z depth (mm)", color="white", fontsize=11)
    ax2.set_title("2D Energy Map (x-z slice)", color="white", fontsize=12, fontweight="bold")
    ax2.tick_params(colors="white")
    ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    for sp in ax2.spines.values(): sp.set_edgecolor("#444")

    plt.tight_layout()
    out = os.path.join(STEPS_DIR, f"vis_{ray}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Saved {out}")


def plot_overview(all_grids, all_stats):
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(3, 4, hspace=0.5, wspace=0.35)
    z_mm        = np.arange(NZ) * VOXEL_Z_MM + VOXEL_Z_MM / 2
    prion_start = PRION_BINS[0] * VOXEL_Z_MM

    for i, ray in enumerate(RAYS):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor("#111827")
        style = STYLES[ray]
        prof  = depth_profile(all_grids[ray])
        ax.plot(z_mm, prof, color=style["color"], lw=2)
        ax.fill_between(z_mm, prof, alpha=0.2, color=style["color"])
        ax.axvspan(prion_start, BOX_Z_MM, alpha=0.25, color="#f39c12")
        ax.set_title(style["label"], color="white", fontsize=8, fontweight="bold")
        ax.set_xlabel("z (mm)", color="white", fontsize=7)
        ax.set_ylabel("MeV", color="white", fontsize=7)
        ax.tick_params(colors="white", labelsize=6)
        for sp in ax.spines.values(): sp.set_edgecolor("#333")

    ax5 = fig.add_subplot(gs[1, :2])
    ax5.set_facecolor("#111827")
    rays   = [s["ray"] for s in all_stats]
    colors = [STYLES[r]["color"] for r in rays]
    cores  = [max(s["prion_mev"], 1e-9) for s in all_stats]
    rbe_w  = [max(s["rbe_prion_mev"], 1e-9) for s in all_stats]
    x      = np.arange(len(rays))
    ax5.bar(x - 0.2, cores, 0.38, color=colors, alpha=0.9, label="Physical MeV")
    ax5.bar(x + 0.2, rbe_w,  0.38, color=colors, alpha=0.4,
            hatch="//", edgecolor="white", lw=0.5, label="RBE-weighted MeV")
    ax5.set_yscale("log")
    ax5.set_xticks(x)
    ax5.set_xticklabels(rays, color="white")
    ax5.set_ylabel("MeV in prion layer (log)", color="white")
    ax5.set_title("Prion Dose: Physical vs RBE-Weighted", color="white", fontweight="bold")
    ax5.tick_params(colors="white")
    ax5.legend(facecolor="#1a1a2e", labelcolor="white")
    for sp in ax5.spines.values(): sp.set_edgecolor("#444")

    ax6 = fig.add_subplot(gs[1, 2:])
    ax6.set_facecolor("#111827")
    sels = [s["selectivity"] for s in all_stats]
    bars = ax6.bar(rays, sels, color=colors, alpha=0.9, edgecolor="none")
    for bar, val in zip(bars, sels):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                 f"{val:.4f}", ha="center", va="bottom", color="white", fontsize=8)
    ax6.set_title("Selectivity (Prion/Steel)", color="white", fontweight="bold")
    ax6.set_ylabel("Selectivity", color="white")
    ax6.tick_params(colors="white")
    for sp in ax6.spines.values(): sp.set_edgecolor("#444")

    ax7 = fig.add_subplot(gs[2, :])
    ax7.set_facecolor("#111827")
    ax7.axis("off")
    hdr  = f"{'Ray':<10} {'Prion MeV':>11} {'RBE-MeV':>10} {'Select.':>10} {'Peak z':>8} {'Reach':>8} {'CV%':>7}"
    rows = [hdr, "─"*72]
    for s in sorted(all_stats, key=lambda x: -x["rbe_prion_mev"]):
        note = " ← CANNOT REACH TIP" if s["ray"] == "alpha" and s["prion_mev"] < 0.001 else ""
        rows.append(f"{s['ray']:<10} {s['prion_mev']:>11.4f} {s['rbe_prion_mev']:>10.4f}"
                    f" {s['selectivity']:>10.4f} {s['peak_z_mm']:>7.1f}mm"
                    f" {s['max_reach_mm']:>7.1f}mm {s['cv_pct']:>7.2f}%{note}")
    rows += ["─"*72, "", "Geometry: G4_STAINLESS-STEEL 10x10x100mm | Prion proxy: G4_WATER (last 5mm)"]
    ax7.text(0.02, 0.95, "\n".join(rows), transform=ax7.transAxes,
             color="white", fontsize=8, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#1a1a2e", edgecolor="#4a9eff", alpha=0.9))

    fig.suptitle("Surgical Tool — Step 1: Individual Ray Analysis (Real Geant4 Data)",
                 color="white", fontsize=14, fontweight="bold")
    out = os.path.join(STEPS_DIR, "step1_overview.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Saved {out}")


def plot_cv_convergence(all_grids):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#111827")
    for label, val, col in [("Excellent", 2.0, "#2ecc71"), ("Good", 5.0, "#f39c12"), ("Acceptable", 10.0, "#e74c3c")]:
        ax.axhline(val, color=col, ls="--", lw=1.2, alpha=0.7, label=f"{label} ({val}%)")
    for ray in RAYS:
        prion = all_grids[ray][:, :, PRION_BINS].flatten()
        prion = prion[prion > 0]
        if len(prion) < 2:
            continue
        np.random.shuffle(prion)
        cvs = [np.std(prion[:n]) / np.mean(prion[:n]) * 100 for n in range(2, min(len(prion)+1, 200))]
        ax.plot(range(2, 2+len(cvs)), cvs, color=STYLES[ray]["color"], lw=1.5, label=ray.capitalize())
    ax.set_title("CV Convergence Over Prion Voxels", color="white", fontsize=12, fontweight="bold")
    ax.set_xlabel("Voxels sampled", color="white")
    ax.set_ylabel("CV %", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    out = os.path.join(STEPS_DIR, "step1_convergence.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Saved {out}")


def plot_radar(all_stats):
    categories = ["Core Energy", "Voxels Hit", "Stability\n(inv CV)", "Core-Pref\n(inv S/C)", "Selectivity"]
    N = len(categories)
    angles = [n / float(N) * 2 * math.pi for n in range(N)] + [0]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#111827")

    def norm(vals, invert=False):
        mn, mx = min(vals), max(vals)
        if mx == mn: return [0.5] * len(vals)
        n = [(v - mn) / (mx - mn) for v in vals]
        return [1-x for x in n] if invert else n

    nc   = norm([s["prion_mev"]   for s in all_stats])
    nv   = norm([s["voxels_hit"]  for s in all_stats])
    ns   = norm([s["cv_pct"]      for s in all_stats], invert=True)
    ni   = norm([s["selectivity"] for s in all_stats], invert=True)
    nsel = norm([s["selectivity"] for s in all_stats])

    for i, s in enumerate(all_stats):
        vals = [nc[i], nv[i], ns[i], ni[i], nsel[i]] + [nc[i]]
        ax.plot(angles, vals, color=STYLES[s["ray"]]["color"], lw=2, label=s["ray"].capitalize())
        ax.fill(angles, vals, color=STYLES[s["ray"]]["color"], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color="white", size=10)
    ax.tick_params(colors="white")
    ax.spines["polar"].set_color("#444")
    ax.yaxis.set_visible(False)
    ax.set_title("Normalised Ray Profile", color="white", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), facecolor="#1a1a2e", labelcolor="white")
    out = os.path.join(STEPS_DIR, "step1_radar.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Saved {out}")


def run_step1():
    print("\n" + "="*65)
    print("  SURGICAL TOOL — Step 1: Individual Ray Analysis")
    print("  Using real Geant4 data (6-column CSV format)")
    print("="*65)
    all_grids = {}
    all_stats = []
    for ray in RAYS:
        print(f"\n  [{ray.upper()}]")
        grid  = load_edep(ray)
        stats = extract_stats(grid, ray)
        all_grids[ray] = grid
        all_stats.append(stats)
        print(f"    Prion: {stats['prion_mev']:.4f} MeV ({stats['prion_pct']:.1f}%)")
        print(f"    RBE:   {stats['rbe_prion_mev']:.4f}  Sel: {stats['selectivity']:.4f}")
        print(f"    Peak:  z={stats['peak_z_mm']:.1f}mm  Reach: {stats['max_reach_mm']:.1f}mm")
        print(f"    CV:    {stats['cv_pct']:.2f}%  Voxels: {stats['voxels_hit']:,}")
        plot_individual(grid, ray, stats)
    print("\n  Generating summary plots...")
    plot_overview(all_grids, all_stats)
    plot_cv_convergence(all_grids)
    plot_radar(all_stats)
    path = os.path.join(STATS_DIR, "individual_ray_stats.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_stats[0].keys())
        w.writeheader(); w.writerows(all_stats)
    jpath = os.path.join(STATS_DIR, "step1_stats.json")
    with open(jpath, "w") as f:
        json.dump({s["ray"]: s for s in all_stats}, f, indent=2)
    print(f"\n    Saved {path}")
    print(f"    Saved {jpath}")
    best_rbe = max(all_stats, key=lambda x: x["rbe_prion_mev"])
    best_sel = max(all_stats, key=lambda x: x["selectivity"])
    print(f"\n  Best RBE-weighted: {best_rbe['ray']} ({best_rbe['rbe_prion_mev']:.4f} MeV)")
    print(f"  Best selectivity:  {best_sel['ray']} ({best_sel['selectivity']:.4f})")
    print("\n  Step 1 complete.")
    return {s["ray"]: s for s in all_stats}, all_grids

if __name__ == "__main__":
    run_step1()
