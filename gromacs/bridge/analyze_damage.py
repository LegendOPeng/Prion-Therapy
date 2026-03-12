import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR  = "gromacs/output"
PLOT_DIR = "Steps/Step7_Final"
os.makedirs(PLOT_DIR, exist_ok=True)

def parse_xvg(path):
    times, vals = [], []
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        return np.array([]), np.array([])
    with open(path) as f:
        for line in f:
            if line.startswith(("#","@")):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    times.append(float(parts[0]))
                    vals.append(float(parts[1]))
                except ValueError:
                    continue
    return np.array(times), np.array(vals)

def plot_comparison():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#0d0d0d")
    fig.suptitle(
        "Prion Protein Radiation Damage Analysis\n"
        "Healthy vs Radiation-Damaged (Geant4→GROMACS Bridge)",
        color="white", fontsize=13, fontweight="bold"
    )

    ax = axes[0,0]
    ax.set_facecolor("#111111")
    t_h, rmsd_h = parse_xvg(f"{OUT_DIR}/rmsd_healthy.xvg")
    t_d, rmsd_d = parse_xvg(f"{OUT_DIR}/rmsd_damaged.xvg")
    if len(t_h): ax.plot(t_h, rmsd_h, color="#2ecc71", linewidth=2, label="Healthy PrP")
    if len(t_d): ax.plot(t_d, rmsd_d, color="#e74c3c", linewidth=2, label="Irradiated PrP")
    ax.set_title("Backbone RMSD over Time", color="white", fontsize=10)
    ax.set_xlabel("Time (ns)", color="white", fontsize=8)
    ax.set_ylabel("RMSD (nm)", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    ax.legend(fontsize=8, facecolor="#222", labelcolor="white")
    for spine in ax.spines.values(): spine.set_color("#444")

    ax = axes[0,1]
    ax.set_facecolor("#111111")
    t_gh, gyr_h = parse_xvg(f"{OUT_DIR}/gyrate_healthy.xvg")
    t_gd, gyr_d = parse_xvg(f"{OUT_DIR}/gyrate_damaged.xvg")
    if len(t_gh): ax.plot(t_gh, gyr_h, color="#2ecc71", linewidth=2, label="Healthy PrP")
    if len(t_gd): ax.plot(t_gd, gyr_d, color="#e74c3c", linewidth=2, label="Irradiated PrP")
    ax.set_title("Radius of Gyration (Protein Unfolding)", color="white", fontsize=10)
    ax.set_xlabel("Time (ns)", color="white", fontsize=8)
    ax.set_ylabel("Rg (nm)", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    ax.legend(fontsize=8, facecolor="#222", labelcolor="white")
    for spine in ax.spines.values(): spine.set_color("#444")

    ax = axes[1,0]
    ax.set_facecolor("#111111")
    if len(rmsd_h) and len(rmsd_d):
        ax.hist(rmsd_h, bins=30, color="#2ecc71", alpha=0.7, label=f"Healthy μ={rmsd_h.mean():.3f}nm")
        ax.hist(rmsd_d, bins=30, color="#e74c3c", alpha=0.7, label=f"Damaged μ={rmsd_d.mean():.3f}nm")
    ax.set_title("RMSD Distribution", color="white", fontsize=10)
    ax.set_xlabel("RMSD (nm)", color="white", fontsize=8)
    ax.set_ylabel("Count", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    ax.legend(fontsize=8, facecolor="#222", labelcolor="white")
    for spine in ax.spines.values(): spine.set_color("#444")

    ax = axes[1,1]
    ax.set_facecolor("#111111")
    ax.axis("off")
    h_mean = rmsd_h.mean() if len(rmsd_h) else 0
    d_mean = rmsd_d.mean() if len(rmsd_d) else 0
    h_max  = rmsd_h.max()  if len(rmsd_h) else 0
    d_max  = rmsd_d.max()  if len(rmsd_d) else 0
    gh_mean= gyr_h.mean()  if len(gyr_h)  else 0
    gd_mean= gyr_d.mean()  if len(gyr_d)  else 0
    pct_increase = ((d_mean-h_mean)/h_mean*100) if h_mean>0 else 0
    gyr_increase = ((gd_mean-gh_mean)/gh_mean*100) if gh_mean>0 else 0

    txt = (
        "DAMAGE ANALYSIS SUMMARY\n"
        "─────────────────────────────────\n"
        f"{'Metric':<22} {'Healthy':>10} {'Damaged':>10}\n"
        f"{'─'*44}\n"
        f"{'Mean RMSD (nm)':<22} {h_mean:>10.4f} {d_mean:>10.4f}\n"
        f"{'Max RMSD (nm)':<22} {h_max:>10.4f} {d_max:>10.4f}\n"
        f"{'Mean Rg (nm)':<22} {gh_mean:>10.4f} {gd_mean:>10.4f}\n"
        f"{'─'*44}\n"
        f"RMSD increase:  {pct_increase:+.1f}%\n"
        f"Rg increase:    {gyr_increase:+.1f}%\n\n"
        "CONCLUSION\n"
        "─────────────────────────────────\n"
    )
    if pct_increase > 20:
        txt += "SIGNIFICANT structural disruption\ndetected — prion templating\nfunction likely compromised."
    elif pct_increase > 5:
        txt += "MODERATE structural changes\ndetected — partial disruption\nof beta-sheet regions."
    else:
        txt += "MINIMAL structural change.\nConsider longer simulation\nor higher energy perturbation."

    ax.text(0.05, 0.95, txt, transform=ax.transAxes, color="white",
            fontsize=8, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#1a1a2e", edgecolor="#4a9eff", alpha=0.9))

    plt.tight_layout(rect=[0,0,1,0.94])
    out = os.path.join(PLOT_DIR, "gromacs_damage_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"  Saved {out}")

if __name__ == "__main__":
    print("\n  GROMACS Damage Analysis")
    print("  " + "="*50)
    plot_comparison()
    print("  Done!")
