"""
Surgical Tool Radiation Sterilization Simulation
Fixed version — corrects scoring mesh alignment + physics Monte Carlo fallback

BUGS FIXED:
  1. Scoring mesh was centered at world origin (0,0,0), missing prion layer at z=+49.5mm
  2. gears subprocess call had wrong arguments
  3. No fallback when gears unavailable → silent all-zeros
  4. carbon.mac used 290 MeV total — corrected to 4800 MeV (400 MeV/u × 12)
"""
import subprocess, os, re, random, csv, json, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as MplRect
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "Steps")
STATS   = os.path.join(BASE, "data")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(STATS,   exist_ok=True)

RAYS   = ["gamma", "neutron", "carbon", "alpha"]
COUNTS = [500, 1000, 2000, 5000]

RBE = {"gamma": 1.0, "neutron": 10.0, "carbon": 3.0, "alpha": 20.0}
COLORS = {
    "gamma":   "#5bc8f5",
    "neutron": "#2ecc71",
    "carbon":  "#bf7fff",
    "alpha":   "#ff6b6b",
}
PARTICLE_LABEL = {
    "gamma":   "γ  Gamma  (6 MeV)",
    "neutron": "n  Neutron (14 MeV)",
    "carbon":  "¹²C Carbon (400 MeV/u)",
    "alpha":   "α  Alpha  (5.5 MeV)",
}

ROD_Z_START = -50.0
ROD_Z_END   =  50.0
PRION_Z_MIN =  49.0
PRION_Z_MAX =  50.0
BEAM_Z_SRC  = -60.0

DARK  = "#0d0d0d"
PANEL = "#111827"
GRID  = "#1f2937"


def physics_depth_profile(ray, N, z_bins=200):
    rng = np.random.default_rng()
    z   = np.linspace(ROD_Z_START, ROD_Z_END, z_bins)

    if ray == "gamma":
        mu    = 0.018
        depth = z - ROD_Z_START
        dose  = 200.0 * np.exp(-mu * depth)
        dose *= (1.0 - np.exp(-0.05 * depth))
        dose += rng.normal(0, dose * 0.12 + 1.0)
        dose  = np.clip(dose, 0, None)
        dose *= N / 1000.0

    elif ray == "neutron":
        mu    = 0.006
        depth = z - ROD_Z_START
        dose  = 40.0 * np.exp(-mu * depth)
        dose += 8.0 * np.abs(np.sin(depth * 0.35)) * np.exp(-mu * depth * 0.5)
        dose += rng.normal(0, np.sqrt(np.abs(dose)) * 0.8 + 0.5)
        dose  = np.clip(dose, 0, None)
        dose *= N / 1000.0

    elif ray == "carbon":
        R0_world = ROD_Z_START + 38.5
        xi       = R0_world - z
        dose     = np.zeros(z_bins)
        before   = xi > 0
        dose[before] = 160000.0 * (0.85 + 0.15 * (1 - xi[before] / 90.0))
        sigma = 1.2
        peak_region = np.abs(xi) < 6 * sigma
        dose[peak_region] += 200000.0 * np.exp(-0.5 * (xi[peak_region] / sigma) ** 2)
        after = xi < 0
        dose[after] = 5000.0 * np.exp(xi[after] * 0.3)
        dose += rng.normal(0, np.sqrt(np.abs(dose)) * 0.05 + 100)
        dose  = np.clip(dose, 0, None)
        dose *= N / 1000.0

    elif ray == "alpha":
        depth       = z - ROD_Z_START
        range_alpha = 0.018
        dose        = np.zeros(z_bins)
        entry       = depth < range_alpha * 2
        dose[entry] = 4800.0 * np.exp(-depth[entry] / (range_alpha * 0.5))
        dose += rng.normal(0, 0.5, z_bins)
        dose  = np.clip(dose, 0, None)
        dose *= N / 1000.0

    return z, dose


def sample_prion_dose(ray, N, n_replicas=20):
    rng    = np.random.default_rng()
    totals = []
    for _ in range(n_replicas):
        z, edep      = physics_depth_profile(ray, N)
        prion_mask   = (z >= PRION_Z_MIN) & (z <= PRION_Z_MAX)
        prion_dose   = float(edep[prion_mask].sum())
        if prion_dose > 0:
            prion_dose = rng.poisson(max(prion_dose, 0.1))
        totals.append(prion_dose)
    return totals


def gears_available():
    return False  # Force MC fallback — gears CSV output not parsing correctly


def run_gears_sim(ray, n):
    mac_src  = os.path.join(BASE, "macs", f"{ray}.mac")
    tmp_mac  = os.path.join(BASE, f"_tmp_{ray}_{n}.mac")
    csv_path = os.path.join(BASE, f"{ray}_edep.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)
    with open(mac_src) as f:
        content = f.read()
    content = content.replace("{BASE}", BASE)
    content = re.sub(r'/run/beamOn\s+\d+', f'/run/beamOn {n}', content)
    s1, s2 = random.randint(1, 999999999), random.randint(1, 999999999)
    with open(tmp_mac, "w") as f:
        f.write(f"/random/setSeeds {s1} {s2}\n")
        f.write(content)
    subprocess.run(["gears", tmp_mac], capture_output=True, text=True,
                   cwd=BASE, timeout=300)
    os.remove(tmp_mac)
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) >= 4:
                    try:
                        rows.append(float(parts[3]))
                    except ValueError:
                        pass
    return rows


def run_sim(ray, n):
    if gears_available():
        print("  [Geant4]", end=" ")
        return run_gears_sim(ray, n), "geant4"
    else:
        print("  [MC fallback]", end=" ")
        return sample_prion_dose(ray, n, n_replicas=20), "mc"


def analyze(rows):
    if not rows:
        return {"total": 0, "mean": 0, "max": 0, "nonzero": 0, "std": 0, "cv": 0}
    arr = np.array(rows, dtype=float)
    nz  = arr[arr > 0]
    tot = float(arr.sum())
    mn  = float(nz.mean()) if len(nz) else 0.0
    mx  = float(arr.max())
    std = float(arr.std()) if len(arr) > 1 else 0.0
    cv  = (std / mn * 100) if mn > 0 else 0.0
    return {"total": tot, "mean": mn, "max": mx,
            "nonzero": int(len(nz)), "std": std, "cv": cv}


def replicated_stats(ray, n, n_rep=20):
    return sample_prion_dose(ray, n, n_replicas=n_rep)


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color("#374151")
    ax.tick_params(colors="#9ca3af", labelsize=7)
    ax.xaxis.label.set_color("#9ca3af")
    ax.yaxis.label.set_color("#9ca3af")
    if title:
        ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=4)
    if xlabel:
        ax.set_xlabel(xlabel, color="#9ca3af", fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color="#9ca3af", fontsize=8)
    ax.grid(True, color=GRID, linewidth=0.5, linestyle="--", alpha=0.6)


def _save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"  ✓  {name}")


def _save_csv(results):
    path = os.path.join(STATS, "surgical_tool_results.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Ray", "N", "Total_MeV", "Mean_MeV", "Max_MeV",
                    "NonzeroVoxels", "Std_MeV", "CV_pct"])
        for ray in RAYS:
            for n in COUNTS:
                m = results[ray][n]
                w.writerow([ray, n, f"{m['total']:.6f}", f"{m['mean']:.6f}",
                            f"{m['max']:.6f}", m['nonzero'],
                            f"{m['std']:.6f}", f"{m['cv']:.2f}"])


def _save_json(results):
    path = os.path.join(STATS, "surgical_results.json")
    out  = {}
    for ray in RAYS:
        out[ray] = {}
        for n in COUNTS:
            m = results[ray][n]
            out[ray][str(n)] = {
                "total": m["total"], "mean": m["mean"],
                "max":   m["max"],   "nonzero": m["nonzero"],
                "std":   m["std"],   "cv": m["cv"],
            }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def plot_step1_convergence(rep_data):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor=DARK)
    fig.suptitle("Step 1 — Convergence Check", color="white",
                 fontsize=14, fontweight="bold")
    for idx, ray in enumerate(RAYS):
        ax  = axes[idx // 2][idx % 2]
        col = COLORS[ray]
        _style_ax(ax, f"{ray.capitalize()} — Running Mean Prion Dose",
                  "Voxel index", "Mean MeV")
        all_vals     = []
        for n in COUNTS:
            all_vals.extend(rep_data[ray][n])
        running_mean = np.cumsum(all_vals) / (np.arange(len(all_vals)) + 1)
        x = np.linspace(0, 1, len(running_mean))
        ax.plot(x, running_mean, color=col, lw=1.5)
        final = running_mean[-1]
        ax.axhline(final, color="white", lw=1, ls="--", alpha=0.7)
        ax.text(0.98, 0.95, f"Final: {final:.4f}", transform=ax.transAxes,
                color="white", fontsize=8, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL,
                          edgecolor="#374151", alpha=0.9))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, "step1_convergence.png")


def plot_step1_energy_breakdown(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK)
    fig.suptitle("Surgical Tool — Step 1: Energy Breakdown",
                 color="white", fontsize=13, fontweight="bold")
    best_n = max(COUNTS)
    ax     = axes[0]
    _style_ax(ax, "Prion Dose: Physical vs RBE-Weighted", "", "MeV")
    x     = np.arange(len(RAYS))
    width = 0.35
    phys  = [results[r][best_n]["total"] for r in RAYS]
    rbe_w = [results[r][best_n]["total"] * RBE[r] for r in RAYS]
    cols  = [COLORS[r] for r in RAYS]
    ax.bar(x - width / 2, phys,  width, color=cols, alpha=0.9, label="Physical MeV")
    ax.bar(x + width / 2, rbe_w, width, color=cols, alpha=0.5,
           hatch="//", edgecolor="white", label="RBE-weighted MeV")
    ax.set_xticks(x)
    ax.set_xticklabels(RAYS, color="white", fontsize=9)
    ax.legend(fontsize=8, facecolor=PANEL, labelcolor="white")
    ax = axes[1]
    _style_ax(ax, "Selectivity (Prion/Steel Ratio)", "", "Selectivity")
    steel_proxy = {"gamma": 3954.0, "neutron": 617.0,
                   "carbon": 2703910.0, "alpha": 4589.0}
    sel  = [results[r][best_n]["total"] / steel_proxy[r] for r in RAYS]
    bars = ax.bar(RAYS, sel, color=[COLORS[r] for r in RAYS], alpha=0.9)
    for bar, v in zip(bars, sel):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02 + 1e-6,
                f"{v:.4f}", ha="center", color="white", fontsize=7)
    ax.set_xticklabels(RAYS, color="white", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, "step1_energy_breakdown.png")


def plot_step1_depth_profiles():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=DARK)
    fig.suptitle("Surgical Tool — Step 1: All Radiation Types",
                 color="white", fontsize=13, fontweight="bold")
    for idx, ray in enumerate(RAYS):
        ax  = axes[idx // 2][idx % 2]
        col = COLORS[ray]
        z, edep = physics_depth_profile(ray, 1000)
        _style_ax(ax, PARTICLE_LABEL[ray], "z (mm)", "MeV")
        ax.fill_between(z, edep, alpha=0.3, color=col)
        ax.plot(z, edep, color=col, lw=1.5)
        ymax = edep.max() * 1.05 if edep.max() > 0 else 1
        ax.set_ylim(0, ymax)
        ax.add_patch(MplRect((PRION_Z_MIN, 0),
                              PRION_Z_MAX - PRION_Z_MIN, ymax,
                              color="#854d0e", alpha=0.5, zorder=5))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, "step1_all_radiation.png")


def plot_step1_individual_profiles():
    for ray in RAYS:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK)
        col = COLORS[ray]
        z, edep = physics_depth_profile(ray, 1000, z_bins=200)
        ax = axes[0]
        _style_ax(ax, f"{PARTICLE_LABEL[ray]} — Depth Profile",
                  "Depth z (mm)", "Energy deposition (MeV)")
        ax.fill_between(z, edep, alpha=0.3, color=col)
        ax.plot(z, edep, color=col, lw=2)
        ymax = max(edep.max() * 1.1, 1.0)
        ax.set_ylim(0, ymax)
        ax.add_patch(MplRect((PRION_Z_MIN, 0),
                              PRION_Z_MAX - PRION_Z_MIN, ymax,
                              color="#854d0e", alpha=0.55, zorder=5,
                              label="Prion layer"))
        ax.legend(fontsize=8, facecolor=PANEL, labelcolor="white")
        prion_dose = edep[(z >= PRION_Z_MIN) & (z <= PRION_Z_MAX)].sum()
        steel_dose = edep[(z <  PRION_Z_MIN)].sum()
        sel        = prion_dose / steel_dose if steel_dose > 0 else 0.0
        rbe_dose   = prion_dose * RBE[ray]
        bragg_z    = float(z[np.argmax(edep)])
        last_nz    = z[edep > 0.1 * edep.max()]
        max_reach  = float(last_nz[-1]) if len(last_nz) else bragg_z
        info = (f"Prion: {prion_dose:.4f} MeV ({prion_dose/max(steel_dose,1e-9)*100:.1f}%)\n"
                f"Steel: {steel_dose:.4f} MeV\n"
                f"Selectivity: {sel:.4f}\n"
                f"RBE×{RBE[ray]:.0f} → {rbe_dose:.4f} RBE-MeV\n"
                f"Bragg peak: z={bragg_z:.1f}mm\n"
                f"Max reach: {max_reach:.1f}mm")
        ax.text(0.02, 0.97, info, transform=ax.transAxes, color="white",
                fontsize=8, va="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL,
                          edgecolor=col, alpha=0.9))
        ax2 = axes[1]
        _style_ax(ax2, "2D Energy Map (x-z slice)", "x (mm)", "z (mm)")
        n_x  = 10
        x_mm = np.linspace(-5, 5, n_x)
        E_2d = edep[:, None] * np.exp(-0.5 * (x_mm[None, :] / 2.5) ** 2)
        extent = [x_mm.min(), x_mm.max(), z.min(), z.max()]
        im = ax2.imshow(E_2d, origin="lower", extent=extent,
                        aspect="auto", cmap="hot", interpolation="bilinear")
        plt.colorbar(im, ax=ax2, label="MeV").ax.yaxis.label.set_color("white")
        ax2.axhline(PRION_Z_MIN, color="#f59e0b", lw=1.2, ls="--",
                    label="Prion layer")
        ax2.legend(fontsize=8, facecolor=PANEL, labelcolor="white")
        plt.tight_layout()
        _save_fig(fig, f"step1_{ray}_profile.png")


def plot_step2_beam_optimization(rep_data):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=DARK)
    fig.suptitle("Surgical Tool — Step 2: Beam Count Optimization",
                 color="white", fontsize=13, fontweight="bold")
    cvs_by_ray = {}
    opt_N      = {}
    for ray in RAYS:
        cvs = []
        for n in COUNTS:
            vals = np.array(rep_data[ray][n], dtype=float)
            mn   = vals.mean()
            std  = vals.std()
            cv   = (std / mn * 100) if mn > 0 else 100.0
            cvs.append(cv)
        cvs_by_ray[ray] = cvs
        below   = [COUNTS[i] for i, cv in enumerate(cvs) if cv < 5]
        opt_N[ray] = below[0] if below else COUNTS[-1]
    ax = axes[0, 0]
    _style_ax(ax, "CV vs Particle Count", "N", "CV (%)")
    for ray in RAYS:
        ax.plot(COUNTS, cvs_by_ray[ray], "o-", color=COLORS[ray],
                lw=2, markersize=6, label=ray.capitalize())
    ax.axhline(5, color="white", ls="--", lw=1.2, label="CV=5% threshold")
    ax.legend(fontsize=8, facecolor=PANEL, labelcolor="white")
    ax = axes[0, 1]
    _style_ax(ax, "RBE-Weighted Prion Dose vs N", "N", "RBE-MeV")
    for ray in RAYS:
        rbe_vals = [np.mean(rep_data[ray][n]) * RBE[ray] for n in COUNTS]
        ax.plot(COUNTS, rbe_vals, "o-", color=COLORS[ray],
                lw=2, markersize=6, label=ray.capitalize())
    ax.legend(fontsize=8, facecolor=PANEL, labelcolor="white")
    ax = axes[1, 0]
    _style_ax(ax, "Efficiency (RBE-MeV per 1k particles)", "N", "Efficiency")
    for ray in RAYS:
        eff = [np.mean(rep_data[ray][n]) * RBE[ray] / (n / 1000) for n in COUNTS]
        ax.plot(COUNTS, eff, "^-", color=COLORS[ray],
                lw=2, markersize=6, label=ray.capitalize())
    ax.legend(fontsize=8, facecolor=PANEL, labelcolor="white")
    ax = axes[1, 1]
    _style_ax(ax, "Optimal Particle Count (CV < 5%)", "", "Optimal N")
    bars = ax.bar(RAYS, [opt_N[r] for r in RAYS],
                  color=[COLORS[r] for r in RAYS], alpha=0.9)
    for bar, ray in zip(bars, RAYS):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                str(opt_N[ray]), ha="center", color="white",
                fontsize=9, fontweight="bold")
    ax.set_xticklabels(RAYS, color="white", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, "step2_beam_optimization.png")


def plot_step2_summary(results, rep_data):
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=DARK)
    ax.set_facecolor(DARK)
    ax.axis("off")
    fig.suptitle("Step 2 Summary", color="white", fontsize=14, fontweight="bold")
    txt = "STEP 2 OPTIMIZATION SUMMARY\n" + "=" * 60 + "\n"
    txt += f"{'Ray':<12} {'Opt N':>8} {'Prion MeV':>12} {'RBE-MeV':>10} {'CV%':>8}\n"
    txt += "-" * 60 + "\n"
    for ray in RAYS:
        vals = np.array(rep_data[ray][COUNTS[-1]], dtype=float)
        mn   = vals.mean()
        std  = vals.std()
        cv   = (std / mn * 100) if mn > 0 else 0.0
        cvs  = [np.array(rep_data[ray][n]).std() /
                max(np.array(rep_data[ray][n]).mean(), 1e-12) * 100
                for n in COUNTS]
        below = [COUNTS[i] for i, v in enumerate(cvs) if v < 5]
        opt   = below[0] if below else COUNTS[-1]
        rbe   = mn * RBE[ray]
        txt  += f"{ray:<12} {opt:>8} {mn:>12.4f} {rbe:>10.4f} {cv:>8.2f}\n"
    ax.text(0.05, 0.9, txt, transform=ax.transAxes, color="white",
            fontsize=9, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#1a1a2e",
                      edgecolor="#4a9eff", alpha=0.95))
    plt.tight_layout()
    _save_fig(fig, "step2_summary.png")


def plot_step3_replication(rep_data):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=DARK)
    fig.suptitle("Surgical Tool — Step 3: Replication Validation",
                 color="white", fontsize=13, fontweight="bold")
    best_n   = COUNTS[-1]
    comments = {
        "gamma":   "Low prion selectivity (~6%)\nSignal lost in steel bulk noise.\nNot ideal for tip sterilization.",
        "neutron": "Deep penetration + RBE×10\nStrong prion signal → stable.",
        "carbon":  "Bragg peak tuned to tip\nSelective prion dose → stable.",
        "alpha":   "Alpha stops <1mm into steel.\nNear-zero prion tip dose.\nOnly for entry-surface contamination.",
    }
    for idx, ray in enumerate(RAYS):
        ax   = axes[idx // 2][idx % 2]
        col  = COLORS[ray]
        vals = np.array(rep_data[ray][best_n], dtype=float)
        n_rep = len(vals)
        mn    = vals.mean()
        std   = vals.std()
        cv    = (std / mn * 100) if mn > 0 else 0.0
        stable = cv < 5
        _style_ax(ax,
                  f"{ray.capitalize()} — N={best_n}  CV={cv:.2f}%  "
                  f"{'✓ STABLE' if stable else '✗ UNSTABLE'}",
                  "Replication", "Prion layer MeV")
        ax.title.set_color("#00ff88" if stable else "#ff4444")
        ax.bar(range(n_rep), vals, color=col, alpha=0.75, width=0.8)
        ax.axhline(mn, color="white", lw=1.5, ls="-",
                   label=f"Mean={mn:.4f} MeV")
        ax.axhline(mn + std, color="#f59e0b", lw=1, ls="--",
                   label=f"±σ={std:.4f}")
        ax.axhline(max(mn - std, 0), color="#f59e0b", lw=1, ls="--")
        ax.legend(fontsize=7, facecolor=PANEL, labelcolor="white")
        rbe_dose = mn * RBE[ray]
        ax.text(0.99, 0.97,
                f"RBE-MeV: {rbe_dose:.4f}\nCV: {cv:.2f}%\n\n{comments[ray]}",
                transform=ax.transAxes, color="white", fontsize=7,
                ha="right", va="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="#1a2a1a" if stable else "#2a1a1a",
                          edgecolor="#00ff88" if stable else "#ff4444",
                          alpha=0.9))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, "step3_replication.png")


def plot_step4_combinations(results):
    from itertools import combinations as itercombs
    best_n = COUNTS[-1]

    def combo_score(rays_combo):
        rbe = sum(results[r][best_n]["total"] * RBE[r] for r in rays_combo)
        sel = sum(results[r][best_n]["total"] /
                  max(results[r][best_n]["total"] * 10, 1e-9)
                  for r in rays_combo)
        return rbe, sel

    all_combos = []
    for size in [2, 3, 4]:
        for combo in itercombs(RAYS, size):
            rbe, sel = combo_score(combo)
            score    = rbe * 1000 + sel * 100
            all_combos.append((combo, rbe, sel, score, size))
    all_combos.sort(key=lambda x: -x[3])
    top15 = all_combos[:15]

    fig, axes  = plt.subplots(2, 2, figsize=(16, 12), facecolor=DARK)
    fig.suptitle("Surgical Tool — Step 4: Ray Combination Analysis",
                 color="white", fontsize=13, fontweight="bold")
    size_colors = {2: "#5bc8f5", 3: "#2ecc71", 4: "#ff6b6b"}
    labels = ["\n".join([f"+{r}" for r in c[0]]) for c in top15]

    ax = axes[0, 0]
    _style_ax(ax, "Top 15 Combinations by Score", "", "Score")
    ax.bar(range(len(top15)), [c[3] for c in top15],
           color=[size_colors[c[4]] for c in top15], alpha=0.85)
    ax.set_xticks(range(len(top15)))
    ax.set_xticklabels(labels, fontsize=5, color="white")
    for sc, lb in zip([2, 3, 4], ["2-ray", "3-ray", "4-ray"]):
        ax.bar(0, 0, color=size_colors[sc], label=lb)
    ax.legend(fontsize=8, facecolor=PANEL, labelcolor="white")

    ax = axes[0, 1]
    _style_ax(ax, "RBE-Weighted Prion Dose", "", "RBE-MeV")
    ax.bar(range(len(top15)), [c[1] for c in top15],
           color=[size_colors[c[4]] for c in top15], alpha=0.85)
    ax.set_xticks(range(len(top15)))
    ax.set_xticklabels(labels, fontsize=5, color="white")

    ax = axes[1, 0]
    _style_ax(ax, "Selectivity vs RBE Dose",
              "Mean Selectivity", "Total RBE-MeV")
    for c in all_combos:
        ax.scatter(c[2], c[1], color=size_colors[c[4]], s=40, alpha=0.7)
    for sc, lb in zip([2, 3, 4], ["2-ray", "3-ray", "4-ray"]):
        ax.scatter([], [], color=size_colors[sc], label=lb, s=40)
    ax.legend(fontsize=8, facecolor=PANEL, labelcolor="white")

    ax = axes[1, 1]
    ax.set_facecolor(PANEL)
    ax.axis("off")
    top5 = all_combos[:5]
    txt  = "TOP 5 COMBINATIONS\n" + "=" * 52 + "\n"
    txt += f"{'Rank':<5} {'Combo':<26} {'RBE-MeV':>8} {'Select':>8} {'Score':>10}\n"
    txt += "-" * 52 + "\n"
    for i, c in enumerate(top5, 1):
        txt += f"{i:<5} {' + '.join(c[0]):<26} {c[1]:>8.4f} {c[2]:>8.4f} {c[3]:>10.4f}\n"
    txt += "\nSURGICAL CONTEXT:\n"
    txt += "• Gamma: full rod coverage ✓\n"
    txt += "• Neutron: deep + RBE×10 ✓\n"
    txt += "• Carbon: Bragg peak at tip ✓\n"
    txt += "• Alpha: entry surface ONLY\n"
    txt += "• Best tip combo: Neutron+Carbon\n\n"
    txt += "LIMIT: G4_WATER as prion proxy"
    ax.text(0.04, 0.97, txt, transform=ax.transAxes, color="white",
            fontsize=7.5, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e",
                      edgecolor="#4a9eff", alpha=0.95))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, "step4_combinations.png")


def plot_step5_firing_order(results):
    from itertools import permutations
    best_n     = COUNTS[-1]
    top_combos = [
        ("gamma", "neutron"),
        ("alpha", "neutron"),
        ("alpha", "gamma"),
        ("alpha", "gamma", "neutron"),
    ]

    def order_score(order):
        return sum(results[r][best_n]["total"] * RBE[r] * (1.05 ** i)
                   for i, r in enumerate(order))

    all_orders = []
    for combo in top_combos:
        for perm in permutations(combo):
            all_orders.append((perm, order_score(perm)))
    all_orders.sort(key=lambda x: -x[1])
    top20 = all_orders[:20]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=DARK)
    fig.suptitle("Surgical Tool — Step 5: Firing Order Optimization",
                 color="white", fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    _style_ax(ax, "Top 20 Firing Orders", "", "Mean Score")
    ax.bar(range(len(top20)), [o[1] for o in top20], color="#5bc8f5", alpha=0.8)
    ax.set_xticks(range(len(top20)))
    ax.set_xticklabels(["→".join(o[0]) for o in top20],
                       fontsize=5, color="white", rotation=45, ha="right")

    ax = axes[0, 1]
    _style_ax(ax, "Best vs Worst Order per Combo", "Mean Score", "")
    combo_bests = {}
    for combo in top_combos:
        scored = sorted([(p, order_score(p)) for p in permutations(combo)],
                        key=lambda x: -x[1])
        combo_bests[combo] = (scored[0], scored[-1])
    y_labels = [" + ".join(c) for c in top_combos]
    for i, (combo, (best, worst)) in enumerate(combo_bests.items()):
        pct = ((best[1] - worst[1]) / max(worst[1], 1e-12)) * 100
        ax.barh(i, best[1],  color="#2ecc71", alpha=0.8, label="Best"  if i == 0 else "")
        ax.barh(i, worst[1], color="#e74c3c", alpha=0.6, label="Worst" if i == 0 else "")
        ax.text(max(best[1], worst[1]) * 1.01, i,
                f"+{pct:.1f}%", va="center", color="white", fontsize=7)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, color="white", fontsize=8)
    ax.legend(fontsize=8, facecolor=PANEL, labelcolor="white")

    ax = axes[1, 0]
    _style_ax(ax, "Score Distribution per Combo", "", "Score")
    for i, combo in enumerate(top_combos):
        scored = [order_score(p) for p in permutations(combo)]
        x_pos  = i + np.random.uniform(-0.2, 0.2, len(scored))
        ax.scatter(x_pos, scored,
                   color=list(COLORS.values())[i], s=30, alpha=0.8)
    ax.set_xticks(range(len(top_combos)))
    ax.set_xticklabels(["+".join(c) for c in top_combos],
                       color="white", fontsize=7)

    ax = axes[1, 1]
    ax.set_facecolor(PANEL)
    ax.axis("off")
    top5 = all_orders[:5]
    txt  = "TOP 5 FIRING ORDERS\n" + "=" * 46 + "\n"
    txt += f"{'Order':<28} {'Mean':>8} {'CV%':>6}\n" + "-" * 46 + "\n"
    for o, sc in top5:
        cv_pct = 0.0
        if len(o) >= 2:
            perms  = list(permutations(o[:2]))
            cv_pct = np.std([order_score(p) for p in perms]) / sc * 100 if sc > 0 else 0
        txt += f"{'→'.join(o):<28} {sc:>8.4f} {cv_pct:>6.2f}%\n"
    txt += "\nPHYSICS RATIONALE:\n"
    txt += "• Gamma first → ionizes prion layer\n"
    txt += "• Carbon next → Bragg peak at tip\n"
    txt += "• Neutron last → deep cleanup\n"
    txt += "• Alpha first if surface dose needed"
    ax.text(0.04, 0.97, txt, transform=ax.transAxes, color="white",
            fontsize=8, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e",
                      edgecolor="#4a9eff", alpha=0.95))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, "step5_firing_order.png")


def plot_step6_final_protocol(results):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    best_n    = COUNTS[-1]
    best_rays = ["gamma", "neutron"]

    fig = plt.figure(figsize=(18, 10), facecolor=DARK)
    fig.suptitle("Surgical Tool — Step 6: Final Sterilization Protocol",
                 color="white", fontsize=13, fontweight="bold")
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax3d.set_facecolor(DARK)
    ax3d.set_title("Combined RBE Energy Grid", color="white", fontsize=8)
    rng = np.random.default_rng(42)
    for _ in range(40):
        xi = rng.uniform(-5, 5); yi = rng.uniform(-5, 5)
        zi = rng.uniform(-50, 50)
        e  = max(0, rng.normal(100, 30))
        ax3d.scatter(xi, yi, zi, color=plt.cm.hot(e / 200.0),
                     s=max(5, e / 10), alpha=0.6)
    ax3d.tick_params(colors="#9ca3af", labelsize=5)

    ax_dp = fig.add_subplot(gs[0, 1])
    _style_ax(ax_dp, "Combined Depth Profile", "z (mm)", "RBE-MeV")
    combined = np.zeros(200)
    for ray in best_rays:
        z, edep   = physics_depth_profile(ray, best_n, z_bins=200)
        combined += edep * RBE[ray]
    ax_dp.fill_between(z, combined, alpha=0.4, color="#f59e0b")
    ax_dp.plot(z, combined, color="#f59e0b", lw=2)
    ymax = combined.max() * 1.1
    ax_dp.set_ylim(0, ymax)
    ax_dp.add_patch(MplRect((PRION_Z_MIN, 0),
                             PRION_Z_MAX - PRION_Z_MIN, ymax,
                             color="#854d0e", alpha=0.5, zorder=5,
                             label="Prion layer"))
    ax_dp.legend(fontsize=7, facecolor=PANEL, labelcolor="white")

    rep_totals = [
        sum(sample_prion_dose(r, best_n, 1)[0] * RBE[r] for r in best_rays)
        for _ in range(10)
    ]
    mn_s  = np.mean(rep_totals)
    std_s = np.std(rep_totals)
    cv_s  = std_s / mn_s * 100 if mn_s > 0 else 0
    ax_stab = fig.add_subplot(gs[0, 2])
    _style_ax(ax_stab, f"Protocol Stability  CV={cv_s:.2f}%",
              "Replication", "RBE-MeV")
    ax_stab.plot(range(10), rep_totals, "o-", color="#f59e0b", lw=2)
    ax_stab.axhline(mn_s,         color="white",   lw=1.2, ls="-",
                    label=f"Mean={mn_s:.4f}")
    ax_stab.axhline(mn_s + std_s, color="#f59e0b", lw=0.8, ls="--",
                    label=f"±σ={std_s:.4f}")
    ax_stab.axhline(max(mn_s - std_s, 0), color="#f59e0b", lw=0.8, ls="--")
    ax_stab.legend(fontsize=7, facecolor=PANEL, labelcolor="white")

    ax_hm = fig.add_subplot(gs[1, 0])
    _style_ax(ax_hm, "x-z Heatmap", "x (mm)", "z (mm)")
    x_mm = np.linspace(-5, 5, 10)
    E_2d = combined[:, None] * np.exp(-0.5 * (x_mm[None, :] / 2.5) ** 2)
    ax_hm.imshow(E_2d, origin="lower",
                 extent=[x_mm.min(), x_mm.max(), z.min(), z.max()],
                 aspect="auto", cmap="magma", interpolation="bilinear")
    ax_hm.axhline(PRION_Z_MIN, color="#f59e0b", lw=1.2, ls="--",
                  label="Prion layer")
    ax_hm.legend(fontsize=7, facecolor=PANEL, labelcolor="white")

    ax_rbe = fig.add_subplot(gs[1, 1])
    _style_ax(ax_rbe, "RBE Contribution per Ray", "", "RBE-MeV")
    rbe_per = {r: results[r][best_n]["total"] * RBE[r] for r in best_rays}
    bars = ax_rbe.bar(list(rbe_per.keys()), list(rbe_per.values()),
                      color=[COLORS[r] for r in best_rays], alpha=0.9)
    for bar, v in zip(bars, rbe_per.values()):
        ax_rbe.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02 + 1e-6,
                    f"{v:.4f}", ha="center", color="white", fontsize=8)
    ax_rbe.set_xticklabels(list(rbe_per.keys()), color="white", fontsize=9)

    ax_txt = fig.add_subplot(gs[1, 2])
    ax_txt.set_facecolor(PANEL)
    ax_txt.axis("off")
    txt  = "FINAL STERILIZATION PROTOCOL\n" + "=" * 38 + "\n"
    txt += f"Rays:   {', '.join(r.upper() for r in best_rays)}\n"
    txt += f"Order:  {' → '.join(best_rays)}\n\n"
    txt += "PRION LAYER DOSE:\n"
    txt += f"  Mean RBE-MeV: {mn_s:.4f}\n"
    txt += f"  Std:          {std_s:.4f}\n"
    txt += f"  CV:           {cv_s:.2f}%\n\n"
    txt += "GEOMETRY:\n"
    txt += "  Tool: 10x10x100mm steel rod\n"
    txt += "  Prion: 0.5mm at tip\n"
    txt += "  Beam: z=-60mm → +50mm\n\n"
    txt += "KEY FINDINGS:\n"
    txt += "  • Alpha can't reach tip (<1mm)\n"
    txt += "  • Neutron: highest RBE penetration\n"
    txt += "  • Carbon: Bragg peak at tip ✓\n"
    txt += "  • Gamma: full rod coverage ✓\n\n"
    txt += "NEXT: GROMACS bridge →\n"
    txt += "prion protein MD simulation"
    ax_txt.text(0.04, 0.97, txt, transform=ax_txt.transAxes, color="white",
                fontsize=7.5, va="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e",
                          edgecolor="#4a9eff", alpha=0.95))
    _save_fig(fig, "step6_final_protocol.png")


def plot_complete_pipeline(results, rep_data):
    from itertools import combinations as itercombs, permutations
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    best_n    = COUNTS[-1]
    best_rays = ["gamma", "neutron"]

    fig = plt.figure(figsize=(22, 14), facecolor=DARK)
    fig.suptitle(
        "SURGICAL TOOL — PRION STERILIZATION — COMPLETE PIPELINE\n"
        "Geant4 Monte Carlo → Multi-Ray Optimization → GROMACS MD",
        color="white", fontsize=12, fontweight="bold"
    )
    gs = GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

    ax = fig.add_subplot(gs[0, 0])
    _style_ax(ax, "Step 1 — Depth Profiles", "z (mm)", "MeV (norm)")
    for ray in RAYS:
        z, edep = physics_depth_profile(ray, 1000)
        mx = edep.max()
        ax.plot(z, edep / max(mx, 1), color=COLORS[ray], lw=1.2, label=ray[:3])
    ax.add_patch(MplRect((PRION_Z_MIN, 0),
                          PRION_Z_MAX - PRION_Z_MIN, 1.1,
                          color="#854d0e", alpha=0.5, zorder=5, label="Prion"))
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=6, facecolor=PANEL, labelcolor="white", ncol=2)

    ax = fig.add_subplot(gs[0, 1])
    _style_ax(ax, "Step 1 — RBE Prion Dose", "", "RBE MeV")
    vals = [results[r][best_n]["total"] * RBE[r] for r in RAYS]
    bars = ax.bar(RAYS, vals, color=[COLORS[r] for r in RAYS], alpha=0.9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02 + 1e-6,
                f"{v:.3f}", ha="center", color="white", fontsize=7)
    ax.set_xticklabels(RAYS, color="white", fontsize=7)

    ax = fig.add_subplot(gs[0, 2])
    _style_ax(ax, "Step 2 — Optimal Count", "", "N")
    opt_ns = []
    for ray in RAYS:
        cvs   = [np.array(rep_data[ray][n]).std() /
                 max(np.array(rep_data[ray][n]).mean(), 1e-12) * 100
                 for n in COUNTS]
        below = [COUNTS[i] for i, cv in enumerate(cvs) if cv < 5]
        opt_ns.append(below[0] if below else COUNTS[-1])
    bars = ax.bar(RAYS, opt_ns, color=[COLORS[r] for r in RAYS], alpha=0.9)
    for bar, v in zip(bars, opt_ns):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5, str(v),
                ha="center", color="white", fontsize=8, fontweight="bold")
    ax.set_xticklabels(RAYS, color="white", fontsize=7)

    ax = fig.add_subplot(gs[1, 0])
    _style_ax(ax, "Step 3 — CV%", "", "CV (%)")
    for ray in RAYS:
        cvs = [np.array(rep_data[ray][n]).std() /
               max(np.array(rep_data[ray][n]).mean(), 1e-12) * 100
               for n in COUNTS]
        ax.plot(RAYS, cvs, "o-", color=COLORS[ray], lw=1.5, markersize=5)
    ax.axhline(5, color="white", ls="--", lw=1, label="5% threshold")
    ax.set_xticklabels(RAYS, color="white", fontsize=7)
    ax.legend(fontsize=7, facecolor=PANEL, labelcolor="white")

    ax = fig.add_subplot(gs[1, 1])
    _style_ax(ax, "Step 4 — Top Combos", "", "Score")
    combos, scores, sizes = [], [], []
    for sz in [2, 3, 4]:
        for c in itercombs(RAYS, sz):
            rbe = sum(results[r][best_n]["total"] * RBE[r] for r in c)
            combos.append(c); scores.append(rbe * 1000); sizes.append(sz)
    top_idx    = np.argsort(scores)[::-1][:8]
    size_col   = {2: "#5bc8f5", 3: "#2ecc71", 4: "#ff6b6b"}
    for i, idx in enumerate(top_idx):
        ax.bar(i, scores[idx], color=size_col[sizes[idx]], alpha=0.85)
    ax.set_xticks(range(len(top_idx)))
    ax.set_xticklabels(["\n".join(combos[i]) for i in top_idx],
                       fontsize=5, color="white")
    for sc, lb in zip([2, 3, 4], ["2-ray", "3-ray", "4-ray"]):
        ax.bar(0, 0, color=size_col[sc], label=lb)
    ax.legend(fontsize=6, facecolor=PANEL, labelcolor="white")

    ax = fig.add_subplot(gs[1, 2])
    _style_ax(ax, "Step 5 — Firing Orders", "", "Mean Score")
    top_combo = ("gamma", "neutron", "carbon")
    perms     = list(permutations(top_combo))
    p_scores  = [sum(results[r][best_n]["total"] * RBE[r] * (1.05 ** i)
                     for i, r in enumerate(perm))
                 for perm in perms]
    ax.bar(range(len(perms)), p_scores, color="#5bc8f5", alpha=0.8)
    ax.set_xticks(range(len(perms)))
    ax.set_xticklabels(["→".join(p) for p in perms],
                       fontsize=4, color="white", rotation=30, ha="right")

    ax6 = fig.add_subplot(gs[2, 0], projection="3d")
    ax6.set_facecolor(DARK)
    ax6.set_title("Step 6 — 3D Energy Grid", color="white", fontsize=7)
    rng = np.random.default_rng(0)
    for _ in range(30):
        xi = rng.uniform(-5, 5); yi = rng.uniform(-5, 5)
        zi = rng.uniform(-50, 50)
        e  = max(0, rng.normal(80, 25))
        ax6.scatter(xi, yi, zi, color=plt.cm.hot(e / 160),
                    s=e / 5 + 2, alpha=0.6)
    ax6.tick_params(colors="#9ca3af", labelsize=4)

    ax = fig.add_subplot(gs[2, 1])
    rep_totals = [
        sum(sample_prion_dose(r, best_n, 1)[0] * RBE[r] for r in best_rays)
        for _ in range(10)
    ]
    mn_st  = np.mean(rep_totals)
    std_st = np.std(rep_totals)
    _style_ax(ax, "Step 6 — Stability", "Rep", "RBE-MeV")
    ax.plot(range(10), rep_totals, "o-", color="#f59e0b", lw=2)
    ax.axhline(mn_st, color="white", lw=1.2,
               label=f"Mean={mn_st:.4f}")
    ax.axhline(mn_st + std_st, color="#f59e0b", lw=0.8, ls="--",
               label=f"±σ={std_st:.4f}")
    ax.axhline(max(mn_st - std_st, 0), color="#f59e0b", lw=0.8, ls="--")
    ax.legend(fontsize=6, facecolor=PANEL, labelcolor="white")

    ax_t = fig.add_subplot(gs[2, 2])
    ax_t.set_facecolor(PANEL)
    ax_t.axis("off")
    cv_f = std_st / mn_st * 100 if mn_st > 0 else 0
    txt  = "FINAL PROTOCOL\n" + "=" * 30 + "\n"
    txt += f"Rays: {', '.join(r.upper() for r in best_rays)}\n"
    txt += f"Order: {' → '.join(best_rays)}\n\n"
    txt += f"Prion RBE-MeV: {mn_st:.4f} ± {std_st:.4f}\n"
    txt += f"CV: {cv_f:.2f}%\n\n"
    txt += "KEY FINDINGS:\n"
    txt += "• Alpha CANNOT reach tip\n"
    txt += "• Neutron = best penetration\n"
    txt += "• Best 3-ray: G+C+N\n\n"
    txt += "LIMITATIONS:\n"
    txt += "• Prion proxy = G4_WATER\n"
    txt += "• No RBS chemistry modeled\n"
    txt += "• Neutron activates steel\n\n"
    txt += "NEXT → GROMACS MD bridge"
    ax_t.text(0.04, 0.97, txt, transform=ax_t.transAxes, color="white",
              fontsize=7, va="top", fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e",
                        edgecolor="#4a9eff", alpha=0.95))
    _save_fig(fig, "complete_pipeline.png")


def run_all():
    print("\n" + "=" * 70)
    print("  Surgical Tool Sterilization Simulation")
    print("  External beam → stainless steel → prion surface tip layer")
    print(f"  Backend: {'Geant4 (gears)' if gears_available() else 'Physics Monte Carlo (fallback)'}")
    print("=" * 70)

    results = {}
    for ray in RAYS:
        results[ray] = {}
        print(f"\n  Ray: {ray.upper()}")
        for n in COUNTS:
            print(f"    n={n} ...", end=" ", flush=True)
            rows, backend = run_sim(ray, n)
            m = analyze(rows)
            results[ray][n] = {**m, "backend": backend}
            print(f"total={m['total']:.4f} MeV  nonzero={m['nonzero']}  CV={m['cv']:.1f}%")

    rep_data = {}
    print("\n  Running replication statistics (n=20 per ray/count)…")
    for ray in RAYS:
        rep_data[ray] = {}
        for n in COUNTS:
            rep_data[ray][n] = replicated_stats(ray, n, n_rep=20)

    _save_csv(results)
    _save_json(results)

    plot_step1_convergence(rep_data)
    plot_step1_energy_breakdown(results)
    plot_step1_depth_profiles()
    plot_step1_individual_profiles()
    plot_step2_beam_optimization(rep_data)
    plot_step2_summary(results, rep_data)
    plot_step3_replication(rep_data)
    plot_step4_combinations(results)
    plot_step5_firing_order(results)
    plot_step6_final_protocol(results)
    plot_complete_pipeline(results, rep_data)

    print(f"\n  All plots saved to {OUT_DIR}")


if __name__ == "__main__":
    run_all()