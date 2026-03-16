"""
Step 3: Validation Reruns — Surgical Tool
Validates stability at optimal counts from Step 2 using
adaptive bootstrap resampling from real Geant4 CSV data.
Mirrors brain project step3 adaptive CV stopping logic.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, csv, json, math

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, "data")
STATS_DIR = os.path.join(ROOT, "data", "stats")
STEPS_DIR = os.path.join(ROOT, "Steps", "Step3_Rerun")
os.makedirs(STEPS_DIR, exist_ok=True)

NX, NY, NZ  = 10, 10, 100
PRION_BINS  = [99]
RAYS        = ["gamma", "neutron", "carbon", "alpha"]
RBE         = {"gamma": 1.0, "neutron": 10.0, "carbon": 3.0, "alpha": 20.0}
STYLES      = {"gamma": "#3498db", "neutron": "#2ecc71", "carbon": "#9b59b6", "alpha": "#e74c3c"}

MIN_RUNS   = 15
MAX_RUNS   = 60
STABLE_WIN = 8
STABLE_THR = 1.0
STABLE_CONF= 3


def load_grid(ray):
    path = os.path.join(DATA_DIR, f"{ray}_edep.csv")
    grid = np.zeros((NX, NY, NZ))
    if not os.path.exists(path):
        return None
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
            except (ValueError, IndexError):
                continue
    return grid if grid.sum() > 0 else None


def bootstrap_sample(flat, n, seed):
    rng     = np.random.default_rng(seed)
    idx     = rng.choice(len(flat), size=min(n, len(flat)), replace=True)
    sample  = flat[idx]
    prion   = float(sample.sum())
    return prion


def run_validation(ray, opt_n, grid):
    """
    Adaptive bootstrap validation — keeps sampling until CV stabilises.
    Returns list of per-run prion MeV values and final stats.
    """
    if grid is None:
        print(f"    [WARN] No grid data for {ray}")
        return [], 999.0

    flat = grid[:, :, PRION_BINS].flatten()
    flat = flat[flat > 0]

    if len(flat) == 0:
        print(f"    [WARN] No prion voxels for {ray} — alpha cannot reach tip")
        return [0.0] * MIN_RUNS, 999.0

    all_prion = []
    cv_history = []
    conf_count = 0
    run_num    = 0

    while run_num < MAX_RUNS:
        run_num += 1
        prion = bootstrap_sample(flat, opt_n, seed=run_num * 1000 + hash(ray) % 10000)
        all_prion.append(prion)

        if len(all_prion) >= 2:
            mean = sum(all_prion) / len(all_prion)
            std  = math.sqrt(sum((v - mean)**2 for v in all_prion) / max(len(all_prion)-1, 1))
            cv   = std / mean * 100 if mean > 0 else 999.0
        else:
            cv = 999.0
        cv_history.append(cv)

        print(f"    Run {run_num:>3}  prion={prion:>10.4f} MeV  CV={cv:.2f}%")

        # Adaptive stopping — like brain project
        if run_num >= MIN_RUNS and len(cv_history) >= STABLE_WIN:
            window = cv_history[-STABLE_WIN:]
            spread = max(window) - min(window)
            if spread < STABLE_THR:
                conf_count += 1
                if conf_count >= STABLE_CONF:
                    print(f"    → Converged at run {run_num} (CV stable within {STABLE_THR}%)")
                    break
            else:
                conf_count = 0

    return all_prion, cv_history[-1] if cv_history else 999.0


def run_step3():
    print("\n" + "="*65)
    print("  SURGICAL TOOL — Step 3: Validation Reruns")
    print("  Adaptive bootstrap at optimal counts from Step 2")
    print("="*65)

    # Load optimal counts from Step 2
    opt_path = os.path.join(STATS_DIR, "step2_optimal.json")
    if not os.path.exists(opt_path):
        print("  [ERROR] step2_optimal.json not found — run Step 2 first")
        return {}
    with open(opt_path) as f:
        step2 = json.load(f)

    results = {}

    for ray in RAYS:
        opt_n = step2.get(ray, {}).get("optimal_count", 1000)
        print(f"\n  [{ray.upper()}]  optimal N={opt_n:,}")
        grid  = load_grid(ray)
        prion_runs, final_cv = run_validation(ray, opt_n, grid)

        if len(prion_runs) == 0 or all(v == 0 for v in prion_runs):
            mean_p = 0.0; std_p = 0.0; final_cv = 999.0
        else:
            nonzero = [v for v in prion_runs if v > 0]
            mean_p  = sum(nonzero) / len(nonzero) if nonzero else 0.0
            std_p   = math.sqrt(sum((v - mean_p)**2 for v in nonzero) / max(len(nonzero)-1, 1)) if len(nonzero) > 1 else 0.0

        rbe_mean = mean_p * RBE[ray]
        quality  = "EXCELLENT" if final_cv < 2 else "GOOD" if final_cv < 5 else "ACCEPTABLE" if final_cv < 10 else "UNSTABLE"

        results[ray] = {
            "optimal_count": opt_n,
            "n_runs":        len(prion_runs),
            "mean_prion_mev":round(mean_p, 4),
            "std_prion_mev": round(std_p,  4),
            "mean_rbe_mev":  round(rbe_mean, 4),
            "final_cv_pct":  round(final_cv, 2),
            "quality":       quality,
            "all_prion":     [round(v, 4) for v in prion_runs],
        }
        print(f"    → Mean={mean_p:.4f} MeV  Std={std_p:.4f}  CV={final_cv:.2f}%  [{quality}]")

    plot_results(results)

    # Save CSV
    csv_path = os.path.join(STATS_DIR, "step3_validation.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Ray", "OptN", "Runs", "MeanPrionMeV", "StdPrionMeV",
                    "MeanRBEMeV", "FinalCV_pct", "Quality"])
        for ray, v in results.items():
            w.writerow([ray, v["optimal_count"], v["n_runs"],
                        v["mean_prion_mev"], v["std_prion_mev"],
                        v["mean_rbe_mev"], v["final_cv_pct"], v["quality"]])

    # Save JSON (without the big all_prion list)
    json_out = {r: {k: v for k, v in d.items() if k != "all_prion"}
                for r, d in results.items()}
    with open(os.path.join(STATS_DIR, "step3_validation.json"), "w") as f:
        json.dump(json_out, f, indent=2)

    print(f"\n    Saved {csv_path}")
    print("\n  Step 3 complete.")
    return results


def plot_results(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("Surgical Tool — Step 3: Validation at Optimal Counts",
                 color="white", fontsize=13, fontweight="bold")

    ray_names = list(results.keys())
    colors    = [STYLES[r] for r in ray_names]

    # CV convergence
    ax = axes[0]; ax.set_facecolor("#111827")
    ax.set_title("CV Convergence per Ray", color="white", fontsize=10)
    ax.set_xlabel("Run number", color="white", fontsize=8)
    ax.set_ylabel("Running CV %", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values(): sp.set_color("#444")
    for ray in ray_names:
        prion_runs = results[ray]["all_prion"]
        if not prion_runs or all(v == 0 for v in prion_runs):
            continue
        cvs = []
        for n in range(2, len(prion_runs) + 1):
            sub  = [v for v in prion_runs[:n] if v > 0]
            if len(sub) < 2:
                cvs.append(999.0); continue
            mean = sum(sub) / len(sub)
            std  = math.sqrt(sum((v - mean)**2 for v in sub) / (len(sub) - 1))
            cvs.append(std / mean * 100 if mean > 0 else 999.0)
        ax.plot(range(2, 2 + len(cvs)), cvs,
                color=STYLES[ray], lw=1.8, label=ray.capitalize())
    for val, col, lbl in [(2, "#2ecc71", "Excellent"), (5, "#f39c12", "Good"), (10, "#e74c3c", "Acceptable")]:
        ax.axhline(val, color=col, ls="--", lw=1, alpha=0.6, label=f"{lbl} ({val}%)")
    ax.set_ylim(0, min(50, ax.get_ylim()[1]))
    ax.legend(fontsize=7, facecolor="#222", labelcolor="white")

    # Mean prion MeV bar
    ax = axes[1]; ax.set_facecolor("#111827")
    ax.set_title("Mean Prion Dose at Optimal N", color="white", fontsize=10)
    ax.set_xlabel("Ray", color="white", fontsize=8)
    ax.set_ylabel("Mean Prion MeV", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values(): sp.set_color("#444")
    means = [results[r]["mean_prion_mev"] for r in ray_names]
    stds  = [results[r]["std_prion_mev"]  for r in ray_names]
    bars  = ax.bar(ray_names, means, color=colors, edgecolor="none")
    ax.errorbar(ray_names, means, yerr=stds,
                fmt="none", color="white", capsize=4, linewidth=1.5)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                f"{val:.2f}", ha="center", va="bottom", color="white", fontsize=7)

    # Quality summary
    ax = axes[2]; ax.set_facecolor("#111827"); ax.axis("off")
    qual_colors = {"EXCELLENT": "#2ecc71", "GOOD": "#f39c12",
                   "ACCEPTABLE": "#e67e22", "UNSTABLE": "#e74c3c"}
    lines = ["VALIDATION SUMMARY", "="*38, ""]
    for ray in ray_names:
        v    = results[ray]
        col  = qual_colors.get(v["quality"], "white")
        lines.append(f"{ray.upper()}")
        lines.append(f"  N={v['optimal_count']:,}  runs={v['n_runs']}")
        lines.append(f"  Mean: {v['mean_prion_mev']:.4f} MeV")
        lines.append(f"  RBE:  {v['mean_rbe_mev']:.4f} MeV")
        lines.append(f"  CV:   {v['final_cv_pct']:.2f}%  [{v['quality']}]")
        lines.append("")
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            color="white", fontsize=8, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#1a1a2e", edgecolor="#4a9eff", alpha=0.9))

    plt.tight_layout()
    out = os.path.join(STEPS_DIR, "step3_validation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Saved {out}")


if __name__ == "__main__":
    run_step3()
