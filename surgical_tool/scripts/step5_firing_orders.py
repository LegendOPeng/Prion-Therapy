"""
Step 5: Firing Order Optimization — Surgical Tool
Tests all permutations of the winning combo from Step 4.
Since grids are additive (linear energy deposition), order
affects timing and biological effectiveness weighting, not
raw dose — modelled here via RBE-weighted sequential scoring.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import permutations
import os, csv, json, math

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, "data")
STATS_DIR = os.path.join(ROOT, "data", "stats")
STEPS_DIR = os.path.join(ROOT, "Steps", "Step5_FiringOrders")
os.makedirs(STEPS_DIR, exist_ok=True)

NX, NY, NZ  = 10, 10, 100
PRION_BINS  = [99]
STEEL_BINS  = list(range(0, 99))
RBE         = {"gamma": 1.0, "neutron": 10.0, "carbon": 3.0, "alpha": 20.0}
STYLES      = {"gamma": "#3498db", "neutron": "#2ecc71", "carbon": "#9b59b6", "alpha": "#e74c3c"}
REPEATS     = 5


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


def scale_grid(grid, opt_n, base_n=2000):
    return grid * (opt_n / base_n)


def order_score(order, grids, opt_counts, rep_seed=0):
    """
    Score a firing order using sequential RBE weighting.
    Earlier rays condition the tissue — modelled by applying
    a cumulative damage factor that boosts subsequent ray effectiveness.
    This is a simplified radiobiological model for ordering effects.
    """
    rng = np.random.default_rng(rep_seed)
    combined = np.zeros((NX, NY, NZ))
    damage_factor = 1.0

    for i, ray in enumerate(order):
        if ray not in grids:
            continue
        g   = grids[ray].copy()
        n   = opt_counts.get(ray, 2000)
        # Bootstrap resample to introduce Monte Carlo variance
        flat = g[:, :, PRION_BINS].flatten()
        flat = flat[flat > 0]
        if len(flat) > 0:
            noise = rng.normal(0, flat.std() * 0.05, size=g.shape)
            g     = np.clip(g + noise, 0, None)
        # Sequential damage model: each successive ray benefits from prior ionisation
        # Gamma (low LET) softens tissue, high-LET rays (carbon, neutron) follow
        if i == 0:
            damage_factor = 1.0
        elif order[i-1] == "gamma":
            damage_factor = 1.08   # gamma pre-conditioning boosts next ray by 8%
        elif order[i-1] in ("neutron", "carbon"):
            damage_factor = 1.04
        combined += g * damage_factor

    prof       = combined.sum(axis=(0, 1))
    prion_dose = float(prof[PRION_BINS].sum())
    steel_dose = float(prof[STEEL_BINS].sum())
    ratio      = steel_dose / prion_dose if prion_dose > 0 else 999.0
    score      = prion_dose / (ratio + 0.01)
    return score, prion_dose, ratio


def run_step5():
    print("\n" + "="*65)
    print("  SURGICAL TOOL — Step 5: Firing Order Optimization")
    print("="*65)

    # Load best combo from Step 4
    s4_path = os.path.join(STATS_DIR, "step4_combinations.json")
    if not os.path.exists(s4_path):
        print("  [ERROR] step4_combinations.json not found — run Step 4 first")
        return []
    with open(s4_path) as f:
        s4 = json.load(f)
    best_combo_name = s4.get("best", "gamma + neutron + carbon")
    best_rays = [r.strip() for r in best_combo_name.split("+")]
    print(f"\n  Best combo from Step 4: {best_combo_name}")
    print(f"  Testing all {math.factorial(len(best_rays))} permutations × {REPEATS} repeats")

    # Load optimal counts
    with open(os.path.join(STATS_DIR, "step2_optimal.json")) as f:
        step2 = json.load(f)
    opt_counts = {ray: step2[ray]["optimal_count"] for ray in best_rays if ray in step2}

    # Load and scale grids
    grids = {}
    for ray in best_rays:
        g = load_grid(ray)
        if g is not None:
            grids[ray] = scale_grid(g, opt_counts.get(ray, 2000))
            print(f"  Loaded {ray} (N={opt_counts.get(ray, 2000):,})")

    # Test all permutations
    all_perms = list(permutations(best_rays))
    results   = []

    print()
    for perm in all_perms:
        order_name = " → ".join(perm)
        run_scores = []
        run_prions = []
        for rep in range(REPEATS):
            score, prion, ratio = order_score(perm, grids, opt_counts, rep_seed=rep * 100)
            run_scores.append(score)
            run_prions.append(prion)

        mean_score = sum(run_scores) / REPEATS
        std_score  = math.sqrt(sum((s - mean_score)**2 for s in run_scores) / max(REPEATS-1, 1))
        mean_prion = sum(run_prions) / REPEATS

        results.append({
            "order":      order_name,
            "perm":       list(perm),
            "mean_score": round(mean_score, 4),
            "std_score":  round(std_score,  4),
            "mean_prion": round(mean_prion, 4),
        })
        print(f"  {order_name:<40}  mean_score={mean_score:>10.2f} ± {std_score:.2f}")

    results.sort(key=lambda x: x["mean_score"], reverse=True)

    print(f"\n  ★ BEST ORDER: {results[0]['order']}")
    print(f"    Score: {results[0]['mean_score']:.4f} ± {results[0]['std_score']:.4f}")

    plot_results(results)

    csv_path = os.path.join(STATS_DIR, "step5_firing_orders.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Order", "MeanScore", "StdScore", "MeanPrionMeV"])
        for r in results:
            w.writerow([r["order"], r["mean_score"], r["std_score"], r["mean_prion"]])

    with open(os.path.join(STATS_DIR, "step5_firing_orders.json"), "w") as f:
        json.dump({"firing_orders": results, "best_order": results[0]["order"],
                   "best_rays": best_rays}, f, indent=2)

    print(f"\n    Saved {csv_path}")
    print(f"\n  Best overall: {results[0]['order']}")
    print("\n  Step 5 complete.")
    return results


def plot_results(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("Surgical Tool — Step 5: Firing Order Optimization",
                 color="white", fontsize=13, fontweight="bold")

    scores = [r["mean_score"] for r in results]
    stds   = [r["std_score"]  for r in results]
    prions = [r["mean_prion"] for r in results]
    orders = [r["order"] for r in results]

    cmap   = plt.get_cmap("RdYlGn")
    norm   = plt.Normalize(min(scores), max(scores))
    colors = [cmap(norm(s)) for s in scores]

    ax = axes[0]; ax.set_facecolor("#111827")
    ax.set_title("Mean Score by Firing Order", color="white", fontsize=10)
    ax.set_ylabel("Mean Score", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values(): sp.set_color("#444")
    bars = ax.bar(range(len(results)), scores, color=colors, edgecolor="none")
    ax.errorbar(range(len(results)), scores, yerr=stds,
                fmt="none", color="white", capsize=4, linewidth=1.5)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(orders, rotation=45, ha="right", fontsize=7, color="white")
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                f"{val:.1f}", ha="center", va="bottom", color="white", fontsize=7)

    ax = axes[1]; ax.set_facecolor("#111827")
    ax.set_title("Mean Prion Dose by Order", color="white", fontsize=10)
    ax.set_ylabel("Mean Prion MeV", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values(): sp.set_color("#444")
    bars = ax.bar(range(len(results)), prions, color=colors, edgecolor="none")
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(orders, rotation=45, ha="right", fontsize=7, color="white")
    for bar, val in zip(bars, prions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                f"{val:.0f}", ha="center", va="bottom", color="white", fontsize=7)

    plt.tight_layout()
    out = os.path.join(STEPS_DIR, "step5_firing_orders.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Saved {out}")


if __name__ == "__main__":
    run_step5()
