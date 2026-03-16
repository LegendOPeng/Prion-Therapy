"""
Step 4: Multi-Ray Combinations — Surgical Tool
Tests all 2, 3, 4-ray combos by additively combining
real Geant4 grids at optimal counts from Step 2/3.
Score = prion_mev / (surf_core_ratio + 0.01)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations
import os, csv, json, math

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, "data")
STATS_DIR = os.path.join(ROOT, "data", "stats")
STEPS_DIR = os.path.join(ROOT, "Steps", "Step4_Combos")
os.makedirs(STEPS_DIR, exist_ok=True)

NX, NY, NZ  = 10, 10, 100
PRION_BINS  = [99]
STEEL_BINS  = list(range(0, 99))
RAYS        = ["gamma", "neutron", "carbon", "alpha"]
RBE         = {"gamma": 1.0, "neutron": 10.0, "carbon": 3.0, "alpha": 20.0}
STYLES      = {"gamma": "#3498db", "neutron": "#2ecc71", "carbon": "#9b59b6", "alpha": "#e74c3c"}
COMBO_COLORS= {2: "#3498db", 3: "#2ecc71", 4: "#e74c3c"}


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
    """Scale grid from base simulation count to optimal N."""
    return grid * (opt_n / base_n)


def combo_metrics(combined_grid):
    """Compute prion dose, steel dose, selectivity and score."""
    prof        = combined_grid.sum(axis=(0, 1))
    prion_dose  = float(prof[PRION_BINS].sum())
    steel_dose  = float(prof[STEEL_BINS].sum())
    total_dose  = float(prof.sum())
    ratio       = steel_dose / prion_dose if prion_dose > 0 else 999.0
    score       = prion_dose / (ratio + 0.01)
    selectivity = prion_dose / steel_dose if steel_dose > 0 else 0.0
    return {
        "prion_mev":   round(prion_dose, 4),
        "steel_mev":   round(steel_dose, 4),
        "total_mev":   round(total_dose, 4),
        "surf_core":   round(ratio, 4),
        "selectivity": round(selectivity, 6),
        "score":       round(score, 4),
    }


def run_step4():
    print("\n" + "="*65)
    print("  SURGICAL TOOL — Step 4: Multi-Ray Combinations")
    print("  Combines real Geant4 grids additively at optimal counts")
    print("="*65)

    # Load optimal counts
    opt_path = os.path.join(STATS_DIR, "step2_optimal.json")
    if not os.path.exists(opt_path):
        print("  [ERROR] step2_optimal.json not found — run Step 2 first")
        return []
    with open(opt_path) as f:
        step2 = json.load(f)

    opt_counts = {ray: step2[ray]["optimal_count"] for ray in RAYS if ray in step2}
    print(f"\n  Optimal counts: {opt_counts}")

    # Load all grids
    grids = {}
    for ray in RAYS:
        g = load_grid(ray)
        if g is not None:
            grids[ray] = scale_grid(g, opt_counts.get(ray, 2000))
            print(f"  Loaded {ray}: prion zone sum = {g[:,:,PRION_BINS].sum():.4f} MeV (unscaled)")
        else:
            print(f"  [SKIP] {ray} — no grid data")

    # Test all combinations
    results = []
    for n_rays in range(2, len(RAYS) + 1):
        for combo in combinations(RAYS, n_rays):
            # Skip combos with no viable rays (alpha always zero)
            viable = [r for r in combo if r in grids and grids[r][:,:,PRION_BINS].sum() > 0]
            combo_name = " + ".join(combo)

            combined = np.zeros((NX, NY, NZ))
            for ray in combo:
                if ray in grids:
                    combined += grids[ray]

            m = combo_metrics(combined)
            results.append({
                "combo":      combo_name,
                "rays":       list(combo),
                "n_rays":     n_rays,
                "viable_rays":len(viable),
                **m
            })
            print(f"  {combo_name:<40}  score={m['score']:>10.2f}  "
                  f"prion={m['prion_mev']:>10.4f} MeV  sel={m['selectivity']:.6f}")

    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n  ★ BEST COMBINATION: {results[0]['combo']}")
    print(f"    Score: {results[0]['score']:.4f}")
    print(f"    Prion MeV: {results[0]['prion_mev']:.4f}")
    print(f"    Selectivity: {results[0]['selectivity']:.6f}")

    plot_results(results)

    # Save CSV
    csv_path = os.path.join(STATS_DIR, "step4_combinations.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Combo", "N_Rays", "PrionMeV", "SteelMeV", "SurfCore", "Selectivity", "Score"])
        for r in results:
            w.writerow([r["combo"], r["n_rays"], r["prion_mev"],
                        r["steel_mev"], r["surf_core"], r["selectivity"], r["score"]])

    with open(os.path.join(STATS_DIR, "step4_combinations.json"), "w") as f:
        json.dump({"combinations": results[:10], "best": results[0]["combo"]}, f, indent=2)

    print(f"\n    Saved {csv_path}")
    print("\n  Step 4 complete.")
    return results


def plot_results(results):
    top = results[:min(10, len(results))]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("Surgical Tool — Step 4: Ray Combination Rankings",
                 color="white", fontsize=13, fontweight="bold")

    labels  = [r["combo"].replace(" + ", "\n+\n") for r in top]
    colors  = [COMBO_COLORS.get(r["n_rays"], "#888") for r in top]

    for ax, key, title, ylabel in [
        (axes[0], "score",       "Score (prion/ratio)",        "Score"),
        (axes[1], "prion_mev",   "Prion Dose (MeV)",           "MeV"),
        (axes[2], "selectivity", "Selectivity (prion/steel)",  "Selectivity"),
    ]:
        ax.set_facecolor("#111827")
        ax.set_title(title, color="white", fontsize=10)
        ax.set_ylabel(ylabel, color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=6)
        for sp in ax.spines.values(): sp.set_color("#444")
        vals = [r[key] for r in top]
        bars = ax.bar(range(len(top)), vals, color=colors, edgecolor="none")
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels([r["combo"] for r in top],
                           rotation=45, ha="right", fontsize=6, color="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    f"{val:.1f}", ha="center", va="bottom", color="white", fontsize=6)

    from matplotlib.patches import Patch
    legend = [Patch(color=c, label=f"{n}-ray") for n, c in COMBO_COLORS.items()]
    axes[0].legend(handles=legend, fontsize=7, facecolor="#222", labelcolor="white")

    plt.tight_layout()
    out = os.path.join(STEPS_DIR, "step4_combinations.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Saved {out}")


if __name__ == "__main__":
    run_step4()
