"""
Step 4: Ray Combinations — Surgical Tool
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.patches import Patch
import os, csv, json

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS_DIR = os.path.join(ROOT, "data", "stats")
STEPS_DIR = os.path.join(ROOT, "Steps", "Step4_Combos")
os.makedirs(STEPS_DIR, exist_ok=True)

RAYS   = ["gamma", "neutron", "carbon", "alpha"]
RBE    = {"gamma":1.0,"neutron":10.0,"alpha":20.0,"carbon":3.0}
STYLES = {"gamma":"#3498db","alpha":"#e74c3c","neutron":"#2ecc71","carbon":"#9b59b6"}
COMBO_COLORS = {2:"#3498db", 3:"#2ecc71", 4:"#e74c3c"}

INTERACTION = {
    ("gamma","neutron"):1.25, ("gamma","carbon"):1.20,  ("gamma","alpha"):1.10,
    ("neutron","gamma"):1.05, ("neutron","carbon"):1.15, ("neutron","alpha"):1.08,
    ("carbon","gamma"):1.10,  ("carbon","neutron"):1.12, ("carbon","alpha"):1.05,
    ("alpha","gamma"):1.15,   ("alpha","neutron"):1.18,  ("alpha","carbon"):1.12,
}


def load_step1_stats():
    path = os.path.join(STATS_DIR, "step1_stats.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "gamma":   {"prion_mev":0.18,"steel_mev":2.1,"selectivity":0.086,"rbe_prion_mev":0.18,"max_reach_mm":95.0},
        "neutron": {"prion_mev":0.45,"steel_mev":3.8,"selectivity":0.118,"rbe_prion_mev":4.50,"max_reach_mm":99.0},
        "carbon":  {"prion_mev":0.62,"steel_mev":2.4,"selectivity":0.258,"rbe_prion_mev":1.86,"max_reach_mm":98.5},
        "alpha":   {"prion_mev":0.001,"steel_mev":8.2,"selectivity":0.0001,"rbe_prion_mev":0.02,"max_reach_mm":1.0},
    }


def score_combination(rays_in_combo, stats):
    total_rbe   = sum(stats[r]["rbe_prion_mev"] for r in rays_in_combo)
    total_steel = sum(stats[r]["steel_mev"]      for r in rays_in_combo)
    mean_sel    = np.mean([stats[r]["selectivity"]   for r in rays_in_combo])
    max_reach   = max(stats[r]["max_reach_mm"]        for r in rays_in_combo)
    coverage    = 1.5 if max_reach > 95.0 else 1.0
    alpha_bonus = 1.2 if "alpha" in rays_in_combo else 1.0
    penalty     = 0.05 * total_steel
    score       = total_rbe * mean_sel * coverage * alpha_bonus - penalty
    return {
        "combo":            " + ".join(sorted(rays_in_combo)),
        "rays":             list(rays_in_combo),
        "n_rays":           len(rays_in_combo),
        "total_rbe_prion":  round(float(total_rbe),  4),
        "total_steel_mev":  round(float(total_steel), 4),
        "mean_selectivity": round(float(mean_sel),    4),
        "max_reach_mm":     round(float(max_reach),   2),
        "coverage_bonus":   coverage,
        "alpha_bonus":      alpha_bonus,
        "score":            round(float(score),        4),
    }


def plot_combinations(results):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor("#0d1117")
    sorted_r = sorted(results, key=lambda x: -x["score"])
    top15    = sorted_r[:15]

    ax = axes[0,0]; ax.set_facecolor("#111827")
    labels = [r["combo"].replace(" + ","\n+") for r in top15]
    scores = [r["score"]  for r in top15]
    nrays  = [r["n_rays"] for r in top15]
    colors = [COMBO_COLORS[n] for n in nrays]
    ax.bar(range(len(top15)), scores, color=colors, edgecolor="none")
    ax.set_xticks(range(len(top15))); ax.set_xticklabels(labels, fontsize=6, color="white")
    ax.set_title("Top 15 Combinations by Score", color="white", fontsize=11, fontweight="bold")
    ax.set_ylabel("Score", color="white"); ax.tick_params(colors="white")
    legend = [Patch(color=c, label=f"{n}-ray") for n,c in COMBO_COLORS.items()]
    ax.legend(handles=legend, facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#444")

    ax = axes[0,1]; ax.set_facecolor("#111827")
    rbe_prion = [r["total_rbe_prion"] for r in top15]
    ax.bar(range(len(top15)), rbe_prion, color=colors, edgecolor="none")
    ax.set_xticks(range(len(top15))); ax.set_xticklabels(labels, fontsize=6, color="white")
    ax.set_title("RBE-Weighted Prion Dose", color="white", fontsize=11, fontweight="bold")
    ax.set_ylabel("RBE-MeV", color="white"); ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")

    ax = axes[1,0]; ax.set_facecolor("#111827")
    for n, c in COMBO_COLORS.items():
        sub = [r for r in results if r["n_rays"] == n]
        if sub:
            ax.scatter([r["mean_selectivity"] for r in sub],
                       [r["total_rbe_prion"]  for r in sub],
                       c=c, s=50, alpha=0.8, label=f"{n}-ray", edgecolors="none")
    for r in sorted_r[:5]:
        ax.annotate(r["combo"].replace(" + ","+"),
                    (r["mean_selectivity"], r["total_rbe_prion"]),
                    fontsize=6, color="white", xytext=(5,5), textcoords="offset points")
    ax.set_xlabel("Mean Selectivity", color="white", fontsize=9)
    ax.set_ylabel("Total RBE-MeV", color="white", fontsize=9)
    ax.set_title("Selectivity vs RBE Dose", color="white", fontsize=11, fontweight="bold")
    ax.tick_params(colors="white"); ax.legend(facecolor="#1a1a2e", labelcolor="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")

    ax = axes[1,1]; ax.set_facecolor("#111827"); ax.axis("off")
    lines = ["TOP 5 COMBINATIONS","="*52,
             f"{'Rank':<5} {'Combo':<26} {'RBE-MeV':>8} {'Select':>8} {'Score':>7}","─"*52]
    for i, r in enumerate(sorted_r[:5], 1):
        lines.append(f"{i:<5} {r['combo']:<26} {r['total_rbe_prion']:>8.4f}"
                     f" {r['mean_selectivity']:>8.4f} {r['score']:>7.4f}")
    lines += ["─"*52,"","SURGICAL CONTEXT:",
              "• Gamma: full rod coverage ✓",
              "• Neutron: deep + RBE×10 ✓",
              "• Carbon: Bragg peak at tip ✓",
              "• Alpha: entry surface ONLY ✗ tip",
              "• Best tip combo: Neutron+Carbon",
              "","LIMIT: G4_WATER as prion proxy"]
    ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, color="white",
            fontsize=8, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#1a1a2e", edgecolor="#4a9eff", alpha=0.9))

    fig.suptitle("Surgical Tool — Step 4: Ray Combination Analysis",
                 color="white", fontsize=14, fontweight="bold")
    out = os.path.join(STEPS_DIR, "step4_combinations.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(); print(f"    Saved {out}")


def run_step4():
    print("\n" + "="*65)
    print("  SURGICAL TOOL — Step 4: Ray Combinations")
    print("="*65)
    stats   = load_step1_stats()
    results = []
    for n in range(2, len(RAYS)+1):
        for combo in combinations(RAYS, n):
            r = score_combination(combo, stats)
            results.append(r)
            print(f"  {r['combo']:<35} score={r['score']:.4f}")
    results_sorted = sorted(results, key=lambda x: -x["score"])
    print(f"\n  Best: {results_sorted[0]['combo']}  score={results_sorted[0]['score']:.4f}")
    plot_combinations(results)
    path = os.path.join(STATS_DIR, "step4_combinations.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader(); w.writerows(results_sorted)
    print(f"\n    Saved {path}")
    payload = {"combinations":results_sorted,
               "best_combo":results_sorted[0]["combo"],
               "best_score":results_sorted[0]["score"]}
    with open(os.path.join(STATS_DIR, "step4_combinations.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print("\n  Step 4 complete.")
    return payload


if __name__ == "__main__":
    run_step4()