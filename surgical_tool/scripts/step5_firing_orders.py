"""
Step 5: Firing Orders — Surgical Tool
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import permutations
import os, csv, json

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS_DIR = os.path.join(ROOT, "data", "stats")
STEPS_DIR = os.path.join(ROOT, "Steps", "Step5_FiringOrders")
os.makedirs(STEPS_DIR, exist_ok=True)

RAYS   = ["gamma", "neutron", "carbon", "alpha"]
RBE    = {"gamma":1.0,"neutron":10.0,"alpha":20.0,"carbon":3.0}
STYLES = {"gamma":"#3498db","alpha":"#e74c3c","neutron":"#2ecc71","carbon":"#9b59b6"}

INTERACTION = {
    ("gamma","neutron"):1.25, ("gamma","carbon"):1.20,  ("gamma","alpha"):1.10,
    ("neutron","gamma"):1.05, ("neutron","carbon"):1.15, ("neutron","alpha"):1.08,
    ("carbon","gamma"):1.10,  ("carbon","neutron"):1.12, ("carbon","alpha"):1.05,
    ("alpha","gamma"):1.15,   ("alpha","neutron"):1.18,  ("alpha","carbon"):1.12,
}


def load_step4():
    path = os.path.join(STATS_DIR, "step4_combinations.json")
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return {"combinations":[], "best_combo":"neutron + carbon", "best_score":0.0}


def load_step1_stats():
    path = os.path.join(STATS_DIR, "step1_stats.json")
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return {
        "gamma":   {"rbe_prion_mev":0.18,"max_reach_mm":95.0},
        "neutron": {"rbe_prion_mev":4.50,"max_reach_mm":99.0},
        "carbon":  {"rbe_prion_mev":1.86,"max_reach_mm":98.5},
        "alpha":   {"rbe_prion_mev":0.02,"max_reach_mm":1.0},
    }


def score_order(order, stats, n_trials=30):
    np.random.seed(hash(order) % 2**31)
    trial_scores = []
    for _ in range(n_trials):
        cumulative = 0.0; prev_ray = None
        for ray in order:
            dose  = stats[ray]["rbe_prion_mev"] * (1.0 + 0.04 * np.random.randn())
            if prev_ray:
                dose *= INTERACTION.get((prev_ray, ray), 1.0)
            dose *= min(1.0, stats[ray]["max_reach_mm"] / 99.0)
            cumulative += dose; prev_ray = ray
        trial_scores.append(cumulative)
    arr = np.array(trial_scores)
    return {
        "order":      " → ".join(order),
        "rays":       list(order),
        "n_rays":     len(order),
        "mean_score": round(float(arr.mean()), 4),
        "std_score":  round(float(arr.std()),  4),
        "cv_pct":     round(float(arr.std()/arr.mean()*100) if arr.mean()>0 else 0, 2),
        "min_score":  round(float(arr.min()), 4),
        "max_score":  round(float(arr.max()), 4),
    }


def get_top_combos(step4, top_n=4):
    combos = step4.get("combinations", [])[:top_n]
    return [tuple(c["rays"]) for c in combos]


def plot_firing_orders(all_orders):
    flat = []
    for orders in all_orders.values(): flat.extend(orders)
    flat_sorted = sorted(flat, key=lambda x: -x["mean_score"])
    top20 = flat_sorted[:20]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor("#0d1117")

    ax = axes[0,0]; ax.set_facecolor("#111827")
    labels = [o["order"].replace(" → ","→\n") for o in top20]
    means  = [o["mean_score"] for o in top20]
    stds   = [o["std_score"]  for o in top20]
    cmap   = plt.get_cmap("RdYlGn")
    norm   = plt.Normalize(min(means), max(means))
    colors = [cmap(norm(s)) for s in means]
    ax.bar(range(len(top20)), means, color=colors, edgecolor="none")
    ax.errorbar(range(len(top20)), means, yerr=stds, fmt="none",
                color="white", capsize=3, lw=1)
    ax.set_xticks(range(len(top20))); ax.set_xticklabels(labels, fontsize=5.5, color="white")
    ax.set_title("Top 20 Firing Orders", color="white", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Score", color="white"); ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")

    ax = axes[0,1]; ax.set_facecolor("#111827")
    combo_keys = list(all_orders.keys())
    for i, ck in enumerate(combo_keys):
        orders = sorted(all_orders[ck], key=lambda x: -x["mean_score"])
        best   = orders[0]["mean_score"]; worst = orders[-1]["mean_score"]
        gain   = (best-worst)/worst*100 if worst > 0 else 0
        ax.barh(i, best,  color="#2ecc71", alpha=0.8, height=0.4, label="Best"  if i==0 else "")
        ax.barh(i, worst, color="#e74c3c", alpha=0.8, height=0.4, label="Worst" if i==0 else "")
        ax.text(best*1.01, i, f"+{gain:.1f}%", va="center", color="white", fontsize=8)
    ax.set_yticks(range(len(combo_keys)))
    ax.set_yticklabels([ck.replace("+"," +\n") for ck in combo_keys], color="white", fontsize=7)
    ax.set_title("Best vs Worst Order per Combo", color="white", fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean Score", color="white"); ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")

    ax = axes[1,0]; ax.set_facecolor("#111827")
    for i, ck in enumerate(combo_keys):
        scores = [o["mean_score"] for o in all_orders[ck]]
        parts  = ax.violinplot([scores], positions=[i], widths=0.6, showmeans=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(["#3498db","#2ecc71","#e74c3c","#9b59b6"][i%4])
            pc.set_alpha(0.7)
    ax.set_xticks(range(len(combo_keys)))
    ax.set_xticklabels([ck.replace(" + ","+\n") for ck in combo_keys], color="white", fontsize=7)
    ax.set_title("Score Distribution per Combo", color="white", fontsize=11, fontweight="bold")
    ax.set_ylabel("Score", color="white"); ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")

    ax = axes[1,1]; ax.set_facecolor("#111827"); ax.axis("off")
    lines = ["TOP 5 FIRING ORDERS","="*55,
             f"{'Order':<35} {'Mean':>7} {'CV%':>6}","─"*55]
    for o in flat_sorted[:5]:
        lines.append(f"{o['order']:<35} {o['mean_score']:>7.4f} {o['cv_pct']:>6.2f}%")
    lines += ["─"*55,"","PHYSICS RATIONALE:",
              "• Gamma first → ionizes prion layer",
              "• Carbon next → Bragg peak at tip",
              "• Neutron last → deep cleanup",
              "• Alpha first if surface dose needed"]
    ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, color="white",
            fontsize=8, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#1a1a2e", edgecolor="#4a9eff", alpha=0.9))

    fig.suptitle("Surgical Tool — Step 5: Firing Order Optimization",
                 color="white", fontsize=14, fontweight="bold")
    out = os.path.join(STEPS_DIR, "step5_firing_orders.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(); print(f"    Saved {out}")


def run_step5():
    print("\n" + "="*65)
    print("  SURGICAL TOOL — Step 5: Firing Orders")
    print("="*65)
    step4      = load_step4()
    stats      = load_step1_stats()
    top_combos = get_top_combos(step4, top_n=4)
    if not top_combos:
        top_combos = [("gamma","neutron","carbon"),("neutron","carbon"),
                      ("gamma","carbon"),("gamma","neutron")]
    all_orders  = {}; all_results = []
    for combo in top_combos:
        combo_key = " + ".join(sorted(combo))
        print(f"\n  Combo: {combo_key}")
        orders = []
        for perm in permutations(combo):
            result = score_order(perm, stats)
            orders.append(result); all_results.append(result)
            print(f"    {result['order']:<40} mean={result['mean_score']:.4f}")
        orders.sort(key=lambda x: -x["mean_score"])
        all_orders[combo_key] = orders
        print(f"  Best: {orders[0]['order']}")
    plot_firing_orders(all_orders)
    flat_sorted = sorted(all_results, key=lambda x: -x["mean_score"])
    path = os.path.join(STATS_DIR, "step5_firing_orders.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat_sorted[0].keys())
        w.writeheader(); w.writerows(flat_sorted)
    print(f"\n    Saved {path}")
    payload = {"firing_orders":flat_sorted,
               "best_order":flat_sorted[0]["order"],
               "best_score":flat_sorted[0]["mean_score"],
               "all_orders_by_combo":{k:v for k,v in all_orders.items()}}
    with open(os.path.join(STATS_DIR, "step5_firing_orders.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Best overall: {flat_sorted[0]['order']}")
    print("\n  Step 5 complete.")
    return payload


if __name__ == "__main__":
    run_step5()