import csv, os, math, subprocess, time, json, random, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import permutations

LOG_FILE  = "simulation_progress_log.json"
OUT_DIR   = "Steps/Step5_FiringOrders"
STATS_DIR = "data/stats"
GRID=50; CENTER=GRID//2; CMIN=CENTER-5; CMAX=CENTER+5

RAYS = {
    "Gamma":      {"mac":"macs/tests/test_gamma.mac",  "csv":"gamma_edep.csv",  "color":"#e74c3c"},
    "Carbon Ion": {"mac":"macs/tests/test_carbon.mac",  "csv":"carbon_edep.csv", "color":"#2ecc71"},
    "Alpha":      {"mac":"macs/tests/test_alpha.mac",   "csv":"alpha_edep.csv",  "color":"#f39c12"},
}

def load_csv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                rows.append((int(parts[0]),int(parts[1]),int(parts[2]),float(parts[3])))
            except ValueError:
                continue
    return rows

def run_sim(mac_path, csv_path, ray_name, n_particles):
    seed1 = random.randint(1,999999999)
    seed2 = random.randint(1,999999999)
    tmp   = f"_tmp_step5_{ray_name.replace(' ','_')}.mac"
    if os.path.exists(csv_path):
        os.remove(csv_path)
    with open(mac_path) as f:
        content = f.read()
    content = re.sub(r'/run/beamOn\s+\d+', f'/run/beamOn {n_particles}', content)
    with open(tmp,"w") as f:
        f.write(f"/random/setSeeds {seed1} {seed2}\n")
        f.write(content)
    subprocess.run(["gears", tmp], capture_output=True)
    os.remove(tmp)
    return load_csv(csv_path)

def grid_from_rows(rows):
    g = np.zeros((GRID,GRID,GRID))
    for x,y,z,e in rows:
        if 0<=x<GRID and 0<=y<GRID and 0<=z<GRID:
            g[x,y,z] += e
    return g

def metrics_from_grid(g):
    total  = g.sum()
    core   = g[CMIN:CMAX,CMIN:CMAX,CMIN:CMAX].sum()
    surf   = total - core
    ratio  = surf/core if core>0 else 999
    voxels = int((g>0).sum())
    nz     = np.where(g>0)
    depth  = float(np.mean(nz[2]))*2 if len(nz[2])>0 else 0
    return {"core_mev":float(core),"total_mev":float(total),
            "ratio":float(ratio),"voxels":voxels,"mean_depth":depth}

def run_step5():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)

    log    = json.load(open(LOG_FILE)) if os.path.exists(LOG_FILE) else {}
    step2  = log.get("step2", {})
    step3  = log.get("step3", {})

    opt_counts = {}
    for ray in RAYS:
        if ray in step3:
            opt_counts[ray] = step3[ray]["optimal_count"]
        elif ray in step2:
            opt_counts[ray] = step2[ray]["optimal_count"]
        else:
            opt_counts[ray] = 500

    SEP = "="*90
    print(f"\n{SEP}")
    print("  STEP 5 — Firing Order Optimization")
    print("  Tests all 6 permutations of the winning combo: Gamma + Carbon Ion + Alpha")
    print("  Each order runs 5 times and averages results to reduce Monte Carlo noise.")
    print(SEP)

    ray_names = list(RAYS.keys())
    all_perms = list(permutations(ray_names))
    print(f"\n  Testing {len(all_perms)} firing orders x 5 repeats each...\n")

    results = []
    REPEATS = 5

    for perm in all_perms:
        order_name = " → ".join(perm)
        print(f"  Order: {order_name}")
        run_scores = []
        run_metrics = []

        for rep in range(REPEATS):
            combined = np.zeros((GRID,GRID,GRID))
            for ray in perm:
                cfg = RAYS[ray]
                n   = opt_counts[ray]
                rows = run_sim(cfg["mac"], cfg["csv"], ray, n)
                if rows:
                    combined += grid_from_rows(rows)
            m = metrics_from_grid(combined)
            score = m["core_mev"] / (m["ratio"] + 0.01)
            run_scores.append(score)
            run_metrics.append(m)
            print(f"    Rep {rep+1}: core={m['core_mev']:.2f} ratio={m['ratio']:.4f} score={score:.2f}")

        mean_score    = sum(run_scores)/len(run_scores)
        mean_core     = sum(m["core_mev"] for m in run_metrics)/REPEATS
        mean_ratio    = sum(m["ratio"] for m in run_metrics)/REPEATS
        mean_depth    = sum(m["mean_depth"] for m in run_metrics)/REPEATS
        std_score     = math.sqrt(sum((s-mean_score)**2 for s in run_scores)/max(REPEATS-1,1))

        results.append({
            "order": order_name,
            "perm": list(perm),
            "mean_score": mean_score,
            "std_score": std_score,
            "mean_core": mean_core,
            "mean_ratio": mean_ratio,
            "mean_depth": mean_depth,
        })
        print(f"  → avg score={mean_score:.2f} ± {std_score:.2f}\n")

    results.sort(key=lambda x: x["mean_score"], reverse=True)

    log["step5"] = {"firing_orders": results}
    with open(LOG_FILE,"w") as f:
        json.dump(log, f, indent=2)

    csv_path = os.path.join(STATS_DIR, "step5_firing_orders.csv")
    with open(csv_path,"w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["Order","Mean_Score","Std_Score","Mean_Core_MeV","Mean_Ratio","Mean_Depth_mm"])
        for r in results:
            w.writerow([r["order"],f"{r['mean_score']:.4f}",f"{r['std_score']:.4f}",
                        f"{r['mean_core']:.4f}",f"{r['mean_ratio']:.4f}",f"{r['mean_depth']:.2f}"])

    print_report(results)
    plot_results(results)
    print(f"\n  CSV: {csv_path}")
    print(f"  Graphs: {OUT_DIR}/\n")

def print_report(results):
    SEP = "="*90
    print(f"\n{SEP}")
    print("  STEP 5 — Firing Order Rankings")
    print(SEP)
    print(f"  {'Rank':<5} {'Order':<40} {'Score':>10} {'±':>8} {'Core MeV':>12} {'Ratio':>8}")
    print(f"  {'-'*82}")
    for i, r in enumerate(results, 1):
        print(f"  {i:<5} {r['order']:<40} {r['mean_score']:>10.2f} "
              f"{r['std_score']:>8.2f} {r['mean_core']:>12.2f} {r['mean_ratio']:>8.4f}")
    print(SEP)
    print(f"\n  ★ BEST FIRING ORDER: {results[0]['order']}")

def plot_results(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0d0d0d")
    fig.suptitle("Step 5 — Firing Order Optimization\n(Gamma + Carbon Ion + Alpha)",
                 color="white", fontsize=13, fontweight="bold")

    orders = [r["order"].replace(" → ","\n→\n") for r in results]
    scores = [r["mean_score"] for r in results]
    stds   = [r["std_score"]  for r in results]
    ratios = [r["mean_ratio"] for r in results]
    cores  = [r["mean_core"]  for r in results]

    cmap   = plt.get_cmap("RdYlGn")
    norm   = plt.Normalize(min(scores), max(scores))
    colors = [cmap(norm(s)) for s in scores]

    ax = axes[0]
    ax.set_facecolor("#111111")
    ax.set_title("Mean Score by Firing Order", color="white", fontsize=10)
    ax.set_ylabel("Mean Score", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values(): spine.set_color("#444")
    bars = ax.bar(range(len(results)), scores, color=colors, edgecolor="none")
    ax.errorbar(range(len(results)), scores, yerr=stds,
                fmt="none", color="white", capsize=4, linewidth=1.5)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([r["order"] for r in results], rotation=45, ha="right", fontsize=7, color="white")
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                f"{val:.0f}", ha="center", va="bottom", color="white", fontsize=7)

    ax = axes[1]
    ax.set_facecolor("#111111")
    ax.set_title("Core MeV vs Surf/Core by Order", color="white", fontsize=10)
    ax.set_xlabel("Core MeV", color="white", fontsize=9)
    ax.set_ylabel("Surf/Core Ratio (lower=better)", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values(): spine.set_color("#444")
    sc = ax.scatter(cores, ratios, c=scores, cmap="RdYlGn", s=120, zorder=5)
    for i, r in enumerate(results):
        ax.annotate(f"#{i+1}", (cores[i], ratios[i]),
                    textcoords="offset points", xytext=(5,5),
                    color="white", fontsize=7)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.03)
    cb.set_label("Score", color="white", fontsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout(rect=[0,0,1,0.93])
    out = os.path.join(OUT_DIR, "step5_firing_orders.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"  Saved {out}")

if __name__ == "__main__":
    run_step5()
