import csv, os, math, subprocess, time, json, random, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations, product

LOG_FILE  = "simulation_progress_log.json"
OUT_DIR   = "Steps/Step4_Combos"
STATS_DIR = "data/stats"
GRID=50; CENTER=GRID//2; CMIN=CENTER-5; CMAX=CENTER+5

RAYS = {
    "Gamma":      {"mac":"macs/tests/test_gamma.mac",  "csv":"gamma_edep.csv",  "color":"#e74c3c"},
    "Neutron":    {"mac":"macs/tests/test_neutron.mac", "csv":"neutron_edep.csv","color":"#3498db"},
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
    tmp   = f"_tmp_step4_{ray_name.replace(' ','_')}.mac"
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
    total    = g.sum()
    core     = g[CMIN:CMAX,CMIN:CMAX,CMIN:CMAX].sum()
    surf     = total - core
    ratio    = surf/core if core>0 else 999
    voxels   = int((g>0).sum())
    nz       = np.where(g>0)
    depth    = float(np.mean(nz[2]))*2 if len(nz[2])>0 else 0
    return {"core_mev":float(core),"total_mev":float(total),
            "ratio":float(ratio),"voxels":voxels,"mean_depth":depth}

def combo_score(m):
    return m["core_mev"] / (m["ratio"] + 0.01)

def run_step4():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)

    log      = json.load(open(LOG_FILE)) if os.path.exists(LOG_FILE) else {}
    step2    = log.get("step2", {})
    step3    = log.get("step3", {})

    # Use step3 optimal counts if available, else step2
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
    print("  STEP 4 — Multi-Ray Combination Testing")
    print("  Tests all 2-ray, 3-ray, and 4-ray combinations using optimal counts.")
    print("  Combines voxel grids additively (energy deposition is linear/additive).")
    print(SEP)
    print(f"\n  Optimal counts from previous steps:")
    for ray, n in opt_counts.items():
        print(f"    {ray:<12} n={n}")

    ray_names = list(RAYS.keys())
    all_combos = []
    for r in range(2, len(ray_names)+1):
        for combo in combinations(ray_names, r):
            all_combos.append(combo)

    print(f"\n  Testing {len(all_combos)} combinations (pairs, triples, quad)...\n")

    results = []

    for combo in all_combos:
        combo_name = " + ".join(combo)
        print(f"  [{combo_name}]")
        combined_grid = np.zeros((GRID,GRID,GRID))

        for ray in combo:
            cfg = RAYS[ray]
            n   = opt_counts[ray]
            print(f"    Running {ray} n={n} ... ", end="", flush=True)
            rows = run_sim(cfg["mac"], cfg["csv"], ray, n)
            if rows:
                combined_grid += grid_from_rows(rows)
                print("ok")
            else:
                print("no data")

        m = metrics_from_grid(combined_grid)
        score = combo_score(m)
        results.append({
            "combo": combo_name,
            "n_rays": len(combo),
            "rays": list(combo),
            "score": score,
            **m
        })
        print(f"    → core={m['core_mev']:.2f} MeV  ratio={m['ratio']:.4f}  score={score:.4f}\n")

    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)

    # Save
    log["step4"] = {"combinations": results[:10]}
    with open(LOG_FILE,"w") as f:
        json.dump(log, f, indent=2)

    csv_path = os.path.join(STATS_DIR, "step4_combinations.csv")
    with open(csv_path,"w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["Combo","N_Rays","Core_MeV","Total_MeV","Ratio","Voxels","Score"])
        for r in results:
            w.writerow([r["combo"],r["n_rays"],f"{r['core_mev']:.4f}",
                        f"{r['total_mev']:.4f}",f"{r['ratio']:.4f}",
                        r["voxels"],f"{r['score']:.4f}"])

    print_report(results)
    plot_results(results)
    print(f"\n  CSV: {csv_path}")
    print(f"  Graphs: {OUT_DIR}/\n")

def print_report(results):
    SEP = "="*90
    print(f"\n{SEP}")
    print("  STEP 4 — Combination Rankings")
    print(SEP)
    print(f"  {'Rank':<5} {'Combination':<35} {'Core MeV':>12} {'Ratio':>8} {'Score':>12}")
    print(f"  {'-'*78}")
    for i, r in enumerate(results[:10], 1):
        print(f"  {i:<5} {r['combo']:<35} {r['core_mev']:>12.2f} {r['ratio']:>8.4f} {r['score']:>12.4f}")
    print(SEP)
    print(f"\n  ★ BEST COMBINATION: {results[0]['combo']}")
    print(f"    Core MeV: {results[0]['core_mev']:.2f}")
    print(f"    Surf/Core: {results[0]['ratio']:.4f}")
    print(f"    Score: {results[0]['score']:.4f}")

def plot_results(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#0d0d0d")
    fig.suptitle("Step 4 — Multi-Ray Combination Results", color="white", fontsize=14, fontweight="bold")

    top10 = results[:10]
    labels = [r["combo"].replace(" + ","\n+\n") for r in top10]
    colors_by_nrays = {2:"#3498db", 3:"#2ecc71", 4:"#e74c3c"}
    bar_colors = [colors_by_nrays[r["n_rays"]] for r in top10]

    for ax, key, title, ylabel in [
        (axes[0], "score",    "Top 10 Combos by Score",    "Score"),
        (axes[1], "core_mev", "Core MeV by Combo",         "Core MeV"),
        (axes[2], "ratio",    "Surf/Core by Combo",        "Surf/Core (lower=better)"),
    ]:
        ax.set_facecolor("#111111")
        ax.set_title(title, color="white", fontsize=10)
        ax.set_ylabel(ylabel, color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=6)
        for spine in ax.spines.values(): spine.set_color("#444")
        vals = [r[key] for r in top10]
        bars = ax.bar(range(len(top10)), vals, color=bar_colors, edgecolor="none")
        ax.set_xticks(range(len(top10)))
        ax.set_xticklabels([r["combo"] for r in top10], rotation=45, ha="right", fontsize=6, color="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                    f"{val:.2f}", ha="center", va="bottom", color="white", fontsize=6)

    from matplotlib.patches import Patch
    legend = [Patch(color=c, label=f"{n}-ray combo") for n,c in colors_by_nrays.items()]
    axes[0].legend(handles=legend, fontsize=7, facecolor="#222", labelcolor="white")

    plt.tight_layout(rect=[0,0,1,0.95])
    out = os.path.join(OUT_DIR, "step4_combinations.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"  Saved {out}")

if __name__ == "__main__":
    run_step4()
