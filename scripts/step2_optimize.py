import csv, os, math, subprocess, time, json, random, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_FILE   = "simulation_progress_log.json"
OUT_DIR    = "Steps/Step2_Opt"
STATS_DIR  = "data/stats"
GRID=50; CENTER=GRID//2; CMIN=CENTER-5; CMAX=CENTER+5
COUNTS     = [100, 300, 500, 1000, 2000, 5000, 10000]

RAYS = {
    "Gamma":      {"mac":"macs/tests/test_gamma.mac",   "csv":"gamma_edep.csv",   "color":"#e74c3c"},
    "Neutron":    {"mac":"macs/tests/test_neutron.mac",  "csv":"neutron_edep.csv", "color":"#3498db"},
    "Carbon Ion": {"mac":"macs/tests/test_carbon.mac",   "csv":"carbon_edep.csv",  "color":"#2ecc71"},
    "Alpha":      {"mac":"macs/tests/test_alpha.mac",    "csv":"alpha_edep.csv",   "color":"#f39c12"},
}

DOSE_LIMITS = {"Gamma":60.0,"Neutron":45.0,"Carbon Ion":55.0,"Alpha":50.0}

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

def compute_metrics(rows, n_particles):
    if not rows:
        return None
    voxels_hit = len(rows)
    core_mev = sum(e for x,y,z,e in rows if CMIN<=x<CMAX and CMIN<=y<CMAX and CMIN<=z<CMAX)
    total_mev = sum(e for _,_,_,e in rows)
    surf_mev  = total_mev - core_mev
    ratio     = surf_mev/core_mev if core_mev>0 else 999
    depths    = [z for _,_,z,e in rows if e>0]
    mean_depth= sum(depths)/len(depths)*2 if depths else 0
    eff       = core_mev/n_particles if n_particles>0 else 0
    score     = eff/(ratio+0.01)
    return {"voxels":voxels_hit,"core_mev":core_mev,"total_mev":total_mev,
            "ratio":ratio,"mean_depth":mean_depth,"efficiency":eff,"score":score,
            "n_particles":n_particles}

def run_sim(mac_path, csv_path, ray_name, n_particles):
    seed1 = random.randint(1,999999999)
    seed2 = random.randint(1,999999999)
    tmp   = f"_tmp_step2_{ray_name.replace(' ','_')}.mac"
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

def interpolate_peak(counts, scores):
    if len(counts) < 3:
        return counts[scores.index(max(scores))]
    best_i = scores.index(max(scores))
    lo = max(0, best_i-1)
    hi = min(len(counts)-1, best_i+1)
    xs = counts[lo:hi+1]
    ys = scores[lo:hi+1]
    try:
        coeffs = np.polyfit(np.log(xs), ys, 2)
        if coeffs[0] < 0:
            peak_log = -coeffs[1]/(2*coeffs[0])
            peak_n   = int(round(math.exp(peak_log)))
            peak_n   = max(counts[0], min(counts[-1]*2, peak_n))
            return peak_n
    except Exception:
        pass
    return counts[best_i]

def run_step2():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)
    SEP = "="*90
    print(f"\n{SEP}")
    print("  STEP 2 — Per-Ray Particle Count Optimization")
    print("  Finds the optimal beamOn count per ray via interpolation search.")
    print(SEP)

    results = {}

    for ray, cfg in RAYS.items():
        print(f"\n  {'─'*60}")
        print(f"  Ray: {ray}")
        print(f"  {'─'*60}")
        counts_tested = []
        metrics_list  = []

        # Coarse sweep
        for n in COUNTS:
            print(f"    Testing n={n:>6} ... ", end="", flush=True)
            rows = run_sim(cfg["mac"], cfg["csv"], ray, n)
            m    = compute_metrics(rows, n)
            if m:
                counts_tested.append(n)
                metrics_list.append(m)
                print(f"core={m['core_mev']:>10.2f} MeV  ratio={m['ratio']:.4f}  eff={m['efficiency']:.6f}  score={m['score']:.6f}")
            else:
                print("no data")

        if not metrics_list:
            print(f"  [SKIP] No data for {ray}")
            continue

        # Interpolate peak
        scores = [m["score"] for m in metrics_list]
        peak_n = interpolate_peak(counts_tested, scores)

        # Test predicted peak if not already tested
        if peak_n not in counts_tested:
            print(f"    Testing interpolated peak n={peak_n} ... ", end="", flush=True)
            rows = run_sim(cfg["mac"], cfg["csv"], ray, peak_n)
            m    = compute_metrics(rows, peak_n)
            if m:
                counts_tested.append(peak_n)
                metrics_list.append(m)
                scores.append(m["score"])
                print(f"core={m['core_mev']:>10.2f} MeV  ratio={m['ratio']:.4f}  score={m['score']:.6f}")

        # Best
        best_idx = scores.index(max(scores))
        best_n   = counts_tested[best_idx]
        best_m   = metrics_list[best_idx]

        results[ray] = {
            "optimal_count": best_n,
            "metrics": best_m,
            "all_counts": counts_tested,
            "all_metrics": metrics_list,
            "color": cfg["color"]
        }
        print(f"\n  ★ OPTIMAL for {ray}: n={best_n}  score={best_m['score']:.6f}")

    # Save results
    log = json.load(open(LOG_FILE)) if os.path.exists(LOG_FILE) else {}
    log["step2"] = {r: {"optimal_count": v["optimal_count"], "metrics": v["metrics"]}
                    for r,v in results.items()}
    with open(LOG_FILE,"w") as f:
        json.dump(log, f, indent=2)

    # CSV
    csv_path = os.path.join(STATS_DIR, "step2_optimal_counts.csv")
    with open(csv_path,"w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["Ray","Optimal_Count","Core_MeV","Surf_Core_Ratio","Efficiency","Score","Mean_Depth_mm"])
        for ray, v in results.items():
            m = v["metrics"]
            w.writerow([ray, v["optimal_count"], f"{m['core_mev']:.4f}",
                        f"{m['ratio']:.4f}", f"{m['efficiency']:.6f}",
                        f"{m['score']:.6f}", f"{m['mean_depth']:.2f}"])

    print_report(results)
    plot_results(results)
    print(f"\n  Graphs saved to {OUT_DIR}/")
    print(f"  CSV saved to {csv_path}")
    print(f"  JSON log updated: {LOG_FILE}\n")

def print_report(results):
    SEP = "="*90
    print(f"\n{SEP}")
    print("  STEP 2 — Optimization Results")
    print(SEP)
    print(f"  {'Ray':<12} {'Optimal N':>10} {'Core MeV':>12} {'Surf/Core':>10} {'Eff MeV/p':>12} {'Score':>12} {'Depth mm':>10}")
    print(f"  {'-'*82}")
    for ray, v in results.items():
        m = v["metrics"]
        print(f"  {ray:<12} {v['optimal_count']:>10} {m['core_mev']:>12.2f} {m['ratio']:>10.4f} "
              f"{m['efficiency']:>12.6f} {m['score']:>12.6f} {m['mean_depth']:>10.2f}mm")
    print(SEP)
    print("\n  Interpretation:")
    for ray, v in results.items():
        m = v["metrics"]
        n = v["optimal_count"]
        print(f"  {ray:<12} — optimal at {n} particles. "
              f"Core energy {m['core_mev']:.2f} MeV, Surf/Core {m['ratio']:.4f}, "
              f"efficiency {m['efficiency']:.6f} MeV/particle.")

def plot_results(results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#0d0d0d")
    fig.suptitle("Step 2 — Per-Ray Particle Count Optimization", color="white", fontsize=14, fontweight="bold")

    plot_data = [
        (axes[0,0], "score",      "Optimization Score",        "Score (efficiency/ratio)"),
        (axes[0,1], "core_mev",   "Core Energy vs Count",      "Core MeV"),
        (axes[1,0], "ratio",      "Surf/Core Ratio vs Count",  "Surf/Core Ratio (lower=better)"),
        (axes[1,1], "efficiency", "Efficiency vs Count",       "MeV per particle"),
    ]

    for ax, key, title, ylabel in plot_data:
        ax.set_facecolor("#111111")
        ax.set_title(title, color="white", fontsize=10)
        ax.set_xlabel("Particle Count (beamOn)", color="white", fontsize=8)
        ax.set_ylabel(ylabel, color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#444")

        for ray, v in results.items():
            counts = v["all_counts"]
            vals   = [m[key] for m in v["all_metrics"]]
            ax.plot(counts, vals, "o-", color=v["color"], label=ray, linewidth=1.5, markersize=4)
            opt_idx = v["all_counts"].index(v["optimal_count"]) if v["optimal_count"] in v["all_counts"] else -1
            if opt_idx >= 0:
                ax.axvline(v["optimal_count"], color=v["color"], alpha=0.3, linestyle="--", linewidth=1)
                ax.plot(counts[opt_idx], vals[opt_idx], "*", color=v["color"], markersize=12)

        ax.set_xscale("log")
        ax.legend(fontsize=7, facecolor="#222", labelcolor="white")

    plt.tight_layout(rect=[0,0,1,0.95])
    out = os.path.join(OUT_DIR, "step2_optimization.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"  Saved {out}")

    # Summary bar chart
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5), facecolor="#0d0d0d")
    fig2.suptitle("Step 2 — Optimal Count Summary per Ray", color="white", fontsize=13, fontweight="bold")
    ray_names = list(results.keys())
    colors    = [results[r]["color"] for r in ray_names]

    for ax, key, title, ylabel in [
        (axes2[0], "optimal_count", "Optimal Particle Count", "beamOn count"),
        (axes2[1], "core_mev",      "Core MeV at Optimal",    "MeV"),
        (axes2[2], "ratio",         "Surf/Core at Optimal",   "ratio"),
    ]:
        ax.set_facecolor("#111111")
        vals = [results[r]["optimal_count"] if key=="optimal_count" else results[r]["metrics"][key]
                for r in ray_names]
        bars = ax.bar(ray_names, vals, color=colors, edgecolor="none")
        ax.set_title(title, color="white", fontsize=10)
        ax.set_ylabel(ylabel, color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#444")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                    f"{val:.0f}" if key=="optimal_count" else f"{val:.3f}",
                    ha="center", va="bottom", color="white", fontsize=8)

    plt.tight_layout(rect=[0,0,1,0.95])
    out2 = os.path.join(OUT_DIR, "step2_summary.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"  Saved {out2}")

if __name__ == "__main__":
    import sys
    if "--plot-only" in sys.argv:
        log = json.load(open(LOG_FILE)) if os.path.exists(LOG_FILE) else {}
        if "step2" not in log:
            print("No Step 2 data in log yet. Run without --plot-only first.")
        else:
            print("Plot-only mode not fully implemented yet — run full simulation.")
    else:
        run_step2()
