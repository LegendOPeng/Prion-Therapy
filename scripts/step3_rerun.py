import csv, os, math, subprocess, time, json, random, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_FILE  = "simulation_progress_log.json"
OUT_DIR   = "Steps/Step3_Rerun"
STATS_DIR = "data/stats"
GRID=50; CENTER=GRID//2; CMIN=CENTER-5; CMAX=CENTER+5
STABLE_WIN=10; STABLE_THR=0.5; STABLE_CONF=3

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

def compute_metrics(rows):
    if not rows:
        return None
    core_mev  = sum(e for x,y,z,e in rows if CMIN<=x<CMAX and CMIN<=y<CMAX and CMIN<=z<CMAX)
    total_mev = sum(e for _,_,_,e in rows)
    surf_mev  = total_mev - core_mev
    ratio     = surf_mev/core_mev if core_mev>0 else 999
    depths    = [z for _,_,z,e in rows if e>0]
    mean_depth= sum(depths)/len(depths)*2 if depths else 0
    voxels    = len(rows)
    return {"core_mev":core_mev,"total_mev":total_mev,"ratio":ratio,
            "mean_depth":mean_depth,"voxels":voxels}

def run_sim(mac_path, csv_path, ray_name, n_particles):
    seed1 = random.randint(1,999999999)
    seed2 = random.randint(1,999999999)
    tmp   = f"_tmp_step3_{ray_name.replace(' ','_')}.mac"
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

def cv(vals):
    if len(vals) < 2:
        return 999.0
    m = sum(vals)/len(vals)
    if m == 0:
        return 999.0
    s = math.sqrt(sum((v-m)**2 for v in vals)/(len(vals)-1))
    return (s/m)*100

def run_step3():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)

    # Load optimal counts from Step 2
    log = json.load(open(LOG_FILE)) if os.path.exists(LOG_FILE) else {}
    step2 = log.get("step2", {})

    SEP = "="*90
    print(f"\n{SEP}")
    print("  STEP 3 — Validation Reruns at Optimal Counts")
    print("  Reruns each ray at its Step 2 optimal count with adaptive CV stopping.")
    print(SEP)

    results = {}

    for ray, cfg in RAYS.items():
        opt_n = step2.get(ray, {}).get("optimal_count", 1000)
        print(f"\n  {'─'*60}")
        print(f"  Ray: {ray}  (optimal n={opt_n} from Step 2)")
        print(f"  {'─'*60}")

        all_core  = []
        all_ratio = []
        all_depth = []
        all_vox   = []
        cv_history= []
        conf_count= 0
        run_num   = 0
        min_runs  = 15
        max_runs  = 80

        while run_num < max_runs:
            run_num += 1
            rows = run_sim(cfg["mac"], cfg["csv"], ray, opt_n)
            m    = compute_metrics(rows)
            if not m:
                continue

            all_core.append(m["core_mev"])
            all_ratio.append(m["ratio"])
            all_depth.append(m["mean_depth"])
            all_vox.append(m["voxels"])

            current_cv = cv(all_core)
            cv_history.append(current_cv)

            print(f"    Run {run_num:>3}  core={m['core_mev']:>10.2f} MeV  "
                  f"ratio={m['ratio']:.4f}  CV={current_cv:.4f}%")

            if run_num >= min_runs and len(cv_history) >= STABLE_WIN:
                window = cv_history[-STABLE_WIN:]
                spread = max(window) - min(window)
                if spread < STABLE_THR:
                    conf_count += 1
                    if conf_count >= STABLE_CONF:
                        print(f"    → Converged at run {run_num} (CV stable within {STABLE_THR}%)")
                        break
                else:
                    conf_count = 0

        # Final stats
        mean_core  = sum(all_core)/len(all_core)
        std_core   = math.sqrt(sum((v-mean_core)**2 for v in all_core)/max(len(all_core)-1,1))
        mean_ratio = sum(all_ratio)/len(all_ratio)
        std_ratio  = math.sqrt(sum((v-mean_ratio)**2 for v in all_ratio)/max(len(all_ratio)-1,1))
        mean_depth = sum(all_depth)/len(all_depth)
        std_depth  = math.sqrt(sum((v-mean_depth)**2 for v in all_depth)/max(len(all_depth)-1,1))
        mean_vox   = sum(all_vox)/len(all_vox)
        final_cv   = cv(all_core)

        results[ray] = {
            "optimal_count": opt_n,
            "n_runs": run_num,
            "mean_core": mean_core, "std_core": std_core,
            "mean_ratio": mean_ratio, "std_ratio": std_ratio,
            "mean_depth": mean_depth, "std_depth": std_depth,
            "mean_vox": mean_vox,
            "final_cv": final_cv,
            "cv_history": cv_history,
            "all_core": all_core,
            "color": cfg["color"]
        }

    # Save to log
    log["step3"] = {r: {k:v for k,v in d.items() if k not in ("cv_history","all_core","color")}
                    for r,d in results.items()}
    with open(LOG_FILE,"w") as f:
        json.dump(log, f, indent=2)

    # CSV
    csv_path = os.path.join(STATS_DIR, "step3_validation.csv")
    with open(csv_path,"w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["Ray","Optimal_N","Runs","Mean_Core_MeV","Std_Core","Mean_Ratio",
                    "Std_Ratio","Mean_Depth_mm","Std_Depth","Final_CV_pct"])
        for ray, v in results.items():
            w.writerow([ray, v["optimal_count"], v["n_runs"],
                        f"{v['mean_core']:.4f}", f"{v['std_core']:.4f}",
                        f"{v['mean_ratio']:.4f}", f"{v['std_ratio']:.4f}",
                        f"{v['mean_depth']:.2f}", f"{v['std_depth']:.2f}",
                        f"{v['final_cv']:.4f}"])

    print_report(results)
    plot_results(results)
    print(f"\n  CSV: {csv_path}")
    print(f"  Graphs: {OUT_DIR}/")
    print(f"  JSON: {LOG_FILE}\n")

def print_report(results):
    SEP = "="*90
    print(f"\n{SEP}")
    print("  STEP 3 — Validation Report")
    print(SEP)
    print(f"  {'Ray':<12} {'N':>6} {'Runs':>5} {'Core MeV':>14} {'Surf/Core':>12} {'Depth mm':>12} {'CV%':>8}")
    print(f"  {'-'*78}")
    for ray, v in results.items():
        print(f"  {ray:<12} {v['optimal_count']:>6} {v['n_runs']:>5} "
              f"{v['mean_core']:>8.2f}+-{v['std_core']:<6.2f} "
              f"{v['mean_ratio']:>6.4f}+-{v['std_ratio']:<6.4f} "
              f"{v['mean_depth']:>6.2f}+-{v['std_depth']:<4.2f}mm "
              f"{v['final_cv']:>8.4f}%")
    print(SEP)

def plot_results(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#0d0d0d")
    fig.suptitle("Step 3 — Validation at Optimal Counts", color="white", fontsize=13, fontweight="bold")

    ray_names = list(results.keys())
    colors    = [results[r]["color"] for r in ray_names]

    # CV convergence
    ax = axes[0]
    ax.set_facecolor("#111111")
    ax.set_title("CV Convergence", color="white", fontsize=10)
    ax.set_xlabel("Run number", color="white", fontsize=8)
    ax.set_ylabel("CV %", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values(): spine.set_color("#444")
    for ray, v in results.items():
        ax.plot(v["cv_history"], color=v["color"], label=ray, linewidth=1.5)
    ax.axhline(0.5, color="white", linestyle="--", alpha=0.4, linewidth=1, label="Threshold")
    ax.legend(fontsize=7, facecolor="#222", labelcolor="white")

    # Core MeV distribution
    ax = axes[1]
    ax.set_facecolor("#111111")
    ax.set_title("Core MeV Distribution", color="white", fontsize=10)
    ax.set_xlabel("Ray", color="white", fontsize=8)
    ax.set_ylabel("Core MeV", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values(): spine.set_color("#444")
    means = [results[r]["mean_core"] for r in ray_names]
    stds  = [results[r]["std_core"]  for r in ray_names]
    bars  = ax.bar(ray_names, means, color=colors, edgecolor="none")
    ax.errorbar(ray_names, means, yerr=stds, fmt="none", color="white", capsize=4, linewidth=1.5)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                f"{val:.1f}", ha="center", va="bottom", color="white", fontsize=7)

    # Surf/Core
    ax = axes[2]
    ax.set_facecolor("#111111")
    ax.set_title("Surf/Core Ratio (lower=better)", color="white", fontsize=10)
    ax.set_xlabel("Ray", color="white", fontsize=8)
    ax.set_ylabel("Surf/Core", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values(): spine.set_color("#444")
    ratios = [results[r]["mean_ratio"] for r in ray_names]
    stds_r = [results[r]["std_ratio"]  for r in ray_names]
    bars   = ax.bar(ray_names, ratios, color=colors, edgecolor="none")
    ax.errorbar(ray_names, ratios, yerr=stds_r, fmt="none", color="white", capsize=4, linewidth=1.5)
    for bar, val in zip(bars, ratios):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                f"{val:.3f}", ha="center", va="bottom", color="white", fontsize=7)

    plt.tight_layout(rect=[0,0,1,0.95])
    out = os.path.join(OUT_DIR, "step3_validation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"  Saved {out}")

if __name__ == "__main__":
    run_step3()
