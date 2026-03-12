import csv, os, math, subprocess, time, json, random, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_FILE  = "simulation_progress_log.json"
OUT_DIR   = "Steps/Step6_Final"
STATS_DIR = "data/stats"
GRID=50; CENTER=GRID//2; CMIN=CENTER-5; CMAX=CENTER+5
REPEATS   = 10

BEST_ORDER = ["Gamma", "Alpha", "Carbon Ion"]
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
    tmp   = f"_tmp_step6_{ray_name.replace(' ','_')}.mac"
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
    eff    = core/sum(1 for _ in BEST_ORDER)
    return {"core_mev":float(core),"total_mev":float(total),"surf_mev":float(surf),
            "ratio":float(ratio),"voxels":voxels,"mean_depth":depth,"efficiency":eff}

def run_step6():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)

    log   = json.load(open(LOG_FILE)) if os.path.exists(LOG_FILE) else {}
    step2 = log.get("step2", {})
    step3 = log.get("step3", {})

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
    print("  STEP 6 — Final Protocol Validation")
    print(f"  Order: {' → '.join(BEST_ORDER)}")
    print(f"  Running {REPEATS} full repetitions to establish final statistics.")
    print(SEP)

    all_metrics = []
    all_grids   = []

    for rep in range(1, REPEATS+1):
        print(f"\n  Rep {rep}/{REPEATS}")
        combined = np.zeros((GRID,GRID,GRID))
        for ray in BEST_ORDER:
            cfg = RAYS[ray]
            n   = opt_counts[ray]
            print(f"    {ray} n={n} ... ", end="", flush=True)
            rows = run_sim(cfg["mac"], cfg["csv"], ray, n)
            if rows:
                combined += grid_from_rows(rows)
                print("ok")
            else:
                print("no data")
        m = metrics_from_grid(combined)
        score = m["core_mev"]/(m["ratio"]+0.01)
        m["score"] = score
        all_metrics.append(m)
        all_grids.append(combined.copy())
        print(f"    → core={m['core_mev']:.2f} MeV  ratio={m['ratio']:.4f}  score={score:.2f}")

    # Final statistics
    def mean_std(vals):
        m = sum(vals)/len(vals)
        s = math.sqrt(sum((v-m)**2 for v in vals)/max(len(vals)-1,1))
        return m, s

    keys = ["core_mev","total_mev","ratio","voxels","mean_depth","score"]
    stats = {k: mean_std([m[k] for m in all_metrics]) for k in keys}

    # Average grid for visualization
    avg_grid = sum(all_grids)/REPEATS

    # Save
    log["step6"] = {
        "best_order": BEST_ORDER,
        "opt_counts": opt_counts,
        "n_reps": REPEATS,
        "final_stats": {k: {"mean": v[0], "std": v[1]} for k,v in stats.items()}
    }
    with open(LOG_FILE,"w") as f:
        json.dump(log, f, indent=2)

    # Save average grid
    np.save(os.path.join(OUT_DIR, "avg_grid.npy"), avg_grid)

    # CSV
    csv_path = os.path.join(STATS_DIR, "step6_final_protocol.csv")
    with open(csv_path,"w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["Rep","Core_MeV","Total_MeV","Ratio","Voxels","Depth_mm","Score"])
        for i,m in enumerate(all_metrics,1):
            w.writerow([i,f"{m['core_mev']:.4f}",f"{m['total_mev']:.4f}",
                        f"{m['ratio']:.4f}",m["voxels"],
                        f"{m['mean_depth']:.2f}",f"{m['score']:.4f}"])

    print_report(stats, opt_counts)
    plot_results(all_metrics, avg_grid, stats)
    print(f"\n  CSV: {csv_path}")
    print(f"  Graphs: {OUT_DIR}/\n")

def print_report(stats, opt_counts):
    SEP = "="*90
    print(f"\n{SEP}")
    print("  STEP 6 — Final Protocol Report")
    print(f"  Protocol: {' → '.join(BEST_ORDER)}")
    print(f"  Particle counts: " + "  ".join(f"{r}={opt_counts.get(r,'?')}" for r in BEST_ORDER))
    print(SEP)
    print(f"  Metric            Mean              Std")
    print(f"  {'-'*50}")
    labels = {"core_mev":"Core MeV","total_mev":"Total MeV","ratio":"Surf/Core",
              "voxels":"Voxels Hit","mean_depth":"Mean Depth mm","score":"Score"}
    for k, label in labels.items():
        m, s = stats[k]
        print(f"  {label:<18} {m:>14.4f}    ±{s:.4f}")
    print(SEP)
    cv = (stats["core_mev"][1]/stats["core_mev"][0])*100 if stats["core_mev"][0]>0 else 0
    print(f"\n  Final CV: {cv:.4f}%")
    if cv < 5:
        print("  Protocol stability: EXCELLENT")
    elif cv < 10:
        print("  Protocol stability: GOOD")
    else:
        print("  Protocol stability: ACCEPTABLE")

def plot_results(all_metrics, avg_grid, stats):
    fig = plt.figure(figsize=(18,10), facecolor="#0d0d0d")
    fig.suptitle(f"Step 6 — Final Protocol: {' → '.join(BEST_ORDER)}",
                 color="white", fontsize=14, fontweight="bold")

    # Rep-by-rep scores
    ax1 = fig.add_subplot(231, facecolor="#111111")
    reps = list(range(1, len(all_metrics)+1))
    scores = [m["score"] for m in all_metrics]
    cores  = [m["core_mev"] for m in all_metrics]
    ratios = [m["ratio"] for m in all_metrics]
    ax1.plot(reps, scores, "o-", color="#2ecc71", linewidth=2, markersize=6)
    ax1.axhline(stats["score"][0], color="white", linestyle="--", alpha=0.5, linewidth=1)
    ax1.set_title("Score per Repetition", color="white", fontsize=10)
    ax1.set_xlabel("Rep", color="white", fontsize=8)
    ax1.set_ylabel("Score", color="white", fontsize=8)
    ax1.tick_params(colors="white", labelsize=7)
    for spine in ax1.spines.values(): spine.set_color("#444")

    # Core MeV per rep
    ax2 = fig.add_subplot(232, facecolor="#111111")
    ax2.bar(reps, cores, color="#e74c3c", edgecolor="none")
    ax2.axhline(stats["core_mev"][0], color="white", linestyle="--", alpha=0.5)
    ax2.set_title("Core MeV per Rep", color="white", fontsize=10)
    ax2.set_xlabel("Rep", color="white", fontsize=8)
    ax2.set_ylabel("Core MeV", color="white", fontsize=8)
    ax2.tick_params(colors="white", labelsize=7)
    for spine in ax2.spines.values(): spine.set_color("#444")

    # Ratio per rep
    ax3 = fig.add_subplot(233, facecolor="#111111")
    ax3.plot(reps, ratios, "s-", color="#f39c12", linewidth=2, markersize=6)
    ax3.axhline(stats["ratio"][0], color="white", linestyle="--", alpha=0.5)
    ax3.set_title("Surf/Core Ratio per Rep", color="white", fontsize=10)
    ax3.set_xlabel("Rep", color="white", fontsize=8)
    ax3.set_ylabel("Ratio", color="white", fontsize=8)
    ax3.tick_params(colors="white", labelsize=7)
    for spine in ax3.spines.values(): spine.set_color("#444")

    # 3D avg grid
    ax4 = fig.add_subplot(234, projection="3d", facecolor="#0d0d0d")
    threshold = avg_grid.max()*0.02
    xi,yi,zi  = np.where(avg_grid>threshold)
    ev        = avg_grid[xi,yi,zi]
    norm      = (ev-ev.min())/(ev.max()-ev.min()+1e-12)
    ax4.scatter(xi,yi,zi, c=ev, cmap="hot", s=norm*15+1, alpha=0.5, linewidths=0)
    for x in [CMIN,CMAX]:
        for y in [CMIN,CMAX]:
            ax4.plot([x,x],[y,y],[CMIN,CMAX],color="cyan",alpha=0.3,lw=0.5)
    for x in [CMIN,CMAX]:
        for z in [CMIN,CMAX]:
            ax4.plot([x,x],[CMIN,CMAX],[z,z],color="cyan",alpha=0.3,lw=0.5)
    for y in [CMIN,CMAX]:
        for z in [CMIN,CMAX]:
            ax4.plot([CMIN,CMAX],[y,y],[z,z],color="cyan",alpha=0.3,lw=0.5)
    ax4.set_title("Avg Energy Deposition\n(cyan=prion core)", color="white", fontsize=9)
    ax4.tick_params(colors="white", labelsize=6)
    ax4.xaxis.pane.fill = False
    ax4.yaxis.pane.fill = False
    ax4.zaxis.pane.fill = False
    ax4.view_init(elev=25, azim=45)

    # Depth profile
    ax5 = fig.add_subplot(235, facecolor="#111111")
    depth_profile = avg_grid.sum(axis=(0,1))
    ax5.bar(range(GRID), depth_profile, color="#3498db", edgecolor="none")
    ax5.axvspan(CMIN, CMAX, alpha=0.2, color="cyan", label="Core")
    ax5.set_title("Avg Depth Profile", color="white", fontsize=10)
    ax5.set_xlabel("Z voxel", color="white", fontsize=8)
    ax5.set_ylabel("Energy (MeV)", color="white", fontsize=8)
    ax5.tick_params(colors="white", labelsize=7)
    for spine in ax5.spines.values(): spine.set_color("#444")
    ax5.legend(fontsize=7, facecolor="#222", labelcolor="white")

    # Summary stats box
    ax6 = fig.add_subplot(236, facecolor="#111111")
    ax6.axis("off")
    labels = {"Core MeV":    f"{stats['core_mev'][0]:.2f} ± {stats['core_mev'][1]:.2f}",
              "Surf/Core":   f"{stats['ratio'][0]:.4f} ± {stats['ratio'][1]:.4f}",
              "Voxels Hit":  f"{stats['voxels'][0]:.0f} ± {stats['voxels'][1]:.0f}",
              "Mean Depth":  f"{stats['mean_depth'][0]:.2f} ± {stats['mean_depth'][1]:.2f} mm",
              "Score":       f"{stats['score'][0]:.2f} ± {stats['score'][1]:.2f}",
              "CV%":         f"{(stats['core_mev'][1]/stats['core_mev'][0])*100:.2f}%"}
    txt = "FINAL PROTOCOL SUMMARY\n"
    txt += f"{'─'*30}\n"
    txt += f"Order: {' → '.join(BEST_ORDER)}\n"
    txt += f"{'─'*30}\n"
    for k,v in labels.items():
        txt += f"{k:<14} {v}\n"
    ax6.text(0.1, 0.9, txt, transform=ax6.transAxes, color="white",
             fontsize=9, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#222", edgecolor="#555", alpha=0.8))

    plt.tight_layout(rect=[0,0,1,0.95])
    out = os.path.join(OUT_DIR, "step6_final_protocol.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"  Saved {out}")

if __name__ == "__main__":
    run_step6()
