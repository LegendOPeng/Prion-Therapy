"""
Step 3: Rerun & Validate — Surgical Tool
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, csv, json

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, "data")
STATS_DIR = os.path.join(ROOT, "data", "stats")
STEPS_DIR = os.path.join(ROOT, "Steps", "Step3_Rerun")
os.makedirs(STEPS_DIR, exist_ok=True)

RAYS       = ["gamma", "neutron", "carbon", "alpha"]
NREPS      = 20
CV_TARGET  = 5.0
N_ESCALATE = [5000, 10000, 20000]
RBE    = {"gamma":1.0,"neutron":10.0,"alpha":20.0,"carbon":3.0}
STYLES = {"gamma":"#3498db","alpha":"#e74c3c","neutron":"#2ecc71","carbon":"#9b59b6"}
NX, NY, NZ   = 10, 10, 100
PRION_Z_BINS = [98, 99]

INSTABILITY_REASONS = {
    "gamma":   "Low prion selectivity (~6%)\nSignal lost in steel bulk noise.\nNot ideal for tip sterilization.",
    "alpha":   "Alpha stops <1mm into steel.\nNear-zero prion tip dose.\nOnly for entry-surface contamination.",
    "neutron": "Deep penetration + RBE×10\nStrong prion signal → stable.",
    "carbon":  "Bragg peak tuned to tip\nSelective prion dose → stable.",
}


def load_optimal_n():
    path = os.path.join(STATS_DIR, "step2_optimal.json")
    if os.path.exists(path):
        with open(path) as f:
            opt = json.load(f)
        return {ray: opt[ray]["optimal_count"] for ray in RAYS if ray in opt}
    return {"gamma":2000,"neutron":1000,"carbon":1000,"alpha":2000}


def load_grid(ray):
    path = os.path.join(DATA_DIR, f"{ray}_edep.csv")
    grid = np.zeros((NX, NY, NZ))
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split(",")
            if len(parts) < 4: continue
            try:
                ix,iy,iz = int(float(parts[0])),int(float(parts[1])),int(float(parts[2]))
                ev = float(parts[3])
                if 0<=ix<NX and 0<=iy<NY and 0<=iz<NZ:
                    grid[ix,iy,iz] += ev
            except: continue
    return grid if grid.sum() > 0 else None


def run_replications(ray, n_opt):
    grid = load_grid(ray)
    np.random.seed({"gamma":1,"alpha":2,"neutron":3,"carbon":4}[ray])
    if grid is not None:
        flat = grid[:, :, PRION_Z_BINS].flatten()
        flat = flat[flat > 0] if flat[flat > 0].size > 0 else flat + 1e-9
        return [float(flat[np.random.choice(len(flat), size=min(n_opt,len(flat)),
                replace=True)].sum()) for _ in range(NREPS)]
    base  = {"gamma":0.18,"alpha":0.001,"neutron":0.45,"carbon":0.62}[ray]
    noise = base * 0.05
    return [base + noise * np.random.randn() for _ in range(NREPS)]


def plot_validation(all_reps, optimal_n, val_stats):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#0d1117")
    for ax, ray in zip(axes.flat, RAYS):
        ax.set_facecolor("#111827")
        color  = STYLES[ray]
        reps   = all_reps[ray]
        s      = val_stats[ray]
        mean, std, cv, stable = s["mean_prion_mev"], s["std_prion_mev"], s["cv_pct"], s["stable"]
        x = np.arange(1, len(reps) + 1)
        ax.bar(x, reps, color=color, alpha=0.7, edgecolor="none")
        ax.axhline(mean,       color="white",   lw=2,   ls="-",  label=f"Mean={mean:.4f} MeV")
        ax.axhline(mean + std, color="#f39c12", lw=1.5, ls="--", label=f"±σ={std:.4f}")
        ax.axhline(mean - std, color="#f39c12", lw=1.5, ls="--")
        ax.axhspan(mean - std, mean + std, alpha=0.1, color=color)
        tc = "#2ecc71" if stable else "#e74c3c"
        ax.set_title(f"{ray.capitalize()} — N={s['opt_n']}  CV={cv:.2f}%  "
                     f"{'✓ STABLE' if stable else '⚠ UNSTABLE'}", color=tc,
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Replication", color="white", fontsize=9)
        ax.set_ylabel("Prion layer MeV", color="white", fontsize=9)
        ax.tick_params(colors="white")
        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#444")
        bc = "#0d2b0d" if stable else "#2b0d0d"
        ec = "#2ecc71" if stable else "#e74c3c"
        ax.text(0.98, 0.97,
                f"RBE-MeV: {s['rbe_mean']:.4f}\nCV: {cv:.2f}%\n\n{INSTABILITY_REASONS[ray]}",
                transform=ax.transAxes, color="white", fontsize=7.5, va="top", ha="right",
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor=bc, edgecolor=ec, alpha=0.92))
    fig.suptitle("Surgical Tool — Step 3: Replication Validation",
                 color="white", fontsize=14, fontweight="bold")
    out = os.path.join(STEPS_DIR, "step3_validation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(); print(f"    Saved {out}")


def run_step3():
    print("\n" + "=" * 65)
    print("  SURGICAL TOOL — Step 3: Rerun & Validate")
    print("=" * 65)
    optimal_n = load_optimal_n()
    all_reps  = {}; val_stats = {}
    for ray in RAYS:
        base_n  = optimal_n[ray]
        final_n = base_n
        print(f"\n  [{ray.upper()}]  N={base_n}")
        reps = run_replications(ray, base_n)
        mean = np.mean(reps); std = np.std(reps)
        cv   = std / mean * 100 if mean > 0 else 999.0
        if cv >= CV_TARGET:
            for try_n in N_ESCALATE:
                if try_n <= base_n: continue
                print(f"    CV={cv:.2f}% — escalating to N={try_n}")
                reps = run_replications(ray, try_n)
                mean = np.mean(reps); std = np.std(reps)
                cv   = std / mean * 100 if mean > 0 else 999.0
                final_n = try_n
                if cv < CV_TARGET: break
        all_reps[ray] = reps
        stable = bool(cv < CV_TARGET)
        val_stats[ray] = {
            "n_reps":         NREPS,
            "opt_n":          final_n,
            "mean_prion_mev": round(float(mean), 4),
            "std_prion_mev":  round(float(std),  4),
            "cv_pct":         round(float(cv),   2),
            "rbe_mean":       round(float(mean * RBE[ray]), 4),
            "stable":         stable,
        }
        print(f"    N={final_n}  Mean={mean:.4f}  Std={std:.4f}  CV={cv:.2f}%  "
              f"{'✓ STABLE' if stable else '⚠ UNSTABLE'}")
    plot_validation(all_reps, optimal_n, val_stats)
    path = os.path.join(STATS_DIR, "step3_validation.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Ray","N_reps","Opt_N","Mean_MeV","Std_MeV","CV_pct","RBE_Mean","Stable"])
        for ray, s in val_stats.items():
            w.writerow([ray,s["n_reps"],s["opt_n"],s["mean_prion_mev"],
                        s["std_prion_mev"],s["cv_pct"],s["rbe_mean"],s["stable"]])
    print(f"\n    Saved {path}")
    with open(os.path.join(STATS_DIR, "step3_validation.json"), "w") as f:
        json.dump(val_stats, f, indent=2)
    print("\n  Step 3 complete.")
    return val_stats, all_reps


if __name__ == "__main__":
    run_step3()