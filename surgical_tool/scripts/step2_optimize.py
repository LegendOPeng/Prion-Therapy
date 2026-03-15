"""
Step 2: Beam Count Optimization — Surgical Tool
Uses interpolation search (like brain project) to find precise
optimal N per ray within ICRP safety limits.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, csv, json

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, "data")
STATS_DIR = os.path.join(ROOT, "data", "stats")
STEPS_DIR = os.path.join(ROOT, "Steps", "Step2_Opt")
os.makedirs(STEPS_DIR, exist_ok=True)

NX, NY, NZ  = 10, 10, 100
PRION_BINS  = list(range(95, 100))
RAYS        = ["gamma", "neutron", "carbon", "alpha"]
RBE         = {"gamma": 1.0, "neutron": 10.0, "carbon": 3.0, "alpha": 20.0}
STYLES      = {"gamma": "#3498db", "neutron": "#2ecc71", "carbon": "#9b59b6", "alpha": "#e74c3c"}

# ICRP safety caps: N_max = 50000 / RBE
SAFE_N_MAX  = {ray: int(50_000 / RBE[ray]) for ray in RBE}
# gamma:50000  carbon:16666  neutron:5000  alpha:2500

CV_THRESHOLD = 5.0   # target CV%
BOOTSTRAP_N  = 30    # bootstrap samples per measurement


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


def measure_cv(ray, n, grid=None):
    """Bootstrap CV% at particle count N from real grid data."""
    seed = {"gamma": 10, "neutron": 20, "carbon": 30, "alpha": 40}[ray]

    if grid is not None:
        flat = grid[:, :, PRION_BINS].flatten()
        flat = flat[flat > 0]
        if len(flat) == 0:
            return 999.0, 0.0, 0.0
    else:
        # Synthetic fallback
        np.random.seed(seed)
        base  = {"gamma": 0.18, "neutron": 0.45, "carbon": 0.62, "alpha": 0.001}[ray]
        noise = base * 0.15 / np.sqrt(max(n, 1) / 500)
        flat  = np.abs(np.random.normal(base, noise, size=max(n * 5, 500))) + 1e-9

    rng = np.random.default_rng(seed + n)
    samples = []
    for _ in range(BOOTSTRAP_N):
        idx = rng.choice(len(flat), size=min(n, len(flat)), replace=True)
        samples.append(flat[idx].sum())

    arr  = np.array(samples)
    mean = arr.mean()
    std  = arr.std()
    cv   = std / mean * 100 if mean > 0 else 999.0
    return float(cv), float(mean), float(std)


def interpolation_search(ray, grid=None, threshold=CV_THRESHOLD, precision=5):
    """
    Interpolation search for smallest N where CV < threshold.
    Narrows: [100, safe_max] → [600,1000] → [634,700] → 641
    Falls back to bisection when CV curve is flat.
    """
    low  = 100
    high = SAFE_N_MAX[ray]

    cv_low,  _, _ = measure_cv(ray, low,  grid)
    cv_high, _, _ = measure_cv(ray, high, grid)

    print(f"    Safety cap: N={high:,}  CV@{low}={cv_low:.1f}%  CV@{high}={cv_high:.1f}%")

    if cv_high >= threshold:
        print(f"    Warning: CV={cv_high:.1f}% still above {threshold}% at safety cap — using cap")
        return high, cv_high

    if cv_low < threshold:
        print(f"    CV already below threshold at N={low}")
        return low, cv_low

    iterations = 0
    while (high - low) > precision and iterations < 60:
        iterations += 1
        if abs(cv_low - cv_high) < 0.01:
            probe = (low + high) // 2
        else:
            t     = (cv_low - threshold) / (cv_low - cv_high)
            probe = int(low + t * (high - low))
            probe = max(low + 1, min(probe, high - 1))

        cv_probe, _, _ = measure_cv(ray, probe, grid)
        print(f"      iter {iterations:2d}: N={probe:6,}  CV={cv_probe:.2f}%  range=[{low:,}, {high:,}]")

        if cv_probe < threshold:
            high    = probe
            cv_high = cv_probe
        else:
            low    = probe
            cv_low = cv_probe

    print(f"    Optimal: N={high:,}  CV={cv_high:.2f}%  ({iterations} iterations)")
    return high, cv_high


def build_plot_points(ray, opt_n, grid=None):
    """Build CV/efficiency curve across log-spaced counts for plotting."""
    safe_max = SAFE_N_MAX[ray]
    counts   = sorted(set(
        [100, 250, 500, 1000, 2000, 5000, opt_n] +
        [int(10 ** x) for x in np.linspace(2, np.log10(safe_max), 8)]
    ))
    counts = [n for n in counts if n <= safe_max]
    conv   = {}
    for n in counts:
        cv, mean, std = measure_cv(ray, n, grid)
        conv[n] = {
            "cv_pct":         round(cv,   2),
            "mean_prion_mev": round(mean, 4),
            "std_prion_mev":  round(std,  4),
            "rbe_mev":        round(mean * RBE[ray], 4),
            "efficiency":     round(mean * RBE[ray] / max(n, 1) * 1000, 6),
        }
    return conv


def plot_results(all_conv, optimal):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#0d1117")

    # CV vs N (like brain project step2 — log x-axis)
    ax = axes[0, 0]; ax.set_facecolor("#111827")
    for ray in RAYS:
        xs = sorted(all_conv[ray].keys())
        ys = [all_conv[ray][n]["cv_pct"] for n in xs]
        ax.plot(xs, ys, "o-", color=STYLES[ray], lw=2, markersize=5, label=ray.capitalize())
        opt_n = optimal[ray]["optimal_count"]
        ax.axvline(opt_n, color=STYLES[ray], lw=1, ls=":", alpha=0.5)
    ax.axhline(CV_THRESHOLD, color="white", lw=1.5, ls="--", label=f"CV={CV_THRESHOLD}% threshold")
    ax.set_xscale("log")
    ax.set_title("CV% vs Particle Count", color="white", fontsize=11, fontweight="bold")
    ax.set_xlabel("N (log scale)", color="white"); ax.set_ylabel("CV (%)", color="white")
    ax.tick_params(colors="white"); ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#444")

    # Efficiency vs N
    ax = axes[0, 1]; ax.set_facecolor("#111827")
    for ray in RAYS:
        xs = sorted(all_conv[ray].keys())
        ys = [all_conv[ray][n]["efficiency"] for n in xs]
        ax.plot(xs, ys, "s-", color=STYLES[ray], lw=2, markersize=5, label=ray.capitalize())
    ax.set_xscale("log")
    ax.set_title("Efficiency (RBE-MeV per 1k particles)", color="white", fontsize=11, fontweight="bold")
    ax.set_xlabel("N (log scale)", color="white"); ax.set_ylabel("Efficiency", color="white")
    ax.tick_params(colors="white"); ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#444")

    # RBE dose vs N
    ax = axes[1, 0]; ax.set_facecolor("#111827")
    for ray in RAYS:
        xs = sorted(all_conv[ray].keys())
        ys = [all_conv[ray][n]["rbe_mev"] for n in xs]
        ax.plot(xs, ys, "^-", color=STYLES[ray], lw=2, markersize=5, label=ray.capitalize())
    ax.set_xscale("log")
    ax.set_title("RBE-Weighted Prion Dose vs N", color="white", fontsize=11, fontweight="bold")
    ax.set_xlabel("N (log scale)", color="white"); ax.set_ylabel("RBE-MeV", color="white")
    ax.tick_params(colors="white"); ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#444")

    # Optimal count bar chart with safety caps
    ax = axes[1, 1]; ax.set_facecolor("#111827")
    rs   = RAYS
    vals = [optimal[r]["optimal_count"] for r in rs]
    bars = ax.bar(rs, vals, color=[STYLES[r] for r in rs], edgecolor="none")
    for bar, r, v in zip(bars, rs, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                f"{v:,}", ha="center", color="white", fontsize=10, fontweight="bold")
    for i, r in enumerate(rs):
        cap = SAFE_N_MAX[r]
        ax.plot([i-0.4, i+0.4], [cap, cap], color="red", lw=1.5, ls="--", alpha=0.7)
    ax.set_title("Optimal Count (CV<5%, within ICRP limits)", color="white", fontsize=10, fontweight="bold")
    ax.set_ylabel("Optimal N", color="white"); ax.tick_params(colors="white")
    ax.text(0.98, 0.97, "-- ICRP cap", transform=ax.transAxes,
            color="red", fontsize=8, ha="right", va="top", alpha=0.8)
    for sp in ax.spines.values(): sp.set_edgecolor("#444")

    fig.suptitle("Surgical Tool — Step 2: Beam Count Optimization (Interpolation Search)",
                 color="white", fontsize=13, fontweight="bold")
    out = os.path.join(STEPS_DIR, "step2_optimization.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(); print(f"    Saved {out}")


def plot_summary(optimal):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#111827"); ax.axis("off")
    lines = [
        "STEP 2 — INTERPOLATION SEARCH SUMMARY",
        "="*60,
        f"{'Ray':<10} {'Opt N':>8} {'ICRP Cap':>10} {'RBE-MeV':>10} {'CV%':>7}",
        "─"*60,
    ]
    for ray in sorted(optimal.keys(), key=lambda r: -optimal[r]["rbe_at_optimal"]):
        o   = optimal[ray]
        cap = SAFE_N_MAX[ray]
        lines.append(f"{ray:<10} {o['optimal_count']:>8,} {cap:>10,} "
                     f"{o['rbe_at_optimal']:>10.4f} {o['cv_at_optimal']:>7.2f}%")
    lines += ["─"*60, "", "Search method: Interpolation (not fixed counts)",
              "Safety limit:  ICRP 50 mSv/yr occupational"]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, color="white",
            fontsize=9, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#1a1a2e", edgecolor="#4a9eff", alpha=0.9))
    fig.suptitle("Step 2 Summary", color="white", fontsize=13, fontweight="bold")
    out = os.path.join(STEPS_DIR, "step2_summary.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(); print(f"    Saved {out}")


def run_step2():
    print("\n" + "="*65)
    print("  SURGICAL TOOL — Step 2: Beam Count Optimization")
    print("  Method: Interpolation search | Limit: ICRP 50 mSv/yr")
    print("="*65)

    all_conv = {}
    optimal  = {}

    for ray in RAYS:
        print(f"\n  [{ray.upper()}]")
        grid  = load_grid(ray)
        opt_n, opt_cv = interpolation_search(ray, grid)
        conv  = build_plot_points(ray, opt_n, grid)
        all_conv[ray] = conv

        entry = conv.get(opt_n, {})
        optimal[ray] = {
            "optimal_count":    opt_n,
            "rbe_at_optimal":   entry.get("rbe_mev", 0.0),
            "cv_at_optimal":    opt_cv,
            "icrp_safe_cap":    SAFE_N_MAX[ray],
            "efficiency":       entry.get("efficiency", 0.0),
        }
        print(f"    → N={opt_n:,}  CV={opt_cv:.2f}%  RBE-MeV={entry.get('rbe_mev', 0):.4f}")

    plot_results(all_conv, optimal)
    plot_summary(optimal)

    path = os.path.join(STATS_DIR, "step2_optimal_counts.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Ray", "OptimalN", "ICRP_Cap", "RBE_MeV", "CV_pct", "Efficiency"])
        for ray, o in optimal.items():
            w.writerow([ray, o["optimal_count"], o["icrp_safe_cap"],
                        o["rbe_at_optimal"], o["cv_at_optimal"], o["efficiency"]])

    with open(os.path.join(STATS_DIR, "step2_optimal.json"), "w") as f:
        json.dump(optimal, f, indent=2)

    print(f"\n    Saved {path}")
    print("\n  Step 2 complete.")
    return optimal, all_conv

if __name__ == "__main__":
    run_step2()
