"""
STEP 1 — Individual Ray Characterization
Prion Multi-Modal Radiation Therapy Simulation

HOW TO RUN (copy-paste any of these into terminal):

  Normal run (simulation + graphs):
    python3 step1_rays.py

  Open GEARS GUI to watch particle tracks for one ray:
    python3 step1_rays.py --gui gamma
    python3 step1_rays.py --gui neutron
    python3 step1_rays.py --gui carbon
    python3 step1_rays.py --gui alpha

  GUI + auto screenshot saved to Steps/Step1_Rays/:
    python3 step1_rays.py --gui gamma --screenshot

  Skip simulation, just replot from saved log:
    python3 step1_rays.py --plot-only
"""

import csv, os, math, subprocess, time, json, random, sys, argparse

# ── Optional matplotlib (graceful fallback if not installed) ──────────────────
try:
    import matplotlib
    matplotlib.use("Agg")          # saves to file, no pop-up window needed
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not found — graphs disabled.")
    print("       Install with:  pip3 install matplotlib numpy\n")

# ─── File paths ───────────────────────────────────────────────────────────────
LOG_FILE  = "simulation_progress_log.json"
PLOT_DIR  = "Steps/Step1_Rays"
DATA_DIR  = "data/raw_output"
STATS_CSV = "data/stats/individual_ray_stats.csv"
for d in (PLOT_DIR, DATA_DIR, "data/stats"):
    os.makedirs(d, exist_ok=True)

# ─── Grid / geometry constants ────────────────────────────────────────────────
GRID   = 50
CENTER = GRID // 2   # = 25
CMIN   = CENTER - 5  # = 20  (core zone lower bound)
CMAX   = CENTER + 5  # = 30  (core zone upper bound)

# ─── Adaptive stopping config ─────────────────────────────────────────────────
STABLE_WIN = 10   # rolling window
STABLE_THR = 0.5  # CV spread < 0.5 percentage points
STABLE_CNF = 3    # confirmed N consecutive times

# ─── Ray configuration ────────────────────────────────────────────────────────
RAYS = {
    "Gamma": {
        "mac": "macs/tests/test_gamma.mac",
        "csv": "gamma_edep.csv",
        "vis": "macs/visualize/vis_individual_G_Gamma.mac",
        "min": 20, "max": 80,
        "color": "#e74c3c",
        "rationale": "Moderate — random Compton scatter angles",
    },
    "Neutron": {
        "mac": "macs/tests/test_neutron.mac",
        "csv": "neutron_edep.csv",
        "vis": "macs/visualize/vis_individual_N_Neutron.mac",
        "min": 50, "max": 150,
        "color": "#3498db",
        "rationale": "Highest — secondary proton production is stochastic",
    },
    "Carbon Ion": {
        "mac": "macs/tests/test_carbon.mac",
        "csv": "carbon_edep.csv",
        "vis": "macs/visualize/vis_individual_C_Carbon_Ion.mac",
        "min": 20, "max": 60,
        "color": "#2ecc71",
        "rationale": "Low — Bragg peak stopping is deterministic",
    },
    "Alpha": {
        "mac": "macs/tests/test_alpha.mac",
        "csv": "alpha_edep.csv",
        "vis": "macs/visualize/vis_individual_A_Alpha.mac",
        "min": 15, "max": 50,
        "color": "#f39c12",
        "rationale": "Lowest — short dense tracks, consistent energy deposition",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
#  CSV / analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(fp):
    """Load energy deposition CSV -> dict {(ix,iy,iz): MeV}."""
    data = {}
    if not os.path.exists(fp):
        return data
    with open(fp) as f:
        for row in csv.reader(f):
            if len(row) < 4:
                continue
            try:
                ix, iy, iz = int(row[0]), int(row[1]), int(row[2])
                e = float(row[3])
                if e > 0:
                    data[(ix, iy, iz)] = data.get((ix, iy, iz), 0) + e
            except ValueError:
                continue
    return data


def classify(ix, iy, iz):
    if CMIN <= ix <= CMAX and CMIN <= iy <= CMAX and CMIN <= iz <= CMAX:
        return "core"
    if ix < 5 or ix > 44 or iy < 5 or iy > 44 or iz < 5 or iz > 44:
        return "edge"
    return "surface"


def analyse(data):
    if not data:
        return None
    total  = sum(data.values())
    voxels = len(data)
    core_e = sum(e for (ix, iy, iz), e in data.items() if classify(ix, iy, iz) == "core")
    surf_e = sum(e for (ix, iy, iz), e in data.items() if classify(ix, iy, iz) == "surface")
    mean_z = sum(iz * e for (ix, iy, iz), e in data.items()) / total
    ratio  = surf_e / core_e if core_e > 0 else float("inf")
    center = any(
        abs(ix - CENTER) <= 2 and abs(iy - CENTER) <= 2 and abs(iz - CENTER) <= 2
        for (ix, iy, iz) in data
    )
    return {
        "voxels":     voxels,
        "total_MeV":  total,
        "core_MeV":   core_e,
        "surf_MeV":   surf_e,
        "surf_core":  ratio,
        "mean_depth": mean_z,
        "core_hit":   center,
    }


def mean_std(vals):
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    m = sum(vals) / n
    s = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return m, s


def cv_pct(vals):
    m, s = mean_std(vals)
    return (s / m * 100) if m > 0 else 0.0


def fmt_time(sec):
    sec = int(sec)
    if sec < 60:   return f"{sec}s"
    if sec < 3600: return f"{sec // 60}m {sec % 60}s"
    return f"{sec // 3600}h {(sec % 3600) // 60}m {sec % 60}s"


def quality_label(cv):
    if cv < 1:   return "PUBLICATION QUALITY"
    if cv < 2:   return "EXCELLENT"
    if cv < 5:   return "GOOD"
    if cv < 10:  return "ACCEPTABLE"
    return "HIGH VARIANCE"

# ─────────────────────────────────────────────────────────────────────────────
#  GEARS runner
# ─────────────────────────────────────────────────────────────────────────────

def run_sim(mac_path, csv_out_path, ray_name):
    s1 = random.randint(1, 999_999_999)
    s2 = random.randint(1, 999_999_999)
    tmp = f"_tmp_{ray_name.replace(' ', '_')}.mac"

    # Delete stale CSV
    if os.path.exists(csv_out_path):
        os.remove(csv_out_path)

    with open(mac_path) as f:
        lines = f.readlines()

    # Find /run/beamOn and inject seed RIGHT before it
    # This overwrites any earlier fixed seed in the mac file
    new_lines = []
    seed_injected = False
    for line in lines:
        if "/run/beamOn" in line and not seed_injected:
            new_lines.append(f"/random/setSeeds {s1} {s2}\n")
            seed_injected = True
        new_lines.append(line)

    # If no beamOn found, append seed at end anyway
    if not seed_injected:
        new_lines.append(f"/random/setSeeds {s1} {s2}\n")

    with open(tmp, "w") as f:
        f.writelines(new_lines)

    subprocess.run(["gears", tmp], capture_output=True)
    os.remove(tmp)
    return load_csv(csv_out_path)

# ─────────────────────────────────────────────────────────────────────────────
#  GUI / visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

def open_gui(ray_name, screenshot=False):
    """Open GEARS GUI for a ray. Optional: save screenshot."""
    matched = None
    for k in RAYS:
        if ray_name.lower() in k.lower():
            matched = k
            break
    if matched is None:
        print(f"[ERROR] Unknown ray '{ray_name}'. Choose: gamma / neutron / carbon / alpha")
        sys.exit(1)

    cfg     = RAYS[matched]
    vis_mac = cfg.get("vis", "")
    tmp     = None

    if vis_mac and os.path.exists(vis_mac):
        if screenshot:
            tmp     = _inject_screenshot(vis_mac, matched)
            vis_mac = tmp
    else:
        # Build a vis mac from the test mac
        vis_mac = _make_vis_mac(cfg["mac"], matched, screenshot)
        tmp     = vis_mac

    safe = matched.replace(" ", "_").lower()
    print(f"\n  Opening GEARS GUI — {matched}")
    print(f"  Mac: {vis_mac}")
    if screenshot:
        print(f"  Screenshot -> {PLOT_DIR}/gui_{safe}.png")
    print("  Close the GUI window when done.\n")

    subprocess.run(["gears", vis_mac])

    if tmp and os.path.exists(tmp):
        os.remove(tmp)


def _make_vis_mac(base_mac, ray_name, screenshot):
    """Generate a temp mac file that opens OpenGL GUI + optionally screenshots."""
    with open(base_mac) as f:
        lines = f.readlines()

    # Split at /run/beamOn so we insert vis commands before it
    beam_idx = next((i for i, l in enumerate(lines) if "/run/beamOn" in l), len(lines))

    vis_cmds = [
        "/vis/open OGL 800x600-0+0\n",
        "/vis/drawVolume\n",
        "/vis/viewer/set/viewpointThetaPhi 70 20\n",
        "/vis/viewer/set/style surface\n",
        "/vis/scene/add/trajectories smooth\n",
        "/vis/scene/endOfEventAction accumulate 50\n",
        "/vis/viewer/set/autoRefresh true\n",
        "/vis/modeling/trajectories/create/drawByParticleID\n",
        "/vis/modeling/trajectories/drawByParticleID-0/set gamma blue\n",
        "/vis/modeling/trajectories/drawByParticleID-0/set neutron green\n",
        "/vis/modeling/trajectories/drawByParticleID-0/set alpha red\n",
        "/vis/modeling/trajectories/drawByParticleID-0/set C12 orange\n",
        "/vis/modeling/trajectories/drawByParticleID-0/set e- yellow\n",
        "/vis/modeling/trajectories/drawByParticleID-0/set proton cyan\n",
    ]

    safe       = ray_name.replace(" ", "_").lower()
    ss_cmd     = []
    if screenshot:
        out    = os.path.abspath(os.path.join(PLOT_DIR, f"gui_{safe}.png"))
        ss_cmd = [f"/vis/ogl/export {out}\n"]

    new_lines = vis_cmds + lines[:beam_idx] + lines[beam_idx:beam_idx + 1] + ss_cmd
    tmp       = f"_tmp_vis_{safe}.mac"
    with open(tmp, "w") as f:
        f.writelines(new_lines)
    return tmp


def _inject_screenshot(vis_mac, ray_name):
    """Append screenshot export command to a copy of an existing vis mac."""
    safe = ray_name.replace(" ", "_").lower()
    out  = os.path.abspath(os.path.join(PLOT_DIR, f"gui_{safe}.png"))
    with open(vis_mac) as f:
        content = f.read()
    content += f"\n/vis/ogl/export {out}\n"
    tmp = f"_tmp_ss_{safe}.mac"
    with open(tmp, "w") as f:
        f.write(content)
    return tmp

# ─────────────────────────────────────────────────────────────────────────────
#  PLAIN TEXT REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_plain_report(all_results):
    S = "=" * 90
    D = "-" * 90
    print(f"\n{S}")
    print("  STEP 1 RESULTS — PLAIN TEXT REPORT")
    print(S)

    for ray, s in all_results.items():
        vm, vs = s["voxels"]["mean"],     s["voxels"]["std"]
        cm, cs = s["core_MeV"]["mean"],   s["core_MeV"]["std"]
        sm, ss = s["surf_MeV"]["mean"],   s["surf_MeV"]["std"]
        rm, rs = s["surf_core"]["mean"],  s["surf_core"]["std"]
        dm, ds = s["mean_depth"]["mean"], s["mean_depth"]["std"]
        print(f"\n  ── {ray} ──")
        print(f"    Runs completed : {s['n_runs']}  ({s['status']})")
        print(f"    CV quality     : {s['final_cv_pct']:.4f}%  -> {s['cv_quality']}")
        print(f"    Voxels hit     : {vm:>12,.0f}  ±  {vs:.0f}")
        print(f"    Total MeV      : {s['total_MeV']['mean']:>12,.2f}  ±  {s['total_MeV']['std']:.2f}")
        print(f"    Core MeV       : {cm:>12,.4f}  ±  {cs:.4f}")
        print(f"    Surface MeV    : {sm:>12,.4f}  ±  {ss:.4f}")
        print(f"    Surf/Core ratio: {rm:>12.4f}  ±  {rs:.4f}")
        print(f"    Mean depth (z) : {dm:>12.4f}  ±  {ds:.4f} mm")
        print(f"    Core hit rate  : {s['core_hit_pct']:.1f}%")
        print(f"    Wall time      : {fmt_time(s['time_sec'])}")

    print(f"\n{D}")
    print(f"  {'Ray':<12} {'Runs':>5} {'CV%':>8}  {'Voxels':>14}  {'Core MeV':>18}  {'Surf/Core':>14}  {'Depth mm':>12}  Quality")
    print(f"  {D}")
    for ray, s in all_results.items():
        vm, vs = s["voxels"]["mean"],     s["voxels"]["std"]
        cm, cs = s["core_MeV"]["mean"],   s["core_MeV"]["std"]
        rm, rs = s["surf_core"]["mean"],  s["surf_core"]["std"]
        dm, ds = s["mean_depth"]["mean"], s["mean_depth"]["std"]
        print(
            f"  {ray:<12} {s['n_runs']:>5} {s['final_cv_pct']:>7.4f}%"
            f"  {vm:>8,.0f} +-{vs:<6.0f}"
            f"  {cm:>12,.2f} +-{cs:<8.2f}"
            f"  {rm:>6.4f} +-{rs:<6.4f}"
            f"  {dm:>6.2f} +-{ds:.2f}mm"
            f"  {s['cv_quality']}"
        )

    print(f"\n{S}")
    print("  INTERPRETATION")
    print(S)
    rays      = list(all_results.keys())
    core_vals = {r: all_results[r]["core_MeV"]["mean"] for r in rays}
    sc_vals   = {r: all_results[r]["surf_core"]["mean"] for r in rays}
    best_core = max(core_vals, key=core_vals.get)
    best_sc   = min(sc_vals,   key=sc_vals.get)
    print(f"  Highest core energy   : {best_core} ({core_vals[best_core]:,.2f} MeV)")
    print(f"  Most core-preferential: {best_sc} (Surf/Core = {sc_vals[best_sc]:.4f})")
    alpha_sc = sc_vals.get("Alpha")
    if alpha_sc is not None and alpha_sc < 1.0:
        print("  Alpha Surf/Core < 1.0 — unique: deposits MORE energy in core than surface")
    print("  -> Recommended combo entering Step 2: Gamma + Neutron + Alpha\n")

# ─────────────────────────────────────────────────────────────────────────────
#  GRAPH PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def make_plots(all_results):
    if not HAS_MPL:
        print("[SKIP] matplotlib not available — pip3 install matplotlib numpy")
        return

    rays   = list(all_results.keys())
    colors = [RAYS[r]["color"] for r in rays]

    # ── 1. Overview 2x2 bar charts ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Step 1 — Individual Ray Characterization", fontsize=15, fontweight="bold")
    metrics = [
        ("core_MeV",   "Core Energy (MeV)",    True),
        ("voxels",     "Voxels Hit",            False),
        ("surf_core",  "Surf / Core Ratio",     False),
        ("mean_depth", "Mean Depth (mm)",       False),
    ]
    for ax, (key, label, logy) in zip(axes.flat, metrics):
        means = [all_results[r][key]["mean"] for r in rays]
        stds  = [all_results[r][key]["std"]  for r in rays]
        bars  = ax.bar(rays, means, yerr=stds, color=colors, capsize=6,
                       edgecolor="black", linewidth=0.8)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel(label)
        if logy:
            ax.set_yscale("log")
            ax.set_ylabel(label + " (log scale)")
        ax.tick_params(axis="x", rotation=15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, val in zip(bars, means):
            ypos = bar.get_height() * (1.3 if logy else 1.02)
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{val:,.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "step1_overview.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

    # ── 2. Energy breakdown: core vs surface stacked bars ────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    x      = list(range(len(rays)))
    core_m = [all_results[r]["core_MeV"]["mean"] for r in rays]
    surf_m = [all_results[r]["surf_MeV"]["mean"] for r in rays]
    core_s = [all_results[r]["core_MeV"]["std"]  for r in rays]
    surf_s = [all_results[r]["surf_MeV"]["std"]  for r in rays]
    ax.bar(x, core_m, yerr=core_s, label="Core",    color="#2c3e50", capsize=5)
    ax.bar(x, surf_m, yerr=surf_s, bottom=core_m, label="Surface", color="#95a5a6", capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(rays)
    ax.set_yscale("log")
    ax.set_title("Energy Breakdown: Core vs Surface (log scale)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Energy (MeV, log scale)")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "step1_energy_breakdown.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

    # ── 3. CV convergence line chart ──────────────────────────────────────────
    log = json.load(open(LOG_FILE)) if os.path.exists(LOG_FILE) else {}
    cv_traces = log.get("step1", {}).get("cv_traces", {})
    if cv_traces:
        fig, ax = plt.subplots(figsize=(10, 5))
        for ray in rays:
            trace = cv_traces.get(ray, [])
            if trace:
                ax.plot(range(1, len(trace) + 1), trace,
                        label=ray, color=RAYS[ray]["color"], linewidth=1.8)
        ax.axhline(10, color="red",    linestyle="--", alpha=0.5, label="Acceptable (10%)")
        ax.axhline(5,  color="orange", linestyle="--", alpha=0.5, label="Good (5%)")
        ax.axhline(2,  color="green",  linestyle="--", alpha=0.5, label="Excellent (2%)")
        ax.set_xlabel("Run number")
        ax.set_ylabel("CV %")
        ax.set_title("CV Convergence Over Runs", fontsize=13, fontweight="bold")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        out = os.path.join(PLOT_DIR, "step1_convergence.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")

    # ── 4. Radar chart ────────────────────────────────────────────────────────
    try:
        cats   = ["Core Energy", "Voxels Hit", "Core-Pref\n(inv S/C)", "Mean Depth", "Stability\n(inv CV)"]
        N      = len(cats)
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
        ax.set_title("Normalised Ray Profile", fontsize=13, fontweight="bold", pad=20)

        # Collect all raw values per axis first for normalisation
        all_raw = {
            "core":    [all_results[r]["core_MeV"]["mean"] for r in rays],
            "voxels":  [all_results[r]["voxels"]["mean"] for r in rays],
            "inv_sc":  [1.0 / (all_results[r]["surf_core"]["mean"] + 1e-9) for r in rays],
            "depth":   [all_results[r]["mean_depth"]["mean"] for r in rays],
            "inv_cv":  [100 - all_results[r]["final_cv_pct"] for r in rays],
        }
        axes_keys = list(all_raw.keys())

        for ray in rays:
            s = all_results[ray]
            raw = [
                s["core_MeV"]["mean"],
                s["voxels"]["mean"],
                1.0 / (s["surf_core"]["mean"] + 1e-9),
                s["mean_depth"]["mean"],
                100 - s["final_cv_pct"],
            ]
            normed = []
            for i, v in enumerate(raw):
                col = all_raw[axes_keys[i]]
                mn, mx = min(col), max(col)
                normed.append((v - mn) / (mx - mn + 1e-12))
            normed += normed[:1]
            ax.plot(angles, normed, color=RAYS[ray]["color"], linewidth=2, label=ray)
            ax.fill(angles, normed, color=RAYS[ray]["color"], alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(cats, size=10)
        ax.set_yticklabels([])
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        out = os.path.join(PLOT_DIR, "step1_radar.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")
    except Exception as e:
        print(f"  [WARN] Radar chart skipped: {e}")

    print()

# ─────────────────────────────────────────────────────────────────────────────
#  SAVE CSV SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def save_stats_csv(all_results):
    with open(STATS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "ray", "runs", "cv_pct", "quality",
            "voxels_mean", "voxels_std",
            "core_mev_mean", "core_mev_std",
            "surf_mev_mean", "surf_mev_std",
            "surf_core_mean", "surf_core_std",
            "depth_mm_mean", "depth_mm_std",
            "core_hit_pct", "time_sec", "stop_condition",
        ])
        for ray, s in all_results.items():
            w.writerow([
                ray, s["n_runs"], round(s["final_cv_pct"], 4), s["cv_quality"],
                round(s["voxels"]["mean"], 1),     round(s["voxels"]["std"], 1),
                round(s["core_MeV"]["mean"], 4),   round(s["core_MeV"]["std"], 4),
                round(s["surf_MeV"]["mean"], 4),   round(s["surf_MeV"]["std"], 4),
                round(s["surf_core"]["mean"], 4),  round(s["surf_core"]["std"], 4),
                round(s["mean_depth"]["mean"], 4), round(s["mean_depth"]["std"], 4),
                round(s["core_hit_pct"], 2), s["time_sec"], s["status"],
            ])
    print(f"  Stats CSV saved: {STATS_CSV}")

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN SIMULATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_all_rays():
    total_start = time.time()
    SEP = "=" * 90
    print(f"\n{SEP}")
    print("  STEP 1 — Individual Ray Simulations")
    print("  Unique random seed per run — essential for Monte Carlo accuracy.")
    print("  Adaptive stopping: halts when CV is stable, not at a fixed count.")
    print(SEP)

    # Pre-flight checks
    print("\n  Pre-flight checks:")
    all_ok = True
    for ray, cfg in RAYS.items():
        ok = "OK" if os.path.exists(cfg["mac"]) else "MISSING"
        if ok == "MISSING":
            all_ok = False
        print(f"    {ray:<12}  {ok}  ({cfg['mac']})")
    brain_ok = os.path.exists("data/brain/brain.tg")
    print(f"    brain.tg     {'OK' if brain_ok else 'MISSING'}  (data/brain/brain.tg)")
    if not brain_ok or not all_ok:
        print("\n  [ERROR] Missing required files. Aborting.")
        return
    print("  All checks passed.\n")

    log = json.load(open(LOG_FILE)) if os.path.exists(LOG_FILE) else {}
    log["step1"] = {
        "rays": {}, "timing": {}, "cv_traces": {},
        "config": {
            "grid": GRID, "center": CENTER, "core_min": CMIN, "core_max": CMAX,
            "stable_window": STABLE_WIN, "stable_threshold_pct": STABLE_THR,
            "stable_confirm": STABLE_CNF,
        },
    }

    all_results = {}

    for ray, cfg in RAYS.items():
        ray_start    = time.time()
        min_r, max_r = cfg["min"], cfg["max"]
        csv_path     = os.path.join(DATA_DIR, os.path.basename(cfg["csv"]))

        print(f"\n{SEP}")
        print(f"  [{ray}]  min={min_r}  max={max_r}  |  {cfg['rationale']}")
        print(SEP)

        runs       = []
        cv_history = []
        stable_cnt = 0
        early_stop = False

        for i in range(1, max_r + 1):
            data = run_sim(cfg["mac"], csv_path, ray)
            r    = analyse(data)
            if r:
                runs.append(r)
            else:
                print(f"    WARNING run {i}: empty CSV — is Geant4 GUI open? Close it.")

            n = len(runs)
            if n >= min_r:
                cur_cv = cv_pct([x["core_MeV"] for x in runs])
                cv_history.append(cur_cv)
                if len(cv_history) >= STABLE_WIN:
                    window = cv_history[-STABLE_WIN:]
                    spread = max(window) - min(window)
                    if spread < STABLE_THR:
                        stable_cnt += 1
                        if stable_cnt >= STABLE_CNF:
                            early_stop = True
                            print(f"\n    Converged at run {i}  CV={cur_cv:.4f}%  spread={spread:.4f}%")
                            break
                    else:
                        stable_cnt = 0

            if i % 10 == 0:
                cur_cv = cv_pct([x["core_MeV"] for x in runs]) if runs else 0
                print(f"    Run {i:>3}/{max_r}  |  CV={cur_cv:.4f}%  |  n={n}  |  {fmt_time(time.time()-ray_start)}")

        ray_time = time.time() - ray_start
        n = len(runs)
        if n == 0:
            print(f"  [{ray}] ERROR — 0 successful runs.")
            continue

        final_cv = cv_pct([x["core_MeV"] for x in runs])
        status   = "early stop (converged)" if early_stop else "reached max runs"
        quality  = quality_label(final_cv)
        print(f"\n  [{ray}] DONE  runs={n}  CV={final_cv:.4f}% -> {quality}  {fmt_time(ray_time)}")

        stats = {}
        for key in ["voxels", "total_MeV", "core_MeV", "surf_MeV", "surf_core", "mean_depth"]:
            m, s = mean_std([x[key] for x in runs])
            stats[key] = {"mean": m, "std": s}
        stats["core_hit_pct"] = sum(1 for x in runs if x["core_hit"]) / n * 100
        stats["n_runs"]       = n
        stats["final_cv_pct"] = final_cv
        stats["time_sec"]     = round(ray_time, 2)
        stats["status"]       = status
        stats["cv_quality"]   = quality

        all_results[ray]               = stats
        log["step1"]["rays"][ray]      = stats
        log["step1"]["timing"][ray]    = round(ray_time, 2)
        log["step1"]["cv_traces"][ray] = cv_history

    total_time = time.time() - total_start
    log["step1"]["total_time_sec"] = round(total_time, 2)
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)

    if not all_results:
        print("[ERROR] No results collected. Check GEARS installation.")
        return

    print_plain_report(all_results)

    print(f"\n{'='*90}")
    print("  GENERATING PLOTS")
    print(f"{'='*90}")
    make_plots(all_results)
    save_stats_csv(all_results)

    total_runs = sum(s["n_runs"] for s in all_results.values())
    print(f"\n{'='*90}")
    print("  TIMING SUMMARY")
    print(f"{'='*90}")
    for ray, s in all_results.items():
        print(f"  {ray:<12}  {s['n_runs']:>3} runs  {fmt_time(s['time_sec'])}")
    print(f"  {'TOTAL':<12}  {total_runs:>3} runs  {fmt_time(total_time)}")
    print(f"\n  Log  : {LOG_FILE}")
    print(f"  CSV  : {STATS_CSV}")
    print(f"  Plots: {PLOT_DIR}/")
    print(f"\n{'='*90}")
    print("  STEP 1 COMPLETE — next: python3 scripts/step2_optimize.py")
    print(f"{'='*90}\n")

# ─────────────────────────────────────────────────────────────────────────────
#  PLOT-ONLY mode
# ─────────────────────────────────────────────────────────────────────────────

def plot_only():
    if not os.path.exists(LOG_FILE):
        print(f"[ERROR] {LOG_FILE} not found. Run the simulation first.")
        sys.exit(1)
    log = json.load(open(LOG_FILE))
    if "step1" not in log or "rays" not in log["step1"]:
        print("[ERROR] No step1 data in log.")
        sys.exit(1)
    all_results = log["step1"]["rays"]
    print("\n  Loaded results from log. Regenerating...\n")
    print_plain_report(all_results)
    make_plots(all_results)
    save_stats_csv(all_results)

# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Individual Ray Characterization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 step1_rays.py                          full simulation run
  python3 step1_rays.py --gui gamma              open GUI for gamma
  python3 step1_rays.py --gui carbon --screenshot  GUI + save PNG
  python3 step1_rays.py --plot-only              replot from saved log
        """,
    )
    parser.add_argument("--gui", metavar="RAY",
                        help="Open GEARS GUI for a ray: gamma / neutron / carbon / alpha")
    parser.add_argument("--screenshot", action="store_true",
                        help="Save GUI screenshot to Steps/Step1_Rays/  (use with --gui)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip simulation, regenerate plots/report from saved log")
    args = parser.parse_args()

    if args.gui:
        open_gui(args.gui, screenshot=args.screenshot)
    elif args.plot_only:
        plot_only()
    else:
        run_all_rays()


if __name__ == "__main__":
    main()