"""
Surgical Tool Radiation Sterilization Simulation
External beam radiation hitting prion-contaminated surgical steel
"""
import subprocess, os, re, random, csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE     = os.path.expanduser("~/Desktop/prion_radiation_research/surgical_tool")
RAYS     = ["gamma","alpha","neutron","carbon"]
COUNTS   = [500, 1000, 2000, 5000]
OUT_DIR  = os.path.join(BASE, "Steps")
STATS    = os.path.join(BASE, "data")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(STATS,   exist_ok=True)

MAC_TEMPLATE = {
    "gamma":   ("gamma",   6.0,   "MeV", "gamma"),
    "alpha":   ("alpha",   5.5,   "MeV", "alpha"),
    "neutron": ("neutron", 14.0,  "MeV", "neutron"),
    "carbon":  ("C12",     290.0, "MeV", "carbon"),
}

COLORS = {"gamma":"#e74c3c","alpha":"#f39c12","neutron":"#3498db","carbon":"#2ecc71"}

def run_sim(ray, n):
    mac_path = os.path.join(BASE, "macs", f"{ray}.mac")
    csv_path = os.path.join(BASE, f"{ray}_edep.csv")
    tmp      = os.path.join(BASE, f"_tmp_{ray}.mac")
    if os.path.exists(csv_path):
        os.remove(csv_path)
    with open(mac_path) as f:
        content = f.read()
    content = re.sub(r'/run/beamOn\s+\d+', f'/run/beamOn {n}', content)
    s1, s2 = random.randint(1,999999999), random.randint(1,999999999)
    with open(tmp,"w") as f:
        f.write(f"/random/setSeeds {s1} {s2}\n")
        f.write(f"/geometry/source {BASE}/structure/surgical_tool.tg\n")
        f.write(content)
    result = subprocess.run(["gears", "structure/surgical_tool.tg", tmp], capture_output=True, cwd=BASE)
    os.remove(tmp)
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                parts = line.split(",")
                if len(parts) >= 4:
                    try: rows.append(float(parts[3]))
                    except: pass
    return rows

def analyze(rows):
    if not rows: return {"total":0,"mean":0,"max":0,"nonzero":0}
    arr = np.array(rows)
    nz  = arr[arr>0]
    return {"total":float(arr.sum()),"mean":float(nz.mean()) if len(nz) else 0,
            "max":float(arr.max()),"nonzero":int(len(nz))}

def run_all():
    print("\n" + "="*70)
    print("  Surgical Tool Sterilization Simulation")
    print("  External beam → stainless steel → prion surface layer")
    print("="*70)

    results = {}
    for ray in RAYS:
        results[ray] = {}
        print(f"\n  Ray: {ray.upper()}")
        for n in COUNTS:
            print(f"    n={n} ... ", end="", flush=True)
            rows = run_sim(ray, n)
            m    = analyze(rows)
            results[ray][n] = m
            print(f"total={m['total']:.4f} MeV  nonzero={m['nonzero']}")

    # Save CSV
    csv_path = os.path.join(STATS, "surgical_tool_results.csv")
    with open(csv_path,"w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["Ray","N","Total_MeV","Mean_MeV","Max_MeV","NonzeroVoxels"])
        for ray in RAYS:
            for n in COUNTS:
                m = results[ray][n]
                w.writerow([ray,n,f"{m['total']:.6f}",f"{m['mean']:.6f}",
                            f"{m['max']:.6f}",m['nonzero']])

    with open(os.path.join(STATS,"surgical_results.json"),"w") as f:
        json.dump(results, f, indent=2)

    plot_results(results)
    print(f"\n  CSV: {csv_path}")

def plot_results(results):
    fig, axes = plt.subplots(2,2, figsize=(14,10), facecolor="#0d0d0d")
    fig.suptitle("Surgical Tool Prion Sterilization — Radiation Dose Analysis\n"
                 "External Beam: Gamma | Alpha | Neutron | Carbon Ion",
                 color="white", fontsize=13, fontweight="bold")

    # Total MeV vs N per ray
    ax = axes[0,0]
    ax.set_facecolor("#111111")
    for ray in RAYS:
        ns     = COUNTS
        totals = [results[ray][n]["total"] for n in ns]
        ax.plot(ns, totals, "o-", color=COLORS[ray], linewidth=2,
                markersize=6, label=ray.capitalize())
    ax.set_title("Total MeV Deposited vs Particle Count", color="white", fontsize=10)
    ax.set_xlabel("Beam Count", color="white", fontsize=8)
    ax.set_ylabel("Total MeV", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    ax.legend(fontsize=8, facecolor="#222", labelcolor="white")
    for spine in ax.spines.values(): spine.set_color("#444")

    # Best count comparison bar
    ax = axes[0,1]
    ax.set_facecolor("#111111")
    best_n  = max(COUNTS)
    totals  = [results[ray][best_n]["total"] for ray in RAYS]
    colors  = [COLORS[r] for r in RAYS]
    bars    = ax.bar(RAYS, totals, color=colors, edgecolor="none")
    ax.set_title(f"Total MeV at N={best_n}", color="white", fontsize=10)
    ax.set_ylabel("Total MeV", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values(): spine.set_color("#444")
    for bar,val in zip(bars,totals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                f"{val:.2f}", ha="center", color="white", fontsize=7)

    # Nonzero voxels (coverage)
    ax = axes[1,0]
    ax.set_facecolor("#111111")
    for ray in RAYS:
        ns  = COUNTS
        nzs = [results[ray][n]["nonzero"] for n in ns]
        ax.plot(ns, nzs, "s-", color=COLORS[ray], linewidth=2,
                markersize=6, label=ray.capitalize())
    ax.set_title("Surface Coverage (Nonzero Voxels)", color="white", fontsize=10)
    ax.set_xlabel("Beam Count", color="white", fontsize=8)
    ax.set_ylabel("Voxels Hit", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    ax.legend(fontsize=8, facecolor="#222", labelcolor="white")
    for spine in ax.spines.values(): spine.set_color("#444")

    # Summary table
    ax = axes[1,1]
    ax.set_facecolor("#111111")
    ax.axis("off")
    best_n = max(COUNTS)
    txt = f"STERILIZATION SUMMARY (N={best_n})\n"
    txt += "─"*38 + "\n"
    txt += f"{'Ray':<10} {'Total MeV':>12} {'Max MeV':>10} {'Voxels':>8}\n"
    txt += "─"*38 + "\n"
    best_ray  = max(RAYS, key=lambda r: results[r][best_n]["total"])
    for ray in RAYS:
        m   = results[ray][best_n]
        tag = " ← BEST" if ray==best_ray else ""
        txt += f"{ray.capitalize():<10} {m['total']:>12.4f} {m['max']:>10.6f} {m['nonzero']:>8}{tag}\n"
    txt += "─"*38 + "\n"
    txt += "\nCONTEXT:\n"
    txt += "Prion sterilization requires\n"
    txt += "~10-50 kGy absorbed dose.\n"
    txt += "Steel tool surface geometry.\n"
    txt += "External beam — no brain tissue."
    ax.text(0.05,0.95, txt, transform=ax.transAxes, color="white",
            fontsize=8, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#1a1a2e", edgecolor="#4a9eff", alpha=0.9))

    plt.tight_layout(rect=[0,0,1,0.95])
    out = os.path.join(OUT_DIR, "surgical_tool_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"  Saved {out}")

if __name__ == "__main__":
    run_all()
