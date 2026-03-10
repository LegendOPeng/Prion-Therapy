import os
import itertools

PARTICLES = {
    "G": {
        "name": "Gamma",
        "comment": "Low LET, full penetration, saturates SSB repair pathways",
        "block": """/gps/particle gamma
/gps/energy 1.17 MeV
/gps/pos/type Volume
/gps/pos/shape Sphere
/gps/pos/radius 15 mm
/gps/pos/centre 0 0 0 mm
/run/beamOn 2000"""
    },
    "N": {
        "name": "Neutron",
        "comment": "Medium-high LET, generates high-LET recoil protons, wide scatter",
        "block": """/gps/particle neutron
/gps/energy 2.0 MeV
/gps/pos/type Volume
/gps/pos/shape Sphere
/gps/pos/radius 15 mm
/gps/pos/centre 0 0 0 mm
/run/beamOn 2000"""
    },
    "C": {
        "name": "Carbon Ion",
        "comment": "Very high LET at Bragg peak, nuclear fragmentation, surface opening",
        "block": """/gps/particle ion
/gps/ion 6 12
/gps/energy 3960 MeV
/gps/pos/type Volume
/gps/pos/shape Sphere
/gps/pos/radius 15 mm
/gps/pos/centre 0 0 0 mm
/run/beamOn 500"""
    },
    "A": {
        "name": "Alpha",
        "comment": "Highest LET, irreparable complex DSBs, most lethal in pre-damaged tissue",
        "block": """/gps/particle alpha
/gps/energy 5.5 MeV
/gps/pos/type Volume
/gps/pos/shape Sphere
/gps/pos/radius 15 mm
/gps/pos/centre 0 0 0 mm
/run/beamOn 2000"""
    },
}

HEADER = """/control/verbose 0
/geometry/source brain.tg
/score/create/boxMesh prionScorer
/score/mesh/boxSize 5 5 5 cm
/score/mesh/nBin 50 50 50
/score/quantity/energyDeposit eDep
/score/close
/run/initialize"""

def make_mac(order, output_csv):
    lines = [HEADER, ""]
    for sym in order:
        p = PARTICLES[sym]
        lines.append(f"# {p['name']} -- {p['comment']}")
        lines.append(p["block"])
        lines.append("")
    lines.append(f"/score/dumpQuantityToFile prionScorer eDep {output_csv}")
    return "\n".join(lines)

def main():
    three_ray_combos = list(itertools.combinations("GNCA", 3))
    four_ray_combos  = [tuple("GNCA")]
    all_combos = three_ray_combos + four_ray_combos

    mac_files  = []
    run_lines  = []

    print("Generating mac files...")

    for combo in all_combos:
        combo_str  = "".join(combo)
        all_orders = list(itertools.permutations(combo))
        for i, order in enumerate(all_orders, 1):
            order_str   = "".join(order)
            mac_name    = f"seq_{combo_str}_{i:02d}_{order_str}.mac"
            csv_name    = f"seq_{combo_str}_{i:02d}_{order_str}.csv"
            with open(mac_name, "w") as f:
                f.write(make_mac(order, csv_name))
            mac_files.append((mac_name, csv_name, order, len(combo)))
            run_lines.append(f"echo 'Running {mac_name}...'")
            run_lines.append(f"gears {mac_name}")
            arrow = " -> ".join(PARTICLES[s]["name"] for s in order)
            print(f"  [{combo_str}] Order {i:02d}: {arrow}")

    with open("run_all_orders.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"echo 'Running {len(mac_files)} simulations...'\n\n")
        f.write("\n".join(run_lines))
        f.write("\n\necho 'All done! Now run: python3 compare_orders.py'\n")

    print(f"\nCreated {len(mac_files)} mac files")
    print("Created run_all_orders.sh")

    # Build registry for compare_orders.py
    registry_lines = []
    for mac_name, csv_name, order, n_rays in mac_files:
        combo = "".join(sorted(set(order)))
        arrow = " -> ".join(PARTICLES[s]["name"] for s in order)
        registry_lines.append(f'    "{csv_name}": {{"order": "{arrow}", "combo": "{combo}", "n_rays": {n_rays}}},')

    compare_script = '''import csv, os

GRID_SIZE = 50
CENTER    = GRID_SIZE // 2
CORE_MIN  = CENTER - 5
CORE_MAX  = CENTER + 5

FILES = {
''' + "\n".join(registry_lines) + '''
}

def load_csv(fp):
    data = {}
    if not os.path.exists(fp): return data
    with open(fp) as f:
        for row in csv.reader(f):
            if len(row) < 4: continue
            try:
                ix,iy,iz = int(row[0]),int(row[1]),int(row[2])
                e = float(row[3])
                if e > 0: data[(ix,iy,iz)] = data.get((ix,iy,iz),0)+e
            except ValueError: continue
    return data

def classify(ix,iy,iz):
    if CORE_MIN<=ix<=CORE_MAX and CORE_MIN<=iy<=CORE_MAX and CORE_MIN<=iz<=CORE_MAX: return "core"
    if ix<5 or ix>44 or iy<5 or iy>44 or iz<5 or iz>44: return "edge"
    return "surface"

def analyse(data, meta):
    if not data: return None
    total  = sum(data.values())
    voxels = len(data)
    core_e = sum(e for (ix,iy,iz),e in data.items() if classify(ix,iy,iz)=="core")
    surf_e = sum(e for (ix,iy,iz),e in data.items() if classify(ix,iy,iz)=="surface")
    mean_z = sum(iz*e for (ix,iy,iz),e in data.items())/total
    ratio  = surf_e/core_e if core_e>0 else float("inf")
    score  = (core_e/total*50) + (voxels/125000*25) + (1/(ratio+0.01)*25)
    return {
        "order": meta["order"], "combo": meta["combo"], "n_rays": meta["n_rays"],
        "voxels": voxels, "total_MeV": round(total,1), "core_MeV": round(core_e,1),
        "surf_core": round(ratio,3), "mean_depth": round(mean_z,2), "score": round(score,3),
    }

def main():
    results = []
    missing = []
    for csv_name, meta in FILES.items():
        data = load_csv(csv_name)
        if data:
            r = analyse(data, meta)
            if r: results.append(r)
        else:
            missing.append(csv_name)

    if missing:
        print(f"Missing {len(missing)} files -- run bash run_all_orders.sh first")
        if not results: return

    results.sort(key=lambda x: x["score"], reverse=True)
    sep = "="*110

    combos = sorted(set(r["combo"] for r in results), key=lambda c:(len(c),c))
    for combo in combos:
        group = [r for r in results if r["combo"]==combo]
        names = " + ".join({"G":"Gamma","N":"Neutron","C":"Carbon Ion","A":"Alpha"}[s] for s in combo)
        print(f"\\n{sep}")
        print(f"  {names}  -- all firing orders ranked")
        print(sep)
        print(f"  {'Rank':<5} {'Firing Order':<52} {'Core MeV':>11} {'Surf/Core':>10} {'Voxels':>8} {'Score':>9}")
        print("-"*100)
        for i,r in enumerate(group,1):
            m = "  << BEST" if i==1 else ("  << WORST" if i==len(group) else "")
            print(f"  {i:<5} {r['order']:<52} {r['core_MeV']:>11,.1f} {r['surf_core']:>10.3f} {r['voxels']:>8,} {r['score']:>9.3f}{m}")
        best=group[0]; worst=group[-1]
        diff=round(((best['score']-worst['score'])/worst['score'])*100,1) if worst['score']>0 else 0
        print(f"\\n  Best order:  {best['order']}")
        print(f"  Worst order: {worst['order']}")
        print(f"  Order effect: {diff}% difference -- this is how much firing sequence matters")

    print(f"\\n{sep}")
    print("  OVERALL WINNER -- Best combination AND best firing order")
    print(sep)
    best = results[0]
    print(f"  Combination:  {best['combo']}")
    print(f"  Firing order: {best['order']}")
    print(f"  Score:        {best['score']}")
    print(f"  Core energy:  {best['core_MeV']:,.1f} MeV")
    print(f"  Surf/Core:    {best['surf_core']}")
    print(f"  Voxels hit:   {best['voxels']:,}")

    with open("order_comparison_results.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)
    print(f"\\n  Saved: order_comparison_results.csv")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        combos_list = sorted(set(r["combo"] for r in results), key=lambda c:(len(c),c))
        fig, axes = plt.subplots(len(combos_list), 1, figsize=(14, 5*len(combos_list)))
        if len(combos_list)==1: axes=[axes]
        for ax, combo in zip(axes, combos_list):
            group  = [r for r in results if r["combo"]==combo]
            labels = [r["order"] for r in group]
            scores = [r["score"] for r in group]
            clr    = "#2E75B6" if len(combo)==3 else "#C44E52"
            bars   = ax.barh(range(len(labels)), scores, color=clr, alpha=0.8)
            bars[0].set_color("#C6EFCE"); bars[0].set_edgecolor("#006100"); bars[0].set_linewidth(2)
            ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlabel("Score")
            names = " + ".join({"G":"Gamma","N":"Neutron","C":"Carbon Ion","A":"Alpha"}[s] for s in combo)
            ax.set_title(f"{names} -- All Orders (green = best)", fontweight="bold")
            ax.invert_yaxis()
        plt.suptitle("Firing Order Comparison -- Does sequence matter?", fontsize=12, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig("order_comparison_plot.png", dpi=150, bbox_inches="tight")
        print("  Saved: order_comparison_plot.png")
        plt.close()
    except ImportError:
        print("  pip install matplotlib --break-system-packages")

if __name__ == "__main__":
    main()
'''

    with open("compare_orders.py", "w") as f:
        f.write(compare_script)
    print("Created compare_orders.py")
    print(f"\nNext step: bash run_all_orders.sh")

if __name__ == "__main__":
    main()
