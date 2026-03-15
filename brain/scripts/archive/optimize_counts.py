import os
import itertools

BASELINE = {"G": 2000, "N": 2000, "C": 1000, "A": 2000}
VARIANTS = {
    "G": [500, 1000, 2000, 5000],
    "N": [500, 1000, 2000, 5000],
    "C": [500, 1000, 2000, 5000],
    "A": [500, 1000, 2000, 5000],
}

HEADER = """/control/verbose 0
/geometry/source brain.tg
/score/create/boxMesh prionScorer
/score/mesh/boxSize 5 5 5 cm
/score/mesh/nBin 50 50 50
/score/quantity/energyDeposit eDep
/score/close
/run/initialize"""

TEMPLATES = {
    "G": """/gps/particle gamma
/gps/energy 1.17 MeV
/gps/pos/type Volume
/gps/pos/shape Sphere
/gps/pos/radius 15 mm
/gps/pos/centre 0 0 0 mm
/run/beamOn {n}""",
    "N": """/gps/particle neutron
/gps/energy 2.0 MeV
/gps/pos/type Volume
/gps/pos/shape Sphere
/gps/pos/radius 15 mm
/gps/pos/centre 0 0 0 mm
/run/beamOn {n}""",
    "C": """/gps/particle ion
/gps/ion 6 12
/gps/energy 3960 MeV
/gps/pos/type Volume
/gps/pos/shape Sphere
/gps/pos/radius 15 mm
/gps/pos/centre 0 0 0 mm
/run/beamOn {n}""",
    "A": """/gps/particle alpha
/gps/energy 5.5 MeV
/gps/pos/type Volume
/gps/pos/shape Sphere
/gps/pos/radius 15 mm
/gps/pos/centre 0 0 0 mm
/run/beamOn {n}""",
}

NAMES = {"G": "Gamma", "N": "Neutron", "C": "Carbon Ion", "A": "Alpha"}
ORDER = ["G", "N", "C", "A"]

def make_mac(counts, csv_name):
    lines = [HEADER, ""]
    for sym in ORDER:
        lines.append(f"# {NAMES[sym]} -- {counts[sym]} particles")
        lines.append(TEMPLATES[sym].format(n=counts[sym]))
        lines.append("")
    lines.append(f"/score/dumpQuantityToFile prionScorer eDep {csv_name}")
    return "\n".join(lines)

def main():
    mac_files = []
    run_lines = []

    print("Phase 1 -- Varying each ray individually:")
    for ray in ["G", "N", "C", "A"]:
        for count in VARIANTS[ray]:
            counts = BASELINE.copy()
            counts[ray] = count
            label    = f"vary_{ray}{count}_G{counts['G']}_N{counts['N']}_C{counts['C']}_A{counts['A']}"
            mac_name = f"opt_{label}.mac"
            csv_name = f"opt_{label}.csv"
            with open(mac_name, "w") as f:
                f.write(make_mac(counts, csv_name))
            mac_files.append({"mac": mac_name, "csv": csv_name,
                "G": counts["G"], "N": counts["N"], "C": counts["C"], "A": counts["A"],
                "varied_ray": NAMES[ray], "varied_count": count, "phase": "single_vary"})
            run_lines.append(f"echo 'Running {mac_name}...'")
            run_lines.append(f"gears {mac_name}")
            print(f"  Vary {NAMES[ray]:10} to {count:5} | G={counts['G']} N={counts['N']} C={counts['C']} A={counts['A']}")

    print("\nPhase 2 -- All equal counts:")
    for n in [500, 1000, 2000, 5000]:
        counts   = {"G": n, "N": n, "C": n, "A": n}
        label    = f"equal_all{n}"
        mac_name = f"opt_{label}.mac"
        csv_name = f"opt_{label}.csv"
        with open(mac_name, "w") as f:
            f.write(make_mac(counts, csv_name))
        mac_files.append({"mac": mac_name, "csv": csv_name,
            "G": n, "N": n, "C": n, "A": n,
            "varied_ray": "All equal", "varied_count": n, "phase": "equal"})
        run_lines.append(f"echo 'Running {mac_name}...'")
        run_lines.append(f"gears {mac_name}")
        print(f"  All equal: {n} particles each")

    print("\nPhase 3 -- Candidate optimal combos:")
    candidates = [
        {"G": 5000, "N": 5000, "C": 1000, "A": 5000, "label": "current_5_5_1_5"},
        {"G": 5000, "N": 5000, "C": 2000, "A": 5000, "label": "test_5_5_2_5"},
        {"G": 5000, "N": 5000, "C": 5000, "A": 5000, "label": "test_5_5_5_5"},
        {"G": 2000, "N": 5000, "C": 1000, "A": 5000, "label": "test_2_5_1_5"},
        {"G": 5000, "N": 2000, "C": 1000, "A": 5000, "label": "test_5_2_1_5"},
        {"G": 5000, "N": 5000, "C": 500,  "A": 5000, "label": "test_5_5_05_5"},
        {"G": 1000, "N": 5000, "C": 500,  "A": 5000, "label": "test_1_5_05_5"},
    ]
    for c in candidates:
        label    = c.pop("label")
        mac_name = f"opt_{label}.mac"
        csv_name = f"opt_{label}.csv"
        with open(mac_name, "w") as f:
            f.write(make_mac(c, csv_name))
        mac_files.append({"mac": mac_name, "csv": csv_name,
            "G": c["G"], "N": c["N"], "C": c["C"], "A": c["A"],
            "varied_ray": "combo", "varied_count": 0, "phase": "candidate"})
        run_lines.append(f"echo 'Running {mac_name}...'")
        run_lines.append(f"gears {mac_name}")
        print(f"  G={c['G']} N={c['N']} C={c['C']} A={c['A']}")

    with open("run_optimize.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"echo 'Running {len(mac_files)} simulations...'\n\n")
        f.write("\n".join(run_lines))
        f.write("\n\necho 'Done! Run: python3 compare_optimize.py'\n")
    print(f"\nCreated {len(mac_files)} mac files")
    print("Created run_optimize.sh")

    registry = "\n".join(
        f'    "{m["csv"]}": {{"G":{m["G"]},"N":{m["N"]},"C":{m["C"]},"A":{m["A"]},"varied_ray":"{m["varied_ray"]}","varied_count":{m["varied_count"]},"phase":"{m["phase"]}"}}, '
        for m in mac_files
    )

    compare = '''import csv, os

GRID_SIZE=50; CENTER=GRID_SIZE//2; CORE_MIN=CENTER-5; CORE_MAX=CENTER+5

FILES = {
''' + registry + '''
}

def load(fp):
    data={}
    if not os.path.exists(fp): return data
    with open(fp) as f:
        for row in csv.reader(f):
            if len(row)<4: continue
            try:
                ix,iy,iz=int(row[0]),int(row[1]),int(row[2]); e=float(row[3])
                if e>0: data[(ix,iy,iz)]=data.get((ix,iy,iz),0)+e
            except: continue
    return data

def classify(ix,iy,iz):
    if CORE_MIN<=ix<=CORE_MAX and CORE_MIN<=iy<=CORE_MAX and CORE_MIN<=iz<=CORE_MAX: return "core"
    if ix<5 or ix>44 or iy<5 or iy>44 or iz<5 or iz>44: return "edge"
    return "surface"

def analyse(data,meta):
    if not data: return None
    total=sum(data.values()); voxels=len(data)
    core_e=sum(e for (ix,iy,iz),e in data.items() if classify(ix,iy,iz)=="core")
    surf_e=sum(e for (ix,iy,iz),e in data.items() if classify(ix,iy,iz)=="surface")
    ratio=surf_e/core_e if core_e>0 else float("inf")
    total_p=meta["G"]+meta["N"]+meta["C"]+meta["A"]
    return {
        "G":meta["G"],"N":meta["N"],"C":meta["C"],"A":meta["A"],
        "phase":meta["phase"],"varied_ray":meta["varied_ray"],
        "total_particles":total_p,"voxels":voxels,
        "total_MeV":round(total,1),"core_MeV":round(core_e,1),
        "surf_core":round(ratio,3),
        "core_per_particle":round(core_e/total_p,4),
        "score":round((core_e/total*50)+(voxels/125000*25)+(1/(ratio+0.01)*25),3),
    }

def main():
    results=[]; missing=[]
    for csv_name,meta in FILES.items():
        data=load(csv_name)
        if data:
            r=analyse(data,meta)
            if r: results.append(r)
        else: missing.append(csv_name)

    if missing:
        print(f"Missing {len(missing)} files -- run bash run_optimize.sh first")
        if not results: return

    results.sort(key=lambda x:x["score"],reverse=True)
    sep="="*100

    print(f"\\n{sep}")
    print("  PHASE 1 -- Varying each ray count individually (others held at baseline)")
    print(sep)
    for ray,fullname in [("G","Gamma"),("N","Neutron"),("C","Carbon Ion"),("A","Alpha")]:
        group=[r for r in results if r["phase"]=="single_vary" and r["varied_ray"]==fullname]
        group.sort(key=lambda x:x["varied_count"])
        print(f"\\n  Varying {fullname}:")
        print(f"  {'Count':<8} {'Core MeV':>11} {'Surf/Core':>10} {'Core/Particle':>14} {'Score':>9}")
        print("  "+"-"*55)
        for r in group:
            print(f"  {r['varied_count']:<8} {r['core_MeV']:>11,.1f} {r['surf_core']:>10.3f} {r['core_per_particle']:>14.4f} {r['score']:>9.3f}")

    print(f"\\n{sep}")
    print("  PHASE 2 -- All equal counts")
    print(sep)
    group=[r for r in results if r["phase"]=="equal"]
    group.sort(key=lambda x:x["G"])
    print(f"  {'Count each':>12} {'Core MeV':>11} {'Surf/Core':>10} {'Voxels':>8} {'Score':>9}")
    print("  "+"-"*55)
    for r in group:
        print(f"  {r['G']:>12} {r['core_MeV']:>11,.1f} {r['surf_core']:>10.3f} {r['voxels']:>8,} {r['score']:>9.3f}")

    print(f"\\n{sep}")
    print("  PHASE 3 -- Candidate combos ranked")
    print(sep)
    group=[r for r in results if r["phase"]=="candidate"]
    group.sort(key=lambda x:x["score"],reverse=True)
    print(f"  {'G':>6} {'N':>6} {'C':>6} {'A':>6} {'Core MeV':>11} {'Surf/Core':>10} {'Score':>9}")
    print("  "+"-"*60)
    for i,r in enumerate(group,1):
        m="  << BEST" if i==1 else ("  << WORST" if i==len(group) else "")
        print(f"  {r['G']:>6} {r['N']:>6} {r['C']:>6} {r['A']:>6} {r['core_MeV']:>11,.1f} {r['surf_core']:>10.3f} {r['score']:>9.3f}{m}")

    print(f"\\n{sep}")
    print("  OVERALL WINNER")
    print(sep)
    best=results[0]
    print(f"  G={best['G']}  N={best['N']}  C={best['C']}  A={best['A']}")
    print(f"  Score:          {best['score']}")
    print(f"  Core energy:    {best['core_MeV']:,.1f} MeV")
    print(f"  Surf/Core:      {best['surf_core']}")
    print(f"  Core/particle:  {best['core_per_particle']} MeV per particle fired")

    with open("optimize_results.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)
    print(f"\\n  Saved: optimize_results.csv")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(14,10))
        fig.suptitle("Particle Count Optimization\\nFiring order: Gamma -> Neutron -> Carbon Ion -> Alpha",fontsize=12,fontweight="bold")
        colors={"Gamma":"#4C72B0","Neutron":"#55A868","Carbon Ion":"#C44E52","Alpha":"#8172B2"}
        for ax,(ray,fullname) in zip(axes.flat,[("G","Gamma"),("N","Neutron"),("C","Carbon Ion"),("A","Alpha")]):
            group=[r for r in results if r["phase"]=="single_vary" and r["varied_ray"]==fullname]
            group.sort(key=lambda x:x["varied_count"])
            if not group: continue
            counts=[r["varied_count"] for r in group]
            scores=[r["score"] for r in group]
            core_e=[r["core_MeV"] for r in group]
            ax2=ax.twinx()
            ax.plot(counts,scores,"o-",color=colors[fullname],linewidth=2,label="Score")
            ax2.plot(counts,core_e,"s--",color=colors[fullname],linewidth=1.5,alpha=0.6,label="Core MeV")
            ax.set_xlabel("Particle Count"); ax.set_ylabel("Score"); ax2.set_ylabel("Core MeV",alpha=0.7)
            ax.set_title(f"Varying {fullname}",fontweight="bold"); ax.set_xticks(counts); ax.grid(True,alpha=0.3)
        plt.tight_layout()
        plt.savefig("optimize_plot.png",dpi=150,bbox_inches="tight")
        print("  Saved: optimize_plot.png")
        plt.close()
    except ImportError:
        print("  pip install matplotlib --break-system-packages")

if __name__=="__main__":
    main()
'''

    with open("compare_optimize.py", "w") as f:
        f.write(compare)
    print("Created compare_optimize.py")
    print(f"\nTotal simulations: {len(mac_files)}")
    print("Next step: bash run_optimize.sh")

if __name__ == "__main__":
    main()
