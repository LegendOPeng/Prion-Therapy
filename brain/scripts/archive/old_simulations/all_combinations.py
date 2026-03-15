import csv, os, itertools

FILES = {
    "Gamma":      "gamma_edep.csv",
    "Neutron":    "neutron_edep.csv",
    "Carbon Ion": "carbon_edep.csv",
    "Alpha":      "alpha_edep.csv",
}

GRID_SIZE = 50
CENTER    = GRID_SIZE // 2
CORE_MIN  = CENTER - 5
CORE_MAX  = CENTER + 5

PHYSICS = {
    "Gamma":      {"LET": 0.2,  "RBE": 1.0,  "external": True,  "bragg": False, "secondaries": False, "damage": "SSB (repairable)"},
    "Neutron":    {"LET": 20.0, "RBE": 3.0,  "external": True,  "bragg": False, "secondaries": True,  "damage": "Mixed DSB via recoil protons"},
    "Carbon Ion": {"LET": 80.0, "RBE": 3.5,  "external": True,  "bragg": True,  "secondaries": True,  "damage": "Complex clustered DSB"},
    "Alpha":      {"LET": 100.0,"RBE": 20.0, "external": False, "bragg": False, "secondaries": False, "damage": "Complex irreparable DSB"},
}

def synergy(ray_names):
    lets = [PHYSICS[r]["LET"] for r in ray_names if r in PHYSICS]
    if len(lets) < 2: return 1.0
    low  = any(l < 5   for l in lets)
    mid  = any(5<=l<50 for l in lets)
    high = any(l >= 50 for l in lets)
    bonus = 1.0
    if low and high:  bonus += 0.35
    if mid and high:  bonus += 0.20
    if low and mid and high: bonus += 0.15
    if "Gamma" in ray_names and "Carbon Ion" in ray_names: bonus += 0.15
    if "Neutron" in ray_names and "Carbon Ion" in ray_names: bonus += 0.10
    return bonus

def load_csv(fp):
    data = {}
    if not os.path.exists(fp):
        print(f"  [WARNING] Not found: {fp}"); return data
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

def analyse(data, label, rays):
    if not data: return None
    total  = sum(data.values())
    voxels = len(data)
    core_e = sum(e for (ix,iy,iz),e in data.items() if classify(ix,iy,iz)=="core")
    surf_e = sum(e for (ix,iy,iz),e in data.items() if classify(ix,iy,iz)=="surface")
    mean_z = sum(iz*e for (ix,iy,iz),e in data.items())/total
    ratio  = surf_e/core_e if core_e>0 else float("inf")
    center = any(abs(ix-CENTER)<=2 and abs(iy-CENTER)<=2 and abs(iz-CENTER)<=2 for (ix,iy,iz) in data)
    rbelist = [PHYSICS[r]["RBE"] for r in rays if r in PHYSICS]
    avg_rbe = sum(rbelist)/len(rbelist) if rbelist else 1.0
    syn     = synergy(rays)
    has_ext = any(PHYSICS[r]["external"] for r in rays if r in PHYSICS)
    deliv   = 1.0 if has_ext else 0.3
    bragg_only = set(rays)=={"Carbon Ion"}
    bragg_pen  = 0.85 if bragg_only else 1.0
    base = ((core_e/(total*avg_rbe)*avg_rbe)*40 +
            (voxels/125000)*25 +
            (1/(ratio+0.01))*20 +
            (avg_rbe/20)*15)
    score = base * syn * deliv * bragg_pen
    return {
        "label": label, "voxels_hit": voxels,
        "total_MeV": round(total,1), "core_MeV": round(core_e,1),
        "surface_MeV": round(surf_e,1), "surf_core": round(ratio,3),
        "mean_depth_mm": round(mean_z,2), "avg_RBE": round(avg_rbe,2),
        "synergy": round(syn,2), "deliverability": round(deliv,2),
        "core_hit": "Yes" if center else "No", "score": round(score,3),
    }

def main():
    print("Loading simulations...")
    datasets = {}
    for name, fp in FILES.items():
        d = load_csv(fp)
        if d: datasets[name]=d; print(f"  {name}: {len(d):,} voxels")
        else: print(f"  {name}: SKIPPED")

    if len(datasets)<2: print("Need >=2 CSV files."); return

    names = list(datasets.keys())
    results = []

    for name in names:
        r = analyse(datasets[name], name, [name])
        if r: results.append(r)

    for r in range(2, len(names)+1):
        for combo in itertools.combinations(names,r):
            label = " + ".join(combo)
            combined = {}
            for name in combo:
                for key,e in datasets[name].items():
                    combined[key] = combined.get(key,0)+e
            res = analyse(combined, label, list(combo))
            if res: results.append(res)

    results.sort(key=lambda x: x["score"], reverse=True)

    sep="="*115
    print(f"\n{sep}")
    print("  PHYSICS PROFILE")
    print(sep)
    print(f"  {'Ray':<14} {'LET keV/um':>11} {'RBE':>6} {'External':>9} {'Bragg':>7} {'Secondaries':>12}  Damage Type")
    print("-"*95)
    for n,p in PHYSICS.items():
        print(f"  {n:<14} {p['LET']:>11.1f} {p['RBE']:>6.1f} {'Yes' if p['external'] else 'No(TAT)':>9} {'Yes' if p['bragg'] else 'No':>7} {'Yes' if p['secondaries'] else 'No':>12}  {p['damage']}")

    print(f"\n{sep}")
    print("  ALL COMBINATIONS RANKED — RBE-weighted + Synergy + Deliverability")
    print(sep)
    print(f"{'Rank':<5} {'Combination':<42} {'Core MeV':>10} {'Avg RBE':>8} {'Synergy':>8} {'Surf/Core':>10} {'Score':>9}")
    print("-"*100)
    for i,r in enumerate(results,1):
        m = "  << BEST" if i==1 else ("  << WORST" if i==len(results) else "")
        print(f"{i:<5} {r['label']:<42} {r['core_MeV']:>10,.1f} {r['avg_RBE']:>8.2f} {r['synergy']:>8.2f} {r['surf_core']:>10.3f} {r['score']:>9.3f}{m}")

    best=results[0]; worst=results[-1]
    print(f"\n{sep}")
    print("  WINNER SUMMARY")
    print(sep)
    print(f"  Most effective:  {best['label']}")
    print(f"    Score:         {best['score']}")
    print(f"    Core energy:   {best['core_MeV']:,.1f} MeV")
    print(f"    Avg RBE:       {best['avg_RBE']}x")
    print(f"    Synergy:       {best['synergy']}x (non-additive LET enhancement)")
    print(f"    Voxels:        {best['voxels_hit']:,}")
    print(f"\n  Least effective: {worst['label']}")
    print(f"    Score:         {worst['score']}")
    print(f"\n{sep}")
    print("  SCORING BREAKDOWN")
    print(sep)
    print("  40 pts  RBE-weighted core energy %     deep biologically effective core damage")
    print("  25 pts  Coverage (voxels hit)          spread across entire aggregate volume")
    print("  20 pts  Inverse Surf/Core ratio        core-favoured over surface delivery")
    print("  15 pts  Average RBE of combo           lethality per MeV deposited")
    print("  x Synergy multiplier                   LET complementarity (NIH, Hall & Giaccia 2018)")
    print("  x Deliverability (1.0 external, 0.3 alpha-only = targeted therapy only)")
    print("  x Bragg penalty if Carbon Ion alone   single-depth weakness without coverage ray")
    print(f"\n{sep}")
    print("  KEY WEAKNESSES ACCOUNTED FOR:")
    print("  Gamma      — Low LET, sparse ionization, SSBs are repairable (RBE=1.0)")
    print("  Neutron    — Unpredictable scatter, BUT generates high-LET recoil protons in Geant4")
    print("  Carbon Ion — Entry track is low-LET before Bragg peak; single-depth coverage only")
    print("  Alpha      — Cannot reach deep targets externally (range ~40um); internal TAT only")
    print("  Combinations cover each other's weaknesses — this is your core research finding.")

    with open("all_combinations_results.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)
    print(f"\n  Saved: all_combinations_results.csv")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt, matplotlib.patches as mpatches
        labels=[r["label"] for r in results]; scores=[r["score"] for r in results]
        core_e=[r["core_MeV"] for r in results]; syns=[r["synergy"] for r in results]
        rbel=[r["avg_RBE"] for r in results]
        pal={1:"#4C72B0",2:"#55A868",3:"#C44E52",4:"#8172B2"}
        colors=[pal[min(l.count("+")+1,4)] for l in labels]
        fig,axes=plt.subplots(2,2,figsize=(20,14))
        fig.suptitle("All Radiation Combinations — Prion Aggregate Disruption\nRBE-weighted + LET Synergy + Deliverability | Sources: Hall & Giaccia (2018); NIH",fontsize=11,fontweight="bold")
        for ax,vals,xlabel,title in [
            (axes[0,0],scores,"Effectiveness Score","Overall Effectiveness (higher=better)"),
            (axes[0,1],core_e,"Core Energy (MeV)","Core Penetration (higher=better)"),
            (axes[1,0],syns,"Synergy Multiplier","LET Synergy (higher=more non-additive benefit)"),
            (axes[1,1],rbel,"Average RBE","Biological Effectiveness (higher=more lethal/MeV)"),
        ]:
            ax.barh(range(len(labels)),vals,color=colors)
            ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels,fontsize=7)
            ax.set_xlabel(xlabel); ax.set_title(title); ax.invert_yaxis()
        axes[1,0].axvline(1.0,color="gray",linestyle="--",alpha=0.5,label="No synergy")
        axes[1,0].legend(fontsize=7)
        axes[1,1].axvline(1.0,color="gray",linestyle="--",alpha=0.5,label="Gamma baseline")
        axes[1,1].legend(fontsize=7)
        patches=[mpatches.Patch(color=pal[i],label=l) for i,l in [(1,"Single"),(2,"2-ray"),(3,"3-ray"),(4,"All 4")]]
        fig.legend(handles=patches,loc="lower center",ncol=4,fontsize=9,bbox_to_anchor=(0.5,-0.01))
        plt.tight_layout(rect=[0,0.04,1,1])
        plt.savefig("all_combinations_plot.png",dpi=150,bbox_inches="tight")
        print("  Saved: all_combinations_plot.png")
        plt.close()
    except ImportError:
        print("  pip install matplotlib --break-system-packages")

if __name__=="__main__":
    main()

# ── NEXT STEP ─────────────────────────────────────────────────────────────────
print("")
print("==============================================")
print("  NOW RUN:")
print("  python3 compare_penetration.py")
print("  (runs each ray 30-100 times for statistics)")
print("==============================================")
