import csv, os

GRID_SIZE=50; CENTER=GRID_SIZE//2; CORE_MIN=CENTER-5; CORE_MAX=CENTER+5

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

def parse_meta(name):
    name = name.replace("opt_","").replace(".csv","")
    meta = {"varied_ray":"unknown","varied_count":0,"phase":"unknown","G":0,"N":0,"C":0,"A":0}
    if name.startswith("equal_all"):
        n=int(name.replace("equal_all",""))
        meta.update({"G":n,"N":n,"C":n,"A":n,"varied_ray":"All equal","varied_count":n,"phase":"equal"})
    elif name.startswith("vary_"):
        # vary_G500_G500_N2000_C1000_A2000
        parts=name.split("_")
        ray_sym=parts[1][0]
        ray_count=int(parts[1][1:])
        names={"G":"Gamma","N":"Neutron","C":"Carbon Ion","A":"Alpha"}
        for part in parts[2:]:
            if part and part[0] in "GNCA" and part[1:].isdigit():
                meta[part[0]]=int(part[1:])
        meta["varied_ray"]=names.get(ray_sym,"?")
        meta["varied_count"]=ray_count
        meta[ray_sym]=ray_count
        meta["phase"]="single_vary"
    else:
        parts=name.split("_")
        nums=[p for p in parts if p.isdigit()]
        if len(nums)>=4:
            meta.update({"G":int(nums[0]),"N":int(nums[1]),"C":int(nums[2]),"A":int(nums[3]),
                         "phase":"candidate","varied_ray":"combo","varied_count":0})
    return meta

def analyse(data,meta):
    if not data: return None
    total=sum(data.values()); voxels=len(data)
    core_e=sum(e for (ix,iy,iz),e in data.items() if classify(ix,iy,iz)=="core")
    surf_e=sum(e for (ix,iy,iz),e in data.items() if classify(ix,iy,iz)=="surface")
    ratio=surf_e/core_e if core_e>0 else float("inf")
    total_p=meta["G"]+meta["N"]+meta["C"]+meta["A"]
    if total_p==0: total_p=1
    return {
        "G":meta["G"],"N":meta["N"],"C":meta["C"],"A":meta["A"],
        "phase":meta["phase"],"varied_ray":meta["varied_ray"],
        "varied_count":int(meta["varied_count"]),
        "total_particles":total_p,"voxels":voxels,
        "total_MeV":round(total,1),"core_MeV":round(core_e,1),
        "surf_core":round(ratio,3),
        "core_per_particle":round(core_e/total_p,4),
        "score":round((core_e/total*50)+(voxels/125000*25)+(1/(ratio+0.01)*25),3),
    }

def main():
    csv_files=[f for f in os.listdir(".") if f.startswith("opt_") and f.endswith(".csv")]
    if not csv_files:
        print("No opt_*.csv files found. Run bash scripts/run_optimize.sh first."); return

    results=[]; missing=[]
    for csv_name in csv_files:
        data=load(csv_name)
        if data:
            meta=parse_meta(csv_name)
            r=analyse(data,meta)
            if r: results.append(r)
        else: missing.append(csv_name)

    if missing: print(f"Missing/empty: {len(missing)} files")
    if not results: print("No results."); return

    results.sort(key=lambda x:x["score"],reverse=True)
    sep="="*100

    print(f"\n{sep}")
    print("  PHASE 1 -- Varying each ray count individually")
    print(sep)
    for fullname in ["Gamma","Neutron","Carbon Ion","Alpha"]:
        group=[r for r in results if r["phase"]=="single_vary" and r["varied_ray"]==fullname]
        if not group: continue
        group.sort(key=lambda x:x["varied_count"])
        print(f"\n  Varying {fullname}:")
        print(f"  {'Count':<8} {'Core MeV':>11} {'Surf/Core':>10} {'Core/Particle':>14} {'Score':>9}")
        print("  "+"-"*55)
        for r in group:
            print(f"  {r['varied_count']:<8} {r['core_MeV']:>11,.1f} {r['surf_core']:>10.3f} {r['core_per_particle']:>14.4f} {r['score']:>9.3f}")

    print(f"\n{sep}")
    print("  PHASE 2 -- All equal counts")
    print(sep)
    group=[r for r in results if r["phase"]=="equal"]
    group.sort(key=lambda x:x["G"])
    print(f"  {'Count each':>12} {'Core MeV':>11} {'Surf/Core':>10} {'Voxels':>8} {'Score':>9}")
    print("  "+"-"*55)
    for r in group:
        print(f"  {r['G']:>12} {r['core_MeV']:>11,.1f} {r['surf_core']:>10.3f} {r['voxels']:>8,} {r['score']:>9.3f}")

    print(f"\n{sep}")
    print("  PHASE 3 -- Candidate combos ranked")
    print(sep)
    group=[r for r in results if r["phase"]=="candidate"]
    group.sort(key=lambda x:x["score"],reverse=True)
    print(f"  {'G':>6} {'N':>6} {'C':>6} {'A':>6} {'Core MeV':>11} {'Surf/Core':>10} {'Score':>9}")
    print("  "+"-"*60)
    for i,r in enumerate(group,1):
        m="  << BEST" if i==1 else ("  << WORST" if i==len(group) else "")
        print(f"  {r['G']:>6} {r['N']:>6} {r['C']:>6} {r['A']:>6} {r['core_MeV']:>11,.1f} {r['surf_core']:>10.3f} {r['score']:>9.3f}{m}")

    print(f"\n{sep}")
    print("  OVERALL WINNER")
    print(sep)
    best=results[0]
    print(f"  G={best['G']}  N={best['N']}  C={best['C']}  A={best['A']}")
    print(f"  Score:         {best['score']}")
    print(f"  Core energy:   {best['core_MeV']:,.1f} MeV")
    print(f"  Surf/Core:     {best['surf_core']}")
    print(f"  Core/particle: {best['core_per_particle']} MeV per particle fired")

    with open("data/optimize_results.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)
    print(f"\n  Saved: data/optimize_results.csv")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(14,10))
        fig.suptitle("Particle Count Optimization\nFiring order: Gamma -> Neutron -> Carbon Ion -> Alpha",fontsize=12,fontweight="bold")
        colors={"Gamma":"#4C72B0","Neutron":"#55A868","Carbon Ion":"#C44E52","Alpha":"#8172B2"}
        for ax,fullname in zip(axes.flat,["Gamma","Neutron","Carbon Ion","Alpha"]):
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
        plt.savefig("plots/optimize_plot.png",dpi=150,bbox_inches="tight")
        print("  Saved: plots/optimize_plot.png")
        plt.close()
    except ImportError:
        print("  pip install matplotlib --break-system-packages")

if __name__=="__main__":
    main()

# ── NEXT STEP ─────────────────────────────────────────────────────────────────
print("")
print("==============================================")
print("  OPTIMIZATION ANALYSIS DONE")
print("")
print("  NOW RUN:")
print("  python3 scripts/generate_final_sim.py")
print("  (creates final simulation with best order")
print("   + best particle counts combined)")
print("==============================================")
