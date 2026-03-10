import csv, os, math, subprocess, time

GRID_SIZE=50; CENTER=GRID_SIZE//2; CORE_MIN=CENTER-5; CORE_MAX=CENTER+5
MIN_RUNS=30; MAX_RUNS=100; STABLE_WINDOW=10; STABLE_THRESHOLD=0.5

MAC_FILES = {
    "Gamma":      "tests/test_gamma.mac",
    "Neutron":    "tests/test_neutron.mac",
    "Carbon Ion": "tests/test_carbon.mac",
    "Alpha":      "tests/test_alpha.mac",
}
CSV_FILES = {
    "Gamma":      "gamma_edep.csv",
    "Neutron":    "neutron_edep.csv",
    "Carbon Ion": "carbon_edep.csv",
    "Alpha":      "alpha_edep.csv",
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

def analyse(data):
    if not data: return None
    total=sum(data.values()); voxels=len(data)
    core_e=sum(e for (ix,iy,iz),e in data.items() if classify(ix,iy,iz)=="core")
    surf_e=sum(e for (ix,iy,iz),e in data.items() if classify(ix,iy,iz)=="surface")
    ratio=surf_e/core_e if core_e>0 else 0
    mean_z=sum(iz*e for (ix,iy,iz),e in data.items())/total
    core_hit=any(abs(ix-CENTER)<=2 and abs(iy-CENTER)<=2 and abs(iz-CENTER)<=2 for (ix,iy,iz) in data)
    return {"voxels":voxels,"total_MeV":round(total,4),
            "core_MeV":round(core_e,4),"surf_core":round(ratio,4),
            "mean_depth":round(mean_z,2),"core_hit":core_hit}

def mean_std(vals):
    n=len(vals)
    if n==0: return 0,0
    m=sum(vals)/n
    std=math.sqrt(sum((v-m)**2 for v in vals)/(n-1)) if n>1 else 0
    return m,std

def cv(vals):
    m,s=mean_std(vals)
    return (s/m*100) if m>0 else 0

def fmt_time(seconds):
    seconds=int(seconds)
    if seconds<60: return f"{seconds}s"
    elif seconds<3600: return f"{seconds//60}m {seconds%60}s"
    else: return f"{seconds//3600}h {(seconds%3600)//60}m {seconds%60}s"

def main():
    sep="="*90
    all_results={}
    total_start=time.time()

    # ── Debug: check files exist first ───────────────────────────────────────
    print("\nChecking files...")
    for ray,mac in MAC_FILES.items():
        exists="OK" if os.path.exists(mac) else "MISSING"
        print(f"  {ray:<12} mac={exists}  ({mac})")

    # ── Check where gears actually saves CSVs by running once ────────────────
    print("\nTest run to find where CSVs are saved...")
    subprocess.run(["gears", "tests/test_gamma.mac"], capture_output=True)
    found=[]
    for root,dirs,files in os.walk("."):
        for f in files:
            if f=="gamma_edep.csv":
                found.append(os.path.join(root,f))
    if found:
        print(f"  gamma_edep.csv found at: {found}")
        # Update CSV path to wherever it actually saves
        actual_path=found[0]
        for key in CSV_FILES:
            CSV_FILES[key]=CSV_FILES[key] if os.path.dirname(actual_path)=="." else \
                           os.path.join(os.path.dirname(actual_path), CSV_FILES[key].split("/")[-1])
    else:
        print("  WARNING: gamma_edep.csv not found anywhere after test run!")
        print("  Listing all CSVs in current directory:")
        for f in os.listdir("."):
            if f.endswith(".csv"): print(f"    {f}")
        return

    print(f"  Using CSV paths: {list(CSV_FILES.values())[0]}")
    print()

    for ray, mac in MAC_FILES.items():
        ray_start=time.time()
        print(f"  [{ray}] starting...", flush=True)
        runs=[]; cv_history=[]; stable_count=0; stopped_early=False

        for i in range(1, MAX_RUNS+1):
            subprocess.run(["gears", mac], capture_output=True)
            data=load(CSV_FILES[ray])
            r=analyse(data)
            if r:
                runs.append(r)
            else:
                if i==1:
                    print(f"    WARNING: run 1 returned no data. CSV size: {os.path.getsize(CSV_FILES[ray]) if os.path.exists(CSV_FILES[ray]) else 'FILE NOT FOUND'}")

            if len(runs)>=MIN_RUNS:
                current_cv=cv([r["core_MeV"] for r in runs])
                cv_history.append(current_cv)
                if len(cv_history)>=STABLE_WINDOW:
                    spread=max(cv_history[-STABLE_WINDOW:])-min(cv_history[-STABLE_WINDOW:])
                    if spread<STABLE_THRESHOLD:
                        stable_count+=1
                        if stable_count>=3:
                            stopped_early=True
                            print(f"    Converged at run {i} (CV={current_cv:.3f}%)")
                            break
                    else:
                        stable_count=0

            if i%10==0:
                current_cv=cv([r["core_MeV"] for r in runs]) if runs else 0
                print(f"    Run {i:>3}/{MAX_RUNS}  CV={current_cv:.3f}%  successful={len(runs)}  elapsed={fmt_time(time.time()-ray_start)}", flush=True)

        ray_time=time.time()-ray_start
        n=len(runs)
        if n==0:
            print(f"  [{ray}] ERROR — 0 successful runs. Skipping.\n"); continue

        final_cv=cv([r["core_MeV"] for r in runs])
        status="(early stop)" if stopped_early else "(full 100 runs)"
        print(f"  [{ray}] DONE — {n} runs, CV={final_cv:.3f}%, time={fmt_time(ray_time)} {status}\n")

        stats={}
        for key in ["voxels","total_MeV","core_MeV","surf_core","mean_depth"]:
            vals=[r[key] for r in runs]
            m,s=mean_std(vals)
            stats[key]=(round(m,4),round(s,4))
        stats["core_hit_pct"]=round(sum(1 for r in runs if r["core_hit"])/n*100,1)
        stats["n_runs"]=n
        stats["final_cv"]=round(final_cv,3)
        stats["time"]=round(ray_time,1)
        all_results[ray]=stats

    if not all_results:
        print("No results collected. Check your mac files and CSV output paths."); return

    total_time=time.time()-total_start

    print(f"\n{sep}")
    print("  COMPARISON TABLE — Mean ± Std Dev")
    print(sep)
    print(f"  {'Radiation':<12} {'Voxels Hit':>20} {'Total MeV':>20} {'Core Hit%':>10} {'Surf/Core':>18} {'Mean Depth':>14}")
    print("  "+"-"*88)
    for ray,s in all_results.items():
        vm,vs=s["voxels"]; tm,ts=s["total_MeV"]
        sm,ss=s["surf_core"]; dm,ds=s["mean_depth"]; ch=s["core_hit_pct"]
        print(f"  {ray:<12} {vm:>10,.0f} ± {vs:<7.0f} {tm:>10,.4f} ± {ts:<7.4f} {ch:>9.1f}% {sm:>8.3f} ± {ss:<6.3f} {dm:>6.1f} ± {ds:.2f}mm")

    print(f"\n{sep}")
    print("  CONSISTENCY CHECK")
    print("  < 2% = publication quality   < 5% = excellent   5-10% = acceptable")
    print(sep)
    for ray,s in all_results.items():
        cv_val=s["final_cv"]; n=s["n_runs"]
        if cv_val<2:    status="PUBLICATION QUALITY"
        elif cv_val<5:  status="EXCELLENT"
        elif cv_val<10: status="ACCEPTABLE"
        else:           status="HIGH VARIANCE"
        print(f"  {ray:<12}  CV={cv_val:>6.3f}%   Runs={n:>3}   {status}")

    print(f"\n{sep}")
    print("  TIMING SUMMARY")
    print(sep)
    for ray,s in all_results.items():
        print(f"  {ray:<12}  {s['n_runs']:>3} runs  {fmt_time(s['time'])}")
    print(f"  {'TOTAL':<12}  {fmt_time(total_time)}")

    rows=[]
    for ray,s in all_results.items():
        row={"ray":ray,"runs":s["n_runs"],"final_cv_pct":s["final_cv"],"core_hit_pct":s["core_hit_pct"]}
        for k in ["voxels","total_MeV","core_MeV","surf_core","mean_depth"]:
            row[f"{k}_mean"]=s[k][0]; row[f"{k}_std"]=s[k][1]
        rows.append(row)
    with open("data/individual_ray_stats.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"\n  Saved: data/individual_ray_stats.csv")
    print(f"  Total time: {fmt_time(total_time)}")
    print(sep)

if __name__=="__main__":
    main()

# ── NEXT STEP ─────────────────────────────────────────────────────────────────
print("")
print("==============================================")
print("  NOW RUN:")
print("  bash scripts/run_all_orders.sh")
print("  (runs all 48 firing order simulations)")
print("==============================================")
