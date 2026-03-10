import csv, os, math, subprocess, time, json, random, re

LOG_FILE   = "simulation_progress_log.json"
STABLE_WIN = 8
STABLE_THR = 1.0
GRID=50; CENTER=GRID//2; CMIN=CENTER-5; CMAX=CENTER+5

COARSE_COUNTS = [100, 300, 500, 1000, 2000, 5000, 10000]

DOSE_LIMITS_GY = {
    "Gamma":24.0,"Neutron":14.0,"Carbon Ion":20.0,"Alpha":20.0
}
MEV_TO_JOULE=1.60218e-13
CORE_MASS_KG=1.0e-6*1350

RAYS = {
    "Gamma":      {"mac":"macs/tests/test_gamma.mac",   "csv":"gamma_edep.csv",   "min":10,"max":40},
    "Neutron":    {"mac":"macs/tests/test_neutron.mac",  "csv":"neutron_edep.csv", "min":15,"max":60},
    "Carbon Ion": {"mac":"macs/tests/test_carbon.mac",   "csv":"carbon_edep.csv",  "min":10,"max":30},
    "Alpha":      {"mac":"macs/tests/test_alpha.mac",    "csv":"alpha_edep.csv",   "min":8, "max":25},
}

def load_csv(fp):
    data={}
    if not os.path.exists(fp): return data
    with open(fp) as f:
        for row in csv.reader(f):
            if len(row)<4: continue
            try:
                ix,iy,iz=int(row[0]),int(row[1]),int(row[2]); e=float(row[3])
                if e>0: data[(ix,iy,iz)]=data.get((ix,iy,iz),0)+e
            except ValueError: continue
    return data

def classify(ix,iy,iz):
    if CMIN<=ix<=CMAX and CMIN<=iy<=CMAX and CMIN<=iz<=CMAX: return "core"
    if ix<5 or ix>44 or iy<5 or iy>44 or iz<5 or iz>44: return "edge"
    return "surface"

def analyse(data):
    if not data: return None
    total=sum(data.values())
    core_e=sum(e for (ix,iy,iz),e in data.items() if classify(ix,iy,iz)=="core")
    surf_e=sum(e for (ix,iy,iz),e in data.items() if classify(ix,iy,iz)=="surface")
    mean_z=sum(iz*e for (ix,iy,iz),e in data.items())/total if total>0 else 0
    return {"voxels":len(data),"total_MeV":total,"core_MeV":core_e,"surf_MeV":surf_e,
            "surf_core":surf_e/core_e if core_e>0 else float("inf"),"mean_depth":mean_z}

def mean_std(vals):
    n=len(vals)
    if n==0: return 0.0,0.0
    m=sum(vals)/n; s=math.sqrt(sum((v-m)**2 for v in vals)/(n-1)) if n>1 else 0.0
    return m,s

def cv_pct(vals):
    m,s=mean_std(vals); return s/m*100 if m>0 else 0.0

def fmt_time(sec):
    sec=int(sec)
    if sec<60: return f"{sec}s"
    if sec<3600: return f"{sec//60}m {sec%60}s"
    return f"{sec//3600}h {(sec%3600)//60}m {sec%60}s"

def get_beamon(mac_path):
    with open(mac_path) as f:
        for line in f:
            m=re.match(r'\s*/run/beamOn\s+(\d+)',line)
            if m: return int(m.group(1))
    return None

def set_beamon_content(mac_path,count):
    with open(mac_path) as f: content=f.read()
    new=re.sub(r'(/run/beamOn\s+)\d+',rf'\g<1>{count}',content)
    if '/run/beamOn' not in new: new+=f'\n/run/beamOn {count}\n'
    return new

def run_sim(mac_content,csv_path,ray_name):
    s1=random.randint(1,999999999); s2=random.randint(1,999999999)
    tmp=f"_tmp2_{ray_name.replace(' ','_')}.mac"
    with open(tmp,"w") as f: f.write(f"/random/setSeeds {s1} {s2}\n"); f.write(mac_content)
    subprocess.run(["gears",tmp],capture_output=True)
    os.remove(tmp)
    return load_csv(csv_path)

def run_adaptive(mac_path,csv_path,ray_name,count,min_r,max_r):
    mc=set_beamon_content(mac_path,count)
    runs=[]; cvh=[]; sc=0
    for _ in range(max_r):
        r=analyse(run_sim(mc,csv_path,ray_name))
        if r: runs.append(r)
        if len(runs)>=min_r:
            cvh.append(cv_pct([x["core_MeV"] for x in runs]))
            if len(cvh)>=STABLE_WIN:
                if max(cvh[-STABLE_WIN:])-min(cvh[-STABLE_WIN:])<STABLE_THR:
                    sc+=1
                    if sc>=3: break
                else: sc=0
    if not runs: return None,0
    st={}
    for k in ["voxels","total_MeV","core_MeV","surf_MeV","surf_core","mean_depth"]:
        m,s=mean_std([x[k] for x in runs]); st[k]={"mean":m,"std":s}
    n=len(runs)
    st["n_runs"]=n; st["final_cv_pct"]=cv_pct([x["core_MeV"] for x in runs])
    st["count"]=count; st["efficiency"]=st["core_MeV"]["mean"]/count*1000
    return st,n

def fit_quad(xs,ys):
    n=len(xs)
    if n<3: return None
    s1=sum(xs); s2=sum(x**2 for x in xs); s3=sum(x**3 for x in xs); s4=sum(x**4 for x in xs)
    t0=sum(ys); t1=sum(x*y for x,y in zip(xs,ys)); t2=sum(x**2*y for x,y in zip(xs,ys))
    A=[[s4,s3,s2],[s3,s2,s1],[s2,s1,float(n)]]
    b=[t2,t1,t0]
    for col in range(3):
        piv=col+max(range(3-col),key=lambda r:abs(A[col+r][col]))
        A[col],A[piv]=A[piv],A[col]; b[col],b[piv]=b[piv],b[col]
        if abs(A[col][col])<1e-20: return None
        for row in range(col+1,3):
            f=A[row][col]/A[col][col]
            for k in range(col,3): A[row][k]-=f*A[col][k]
            b[row]-=f*b[col]
    x=[0.0]*3
    for i in range(2,-1,-1):
        x[i]=b[i]
        for j in range(i+1,3): x[i]-=A[i][j]*x[j]
        x[i]/=A[i][i]
    return x[0],x[1],x[2]

def interp_peak(counts,effs):
    r=fit_quad([float(c) for c in counts],effs)
    if not r: return None
    a,b,c=r
    if a>=0: return None
    peak=-b/(2*a)
    if peak<min(counts) or peak>max(counts): return None
    return max(1,int(round(peak)))

def dose_gy(core_mev):
    return core_mev*MEV_TO_JOULE/CORE_MASS_KG

def write_beamon(mac_path,count):
    with open(mac_path) as f: content=f.read()
    new=re.sub(r'(/run/beamOn\s+)\d+',rf'\g<1>{count}',content)
    if '/run/beamOn' not in new: new+=f'\n/run/beamOn {count}\n'
    with open(mac_path,"w") as f: f.write(new)

def main():
    t0=time.time(); sep="="*95
    print(f"\n{sep}")
    print("  STEP 2 — Particle Count Optimization with Quadratic Interpolation")
    print("  Phase 1: Coarse grid (7 preset counts, adaptive runs each)")
    print("  Phase 2: Quadratic interpolation — find efficiency PEAK between points")
    print("  Phase 3: Verify interpolated optimum with fresh adaptive runs")
    print("  Phase 4: Safety check vs worst-case single-fraction brain dose limits")
    print(sep)

    print("\n  Pre-flight:")
    ok=True
    for ray,cfg in RAYS.items():
        ex="OK" if os.path.exists(cfg["mac"]) else "MISSING"
        if ex=="MISSING": ok=False
        print(f"    {ray:<12} mac={ex}  beamOn={get_beamon(cfg['mac'])}")
    if not os.path.exists("data/brain/brain.tg"): print("  ERROR: brain.tg missing"); return
    if not os.path.exists(LOG_FILE): print("  ERROR: run Step 1 first"); return
    if not ok: print("  ERROR: missing macs"); return
    print(f"  Coarse counts: {COARSE_COUNTS}")

    with open(LOG_FILE) as f: log=json.load(f)
    if "step1" not in log: print("  ERROR: Step 1 missing from log"); return

    log["step2"]={"results":{},"optimal":{},"timing":{},"interpolation":{}}
    all_opt={}; grand=0

    for ray,cfg in RAYS.items():
        rs=time.time()
        print(f"\n{sep}\n  [{ray}]  PHASE 1 — coarse grid\n{sep}")
        cr={}

        for count in COARSE_COUNTS:
            ct=time.time()
            print(f"    beamOn={count:>6} ...",end="",flush=True)
            st,nr=run_adaptive(cfg["mac"],cfg["csv"],ray,count,cfg["min"],cfg["max"])
            grand+=nr
            if st is None: print("  FAILED"); continue
            print(f"  n={nr:>3}  core={st['core_MeV']['mean']:>12,.2f} MeV  "
                  f"eff={st['efficiency']:.5f}  CV={st['final_cv_pct']:.3f}%  ({fmt_time(time.time()-ct)})")
            cr[count]=st

        note=""; best=None
        print(f"\n  PHASE 2 — Quadratic interpolation")
        if len(cr)<3:
            print(f"    Too few points — using grid best")
            best=max(cr,key=lambda c:cr[c]["efficiency"]); note="too few for interpolation"
        else:
            xs=sorted(cr.keys()); ys=[cr[c]["efficiency"] for c in xs]
            print(f"    Efficiency at coarse points:")
            for c,e in zip(xs,ys): print(f"      {c:>6} particles → eff={e:.5f} MeV/1kP")
            pk=interp_peak(xs,ys)
            if pk is None:
                print(f"    No interior maximum — efficiency monotone increasing.")
                print(f"    More particles always helps in this range. Using highest count.")
                best=xs[-1]; note="monotone — used highest tested count"
            else:
                pkr=max(50,int(round(pk/50)*50))
                print(f"    Raw interpolated peak: {pk:.1f}  →  rounded: {pkr} particles")

                print(f"\n  PHASE 3 — Verify at {pkr} particles")
                if pkr in cr:
                    print(f"    Already in coarse grid — using cached result.")
                    vs=cr[pkr]; nvr=vs["n_runs"]
                else:
                    ct=time.time()
                    print(f"    Running beamOn={pkr} ...",end="",flush=True)
                    vs,nvr=run_adaptive(cfg["mac"],cfg["csv"],ray,pkr,cfg["min"],cfg["max"])
                    grand+=nvr
                    if vs is None:
                        print("  FAILED"); best=max(cr,key=lambda c:cr[c]["efficiency"]); note="verify failed"
                    else:
                        print(f"  n={nvr:>3}  core={vs['core_MeV']['mean']:>12,.2f} MeV  "
                              f"eff={vs['efficiency']:.5f}  ({fmt_time(time.time()-ct)})")
                    cr[pkr]=vs

                if best is None:
                    gb=max(cr,key=lambda c:cr[c]["efficiency"])
                    if cr[pkr]["efficiency"]>=cr[gb]["efficiency"]:
                        best=pkr; note=f"interpolated peak {pkr} BETTER than grid best {gb}"
                    else:
                        best=gb; note=f"interpolated peak {pkr} tested; grid best {gb} won"
                    print(f"    → {note}")

        log["step2"]["interpolation"][ray]={"note":note,"best_count":best}

        print(f"\n  PHASE 4 — Safety check")
        bst=cr[best]; cmev=bst["core_MeV"]["mean"]
        dgy=dose_gy(cmev); lim=DOSE_LIMITS_GY[ray]; safe=dgy<=lim
        print(f"    Optimal count:  {best} particles")
        print(f"    Core MeV:       {cmev:,.4f} MeV")
        print(f"    Approx dose:    {dgy:.4e} Gy")
        print(f"    Brain limit:    {lim} Gy  ({ray})")
        print(f"    Status:         {'SAFE' if safe else '*** EXCEEDS BRAIN LIMIT ***'}")

        print(f"\n  {'Count':>8}  {'Core MeV':>16}  {'Eff MeV/1kP':>13}  {'Dose Gy':>13}  {'CV%':>8}  {'N':>4}")
        print("  "+"-"*72)
        for c in sorted(cr.keys()):
            r=cr[c]; d=dose_gy(r["core_MeV"]["mean"])
            fl=" <- OPTIMAL" if c==best else ("  UNSAFE" if d>lim else "")
            print(f"  {c:>8}  {r['core_MeV']['mean']:>16,.4f}  {r['efficiency']:>13.5f}  "
                  f"{d:>13.4e}  {r['final_cv_pct']:>8.4f}%  {r['n_runs']:>4}{fl}")

        rt=time.time()-rs
        all_opt[ray]={"optimal_count":best,"stats":bst,"dose_gy":dgy,"safety_ok":safe}
        log["step2"]["results"][ray]={str(c):s for c,s in cr.items()}
        log["step2"]["optimal"][ray]={"optimal_count":best,"stats":bst,"dose_gy":dgy,"safety_ok":safe}
        log["step2"]["timing"][ray]=round(rt,2)
        print(f"\n  [{ray}] complete — {fmt_time(rt)}")

    print(f"\n{sep}\n  UPDATING MAC FILES\n{sep}")
    for ray,opt in all_opt.items():
        mac=RAYS[ray]["mac"]; old=get_beamon(mac); new=opt["optimal_count"]
        write_beamon(mac,new)
        print(f"  {ray:<12}  {old} → {new}  dose≈{opt['dose_gy']:.3e} Gy  "
              f"{'SAFE' if opt['safety_ok'] else 'UNSAFE'}")

    tt=time.time()-t0
    log["step2"]["total_time_sec"]=round(tt,2); log["step2"]["total_runs"]=grand
    with open(LOG_FILE,"w") as f: json.dump(log,f,indent=2)

    print(f"\n{sep}\n  STEP 2 SUMMARY\n{sep}")
    print(f"  {'Ray':<12}  {'Count':>7}  {'Core MeV':>14}  {'Eff MeV/1kP':>13}  "
          f"{'Dose Gy':>13}  {'Limit':>7}  {'OK':>4}")
    print("  "+"-"*78)
    for ray,opt in all_opt.items():
        s=opt["stats"]; c=opt["optimal_count"]
        print(f"  {ray:<12}  {c:>7}  {s['core_MeV']['mean']:>14,.4f}  "
              f"{s['efficiency']:>13.5f}  {opt['dose_gy']:>13.4e}  "
              f"{DOSE_LIMITS_GY[ray]:>7.1f}  {'YES' if opt['safety_ok'] else 'NO':>4}")

    print(f"\n  Total time: {fmt_time(tt)}  ({grand} runs)")
    print(f"  Saved: {LOG_FILE}")
    print(f"\n{sep}\n  STEP 2 COMPLETE — NOW RUN:\n  python3 scripts/step3_rerun_optimal.py\n{sep}\n")

if __name__=="__main__": main()
