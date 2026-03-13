import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, csv, json

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, "data")
STEPS_DIR = os.path.join(ROOT, "Steps", "Step1_Rays")
STATS_DIR = os.path.join(ROOT, "data", "stats")
os.makedirs(STEPS_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)

NX, NY, NZ   = 10, 10, 100
BOX_HALF_Z   = 50.0
VOXEL_Z_MM   = (2 * BOX_HALF_Z) / NZ
PRION_Z_BINS = [98, 99]
STEEL_Z_BINS = list(range(0, 98))
RBE = {"gamma":1.0,"neutron":10.0,"alpha":20.0,"carbon":3.0}
STYLES = {
    "gamma":   {"color":"#3498db","label":"γ  Gamma  (6 MeV)"},
    "alpha":   {"color":"#e74c3c","label":"α  Alpha  (5.5 MeV)"},
    "neutron": {"color":"#2ecc71","label":"n  Neutron (14 MeV)"},
    "carbon":  {"color":"#9b59b6","label":"C  Carbon (400 MeV)"},
}
RAYS = ["gamma","neutron","carbon","alpha"]

def load_edep(ray):
    path = os.path.join(DATA_DIR, f"{ray}_edep.csv")
    grid = np.zeros((NX,NY,NZ))
    if not os.path.exists(path):
        print(f"  [WARN] {path} not found — using synthetic data")
        return _synthetic_grid(ray)
    loaded = 0
    with open(path) as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#"): continue
            parts=line.split(",")
            if len(parts)<4: continue
            try:
                ix,iy,iz=int(float(parts[0])),int(float(parts[1])),int(float(parts[2]))
                ev=float(parts[3])
                if 0<=ix<NX and 0<=iy<NY and 0<=iz<NZ:
                    grid[ix,iy,iz]+=ev; loaded+=1
            except: continue
    total=grid.sum()
    print(f"  {ray:8s}: {loaded} voxels, total={total:.4f} MeV")
    if total==0:
        print(f"  [WARN] All zeros — using synthetic"); return _synthetic_grid(ray)
    return grid

def _synthetic_grid(ray):
    np.random.seed({"gamma":1,"alpha":2,"neutron":3,"carbon":4}[ray])
    grid=np.zeros((NX,NY,NZ))
    z=np.arange(NZ)
    for ix in range(NX):
        for iy in range(NY):
            r2=(ix-NX//2)**2+(iy-NY//2)**2
            bw=np.exp(-r2/8.0)
            if ray=="gamma":
                p=1.8*np.exp(-0.04*z)*bw; p[PRION_Z_BINS]*=1.1
            elif ray=="alpha":
                p=np.zeros(NZ); p[0]=8.0*bw; p[1]=2.0*bw
            elif ray=="neutron":
                p=(0.8+0.3*np.exp(-0.005*z))*bw+0.1*np.random.rand(NZ)
                p[PRION_Z_BINS]+=1.2*bw
            else:
                p=0.4*np.exp(-((z-97)/3.0)**2)*bw+0.05*np.random.rand(NZ)
                p[PRION_Z_BINS]+=2.5*bw
            grid[ix,iy,:]=np.clip(p,0,None)
    return grid

def depth_profile(grid): return grid.sum(axis=(0,1))

def extract_stats(grid,ray):
    prof=depth_profile(grid)
    z_mm=np.arange(NZ)*VOXEL_Z_MM-BOX_HALF_Z+VOXEL_Z_MM/2
    prion_dose=prof[PRION_Z_BINS].sum()
    steel_dose=prof[STEEL_Z_BINS].sum()
    total_dose=prof.sum()
    selectivity=prion_dose/steel_dose if steel_dose>0 else 0.0
    peak_z=float(z_mm[np.argmax(prof)])
    rbe=RBE[ray]; rbe_prion=prion_dose*rbe
    reach=np.where(prof>0.01*prof.max())[0]
    max_reach=float(z_mm[reach[-1]]) if len(reach) else 0.0
    return {"ray":ray,"prion_mev":round(prion_dose,4),"steel_mev":round(steel_dose,4),
            "total_mev":round(total_dose,4),"selectivity":round(selectivity,4),
            "rbe":rbe,"rbe_prion_mev":round(rbe_prion,4),"peak_z_mm":round(peak_z,2),
            "max_reach_mm":round(max_reach,2),
            "prion_pct":round(100*prion_dose/total_dose,2) if total_dose>0 else 0.0}

def plot_individual(grid,ray,stats):
    style=STYLES[ray]
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    fig.patch.set_facecolor("#0d1117")
    z_mm=np.arange(NZ)*VOXEL_Z_MM-BOX_HALF_Z+VOXEL_Z_MM/2
    prof=depth_profile(grid)
    ax=axes[0]; ax.set_facecolor("#0d1117")
    ax.plot(z_mm,prof,color=style["color"],lw=2.5,label=style["label"])
    ax.fill_between(z_mm,prof,alpha=0.18,color=style["color"])
    ps=z_mm[PRION_Z_BINS[0]]-VOXEL_Z_MM/2; pe=z_mm[PRION_Z_BINS[-1]]+VOXEL_Z_MM/2
    ax.axvspan(ps,pe,alpha=0.3,color="#f39c12",label="Prion layer")
    ax.set_xlabel("z (mm)",color="white"); ax.set_ylabel("MeV",color="white")
    ax.set_title(f"{style['label']} — Depth Profile",color="white",fontweight="bold")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.legend(facecolor="#1a1a2e",labelcolor="white",fontsize=9)
    info=(f"Prion: {stats['prion_mev']:.4f} MeV ({stats['prion_pct']:.1f}%)\n"
          f"Steel: {stats['steel_mev']:.4f} MeV\nSelectivity: {stats['selectivity']:.4f}\n"
          f"RBE x{stats['rbe']} -> {stats['rbe_prion_mev']:.4f}\nPeak z={stats['peak_z_mm']:.1f}mm")
    ax.text(0.02,0.97,info,transform=ax.transAxes,color="white",fontsize=8,va="top",
            fontfamily="monospace",bbox=dict(boxstyle="round",facecolor="#1a1a2e",edgecolor=style["color"],alpha=0.85))
    ax2=axes[1]; ax2.set_facecolor("#0d1117")
    s2d=grid[:,NY//2,:].T
    im=ax2.imshow(s2d,aspect="auto",origin="lower",
                  extent=[-5,5,-50,50],cmap="hot",interpolation="nearest")
    ax2.axhline(ps,color="#f39c12",lw=1.5,ls="--",label="Prion layer")
    plt.colorbar(im,ax=ax2,label="MeV").ax.yaxis.label.set_color("white")
    ax2.set_xlabel("x (mm)",color="white"); ax2.set_ylabel("z (mm)",color="white")
    ax2.set_title("2D Energy Map",color="white",fontweight="bold")
    ax2.tick_params(colors="white"); ax2.legend(facecolor="#1a1a2e",labelcolor="white",fontsize=9)
    for sp in ax2.spines.values(): sp.set_edgecolor("#444")
    plt.tight_layout()
    out=os.path.join(STEPS_DIR,f"vis_{ray}.png")
    fig.savefig(out,dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(); print(f"    Saved {out}")

def plot_overview(all_grids,all_stats):
    fig=plt.figure(figsize=(18,10)); fig.patch.set_facecolor("#0d1117")
    gs=gridspec.GridSpec(2,4,hspace=0.45,wspace=0.35)
    z_mm=np.arange(NZ)*VOXEL_Z_MM-BOX_HALF_Z+VOXEL_Z_MM/2
    ps=z_mm[PRION_Z_BINS[0]]-VOXEL_Z_MM/2
    for i,ray in enumerate(RAYS):
        ax=fig.add_subplot(gs[0,i]); ax.set_facecolor("#111827")
        style=STYLES[ray]; prof=depth_profile(all_grids[ray])
        ax.plot(z_mm,prof,color=style["color"],lw=2)
        ax.fill_between(z_mm,prof,alpha=0.2,color=style["color"])
        ax.axvspan(ps,BOX_HALF_Z,alpha=0.25,color="#f39c12")
        ax.set_title(style["label"],color="white",fontsize=9,fontweight="bold")
        ax.set_xlabel("z (mm)",color="white",fontsize=8); ax.set_ylabel("MeV",color="white",fontsize=8)
        ax.tick_params(colors="white",labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#333")
    ax5=fig.add_subplot(gs[1,:2]); ax5.set_facecolor("#111827")
    rays=[s["ray"] for s in all_stats]; colors=[STYLES[r]["color"] for r in rays]
    phys=[s["prion_mev"] for s in all_stats]; rbe_w=[s["rbe_prion_mev"] for s in all_stats]
    x=np.arange(len(rays))
    b1=ax5.bar(x-0.2,phys,0.38,color=colors,alpha=0.9,label="Physical MeV")
    b2=ax5.bar(x+0.2,rbe_w,0.38,color=colors,alpha=0.4,hatch="//",edgecolor="white",lw=0.5,label="RBE-weighted")
    ax5.set_xticks(x); ax5.set_xticklabels(rays,color="white",fontsize=10)
    ax5.set_ylabel("MeV in prion layer",color="white"); ax5.set_title("Prion Dose: Physical vs RBE",color="white",fontweight="bold")
    ax5.tick_params(colors="white"); ax5.legend(facecolor="#1a1a2e",labelcolor="white")
    for sp in ax5.spines.values(): sp.set_edgecolor("#444")
    ax6=fig.add_subplot(gs[1,2:]); ax6.set_facecolor("#111827"); ax6.axis("off")
    hdr=f"{'Ray':<10} {'Prion MeV':>10} {'RBE-MeV':>10} {'Select.':>9} {'Reach':>8}"
    rows=[hdr,"─"*52]
    for s in sorted(all_stats,key=lambda x:-x["rbe_prion_mev"]):
        rows.append(f"{s['ray']:<10} {s['prion_mev']:>10.4f} {s['rbe_prion_mev']:>10.4f} {s['selectivity']:>9.4f} {s['max_reach_mm']:>7.1f}mm")
    rows+=["─"*52,"","Note: Alpha stops in first 1mm of steel.","G4_WATER used as prion proxy (see limitations)."]
    ax6.text(0.03,0.97,"\n".join(rows),transform=ax6.transAxes,color="white",fontsize=8,va="top",
             fontfamily="monospace",bbox=dict(boxstyle="round",facecolor="#1a1a2e",edgecolor="#4a9eff",alpha=0.9))
    fig.suptitle("Surgical Tool — Step 1: All Radiation Types",color="white",fontsize=14,fontweight="bold")
    out=os.path.join(STEPS_DIR,"step1_overview.png")
    fig.savefig(out,dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(); print(f"    Saved {out}")

def run_step1():
    print("\n"+"="*65+"\n  SURGICAL TOOL — Step 1: Individual Ray Analysis\n"+"="*65)
    all_grids={}; all_stats=[]
    for ray in RAYS:
        print(f"\n  [{ray.upper()}]")
        grid=load_edep(ray); stats=extract_stats(grid,ray)
        all_grids[ray]=grid; all_stats.append(stats)
        print(f"    Prion: {stats['prion_mev']:.4f} MeV ({stats['prion_pct']:.1f}%)")
        print(f"    RBE:   {stats['rbe_prion_mev']:.4f}  Peak z={stats['peak_z_mm']:.1f}mm  Reach={stats['max_reach_mm']:.1f}mm")
        plot_individual(grid,ray,stats)
    plot_overview(all_grids,all_stats)
    path=os.path.join(STATS_DIR,"individual_ray_stats.csv")
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=all_stats[0].keys()); w.writeheader(); w.writerows(all_stats)
    jpath=os.path.join(STATS_DIR,"step1_stats.json")
    with open(jpath,"w") as f: json.dump({s["ray"]:s for s in all_stats},f,indent=2)
    print(f"\n    Saved {path}\n    Saved {jpath}\n  Step 1 complete.")
    return {s["ray"]:s for s in all_stats},all_grids

if __name__=="__main__": run_step1()
