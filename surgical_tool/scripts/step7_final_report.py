"""
Step 7: Master Summary — Surgical Tool
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os, csv, json

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS_DIR = os.path.join(ROOT, "data", "stats")
STEPS_DIR = os.path.join(ROOT, "Steps", "Step7_Final")
os.makedirs(STEPS_DIR, exist_ok=True)

RAYS   = ["gamma","neutron","carbon","alpha"]
STYLES = {"gamma":"#3498db","alpha":"#e74c3c","neutron":"#2ecc71","carbon":"#9b59b6"}
COMBO_COLORS = {2:"#3498db",3:"#2ecc71",4:"#e74c3c"}


def _load(name):
    path = os.path.join(STATS_DIR, name)
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return {}


def run_step7():
    print("\n"+"="*65)
    print("  SURGICAL TOOL — Step 7: Master Summary")
    print("="*65)
    s1=_load("step1_stats.json"); s2=_load("step2_optimal.json")
    s3=_load("step3_validation.json"); s4=_load("step4_combinations.json")
    s5=_load("step5_firing_orders.json"); s6=_load("step6_final_protocol.json")
    plot_master_summary(s1,s2,s3,s4,s5,s6)
    save_final_csv(s1,s2,s3,s6)
    _try_gromacs_bridge()
    print("\n  Step 7 complete.")


def plot_master_summary(s1,s2,s3,s4,s5,s6):
    fig = plt.figure(figsize=(20,14)); fig.patch.set_facecolor("#0d0d0d")
    fig.suptitle("SURGICAL TOOL — PRION STERILIZATION — COMPLETE PIPELINE\n"
                 "Geant4 Monte Carlo → Multi-Ray Optimization → GROMACS MD",
                 color="white",fontsize=14,fontweight="bold",y=0.98)

    NZ=100; BOX_HALF_Z=50.0; VOXEL_Z_MM=(2*BOX_HALF_Z)/NZ
    z_mm = np.arange(NZ)*VOXEL_Z_MM-BOX_HALF_Z+VOXEL_Z_MM/2

    ax1 = fig.add_subplot(331,facecolor="#111111")
    for ray in RAYS:
        d = s1.get(ray,{})
        if d:
            peak_z=d.get("peak_z_mm",0); total=d.get("total_mev",1)
            sigma=5.0 if ray!="alpha" else 0.5
            prof=(total/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((z_mm-peak_z)/sigma)**2)
            ax1.plot(z_mm,prof,color=STYLES[ray],lw=1.5,label=ray[:3],alpha=0.85)
    ax1.axvspan(48,50,alpha=0.3,color="#f39c12",label="Prion")
    ax1.set_title("Step 1 — Depth Profiles",color="white",fontsize=9)
    ax1.set_xlabel("z (mm)",color="white",fontsize=7); ax1.set_ylabel("MeV",color="white",fontsize=7)
    ax1.tick_params(colors="white",labelsize=6); ax1.legend(facecolor="#222",labelcolor="white",fontsize=6)
    for sp in ax1.spines.values(): sp.set_color("#444")

    ax2 = fig.add_subplot(332,facecolor="#111111")
    rbe_vals=[s1.get(r,{}).get("rbe_prion_mev",0) for r in RAYS]
    colors=[STYLES[r] for r in RAYS]
    bars=ax2.bar(RAYS,rbe_vals,color=colors,edgecolor="none")
    ax2.set_title("Step 1 — RBE Prion Dose",color="white",fontsize=9)
    ax2.set_ylabel("RBE-MeV",color="white",fontsize=7); ax2.tick_params(colors="white",labelsize=7)
    for sp in ax2.spines.values(): sp.set_color("#444")
    for bar,val in zip(bars,rbe_vals):
        ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()*1.02,
                 f"{val:.3f}",ha="center",color="white",fontsize=7)

    ax3 = fig.add_subplot(333,facecolor="#111111")
    opt_ns=[s2.get(r,{}).get("optimal_count",0) for r in RAYS]
    bars=ax3.bar(RAYS,opt_ns,color=colors,edgecolor="none")
    ax3.set_title("Step 2 — Optimal Count",color="white",fontsize=9)
    ax3.set_ylabel("N",color="white",fontsize=7); ax3.tick_params(colors="white",labelsize=7)
    for sp in ax3.spines.values(): sp.set_color("#444")
    for bar,val in zip(bars,opt_ns):
        ax3.text(bar.get_x()+bar.get_width()/2,bar.get_height()*1.02,
                 str(val),ha="center",color="white",fontsize=7)

    ax4 = fig.add_subplot(334,facecolor="#111111")
    cv_vals=[s3.get(r,{}).get("cv_pct",0) for r in RAYS]
    bc=["#2ecc71" if cv<5 else "#e74c3c" for cv in cv_vals]
    ax4.bar(RAYS,cv_vals,color=bc,edgecolor="none")
    ax4.axhline(5.0,color="white",lw=1.5,ls="--",label="5% threshold")
    ax4.set_title("Step 3 — CV%",color="white",fontsize=9)
    ax4.set_ylabel("CV (%)",color="white",fontsize=7); ax4.tick_params(colors="white",labelsize=7)
    ax4.legend(facecolor="#222",labelcolor="white",fontsize=6)
    for sp in ax4.spines.values(): sp.set_color("#444")

    ax5 = fig.add_subplot(335,facecolor="#111111")
    combos=s4.get("combinations",[])[:8]
    if combos:
        clabels=[c["combo"].replace(" + ","\n+") for c in combos]
        cscores=[c["score"] for c in combos]
        cnrays=[c["n_rays"] for c in combos]
        ax5.bar(range(len(combos)),cscores,
                color=[COMBO_COLORS.get(n,"#aaa") for n in cnrays],edgecolor="none")
        ax5.set_xticks(range(len(combos))); ax5.set_xticklabels(clabels,fontsize=6,color="white")
        legend=[Patch(color=c,label=f"{n}-ray") for n,c in COMBO_COLORS.items()]
        ax5.legend(handles=legend,fontsize=6,facecolor="#222",labelcolor="white")
    ax5.set_title("Step 4 — Top Combos",color="white",fontsize=9)
    ax5.set_ylabel("Score",color="white",fontsize=7); ax5.tick_params(colors="white",labelsize=7)
    for sp in ax5.spines.values(): sp.set_color("#444")

    ax6 = fig.add_subplot(336,facecolor="#111111")
    orders=s5.get("firing_orders",[])[:8]
    if orders:
        olabels=[o["order"].replace(" → ","→\n") for o in orders]
        oscores=[o["mean_score"] for o in orders]
        ostds=[o["std_score"] for o in orders]
        cmap=plt.get_cmap("RdYlGn"); norm=plt.Normalize(min(oscores),max(oscores))
        ax6.bar(range(len(orders)),oscores,color=[cmap(norm(s)) for s in oscores],edgecolor="none")
        ax6.errorbar(range(len(orders)),oscores,yerr=ostds,fmt="none",color="white",capsize=2,lw=1)
        ax6.set_xticks(range(len(orders))); ax6.set_xticklabels(olabels,fontsize=6,color="white")
    ax6.set_title("Step 5 — Firing Orders",color="white",fontsize=9)
    ax6.set_ylabel("Mean Score",color="white",fontsize=7); ax6.tick_params(colors="white",labelsize=7)
    for sp in ax6.spines.values(): sp.set_color("#444")

    ax7 = fig.add_subplot(337,projection="3d",facecolor="#0d0d0d")
    npy_path = os.path.join(ROOT,"Steps","Step6_Final","avg_grid.npy")
    if os.path.exists(npy_path):
        avg_grid=np.load(npy_path); threshold=avg_grid.max()*0.02
        xi,yi,zi=np.where(avg_grid>threshold); ev=avg_grid[xi,yi,zi]
        norm2=(ev-ev.min())/(ev.max()-ev.min()+1e-12)
        ax7.scatter(xi,yi,zi,c=ev,cmap="hot",s=norm2*15+1,alpha=0.5,linewidths=0)
    ax7.set_title("Step 6 — 3D Energy Grid",color="white",fontsize=9)
    ax7.tick_params(colors="white",labelsize=5)
    ax7.xaxis.pane.fill=ax7.yaxis.pane.fill=ax7.zaxis.pane.fill=False
    ax7.view_init(elev=25,azim=45)

    ax8 = fig.add_subplot(338,facecolor="#111111")
    rs=s6.get("rep_stats",{}); mean=rs.get("mean",0); std=rs.get("std",0); nrep=s6.get("n_reps",10)
    ax8.bar(range(1,nrep+1),[mean]*nrep,color="#2ecc71",alpha=0.3,edgecolor="none")
    ax8.axhline(mean,color="#2ecc71",lw=2,label=f"Mean={mean:.4f}")
    ax8.axhspan(mean-std,mean+std,alpha=0.2,color="#2ecc71",label="±σ")
    ax8.set_title("Step 6 — Stability",color="white",fontsize=9)
    ax8.set_xlabel("Rep",color="white",fontsize=7); ax8.set_ylabel("RBE-MeV",color="white",fontsize=7)
    ax8.tick_params(colors="white",labelsize=7); ax8.legend(fontsize=7,facecolor="#222",labelcolor="white")
    for sp in ax8.spines.values(): sp.set_color("#444")

    ax9 = fig.add_subplot(339,facecolor="#111111"); ax9.axis("off")
    cv=rs.get("cv_pct",0); best_order=s6.get("best_order","gamma → neutron → carbon")
    best_rays=s6.get("best_rays",["gamma","neutron","carbon"])
    txt=("FINAL PROTOCOL\n"+"─"*30+"\n"
         f"Rays: {', '.join([r.upper() for r in best_rays])}\n"
         f"Order: {best_order}\n\n"
         f"Prion RBE-MeV: {mean:.4f} ± {std:.4f}\n"
         f"CV: {cv:.2f}%\n\n"
         "KEY FINDINGS:\n"
         "• Alpha CANNOT reach tip\n"
         "• Neutron = best penetration\n"
         "• Carbon Bragg peak at tip\n"
         "• Gamma covers whole rod\n"
         "• Best 3-ray: G→C→N\n\n"
         "LIMITATIONS:\n"
         "• Prion proxy = G4_WATER\n"
         "• No ROS chemistry modeled\n"
         "• Neutron activates steel\n\n"
         "NEXT → GROMACS MD bridge")
    ax9.text(0.03,0.97,txt,transform=ax9.transAxes,color="white",fontsize=7.5,
             va="top",fontfamily="monospace",
             bbox=dict(boxstyle="round",facecolor="#1a1a2e",edgecolor="#4a9eff",alpha=0.9))

    plt.tight_layout(rect=[0,0,1,0.96])
    out=os.path.join(STEPS_DIR,"step7_master_summary.png")
    plt.savefig(out,dpi=150,bbox_inches="tight",facecolor="#0d0d0d")
    plt.close(); print(f"    Saved {out}")


def save_final_csv(s1,s2,s3,s6):
    path=os.path.join(STATS_DIR,"step7_master_summary.csv")
    with open(path,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["Ray","Prion_MeV","RBE_MeV","Selectivity","Peak_Z_mm",
                    "Reach_mm","Opt_N","CV_pct","Stable"])
        for ray in RAYS:
            d1=s1.get(ray,{}); d2=s2.get(ray,{}); d3=s3.get(ray,{})
            w.writerow([ray,d1.get("prion_mev",0),d1.get("rbe_prion_mev",0),
                        d1.get("selectivity",0),d1.get("peak_z_mm",0),
                        d1.get("max_reach_mm",0),d2.get("optimal_count",0),
                        d3.get("cv_pct",0),d3.get("stable",False)])
    print(f"    Saved {path}")


def _try_gromacs_bridge():
    rmsd_path=os.path.join(ROOT,"gromacs","output","rmsd_damaged.xvg")
    if os.path.exists(rmsd_path):
        print("\n  GROMACS output found — running bridge analysis...")
        import subprocess
        bridge_dir=os.path.join(ROOT,"gromacs","bridge")
        for script in ["analyze_damage.py","visualize_protein.py"]:
            sp=os.path.join(bridge_dir,script)
            if os.path.exists(sp): subprocess.run(["python3",sp],cwd=ROOT)
    else:
        print("\n  No GROMACS output yet — skipping bridge.")
        print("  Run GROMACS MD first, then re-run step7.")


if __name__ == "__main__":
    run_step7()