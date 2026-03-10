import csv, os

GRID_SIZE = 50
CENTER    = GRID_SIZE // 2
CORE_MIN  = CENTER - 5
CORE_MAX  = CENTER + 5

FILES = {
    "seq_GNC_01_GNC.csv": {"order": "Gamma -> Neutron -> Carbon Ion", "combo": "CGN", "n_rays": 3},
    "seq_GNC_02_GCN.csv": {"order": "Gamma -> Carbon Ion -> Neutron", "combo": "CGN", "n_rays": 3},
    "seq_GNC_03_NGC.csv": {"order": "Neutron -> Gamma -> Carbon Ion", "combo": "CGN", "n_rays": 3},
    "seq_GNC_04_NCG.csv": {"order": "Neutron -> Carbon Ion -> Gamma", "combo": "CGN", "n_rays": 3},
    "seq_GNC_05_CGN.csv": {"order": "Carbon Ion -> Gamma -> Neutron", "combo": "CGN", "n_rays": 3},
    "seq_GNC_06_CNG.csv": {"order": "Carbon Ion -> Neutron -> Gamma", "combo": "CGN", "n_rays": 3},
    "seq_GNA_01_GNA.csv": {"order": "Gamma -> Neutron -> Alpha", "combo": "AGN", "n_rays": 3},
    "seq_GNA_02_GAN.csv": {"order": "Gamma -> Alpha -> Neutron", "combo": "AGN", "n_rays": 3},
    "seq_GNA_03_NGA.csv": {"order": "Neutron -> Gamma -> Alpha", "combo": "AGN", "n_rays": 3},
    "seq_GNA_04_NAG.csv": {"order": "Neutron -> Alpha -> Gamma", "combo": "AGN", "n_rays": 3},
    "seq_GNA_05_AGN.csv": {"order": "Alpha -> Gamma -> Neutron", "combo": "AGN", "n_rays": 3},
    "seq_GNA_06_ANG.csv": {"order": "Alpha -> Neutron -> Gamma", "combo": "AGN", "n_rays": 3},
    "seq_GCA_01_GCA.csv": {"order": "Gamma -> Carbon Ion -> Alpha", "combo": "ACG", "n_rays": 3},
    "seq_GCA_02_GAC.csv": {"order": "Gamma -> Alpha -> Carbon Ion", "combo": "ACG", "n_rays": 3},
    "seq_GCA_03_CGA.csv": {"order": "Carbon Ion -> Gamma -> Alpha", "combo": "ACG", "n_rays": 3},
    "seq_GCA_04_CAG.csv": {"order": "Carbon Ion -> Alpha -> Gamma", "combo": "ACG", "n_rays": 3},
    "seq_GCA_05_AGC.csv": {"order": "Alpha -> Gamma -> Carbon Ion", "combo": "ACG", "n_rays": 3},
    "seq_GCA_06_ACG.csv": {"order": "Alpha -> Carbon Ion -> Gamma", "combo": "ACG", "n_rays": 3},
    "seq_NCA_01_NCA.csv": {"order": "Neutron -> Carbon Ion -> Alpha", "combo": "ACN", "n_rays": 3},
    "seq_NCA_02_NAC.csv": {"order": "Neutron -> Alpha -> Carbon Ion", "combo": "ACN", "n_rays": 3},
    "seq_NCA_03_CNA.csv": {"order": "Carbon Ion -> Neutron -> Alpha", "combo": "ACN", "n_rays": 3},
    "seq_NCA_04_CAN.csv": {"order": "Carbon Ion -> Alpha -> Neutron", "combo": "ACN", "n_rays": 3},
    "seq_NCA_05_ANC.csv": {"order": "Alpha -> Neutron -> Carbon Ion", "combo": "ACN", "n_rays": 3},
    "seq_NCA_06_ACN.csv": {"order": "Alpha -> Carbon Ion -> Neutron", "combo": "ACN", "n_rays": 3},
    "seq_GNCA_01_GNCA.csv": {"order": "Gamma -> Neutron -> Carbon Ion -> Alpha", "combo": "ACGN", "n_rays": 4},
    "seq_GNCA_02_GNAC.csv": {"order": "Gamma -> Neutron -> Alpha -> Carbon Ion", "combo": "ACGN", "n_rays": 4},
    "seq_GNCA_03_GCNA.csv": {"order": "Gamma -> Carbon Ion -> Neutron -> Alpha", "combo": "ACGN", "n_rays": 4},
    "seq_GNCA_04_GCAN.csv": {"order": "Gamma -> Carbon Ion -> Alpha -> Neutron", "combo": "ACGN", "n_rays": 4},
    "seq_GNCA_05_GANC.csv": {"order": "Gamma -> Alpha -> Neutron -> Carbon Ion", "combo": "ACGN", "n_rays": 4},
    "seq_GNCA_06_GACN.csv": {"order": "Gamma -> Alpha -> Carbon Ion -> Neutron", "combo": "ACGN", "n_rays": 4},
    "seq_GNCA_07_NGCA.csv": {"order": "Neutron -> Gamma -> Carbon Ion -> Alpha", "combo": "ACGN", "n_rays": 4},
    "seq_GNCA_08_NGAC.csv": {"order": "Neutron -> Gamma -> Alpha -> Carbon Ion", "combo": "ACGN", "n_rays": 4},
    "seq_GNCA_09_NCGA.csv": {"order": "Neutron -> Carbon Ion -> Gamma -> Alpha", "combo": "ACGN", "n_rays": 4},
    "seq_GNCA_10_NCAG.csv": {"order": "Neutron -> Carbon Ion -> Alpha -> Gamma", "combo": "ACGN", "n_rays": 4},
    "seq_GNCA_11_NAGC.csv": {"order": "Neutron -> Alpha -> Gamma -> Carbon Ion", "combo": "ACGN", "n_rays": 4},
    "seq_GNCA_12_NACG.csv": {"order": "Neutron -> Alpha -> Carbon Ion -> Gamma", "combo": "ACGN", "n_rays": 4},
    "seq_GNCA_13_CGNA.csv": {"order": "Carbon Ion -> Gamma -> Neutron -> Alpha", "combo": "ACGN", "n_rays": 4},
    "seq_GNCA_14_CGAN.csv": {"order": "Carbon Ion -> Gamma -> Alpha -> Neutron", "combo": "ACGN", "n_rays": 4},
    "seq_GNCA_15_CNGA.csv": {"order": "Carbon Ion -> Neutron -> Gamma -> Alpha", "combo": "ACGN", "n_rays": 4},
    "seq_GNCA_16_CNAG.csv": {"order": "Carbon Ion -> Neutron -> Alpha -> Gamma", "combo": "ACGN", "n_rays": 4},
    "seq_GNCA_17_CAGN.csv": {"order": "Carbon Ion -> Alpha -> Gamma -> Neutron", "combo": "ACGN", "n_rays": 4},
    "seq_GNCA_18_CANG.csv": {"order": "Carbon Ion -> Alpha -> Neutron -> Gamma", "combo": "ACGN", "n_rays": 4},
    "seq_GNCA_19_AGNC.csv": {"order": "Alpha -> Gamma -> Neutron -> Carbon Ion", "combo": "ACGN", "n_rays": 4},
    "seq_GNCA_20_AGCN.csv": {"order": "Alpha -> Gamma -> Carbon Ion -> Neutron", "combo": "ACGN", "n_rays": 4},
    "seq_GNCA_21_ANGC.csv": {"order": "Alpha -> Neutron -> Gamma -> Carbon Ion", "combo": "ACGN", "n_rays": 4},
    "seq_GNCA_22_ANCG.csv": {"order": "Alpha -> Neutron -> Carbon Ion -> Gamma", "combo": "ACGN", "n_rays": 4},
    "seq_GNCA_23_ACGN.csv": {"order": "Alpha -> Carbon Ion -> Gamma -> Neutron", "combo": "ACGN", "n_rays": 4},
    "seq_GNCA_24_ACNG.csv": {"order": "Alpha -> Carbon Ion -> Neutron -> Gamma", "combo": "ACGN", "n_rays": 4},
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
        print(f"\n{sep}")
        print(f"  {names}  -- all firing orders ranked")
        print(sep)
        print(f"  {'Rank':<5} {'Firing Order':<52} {'Core MeV':>11} {'Surf/Core':>10} {'Voxels':>8} {'Score':>9}")
        print("-"*100)
        for i,r in enumerate(group,1):
            m = "  << BEST" if i==1 else ("  << WORST" if i==len(group) else "")
            print(f"  {i:<5} {r['order']:<52} {r['core_MeV']:>11,.1f} {r['surf_core']:>10.3f} {r['voxels']:>8,} {r['score']:>9.3f}{m}")
        best=group[0]; worst=group[-1]
        diff=round(((best['score']-worst['score'])/worst['score'])*100,1) if worst['score']>0 else 0
        print(f"\n  Best order:  {best['order']}")
        print(f"  Worst order: {worst['order']}")
        print(f"  Order effect: {diff}% difference -- this is how much firing sequence matters")

    print(f"\n{sep}")
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
    print(f"\n  Saved: order_comparison_results.csv")

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

# ── NEXT STEP ─────────────────────────────────────────────────────────────────
print("")
print("==============================================")
print("  NOW RUN:")
print("  bash scripts/run_optimize.sh")
print("  (runs all particle count optimization sims)")
print("==============================================")
