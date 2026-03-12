"""
Python-based 3D protein visualizer (VMD replacement)
Shows healthy vs radiation-damaged prion protein structure
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

HEALTHY_GRO  = "gromacs/structure/final_healthy.gro"
DAMAGED_GRO  = "gromacs/structure/final_damaged.gro"
OUT_DIR      = "Steps/Step7_Final"
os.makedirs(OUT_DIR, exist_ok=True)

RESIDUE_COLORS = {
    "ALA":"#ff6b6b","ARG":"#4ecdc4","ASN":"#45b7d1","ASP":"#f7dc6f",
    "CYS":"#f0e68c","GLN":"#dda0dd","GLU":"#ff7f50","GLY":"#98fb98",
    "HIS":"#87ceeb","ILE":"#ffa07a","LEU":"#20b2aa","LYS":"#ff69b4",
    "MET":"#cd853f","PHE":"#6495ed","PRO":"#dc143c","SER":"#00ced1",
    "THR":"#ff8c00","TRP":"#9370db","TYR":"#3cb371","VAL":"#b8860b",
}

def parse_gro(path):
    atoms = []
    if not os.path.exists(path):
        print(f"[ERROR] {path} not found")
        return atoms
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[1].strip())
    for line in lines[2:2+n]:
        resname = line[5:10].strip()
        atname  = line[10:15].strip()
        if resname in ("SOL","NA","CL","HOH"):
            continue
        try:
            x = float(line[20:28])
            y = float(line[28:36])
            z = float(line[36:44])
            atoms.append({"resname":resname,"atname":atname,"x":x,"y":y,"z":z})
        except:
            continue
    return atoms

def get_ca_trace(atoms):
    """Get C-alpha backbone trace for ribbon visualization"""
    ca = [a for a in atoms if a["atname"] == "CA"]
    return ca

def get_color(resname):
    return RESIDUE_COLORS.get(resname, "#ffffff")

def plot_protein(ax, atoms, title, color_scheme="residue"):
    ca_atoms = get_ca_trace(atoms)
    if not ca_atoms:
        return

    xs = np.array([a["x"] for a in ca_atoms])
    ys = np.array([a["y"] for a in ca_atoms])
    zs = np.array([a["z"] for a in ca_atoms])

    # Draw backbone ribbon
    for i in range(len(ca_atoms)-1):
        c = get_color(ca_atoms[i]["resname"])
        ax.plot([xs[i],xs[i+1]], [ys[i],ys[i+1]], [zs[i],zs[i+1]],
                color=c, linewidth=2.5, alpha=0.85)

    # Draw residue spheres
    colors = [get_color(a["resname"]) for a in ca_atoms]
    ax.scatter(xs, ys, zs, c=colors, s=18, alpha=0.9, linewidths=0, depthshade=True)

    # Highlight cysteines (disulfide bond — critical for prion structure)
    cys_atoms = [a for a in ca_atoms if a["resname"] == "CYS"]
    if cys_atoms:
        cx = [a["x"] for a in cys_atoms]
        cy = [a["y"] for a in cys_atoms]
        cz = [a["z"] for a in cys_atoms]
        ax.scatter(cx, cy, cz, c="#ffff00", s=60, alpha=1.0,
                   linewidths=1, edgecolors="white", zorder=5, label="CYS (disulfide)")

    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
    ax.set_facecolor("#0a0a0a")
    ax.tick_params(colors="white", labelsize=6)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#333")
    ax.yaxis.pane.set_edgecolor("#333")
    ax.zaxis.pane.set_edgecolor("#333")
    ax.set_xlabel("x (nm)", color="white", fontsize=7)
    ax.set_ylabel("y (nm)", color="white", fontsize=7)
    ax.set_zlabel("z (nm)", color="white", fontsize=7)
    if cys_atoms:
        ax.legend(fontsize=7, facecolor="#222", labelcolor="white", loc="upper left")

def plot_difference(ax, healthy, damaged):
    """Show per-residue displacement between healthy and damaged"""
    ca_h = get_ca_trace(healthy)
    ca_d = get_ca_trace(damaged)
    n = min(len(ca_h), len(ca_d))
    if n == 0:
        return

    displacements = []
    for i in range(n):
        dx = ca_d[i]["x"] - ca_h[i]["x"]
        dy = ca_d[i]["y"] - ca_h[i]["y"]
        dz = ca_d[i]["z"] - ca_h[i]["z"]
        displacements.append(np.sqrt(dx**2+dy**2+dz**2))

    displacements = np.array(displacements)
    residues = list(range(1, n+1))
    colors = plt.cm.hot(displacements / displacements.max())

    ax.bar(residues, displacements*10, color=colors, edgecolor="none", width=1.0)
    ax.axhline(np.mean(displacements)*10, color="cyan", linestyle="--",
               linewidth=1, label=f"Mean Δ={np.mean(displacements)*10:.3f} Å")

    # Highlight top 10 most displaced residues
    top10 = np.argsort(displacements)[-10:]
    for idx in top10:
        resname = ca_h[idx]["resname"]
        ax.text(residues[idx], displacements[idx]*10+0.002,
                resname[:3], color="white", fontsize=5, ha="center", va="bottom")

    ax.set_title("Per-Residue Displacement After Radiation (Å)", color="white", fontsize=10)
    ax.set_xlabel("Residue Number", color="white", fontsize=8)
    ax.set_ylabel("Displacement (Å)", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    ax.set_facecolor("#111111")
    for spine in ax.spines.values(): spine.set_color("#444")
    ax.legend(fontsize=8, facecolor="#222", labelcolor="white")

    return displacements

def main():
    print("\n  Python Protein Visualizer")
    print("  " + "="*50)

    healthy = parse_gro(HEALTHY_GRO)
    damaged = parse_gro(DAMAGED_GRO)
    print(f"  Healthy atoms: {len(healthy)}")
    print(f"  Damaged atoms: {len(damaged)}")

    fig = plt.figure(figsize=(20,14), facecolor="#0d0d0d")
    fig.suptitle(
        "Human Prion Protein (PrP) — Radiation Damage Visualization\n"
        "Geant4 → GROMACS Bridge — 1QLX Structure",
        color="white", fontsize=14, fontweight="bold"
    )

    # Healthy 3D
    ax1 = fig.add_subplot(231, projection="3d")
    plot_protein(ax1, healthy, "Healthy PrP (Pre-Radiation)")
    ax1.view_init(elev=20, azim=45)

    # Damaged 3D
    ax2 = fig.add_subplot(232, projection="3d")
    plot_protein(ax2, damaged, "Irradiated PrP (Post-Radiation)")
    ax2.view_init(elev=20, azim=45)

    # Second angle healthy
    ax3 = fig.add_subplot(233, projection="3d")
    plot_protein(ax3, healthy, "Healthy PrP (Top View)")
    ax3.view_init(elev=90, azim=0)

    # Second angle damaged
    ax4 = fig.add_subplot(234, projection="3d")
    plot_protein(ax4, damaged, "Irradiated PrP (Top View)")
    ax4.view_init(elev=90, azim=0)

    # Per-residue displacement
    ax5 = fig.add_subplot(212)
    ax5.set_facecolor("#111111")
    displacements = plot_difference(ax5, healthy, damaged)

    # Summary
    if displacements is not None:
        ca_h = get_ca_trace(healthy)
        ca_d = get_ca_trace(damaged)
        n = min(len(ca_h), len(ca_d))
        top5_idx = np.argsort(displacements)[-5:]
        print(f"\n  Top 5 most displaced residues:")
        for idx in reversed(top5_idx):
            print(f"    Residue {idx+1} ({ca_h[idx]['resname']}): "
                  f"{displacements[idx]*10:.4f} Å")
        print(f"\n  Mean displacement: {np.mean(displacements)*10:.4f} Å")
        print(f"  Max displacement:  {np.max(displacements)*10:.4f} Å")

    plt.tight_layout(rect=[0,0,1,0.95])
    out = os.path.join(OUT_DIR, "protein_3d_visualization.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"\n  Saved: {out}")
    print("  Done!")

if __name__ == "__main__":
    main()
