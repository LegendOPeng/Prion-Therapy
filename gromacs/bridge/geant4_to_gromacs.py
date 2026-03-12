"""
Bridge: Geant4 → GROMACS
Reads avg_grid.npy from Step 6, maps energy deposition onto prion protein
atoms, creates a perturbed .gro file for damaged MD simulation.
"""
import numpy as np
import os, math

# Paths
GRID_PATH   = "../Steps/Step6_Final/avg_grid.npy"
INPUT_GRO   = "../gromacs/npt.gro"
OUTPUT_GRO  = "../gromacs/structure/prion_damaged.gro"
PRION_RADIUS_NM = 1.5   # 15mm sphere = 1.5nm in GROMACS units
CENTER_NM   = None      # auto-detect from protein center of mass

GRID = 50
VOXEL_SIZE_NM = 10.0 / GRID  # 10cm box / 50 voxels = 2mm = 0.2nm per voxel

# Energy to velocity scaling
# 1 MeV = 1.602e-13 J
# Average atom mass ~12 amu = 12 * 1.66e-27 kg
# v = sqrt(2E/m) but we scale conservatively to avoid blowing up simulation
SCALE_FACTOR = 0.0001  # nm/ps per MeV — conservative perturbation

def load_grid():
    if not os.path.exists(GRID_PATH):
        print(f"[ERROR] Grid not found: {GRID_PATH}")
        return None
    g = np.load(GRID_PATH)
    print(f"  Grid loaded: shape={g.shape} max={g.max():.2f} total={g.sum():.2f} MeV")
    return g

def parse_gro(path):
    with open(path) as f:
        lines = f.readlines()
    title   = lines[0]
    n_atoms = int(lines[1].strip())
    atoms   = []
    for line in lines[2:2+n_atoms]:
        resnum  = int(line[0:5])
        resname = line[5:10].strip()
        atname  = line[10:15].strip()
        atnum   = int(line[15:20])
        x = float(line[20:28])
        y = float(line[28:36])
        z = float(line[36:44])
        try:
            vx = float(line[44:52])
            vy = float(line[52:60])
            vz = float(line[60:68])
        except:
            vx = vy = vz = 0.0
        atoms.append({"resnum":resnum,"resname":resname,"atname":atname,
                      "atnum":atnum,"x":x,"y":y,"z":z,"vx":vx,"vy":vy,"vz":vz})
    box = lines[2+n_atoms].strip()
    return title, atoms, box

def get_protein_atoms(atoms):
    return [a for a in atoms if a["resname"] not in ("SOL","NA","CL","HOH")]

def get_center(protein_atoms):
    xs = [a["x"] for a in protein_atoms]
    ys = [a["y"] for a in protein_atoms]
    zs = [a["z"] for a in protein_atoms]
    return sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)

def coord_to_voxel(x, y, z, cx, cy, cz):
    # Map protein coordinate to grid voxel
    # Grid center = voxel 25, each voxel = 0.2nm
    ix = int((x - cx) / VOXEL_SIZE_NM + GRID//2)
    iy = int((y - cy) / VOXEL_SIZE_NM + GRID//2)
    iz = int((z - cz) / VOXEL_SIZE_NM + GRID//2)
    ix = max(0, min(GRID-1, ix))
    iy = max(0, min(GRID-1, iy))
    iz = max(0, min(GRID-1, iz))
    return ix, iy, iz

def apply_perturbation(atoms, grid):
    protein_atoms = get_protein_atoms(atoms)
    cx, cy, cz    = get_center(protein_atoms)
    print(f"  Protein center: ({cx:.3f}, {cy:.3f}, {cz:.3f}) nm")

    perturbed = 0
    total_energy_applied = 0.0

    for a in atoms:
        if a["resname"] in ("SOL","NA","CL","HOH"):
            continue
        ix, iy, iz = coord_to_voxel(a["x"], a["y"], a["z"], cx, cy, cz)
        edep = grid[ix, iy, iz]
        if edep > 0:
            # Apply velocity perturbation in random direction
            rng   = np.random.default_rng(a["atnum"])
            dv    = edep * SCALE_FACTOR
            theta = rng.uniform(0, math.pi)
            phi   = rng.uniform(0, 2*math.pi)
            dvx   = dv * math.sin(theta) * math.cos(phi)
            dvy   = dv * math.sin(theta) * math.sin(phi)
            dvz   = dv * math.cos(theta)
            a["vx"] += dvx
            a["vy"] += dvy
            a["vz"] += dvz
            perturbed += 1
            total_energy_applied += edep

    print(f"  Atoms perturbed: {perturbed}")
    print(f"  Total energy applied: {total_energy_applied:.2f} MeV")
    return atoms

def write_gro(path, title, atoms, box):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"Radiation-damaged prion protein (Geant4 bridge)\n")
        f.write(f"{len(atoms):>5}\n")
        for a in atoms:
            f.write(f"{a['resnum']:>5}{a['resname']:<5}{a['atname']:>5}{a['atnum']:>5}"
                    f"{a['x']:>8.3f}{a['y']:>8.3f}{a['z']:>8.3f}"
                    f"{a['vx']:>8.4f}{a['vy']:>8.4f}{a['vz']:>8.4f}\n")
        f.write(f" {box}\n")
    print(f"  Written: {path}")

def main():
    print("\n  Geant4 → GROMACS Bridge")
    print("  " + "="*50)

    grid = load_grid()
    if grid is None:
        return

    print(f"  Parsing: {INPUT_GRO}")
    title, atoms, box = parse_gro(INPUT_GRO)
    print(f"  Atoms: {len(atoms)}")

    print("  Applying radiation perturbation...")
    atoms = apply_perturbation(atoms, grid)

    write_gro(OUTPUT_GRO, title, atoms, box)
    print("\n  Bridge complete!")
    print(f"  Use {OUTPUT_GRO} for damaged MD simulation.")

if __name__ == "__main__":
    main()
