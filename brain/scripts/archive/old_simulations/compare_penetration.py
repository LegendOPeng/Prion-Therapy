#!/usr/bin/env python3
"""
compare_penetration.py
Analyzes Geant4 energy deposition CSV files for all 4 radiation types.
Run after all 4 simulations complete.

Usage:
    python3 compare_penetration.py

Output:
    - penetration_comparison.png  (bar chart)
    - depth_profiles.png          (depth vs energy plot)
    - comparison_table.csv        (numbers for your report)
"""

import csv
import os
import sys
import math

# ── CONFIG ────────────────────────────────────────────────────────────────────
FILES = {
    "Gamma":     "gamma_edep.csv",   # Already exists from your test run
    "Neutron":   "neutron_edep.csv",
    "Carbon Ion":"carbon_edep.csv",
    "Alpha":     "alpha_edep.csv",
}

GRID_SIZE  = 50          # 50×50×50 voxels
BOX_MM     = 50.0        # Total box is 50mm × 50mm × 50mm
VOXEL_MM   = BOX_MM / GRID_SIZE   # 1mm per voxel
CENTER     = GRID_SIZE // 2        # Voxel index of center = 25

# "Core" = central 10×10×10 voxels (±5mm from center)
# "Surface" = anything within 5 voxels of edge
CORE_MIN   = CENTER - 5   # voxel 20
CORE_MAX   = CENTER + 5   # voxel 30


# ── HELPERS ───────────────────────────────────────────────────────────────────
def load_csv(filepath):
    """Load a Geant4 score CSV. Returns list of (ix, iy, iz, energy) tuples."""
    data = []
    if not os.path.exists(filepath):
        print(f"  [WARNING] File not found: {filepath}")
        return data
    with open(filepath) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            try:
                ix, iy, iz = int(row[0]), int(row[1]), int(row[2])
                energy = float(row[3])
                if energy > 0:
                    data.append((ix, iy, iz, energy))
            except ValueError:
                continue
    return data


def classify_voxel(ix, iy, iz):
    """Return 'core', 'surface', or 'edge' based on voxel position."""
    in_core = (CORE_MIN <= ix <= CORE_MAX and
               CORE_MIN <= iy <= CORE_MAX and
               CORE_MIN <= iz <= CORE_MAX)
    if in_core:
        return "core"
    on_edge = (ix < 5 or ix > 44 or
               iy < 5 or iy > 44 or
               iz < 5 or iz > 44)
    return "edge" if on_edge else "surface"


def distance_from_center(ix, iy, iz):
    """Return Euclidean distance (in voxels) from center."""
    return math.sqrt((ix - CENTER)**2 + (iy - CENTER)**2 + (iz - CENTER)**2)


def analyze(data, label):
    """Compute summary statistics for one radiation type."""
    if not data:
        return None

    total_energy  = sum(e for *_, e in data)
    voxels_hit    = len(data)
    max_energy    = max(e for *_, e in data)

    core_energy    = sum(e for ix, iy, iz, e in data if classify_voxel(ix, iy, iz) == "core")
    surface_energy = sum(e for ix, iy, iz, e in data if classify_voxel(ix, iy, iz) == "surface")

    # Mean depth of energy deposition (distance from entry face, iz direction)
    mean_iz = sum(iz * e for ix, iy, iz, e in data) / total_energy

    # Check if CENTER voxel got any hits
    center_hits = sum(e for ix, iy, iz, e in data
                      if abs(ix - CENTER) <= 2 and
                         abs(iy - CENTER) <= 2 and
                         abs(iz - CENTER) <= 2)

    ratio = surface_energy / core_energy if core_energy > 0 else float("inf")

    return {
        "label":          label,
        "voxels_hit":     voxels_hit,
        "total_energy":   total_energy,
        "max_energy":     max_energy,
        "core_energy":    core_energy,
        "surface_energy": surface_energy,
        "surf_core_ratio":ratio,
        "mean_depth_vox": mean_iz,
        "center_hits":    center_hits > 0,
    }


def depth_profile(data):
    """Return list of (iz_voxel, total_energy_at_that_depth) for plotting."""
    by_depth = {}
    for ix, iy, iz, e in data:
        by_depth[iz] = by_depth.get(iz, 0) + e
    return sorted(by_depth.items())


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Prion Radiation Penetration Comparison")
    print("=" * 60)

    results = []
    profiles = {}

    for label, filepath in FILES.items():
        print(f"\nLoading {label}: {filepath}")
        data = load_csv(filepath)
        if data:
            print(f"  → {len(data)} voxels with energy deposition")
            r = analyze(data, label)
            results.append(r)
            profiles[label] = depth_profile(data)
        else:
            print(f"  → No data (run simulation first)")

    if not results:
        print("\nNo data found. Run your Geant4 simulations first, then rerun this script.")
        sys.exit(0)

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  COMPARISON TABLE")
    print("=" * 60)
    header = f"{'Radiation':<14} {'Voxels Hit':>10} {'Max E (MeV)':>12} {'Core Hit?':>10} {'Surf/Core':>10} {'Mean Depth':>11}"
    print(header)
    print("-" * 70)
    for r in results:
        print(f"{r['label']:<14} "
              f"{r['voxels_hit']:>10,} "
              f"{r['max_energy']:>12.4f} "
              f"{'Yes' if r['center_hits'] else 'No':>10} "
              f"{r['surf_core_ratio']:>10.2f} "
              f"{r['mean_depth_vox']:>10.1f}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_csv = "comparison_table.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {out_csv}")

    # ── Try to plot (optional - only if matplotlib available) ─────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Bar chart comparison
        labels    = [r["label"] for r in results]
        voxels    = [r["voxels_hit"] for r in results]
        core_e    = [r["core_energy"] for r in results]
        surf_e    = [r["surface_energy"] for r in results]

        x = range(len(labels))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: voxels hit
        axes[0].bar(x, voxels, color=["steelblue", "orange", "green", "red"])
        axes[0].set_xticks(list(x))
        axes[0].set_xticklabels(labels)
        axes[0].set_ylabel("Voxels with Energy Deposition")
        axes[0].set_title("Coverage (Voxels Hit)")

        # Right: core vs surface energy
        width = 0.35
        axes[1].bar([i - width/2 for i in x], surf_e, width, label="Surface", color="tomato")
        axes[1].bar([i + width/2 for i in x], core_e,  width, label="Core",    color="steelblue")
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels(labels)
        axes[1].set_ylabel("Energy Deposited (MeV)")
        axes[1].set_title("Surface vs Core Energy Deposition")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig("penetration_comparison.png", dpi=150)
        print("Saved: penetration_comparison.png")
        plt.close()

        # Depth profile plot
        fig2, ax = plt.subplots(figsize=(10, 5))
        colors = {"Gamma": "blue", "Neutron": "orange", "Carbon Ion": "green", "Alpha": "red"}
        for label, profile in profiles.items():
            if profile:
                depths, energies = zip(*profile)
                depths_mm = [(d - CENTER) * VOXEL_MM for d in depths]
                ax.plot(depths_mm, energies, label=label,
                        color=colors.get(label, "gray"), linewidth=2)

        ax.axvline(0,  color="black", linestyle="--", alpha=0.4, label="Center")
        ax.axvline(-15, color="gray", linestyle=":",  alpha=0.4, label="Sphere edge")
        ax.axvline(+15, color="gray", linestyle=":",  alpha=0.4)
        ax.set_xlabel("Depth from center (mm)")
        ax.set_ylabel("Energy deposited (MeV)")
        ax.set_title("Depth Profile of Energy Deposition\n(negative = entry side, positive = exit side)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("depth_profiles.png", dpi=150)
        print("Saved: depth_profiles.png")
        plt.close()

    except ImportError:
        print("\nMatplotlib not installed - skipping plots.")
        print("Install with: pip install matplotlib --break-system-packages")

    print("\nDone! Check comparison_table.csv for your report numbers.")


if __name__ == "__main__":
    main()
