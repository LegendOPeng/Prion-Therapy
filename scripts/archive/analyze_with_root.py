#!/usr/bin/env python3
import ROOT
import re

print("Analyzing Geant4 data with ROOT...")

# Parse data
data = {'dE': [], 'x': [], 'y': [], 'z': []}
try:
    with open("geant4_output.txt", 'r') as f:
        for line in f:
            if 'prion_region' in line:
                parts = line.split()
                try:
                    if len(parts) >= 10:
                        dE = float(parts[5])
                        if dE > 0:
                            data['x'].append(float(parts[1]))
                            data['y'].append(float(parts[2]))
                            data['z'].append(float(parts[3]))
                            data['dE'].append(dE)
                except: pass
except FileNotFoundError:
    print("ERROR: geant4_output.txt not found!")
    exit(1)

print(f"Found {len(data['dE'])} events")

# Create histograms
h_energy = ROOT.TH1F("h_energy", "Energy;dE (MeV);Counts", 50, 0, max(data['dE'])*1.1)
for e in data['dE']: h_energy.Fill(e)

h_xy = ROOT.TH2F("h_xy", "Spatial;X (mm);Y (mm)", 30, -20, 20, 30, -20, 20)
for x, y in zip(data['x'], data['y']): h_xy.Fill(x, y)

# Draw and save
c1 = ROOT.TCanvas("c1", "Summary", 1200, 600)
c1.Divide(2, 1)
c1.cd(1)
h_energy.SetFillColor(ROOT.kBlue-7)
h_energy.Draw()
c1.cd(2)
h_xy.Draw("COLZ")
c1.SaveAs("summary_plots.png")

print(f"Saved: summary_plots.png")
print(f"Total energy: {sum(data['dE']):.4f} MeV")#!/usr/bin/env python3
"""
Analyze Geant4 Prion Simulation Data with ROOT
Creates histograms and charts using PyROOT
"""

import ROOT
import re

print("="*70)
print("GEANT4 PRION SIMULATION - ROOT ANALYSIS")
print("="*70)

# ============================================
# STEP 1: PARSE GEANT4 OUTPUT
# ============================================
print("\nStep 1: Parsing Geant4 output file...")

def parse_geant4_data(filename):
    """Extract data from Geant4 output"""
    data = {
        'energy_deposited': [],
        'x_position': [],
        'y_position': [],
        'z_position': [],
        'kinetic_energy': [],
        'step_length': [],
        'process': [],
        'volume': []
    }
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                # Look for lines with prion_region
                if 'prion_region' in line:
                    parts = line.split()
                    try:
                        if len(parts) >= 10:
                            # Extract numerical data
                            x = float(parts[1])
                            y = float(parts[2])
                            z = float(parts[3])
                            kinE = float(parts[4])
                            dE = float(parts[5])
                            stepLen = float(parts[6])
                            volume = parts[9]
                            process = parts[10] if len(parts) > 10 else "unknown"
                            
                            # Only record if energy was deposited
                            if dE > 0:
                                data['x_position'].append(x)
                                data['y_position'].append(y)
                                data['z_position'].append(z)
                                data['kinetic_energy'].append(kinE)
                                data['energy_deposited'].append(dE)
                                data['step_length'].append(stepLen)
                                data['volume'].append(volume)
                                data['process'].append(process)
                    except (ValueError, IndexError):
                        continue
    except FileNotFoundError:
        print(f"ERROR: Could not find {filename}")
        return None
    
    return data

# Parse the data
data = parse_geant4_data("geant4_output.txt")

if data is None or len(data['energy_deposited']) == 0:
    print("ERROR: No data found. Please run Geant4 simulation first.")
    exit(1)

print(f"  Found {len(data['energy_deposited'])} energy deposition events")
print(f"  Total energy: {sum(data['energy_deposited']):.4f} MeV")

# ============================================
# STEP 2: CREATE ROOT HISTOGRAMS
# ============================================
print("\nStep 2: Creating ROOT histograms...")

# Create a ROOT file to save results
output_file = ROOT.TFile("prion_analysis.root", "RECREATE")

# Histogram 1: Energy Deposition Distribution
h_energy = ROOT.TH1F("h_energy", "Energy Deposition in Prion Region;Energy Deposited (MeV);Counts", 
                     50, 0, max(data['energy_deposited'])*1.1)
for e in data['energy_deposited']:
    h_energy.Fill(e)

# Histogram 2: Spatial Distribution (X position)
h_x = ROOT.TH1F("h_x", "X Position of Energy Depositions;X Position (mm);Counts",
                50, min(data['x_position'])-5, max(data['x_position'])+5)
for x in data['x_position']:
    h_x.Fill(x)

# Histogram 3: Spatial Distribution (Y position)
h_y = ROOT.TH1F("h_y", "Y Position of Energy Depositions;Y Position (mm);Counts",
                50, min(data['y_position'])-5, max(data['y_position'])+5)
for y in data['y_position']:
    h_y.Fill(y)

# Histogram 4: Spatial Distribution (Z position)
h_z = ROOT.TH1F("h_z", "Z Position of Energy Depositions;Z Position (mm);Counts",
                50, min(data['z_position'])-5, max(data['z_position'])+5)
for z in data['z_position']:
    h_z.Fill(z)

# Histogram 5: 2D Spatial Distribution (X vs Y)
h_xy = ROOT.TH2F("h_xy", "Spatial Distribution (Top View);X Position (mm);Y Position (mm)",
                 30, min(data['x_position'])-5, max(data['x_position'])+5,
                 30, min(data['y_position'])-5, max(data['y_position'])+5)
for x, y in zip(data['x_position'], data['y_position']):
    h_xy.Fill(x, y)

# Histogram 6: Process types
processes = list(set(data['process']))
h_process = ROOT.TH1F("h_process", "Physics Processes;Process;Counts",
                      len(processes), 0, len(processes))
for i, proc in enumerate(processes):
    h_process.GetXaxis().SetBinLabel(i+1, proc)
    count = data['process'].count(proc)
    for _ in range(count):
        h_process.Fill(i)

print("  Created 6 histograms")

# ============================================
# STEP 3: CREATE VISUALIZATIONS
# ============================================
print("\nStep 3: Creating ROOT canvases...")

# Set ROOT style
ROOT.gStyle.SetOptStat(1111)
ROOT.gStyle.SetOptFit(1)
ROOT.gStyle.SetPalette(ROOT.kRainBow)

# Canvas 1: Energy Deposition
c1 = ROOT.TCanvas("c1", "Energy Deposition", 800, 600)
h_energy.SetFillColor(ROOT.kBlue-7)
h_energy.SetLineColor(ROOT.kBlue+2)
h_energy.Draw()
c1.SaveAs("energy_deposition.png")
print("  Saved: energy_deposition.png")

# Canvas 2: Spatial Distributions (1D)
c2 = ROOT.TCanvas("c2", "Spatial Distributions", 1200, 400)
c2.Divide(3, 1)

c2.cd(1)
h_x.SetFillColor(ROOT.kRed-7)
h_x.SetLineColor(ROOT.kRed+2)
h_x.Draw()

c2.cd(2)
h_y.SetFillColor(ROOT.kGreen-7)
h_y.SetLineColor(ROOT.kGreen+2)
h_y.Draw()

c2.cd(3)
h_z.SetFillColor(ROOT.kMagenta-7)
h_z.SetLineColor(ROOT.kMagenta+2)
h_z.Draw()

c2.SaveAs("spatial_distributions.png")
print("  Saved: spatial_distributions.png")

# Canvas 3: 2D Spatial Distribution
c3 = ROOT.TCanvas("c3", "2D Spatial Distribution", 800, 700)
c3.SetRightMargin(0.15)
h_xy.SetContour(50)
h_xy.Draw("COLZ")
c3.SaveAs("spatial_2d.png")
print("  Saved: spatial_2d.png")

# Canvas 4: Physics Processes
c4 = ROOT.TCanvas("c4", "Physics Processes", 800, 600)
h_process.SetFillColor(ROOT.kOrange-3)
h_process.SetLineColor(ROOT.kOrange+2)
h_process.Draw()
c4.SaveAs("physics_processes.png")
print("  Saved: physics_processes.png")

# Canvas 5: Summary (4-panel)
c5 = ROOT.TCanvas("c5", "Summary", 1400, 1000)
c5.Divide(2, 2)

c5.cd(1)
h_energy.Draw()

c5.cd(2)
h_xy.Draw("COLZ")

c5.cd(3)
h_x.Draw()

c5.cd(4)
h_process.Draw()

c5.SaveAs("summary_plots.png")
print("  Saved: summary_plots.png")

# ============================================
# STEP 4: CALCULATE STATISTICS
# ============================================
print("\nStep 4: Calculating statistics...")

# Calculate dose
PRION_RADIUS_M = 0.015  # 15 mm
prion_volume_m3 = (4/3) * 3.14159 * (PRION_RADIUS_M**3)
prion_mass_kg = prion_volume_m3 * 1040  # brain density

total_energy_MeV = sum(data['energy_deposited'])
total_energy_J = total_energy_MeV * 1.602e-13
dose_Gy = total_energy_J / prion_mass_kg
dose_mGy = dose_Gy * 1000

print(f"\n  STATISTICS:")
print(f"  -----------")
print(f"  Total events: {len(data['energy_deposited'])}")
print(f"  Total energy deposited: {total_energy_MeV:.4f} MeV")
print(f"  Mean energy per event: {h_energy.GetMean():.4f} MeV")
print(f"  Dose delivered: {dose_mGy:.6f} mGy")
print(f"  Prion region mass: {prion_mass_kg*1000:.2f} grams")

# Process breakdown
print(f"\n  PHYSICS PROCESSES:")
for proc in processes:
    count = data['process'].count(proc)
    percent = (count / len(data['process'])) * 100
    print(f"    {proc}: {count} events ({percent:.1f}%)")

# ============================================
# STEP 5: SAVE ROOT FILE
# ============================================
print("\nStep 5: Saving ROOT file...")

# Write histograms to file
output_file.cd()
h_energy.Write()
h_x.Write()
h_y.Write()
h_z.Write()
h_xy.Write()
h_process.Write()

# Create a TTree for detailed analysis
tree = ROOT.TTree("prion_data", "Prion Region Energy Depositions")

# Create branches (need to use arrays for ROOT)
import array
x_arr = array.array('f', [0])
y_arr = array.array('f', [0])
z_arr = array.array('f', [0])
dE_arr = array.array('f', [0])
kinE_arr = array.array('f', [0])

tree.Branch('x', x_arr, 'x/F')
tree.Branch('y', y_arr, 'y/F')
tree.Branch('z', z_arr, 'z/F')
tree.Branch('dE', dE_arr, 'dE/F')
tree.Branch('kinE', kinE_arr, 'kinE/F')

# Fill the tree
for i in range(len(data['energy_deposited'])):
    x_arr[0] = data['x_position'][i]
    y_arr[0] = data['y_position'][i]
    z_arr[0] = data['z_position'][i]
    dE_arr[0] = data['energy_deposited'][i]
    kinE_arr[0] = data['kinetic_energy'][i]
    tree.Fill()

tree.Write()
output_file.Close()

print("  Saved: prion_analysis.root")
print(f"  Contains: {len(data['energy_deposited'])} events in TTree")

# ============================================
# DONE
# ============================================
print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print("\nOUTPUTS:")
print("  • energy_deposition.png - Energy histogram")
print("  • spatial_distributions.png - X, Y, Z distributions")
print("  • spatial_2d.png - 2D dose map (top view)")
print("  • physics_processes.png - Process breakdown")
print("  • summary_plots.png - 4-panel summary")
print("  • prion_analysis.root - ROOT file with histograms and TTree")
print("\nYou can open the ROOT file with:")
print("  root prion_analysis.root")
print("  root [0] prion_data->Draw(\"dE\")")
print("="*70)
