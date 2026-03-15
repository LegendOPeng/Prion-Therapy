import os

VIS_HEADER = """/control/verbose 1
/geometry/source brain.tg
/vis/open OGL 800x800-0+0
/vis/drawVolume
/vis/scene/create
/vis/scene/add/volume
/vis/scene/add/axes 0 0 0 10 cm
/vis/geometry/set/colour prion_region 0 0.9 0.1 0.1 0.8
/vis/geometry/set/colour brain        0 0.9 0.7 0.7 0.3
/vis/geometry/set/colour head         0 0.95 0.95 0.9 0.15
/vis/geometry/set/visibility air  0 false
/vis/geometry/set/visibility room 0 false
/vis/scene/add/trajectories smooth
/vis/modeling/trajectories/create/drawByParticleID
/vis/modeling/trajectories/drawByParticleID-0/set gamma     blue
/vis/modeling/trajectories/drawByParticleID-0/set neutron   green
/vis/modeling/trajectories/drawByParticleID-0/set C12[0.0] orange
/vis/modeling/trajectories/drawByParticleID-0/set alpha     red
/vis/modeling/trajectories/drawByParticleID-0/set proton    yellow
/vis/modeling/trajectories/drawByParticleID-0/set e-        magenta
/vis/scene/add/hits
/vis/scene/endOfEventAction accumulate
/vis/viewer/set/viewpointThetaPhi 70 20
/vis/viewer/set/lightsThetaPhi 100 30
/vis/viewer/zoom 1.5
/run/initialize"""

VIS_FOOTER = """/vis/viewer/flush
/vis/viewer/refresh"""

PARTICLES = {
    "G": {"name":"Gamma",      "color":"Blue",   "LET":"Low LET, wide spread blue tracks",
          "block":"""/gps/particle gamma
/gps/energy 1.17 MeV
/gps/pos/type Volume
/gps/pos/shape Sphere
/gps/pos/radius 15 mm
/gps/pos/centre 0 0 0 mm
/run/beamOn 200"""},
    "N": {"name":"Neutron",    "color":"Green",  "LET":"Medium-high LET, green tracks with yellow recoil protons",
          "block":"""/gps/particle neutron
/gps/energy 2.0 MeV
/gps/pos/type Volume
/gps/pos/shape Sphere
/gps/pos/radius 15 mm
/gps/pos/centre 0 0 0 mm
/run/beamOn 200"""},
    "C": {"name":"Carbon Ion", "color":"Orange", "LET":"Very high LET, dense orange tracks with Bragg peak cluster",
          "block":"""/gps/particle ion
/gps/ion 6 12
/gps/energy 3960 MeV
/gps/pos/type Volume
/gps/pos/shape Sphere
/gps/pos/radius 15 mm
/gps/pos/centre 0 0 0 mm
/run/beamOn 50"""},
    "A": {"name":"Alpha",      "color":"Red",    "LET":"Highest LET, short dense red tracks",
          "block":"""/gps/particle alpha
/gps/energy 5.5 MeV
/gps/pos/type Volume
/gps/pos/shape Sphere
/gps/pos/radius 15 mm
/gps/pos/centre 0 0 0 mm
/run/beamOn 200"""},
}

def make_mac(title, desc, blocks, filename):
    lines = [f"# {title}", f"# {desc}",
             f"# Colors: Gamma=Blue  Neutron=Green  Carbon=Orange  Alpha=Red  Proton=Yellow  Electron=Magenta",
             "", VIS_HEADER, ""]
    for b in blocks:
        lines.append(b); lines.append("")
    lines.append(VIS_FOOTER)
    path = os.path.join("simulations", filename)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Created: {filename}")

def main():
    print("Individual rays:")
    for sym, p in PARTICLES.items():
        make_mac(f"Individual: {p['name']}", p["LET"],
                 [f"# {p['name']} -- {p['color']}", p["block"]],
                 f"vis_individual_{sym}_{p['name'].replace(' ','_')}.mac")

    print("\nAll 4 individually in one window:")
    blocks = []
    for sym,p in PARTICLES.items():
        blocks += [f"# {p['name']} ({p['color']})", p["block"]]
    make_mac("All 4 Individual Rays", "All 4 fire sequentially -- watch how each type looks different",
             blocks, "vis_all_individual.mac")

    print("\n3-ray combinations:")
    combos = [
        ("GNA","Gamma + Neutron + Alpha -- WINNER COMBO",
         "Gamma saturates repair, Neutron scatters secondaries, Alpha devastates core",
         ["G","N","A"],"vis_3ray_GNA_best_order.mac"),
        ("GNC","Gamma + Neutron + Carbon Ion",
         "Carbon Bragg peak dense orange cluster -- Gamma covers entry track weakness",
         ["C","G","N"],"vis_3ray_GNC_best_order.mac"),
        ("GCA","Gamma + Carbon Ion + Alpha -- Windshield Effect",
         "Gamma opens path, Carbon peaks deep, Alpha follows into pre-damaged tissue",
         ["G","C","A"],"vis_3ray_GCA_best_order.mac"),
        ("NCA","Neutron + Carbon Ion + Alpha",
         "Neutron secondaries flood volume first, Carbon precision, Alpha finishes",
         ["C","N","A"],"vis_3ray_NCA_best_order.mac"),
    ]
    for _,title,desc,order,fname in combos:
        blocks = []
        for sym in order:
            p=PARTICLES[sym]
            blocks += [f"# {p['name']} ({p['color']})", p["block"]]
        make_mac(title, desc, blocks, fname)

    print("\n4-ray best order:")
    blocks = []
    for sym in ["G","N","C","A"]:
        p=PARTICLES[sym]
        blocks += [f"# {p['name']} ({p['color']})", p["block"]]
    make_mac("All 4 Rays BEST ORDER: Gamma -> Neutron -> Carbon -> Alpha",
             "Windshield effect -- each ray builds on the last",
             blocks, "vis_4ray_GNCA_best_order.mac")

    print("\n4-ray worst order (comparison):")
    blocks = []
    for sym in ["C","N","A","G"]:
        p=PARTICLES[sym]
        blocks += [f"# {p['name']} ({p['color']})", p["block"]]
    make_mac("All 4 Rays WORST ORDER: Carbon -> Neutron -> Alpha -> Gamma",
             "Compare to best order -- shows why sequence matters",
             blocks, "vis_4ray_CNAG_worst_order.mac")

    print("\n" + "="*55)
    print("All done! Run any with:")
    print("  gears simulations/vis_individual_G_Gamma.mac")
    print("  gears simulations/vis_individual_N_Neutron.mac")
    print("  gears simulations/vis_individual_C_Carbon_Ion.mac")
    print("  gears simulations/vis_individual_A_Alpha.mac")
    print("  gears simulations/vis_all_individual.mac")
    print("  gears simulations/vis_3ray_GNA_best_order.mac  <- WINNER")
    print("  gears simulations/vis_3ray_GNC_best_order.mac")
    print("  gears simulations/vis_3ray_GCA_best_order.mac")
    print("  gears simulations/vis_3ray_NCA_best_order.mac")
    print("  gears simulations/vis_4ray_GNCA_best_order.mac")
    print("  gears simulations/vis_4ray_CNAG_worst_order.mac")

if __name__ == "__main__":
    main()
