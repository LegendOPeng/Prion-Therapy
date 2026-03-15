import subprocess, os, sys, time

ROOT    = os.path.dirname(os.path.abspath(__file__))
GEARS   = "/Users/penguin/geant4/11.4.0/bin/gears"
MACS    = os.path.join(ROOT, "macs")
DATA    = os.path.join(ROOT, "data")
SCRIPTS = os.path.join(ROOT, "scripts")
RAYS    = ["gamma", "neutron", "carbon", "alpha"]

def banner(msg):
    print("\n" + "="*65)
    print(f"  {msg}")
    print("="*65)

def run_geant4():
    banner("GEANT4 SIMULATIONS")
    os.makedirs(DATA, exist_ok=True)
    for ray in RAYS:
        mac     = os.path.join(MACS, f"{ray}.mac")
        out_csv = os.path.join(DATA, f"{ray}_edep.csv")
        print(f"\n  [{ray.upper()}] running...")
        t0 = time.time()
        # Run from ROOT so relative paths in mac files resolve correctly
        r  = subprocess.run([GEARS, mac], cwd=ROOT, capture_output=True, text=True)
        elapsed = time.time() - t0
        # gears writes CSV to cwd — move it to data/
        produced = os.path.join(ROOT, f"{ray}_edep.csv")
        if os.path.exists(produced):
            os.replace(produced, out_csv)
        if r.returncode != 0:
            print(f"  [ERROR]\n{r.stderr[-600:]}")
        elif os.path.exists(out_csv):
            n = sum(1 for _ in open(out_csv))
            print(f"  [OK] {n} lines  {elapsed:.1f}s")
        else:
            print(f"  [WARN] CSV not created")
            print(r.stdout[-400:])

def run_step(n, script, label):
    banner(f"STEP {n} — {label}")
    path = os.path.join(SCRIPTS, script)
    if not os.path.exists(path):
        print(f"  [SKIP] {script} not found")
        return False
    r = subprocess.run([sys.executable, path], cwd=ROOT)
    if r.returncode != 0:
        print(f"  [ERROR] Step {n} failed")
        return False
    return True

def main():
    skip_sim  = "--skip-sim" in sys.argv
    only_step = None
    if "--step" in sys.argv:
        only_step = int(sys.argv[sys.argv.index("--step") + 1])

    banner("SURGICAL TOOL — FULL PIPELINE")
    print(f"  Root:  {ROOT}")
    print(f"  Gears: {GEARS}")

    if not skip_sim and only_step is None:
        run_geant4()

    steps = [
        (1, "step1_rays.py",          "Individual Ray Analysis"),
        (2, "step2_optimize.py",       "Beam Count Optimization"),
        (3, "step3_rerun.py",          "Validation Reruns"),
        (4, "step4_combinations.py",   "Multi-Ray Combinations"),
        (5, "step5_firing_orders.py",  "Firing Order Optimization"),
        (6, "step6_final_protocol.py", "Final Protocol Validation"),
        (7, "step7_final_report.py",   "Master Summary Report"),
    ]
    for n, script, label in steps:
        if only_step is not None and n != only_step:
            continue
        ok = run_step(n, script, label)
        if not ok and only_step is None:
            print(f"\n  Stopped at Step {n}.")
            sys.exit(1)

    banner("PIPELINE COMPLETE")
    print(f"  Outputs → {ROOT}/Steps/")

if __name__ == "__main__":
    main()
