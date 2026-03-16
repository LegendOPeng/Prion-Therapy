"""
run_all.py — Surgical Tool Full Pipeline
Runs Geant4 simulations then all analysis steps in order.
Usage: python3 run_all.py
       python3 run_all.py --skip-sim   (skip Geant4, use existing CSVs)
       python3 run_all.py --step 2     (run only a specific step)
"""
import subprocess, os, sys, time, shutil

ROOT    = os.path.dirname(os.path.abspath(__file__))
GEARS   = "/Users/penguin/geant4/11.4.0/bin/gears"
MACS    = os.path.join(ROOT, "macs")
DATA    = os.path.join(ROOT, "data")
SCRIPTS = os.path.join(ROOT, "scripts")

RAYS = ["gamma", "neutron", "carbon", "alpha"]

def banner(msg):
    print("\n" + "="*65)
    print(f"  {msg}")
    print("="*65)

def run_geant4():
    banner("GEANT4 SIMULATIONS")
    os.makedirs(DATA, exist_ok=True)

    for ray in RAYS:
        mac = os.path.join(MACS, f"{ray}.mac")
        out_csv = os.path.join(DATA, f"{ray}_edep.csv")

        print(f"\n  [{ray.upper()}] Running {mac}")
        t0 = time.time()

        # gears writes CSV to cwd, so run from DATA dir
        result = subprocess.run(
            [GEARS, mac],
            cwd=DATA,
            capture_output=True,
            text=True
        )

        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"  [ERROR] gears failed for {ray}:")
            print(result.stderr[-500:])
        elif os.path.exists(out_csv):
            lines = sum(1 for _ in open(out_csv))
            print(f"  [OK] {out_csv}  ({lines} lines, {elapsed:.1f}s)")
        else:
            print(f"  [WARN] {out_csv} not created — check mac file")
            print(result.stdout[-300:])

def run_step(n, script_name, label):
    banner(f"STEP {n} — {label}")
    script = os.path.join(SCRIPTS, script_name)
    if not os.path.exists(script):
        print(f"  [SKIP] {script} not found")
        return False
    result = subprocess.run(
        [sys.executable, script],
        cwd=ROOT,
        capture_output=False   # show output live
    )
    if result.returncode != 0:
        print(f"\n  [ERROR] Step {n} failed (exit {result.returncode})")
        return False
    return True

def main():
    skip_sim  = "--skip-sim" in sys.argv
    only_step = None
    if "--step" in sys.argv:
        idx = sys.argv.index("--step")
        only_step = int(sys.argv[idx + 1])

    banner("SURGICAL TOOL — FULL PIPELINE")
    print(f"  Root:  {ROOT}")
    print(f"  Gears: {GEARS}")
    print(f"  Mode:  {'skip sim' if skip_sim else 'full run'}")

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
            print(f"\n  Pipeline stopped at Step {n}. Fix errors then re-run.")
            sys.exit(1)

    banner("PIPELINE COMPLETE")
    print(f"  Outputs in: {os.path.join(ROOT, 'Steps')}/")

if __name__ == "__main__":
    main()