import sys, os, time
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

def run_step(n):
    t0 = time.time()
    print(f"\n{'='*65}\n  RUNNING STEP {n}\n{'='*65}")
    if n == 1:
        from step1_rays import run_step1; run_step1()
    elif n == 2:
        from step2_optimize import run_step2; run_step2()
    elif n == 3:
        from step3_rerun import run_step3; run_step3()
    elif n == 4:
        from step4_combinations import run_step4; run_step4()
    elif n == 5:
        from step5_firing_orders import run_step5; run_step5()
    elif n == 6:
        from step6_final_protocol import run_step6; run_step6()
    elif n == 7:
        from step7_final_report import run_step7; run_step7()
    print(f"\n  Step {n} done in {time.time()-t0:.1f}s")

def main():
    args = sys.argv[1:]
    if not args:
        steps = list(range(1, 8))
    elif args[0] == "--from":
        steps = list(range(int(args[1]), 8))
    else:
        steps = [int(a) for a in args if a.isdigit()]
    print(f"\n{'#'*65}\n  SURGICAL TOOL PIPELINE — Steps: {steps}\n{'#'*65}")
    t_total = time.time()
    for step in steps:
        try:
            run_step(step)
        except Exception as e:
            import traceback
            print(f"\n  [ERROR] Step {step} failed: {e}")
            traceback.print_exc()
    print(f"\n  All done! {time.time()-t_total:.1f}s — Results in Steps/ and data/stats/")

if __name__ == "__main__":
    main()
