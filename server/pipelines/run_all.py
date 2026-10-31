"""Run the offline engine pipeline in order: ingest -> precompute_pc -> train -> build_events -> predict_test -> build_scenarios.

Usage: cd server && .venv/bin/python pipelines/run_all.py [--skip-train]
"""
try:
    from . import _boot  # noqa: F401
except ImportError:
    import _boot  # noqa: F401
import sys
import time

try:
    from . import build_events, build_scenarios, ingest, precompute_pc, predict_test, train
except ImportError:
    import build_events, build_scenarios, ingest, precompute_pc, predict_test, train  # noqa: E401


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    t0 = time.time()
    steps = [("ingest", ingest.main), ("precompute_pc", precompute_pc.main), ("train", train.main),
             ("build_events", build_events.main), ("predict_test", predict_test.main),
             ("build_scenarios", build_scenarios.main)]
    for name, fn in steps:
        if name == "train" and "--skip-train" in argv:
            print("[run_all] skipping train (using existing registry)")
            continue
        print(f"[run_all] {name} ...", flush=True)
        fn()
    print(f"[run_all] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
