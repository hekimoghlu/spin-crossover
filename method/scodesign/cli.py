"""Command-line entry points.

  python3 -m scodesign run        [--quick] [--backend auto|cpp|numpy]   full pipeline -> results/results.json
  python3 -m scodesign figures                                           figures/reimpl_fig*.pdf|png
  python3 -m scodesign qasm                                              qasm/ export + verification
  python3 -m scodesign validate                                          VALIDATION.md
  python3 -m scodesign all                                               run + figures + qasm + validate
"""
import argparse
import json
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def cmd_run(args):
    from .loop import Config, Loop
    cfg = Config(backend=args.backend)
    if args.quick:
        cfg = Config(backend=args.backend, n_epochs=3, n_epochs_cal=4, rsem_epochs=3, budget_subsets=5,
                     rough_reps=4, rough_seeds=1, eps_epochs=2, quick=True,
                     rdm_steps=300, rl_episodes=3000)
    loop = Loop(cfg)
    R = loop.run(out_dir=RESULTS)
    print(f"[run] done in {R['timing']['total']:.1f} s -> {RESULTS}/results.json")
    return loop


def cmd_figures(args):
    from . import figures
    figures.make_all(os.path.join(RESULTS, "results.json"), os.path.join(ROOT, "figures"))


def cmd_qasm(args):
    from . import qasm
    qasm.export_and_verify(os.path.join(RESULTS, "results.json"), os.path.join(ROOT, "qasm"))


def cmd_validate(args):
    from . import validation
    validation.write(os.path.join(RESULTS, "results.json"), os.path.join(ROOT, "VALIDATION.md"), args.paper)


def main():
    ap = argparse.ArgumentParser(prog="scodesign")
    ap.add_argument("command", choices=["run", "figures", "qasm", "validate", "all"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--backend", default="auto", choices=["auto", "cpp", "numpy", "cuda"])
    ap.add_argument("--paper", default=os.environ.get("SCODESIGN_PAPER", ""))
    args = ap.parse_args()
    t0 = time.time()
    if args.command in ("run", "all"):
        cmd_run(args)
    if args.command in ("figures", "all"):
        cmd_figures(args)
    if args.command in ("qasm", "all"):
        cmd_qasm(args)
    if args.command in ("validate", "all"):
        cmd_validate(args)
    print(f"[scodesign] {args.command} finished in {time.time() - t0:.1f} s")
