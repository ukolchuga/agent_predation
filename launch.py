"""Train one config for several seeds in parallel (one process and one CPU thread per seed).

Usage (from anywhere; relative config paths are also looked up next to predation/):
    .venv/bin/python -m predation.launch predation/configs/victim_unconstrained.yaml \\
        --seeds 0-4 --workers 4 --iterations 150 --B 512 --T 200 --name victim_big

    --seeds       list and/or ranges: "0-4", "0 1 2", "0-2 7"         (default 0-4)
    --workers     processes at the same time                          (default: #seeds, max #cores)
    --iterations  training iterations         (= --set train.iterations=N)
    --B           parallel episodes per iteration (= --set env.B=N)
    --T           steps per episode           (= --set env.T=N)
    --name        run folder runs/<name>/     (= --set name=NAME)
    --set         any other config field: env.*, ppo.*, train.*       (see the YAML file)
    --force       overwrite runs that already exist

Per seed: runs/<name>/seed<N>/ (see train_victim.py), console output in runs/<name>/logs/.
At the end: mean and std over seeds in runs/<name>/summary_over_seeds.json.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .train_victim import (RUNS_DIR, add_override_args, apply_overrides, collect_overrides,
                           load_config, validate_config)

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # folder that contains predation/
DEFAULT_CONFIG = "predation/configs/victim_unconstrained.yaml"
POLICIES = ["trained victim", "always-max", "always-half"]


def parse_seeds(tokens: list[str]) -> list[int]:
    """['0-2', '7'] -> [0, 1, 2, 7]"""
    seeds = []
    for tok in tokens:
        lo, _, hi = tok.partition("-")
        seeds += list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]
    return sorted(set(seeds))


def resolve_config(path: str) -> Path:
    p = Path(path)
    if not p.exists() and (PROJECT_ROOT / p).exists():
        p = PROJECT_ROOT / p
    if not p.exists():
        sys.exit(f"config not found: {path}")
    return p.resolve()


def run_seed(seed: int, config: Path, overrides: list[str], force: bool, log_dir: Path):
    cmd = [sys.executable, "-m", "predation.train_victim", str(config), "--seed", str(seed)]
    if overrides:
        cmd += ["--set", *overrides]
    if force:
        cmd.append("--force")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    t0 = time.time()
    with open(log_dir / f"seed{seed}.log", "w") as log:
        code = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    return seed, code, time.time() - t0


def summarize(run_root: Path, seeds: list[int]) -> dict:
    """Mean and std over seeds of the final evaluation of every policy."""
    runs = [json.loads((run_root / f"seed{s}" / "summary.json").read_text()) for s in seeds]
    out = {"seeds": seeds}
    for policy in POLICIES:
        out[policy] = {}
        for metric in ["pnl_mean", "mean_frac", "ep_cost"]:
            values = np.array([r[policy][metric] for r in runs])
            out[policy][metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    return out


def main():
    parser = argparse.ArgumentParser(description="Train one config for several seeds in parallel.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", nargs="+", default=["0-4"])
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    add_override_args(parser)
    args = parser.parse_args()

    config = resolve_config(args.config)
    overrides = collect_overrides(args)
    cfg = apply_overrides(load_config(config), overrides)
    validate_config(cfg)
    seeds = parse_seeds(args.seeds)
    workers = args.workers or min(len(seeds), os.cpu_count() or 1)

    run_root = RUNS_DIR / cfg["name"]
    existing = [s for s in seeds if (run_root / f"seed{s}").exists()]
    if existing and not args.force:
        sys.exit(f"runs for seeds {existing} already exist in {run_root}; use --force or --name ...")
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    env, train = cfg.get("env", {}), cfg["train"]
    print(f"{cfg['name']} | seeds {seeds} | {workers} in parallel | "
          f"B={env.get('B', 'default')} T={env.get('T', 'default')} iterations={train['iterations']}")
    if overrides:
        print("overrides:", " ".join(overrides))
    print(f"logs: {log_dir}", flush=True)

    t0, failed = time.time(), []
    with ThreadPoolExecutor(workers) as pool:
        jobs = [pool.submit(run_seed, s, config, overrides, args.force, log_dir) for s in seeds]
        for job in as_completed(jobs):
            seed, code, sec = job.result()
            status = "ok" if code == 0 else f"FAILED (exit {code}), see {log_dir / f'seed{seed}.log'}"
            print(f"  seed {seed} done in {sec:5.0f}s  {status}", flush=True)
            if code != 0:
                failed.append(seed)
    print(f"all done in {time.time() - t0:.0f}s")

    done = [s for s in seeds if s not in failed]
    if done:
        summary = summarize(run_root, done)
        (run_root / "summary_over_seeds.json").write_text(json.dumps(summary, indent=2))
        print(f"\nEvaluation, mean ± std over {len(done)} seeds:")
        print(f"  {'':16} {'PnL':>16} {'frac':>14} {'cost':>12}")
        for policy in POLICIES:
            m = summary[policy]
            print(f"  {policy:16} {m['pnl_mean']['mean']:8.0f} ± {m['pnl_mean']['std']:<5.0f} "
                  f"{m['mean_frac']['mean']:7.3f} ± {m['mean_frac']['std']:.3f} "
                  f"{m['ep_cost']['mean']:6.2f} ± {m['ep_cost']['std']:.2f}")
        print(f"\nSaved to {run_root}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
