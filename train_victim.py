"""Train a lone PPO victim.

Usage (from the folder that contains predation/):
    .venv/bin/python -m predation.train_victim predation/configs/victim_unconstrained.yaml --seed 0
    ... --set env.B=512 train.iterations=200     # override any config field
    ... --iterations 200 --B 512 --T 60 --name my_run   # shortcuts for the common ones
For several seeds in parallel use predation.launch.

Writes predation/runs/<name>/seed<seed>/:
    config.yaml   resolved config incl. seed (what was actually run)
    log.csv       one row per iteration
    model.pt      actor + critic weights
    summary.json  final evaluation vs. constant baselines
    training.png  learning curves
    events.*      TensorBoard scalars, updated live during training:
                  .venv/bin/tensorboard --logdir predation/runs
"""
import argparse
import copy
import csv
import dataclasses
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # tensorboard not installed: training works, only live plots are off
    SummaryWriter = None

from .agents.ppo import PPO, PPOConfig, collect_rollout
from .envs import VecMarket
from .plotting import plot_training

RUNS_DIR = Path(__file__).resolve().parent / "runs"
EVAL_SEED_OFFSET = 10_000
LOG_FIELDS = ["iteration", "pnl_mean", "mean_frac", "ep_cost", "std", "approx_kl", "clip_frac",
              "policy_loss", "value_loss", "entropy", "time_s"]
# log.csv column -> TensorBoard tag (the prefix groups the plots)
TB_TAGS = {"pnl_mean": "victim/pnl", "mean_frac": "victim/mean_frac", "ep_cost": "victim/ep_cost",
           "std": "policy/std", "entropy": "policy/entropy", "approx_kl": "ppo/approx_kl",
           "clip_frac": "ppo/clip_frac", "policy_loss": "loss/policy", "value_loss": "loss/value"}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_value(text: str):
    """'512' -> 512, '3e-4' -> 0.0003, 'null' -> None, 'true' -> True, 'abc' -> 'abc'."""
    value = yaml.safe_load(text)
    if isinstance(value, str):  # YAML reads '3e-4' (no dot) as a string
        try:
            return float(value)
        except ValueError:
            pass
    return value


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """['env.B=512', 'train.iterations=50'] -> copy of cfg with these fields replaced."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        key, sep, text = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        *path, last = key.split(".")
        node = cfg
        for part in path:
            node = node.setdefault(part, {})
        node[last] = parse_value(text)
    return cfg


# shortcut flag -> config field it sets
SHORTCUTS = {"iterations": "train.iterations", "B": "env.B", "T": "env.T", "name": "name"}


def add_override_args(parser: argparse.ArgumentParser):
    """--set KEY=VALUE ... plus shortcut flags for the most common fields."""
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                        help="override config fields, e.g. env.mu=5e-4 ppo.lr=1e-3")
    group = parser.add_argument_group("shortcuts for --set")
    group.add_argument("--iterations", type=int, help="= --set train.iterations=N")
    group.add_argument("--B", type=int, help="= --set env.B=N (parallel episodes per iteration)")
    group.add_argument("--T", type=int, help="= --set env.T=N (steps per episode)")
    group.add_argument("--name", help="= --set name=NAME (folder runs/NAME/)")


def collect_overrides(args) -> list[str]:
    """--set entries plus the shortcut flags, as one list of 'key=value' strings."""
    overrides = list(args.set)
    set_keys = {item.partition("=")[0] for item in overrides}
    for flag, key in SHORTCUTS.items():
        value = getattr(args, flag)
        if value is None:
            continue
        if key in set_keys:
            raise ValueError(f"--{flag} and --set {key}=... both given")
        overrides.append(f"{key}={value}")
    return overrides


def validate_config(cfg: dict):
    """Fail early on typos such as 'evn.B' or 'ppo.learning_rate'."""
    allowed = {
        "env": set(inspect.signature(VecMarket.__init__).parameters) - {"self", "seed"},
        "ppo": {f.name for f in dataclasses.fields(PPOConfig)},
        "train": {"iterations", "log_every", "eval_episodes"},
    }
    unknown = set(cfg) - set(allowed) - {"name"}
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    for section, keys in allowed.items():
        bad = set(cfg.get(section, {})) - keys
        if bad:
            raise ValueError(f"unknown keys in '{section}': {sorted(bad)}; allowed: {sorted(keys)}")


@torch.no_grad()
def evaluate(policy, env_kwargs: dict, episodes: int, seed: int) -> dict:
    """Run `policy` (numpy obs (B, obs_dim) -> numpy action (B,)) on fresh episodes."""
    env = VecMarket(**{**env_kwargs, "B": episodes, "seed": seed})
    obs, cost, frac = env.reset(), np.zeros(episodes), []
    for _ in range(env.T):
        obs, _, c, _, _ = env.step(policy(obs))
        cost += c
        frac.append(env.pos_v / env.max_pos)
    pnl = env._equity(env.cash_v, env.pos_v) - env.cash0
    return {"pnl_mean": float(pnl.mean()), "pnl_ci95": float(1.96 * pnl.std() / np.sqrt(episodes)),
            "mean_frac": float(np.mean(frac)), "ep_cost": float(cost.mean())}


def train(cfg: dict, seed: int, run_dir: Path) -> PPO:
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = VecMarket(**cfg["env"], seed=seed)
    agent = PPO(env.obs_dim, PPOConfig(**cfg["ppo"]))
    n_iter, log_every = cfg["train"]["iterations"], cfg["train"]["log_every"]

    print(f"{cfg['name']} | seed {seed} | {n_iter} iterations x {env.B} episodes -> {run_dir}")
    print(f"{'iter':>5} {'PnL':>8} {'frac':>6} {'cost':>6} {'std':>6} {'kl':>7} {'clip':>6} {'time':>6}")
    tb = SummaryWriter(log_dir=str(run_dir), flush_secs=5) if SummaryWriter else None
    t0 = time.time()
    with open(run_dir / "log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        for it in range(n_iter):
            batch = collect_rollout(env, agent)
            stats = agent.update(batch)
            row = {"iteration": it, "pnl_mean": float(batch["pnl"].mean()),
                   "mean_frac": float(batch["mean_frac"].mean()),
                   "ep_cost": float(batch["ep_cost"].mean()), "time_s": time.time() - t0, **stats}
            writer.writerow(row)
            f.flush()  # so that log.csv can be read while training
            if tb:
                for key, tag in TB_TAGS.items():
                    tb.add_scalar(tag, row[key], it)
            if it % log_every == 0 or it == n_iter - 1:
                print(f"{it:5d} {row['pnl_mean']:8.0f} {row['mean_frac']:6.3f} {row['ep_cost']:6.2f} "
                      f"{row['std']:6.3f} {row['approx_kl']:7.4f} {row['clip_frac']:6.3f} "
                      f"{row['time_s']:5.0f}s", flush=True)
    if tb:
        tb.close()
    agent.save(run_dir / "model.pt")
    return agent


def main():
    parser = argparse.ArgumentParser(description="Train a victim alone with PPO.")
    parser.add_argument("config", help="path to a YAML config, e.g. predation/configs/victim_unconstrained.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="overwrite an existing run directory")
    add_override_args(parser)
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), collect_overrides(args))
    validate_config(cfg)
    run_dir = RUNS_DIR / cfg["name"] / f"seed{args.seed}"
    if run_dir.exists() and not args.force:
        sys.exit(f"{run_dir} already exists; use --force to overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)
    for old in run_dir.glob("events.out.tfevents.*"):  # --force: drop curves of the previous run
        old.unlink()
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump({**cfg, "seed": args.seed}, f, sort_keys=False)

    agent = train(cfg, args.seed, run_dir)

    episodes, eval_seed = cfg["train"]["eval_episodes"], args.seed + EVAL_SEED_OFFSET
    policies = {
        "trained victim": lambda o: agent.act(o, deterministic=True)[1].numpy(),
        "always-max": lambda o: np.ones(len(o)),
        "always-half": lambda o: np.zeros(len(o)),
    }
    summary = {name: evaluate(p, cfg["env"], episodes, eval_seed) for name, p in policies.items()}
    with open(run_dir / "summary.json", "w") as f:
        json.dump({"seed": args.seed, "eval_seed": eval_seed, "eval_episodes": episodes, **summary}, f, indent=2)
    plot_training(run_dir / "log.csv", run_dir / "training.png", f"{cfg['name']}, seed {args.seed}")

    print(f"\nEvaluation on {episodes} fresh episodes (deterministic policy, eval seed {eval_seed}):")
    print(f"  {'':16} {'PnL (95% CI)':>16} {'frac':>7} {'cost':>6}")
    for name, s in summary.items():
        print(f"  {name:16} {s['pnl_mean']:8.0f} ± {s['pnl_ci95']:<5.0f} {s['mean_frac']:7.3f} {s['ep_cost']:6.2f}")
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()
