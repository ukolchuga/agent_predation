# Exploitation of Constrained Agents by Self-Interested Opponents

Code for the bachelor thesis: *a multi-agent reinforcement learning study in a simulated market*.

**Question.** A trading agent that obeys a risk constraint (stop-loss or drawdown budget) reacts
predictably under pressure: it sells. Can a self-interested opponent learn to push the price,
trigger that reaction and profit from it? Is a hard constraint (shield) or a soft one
(PPO-Lagrangian) more exploitable, and can the victim learn to defend itself when both are
trained together?

**Setup.**
- Synthetic one-asset market (Almgren-Chriss style): drift, noise, permanent and temporary price
  impact, equity marked at liquidation value. 120 steps, many episodes in parallel.
- Victims (PPO): unconstrained (control), shield (hard stop-loss), PPO-Lagrangian (soft budget).
- Predator (PPO): rewarded only by its own profit, does not see the victim's hidden state.
- Training: each victim alone, predator as best response to a frozen victim, then co-training
  with independent PPO.

**Status.** Market and PPO done; the unconstrained victim learns to hold the maximum position
(5 seeds, within the confidence interval of the always-max baseline). Next: shield and
Lagrangian victims, predator, co-training.

## Layout

```
envs/vec_env.py           vectorised two-agent market
agents/networks.py        actor (Gaussian policy) and critic
agents/ppo.py             PPO and collect_rollout()
train_victim.py           train one victim, one seed: config -> runs/<name>/seed<N>/
launch.py                 several seeds in parallel + mean/std over seeds
plotting.py               learning curves
configs/                  one YAML per experiment
scripts/                  start training on the server
```

## Quick start

Run from the folder that **contains** `predation/`:

```bash
python -m venv .venv && .venv/bin/pip install -r predation/requirements.txt
.venv/bin/python -m predation.train_victim predation/configs/victim_unconstrained.yaml --seed 0
.venv/bin/tensorboard --logdir predation/runs        # live curves at http://localhost:6006
```

Several seeds in parallel; `--iterations`, `--B`, `--T`, `--name` are shortcuts, any other
config field can be overridden with `--set`:

```bash
.venv/bin/python -m predation.launch predation/configs/victim_unconstrained.yaml \
    --seeds 0-4 --workers 4 --iterations 150 --B 512 --T 200 --name victim_big --set ppo.lr=1e-3
```

Same on the server (after `git push svoi master`), as a CPU-capped background job:

```bash
predation/scripts/train_on_server.sh --seeds 0-4 --iterations 150
```

Each run writes `config.yaml`, `log.csv`, `model.pt`, `summary.json`, `training.png` and
TensorBoard events; `launch` adds `summary_over_seeds.json`. Same seed on the same machine
gives identical results.
