"""PPO for one agent in the vectorised market."""
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn

from .networks import Actor, Critic


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99          # discount
    lam: float = 0.95            # GAE lambda
    clip: float = 0.2            # PPO ratio clip
    epochs: int = 10             # passes over one rollout
    minibatch: int = 2048        # samples per gradient step
    vf_coef: float = 0.5         # weight of the value loss
    ent_coef: float = 0.0        # weight of the entropy bonus
    max_grad_norm: float = 0.5
    reward_scale: float = 100.0
    hidden: int = 64

class PPO:
    def __init__(self, obs_dim: int, cfg: PPOConfig = PPOConfig()):
        self.cfg = cfg
        self.actor = Actor(obs_dim, cfg.hidden)
        self.critic = Critic(obs_dim, cfg.hidden)
        self.params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=cfg.lr)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        dist = self.actor(obs)
        action = dist.mean if deterministic else dist.sample()
        return obs, action, dist.log_prob(action), self.critic(obs)

    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor):
        T = rewards.shape[0]
        adv = torch.zeros_like(rewards)
        last = torch.zeros_like(rewards[0])
        for t in reversed(range(T)):
            next_value = values[t + 1] if t + 1 < T else torch.zeros_like(values[t])
            delta = rewards[t] + self.cfg.gamma * next_value - values[t]
            last = delta + self.cfg.gamma * self.cfg.lam * last
            adv[t] = last
        return adv, adv + values

    def update(self, batch: dict) -> dict:
        """batch from collect_rollout() -> training statistics"""
        cfg = self.cfg
        adv, ret = self.compute_gae(batch["rewards"] * cfg.reward_scale, batch["values"])
        obs = batch["obs"].flatten(0, 1)
        actions, old_logp = batch["actions"].flatten(), batch["logp"].flatten()
        adv, ret = adv.flatten(), ret.flatten()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = obs.shape[0]
        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}
        for _ in range(cfg.epochs):
            for idx in torch.randperm(n).split(cfg.minibatch):
                dist = self.actor(obs[idx])
                logp = dist.log_prob(actions[idx])
                ratio = torch.exp(logp - old_logp[idx])
                surr1 = ratio * adv[idx]
                surr2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((self.critic(obs[idx]) - ret[idx]) ** 2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    stats["policy_loss"].append(policy_loss.item())
                    stats["value_loss"].append(value_loss.item())
                    stats["entropy"].append(entropy.item())
                    stats["approx_kl"].append((old_logp[idx] - logp).mean().item())
                    stats["clip_frac"].append(((ratio - 1).abs() > cfg.clip).float().mean().item())
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        out["std"] = self.actor.log_std.exp().item()
        return out

    def save(self, path: str):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "cfg": asdict(self.cfg)}, path)

    def load(self, path: str):
        state = torch.load(path)
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])


def collect_rollout(env, agent: PPO) -> dict:
    """Plays one full episode on all B parallel markets"""
    T, B = env.T, env.B
    batch = {"obs": torch.zeros(T, B, env.obs_dim)}
    for k in ("actions", "logp", "values", "rewards", "costs"):
        batch[k] = torch.zeros(T, B)
    frac = np.zeros((T, B))

    obs_np = env.reset()
    for t in range(T):
        obs, action, logp, value = agent.act(obs_np)
        obs_np, r_v, cost, _, _ = env.step(action.numpy())
        batch["obs"][t], batch["actions"][t], batch["logp"][t], batch["values"][t] = obs, action, logp, value
        batch["rewards"][t] = torch.as_tensor(r_v, dtype=torch.float32)
        batch["costs"][t] = torch.as_tensor(cost, dtype=torch.float32)
        frac[t] = env.pos_v / env.max_pos

    batch["pnl"] = env._equity(env.cash_v, env.pos_v) - env.cash0
    batch["ep_cost"] = batch["costs"].sum(0).numpy()
    batch["mean_frac"] = frac.mean(0)
    return batch
