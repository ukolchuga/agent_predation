
from dataclasses import dataclass, asdict
import torch.nn as nn
import numpy as np
import torch

from .networks import Critic
from .ppo import PPO, PPOConfig


@dataclass
class LagrangianConfig(PPOConfig):
    cost_budget: float = 0.6     # allowed mean episode cost (always-half has ~0.6)
    lambda_lr: float = 0.05      # step size of the multiplier update
    lambda_init: float = 0.0     # start without a penalty
    cost_scale: float = 1.0      # episode cost is already O(1)


class PPOLagrangian(PPO):
    def __init__(self, obs_dim: int, cfg: LagrangianConfig = LagrangianConfig()):
        super().__init__(obs_dim, cfg)
        self.cost_critic = Critic(obs_dim, cfg.hidden)
        # one optimizer over actor, reward critic and cost critic
        self.params = self.params + list(self.cost_critic.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=cfg.lr)
        self.lagrange = cfg.lambda_init  # lam; plain float, it is not trained by the optimizer

    def update_lambda(self, mean_episode_cost: float) -> float:

        self.lagrange = max(0, self.lagrange + self.cfg.lambda_lr * (mean_episode_cost - self.cfg.cost_budget))
        return self.lagrange

    def update(self, batch: dict) -> dict:

        cfg = self.cfg
        with torch.no_grad():
            cost_values = self.cost_critic(batch["obs"])

        adv_r, ret_r = self.compute_gae(batch["rewards"] * cfg.reward_scale, batch["values"])
        adv_c, ret_c = self.compute_gae(batch["costs"] * cfg.cost_scale, cost_values)
        adv = adv_r - self.lagrange * adv_c
        obs = batch["obs"].flatten(0, 1)
        actions, old_logp = batch["actions"].flatten(), batch["logp"].flatten()
        adv, ret_r, ret_c = adv.flatten(), ret_r.flatten(), ret_c.flatten()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = obs.shape[0]
        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": [], "cost_value_loss":[]}
        for _ in range(cfg.epochs):
            for idx in torch.randperm(n).split(cfg.minibatch):
                dist = self.actor(obs[idx])
                logp = dist.log_prob(actions[idx])
                ratio = torch.exp(logp - old_logp[idx])
                surr1 = ratio * adv[idx]
                surr2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((self.critic(obs[idx]) - ret_r[idx]) ** 2).mean()
                cost_value_loss = ((self.cost_critic(obs[idx]) - ret_c[idx]) ** 2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + cfg.vf_coef * value_loss + cost_value_loss * cfg.vf_coef - cfg.ent_coef * entropy

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
                    stats["cost_value_loss"].append(cost_value_loss.item())

        mean_ep_cost = batch["ep_cost"].mean()
        self.update_lambda(mean_ep_cost)

        out = {k: float(np.mean(v)) for k, v in stats.items()}
        out["std"] = self.actor.log_std.exp().item()
        out["ep_cost"] = mean_ep_cost
        out["lagrange"] = self.lagrange
        return out

    def save(self, path: str):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "cost_critic": self.cost_critic.state_dict(), "lagrange": float(self.lagrange),
                    "cfg": asdict(self.cfg)}, path)

    def load(self, path: str):
        state = torch.load(path)
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.cost_critic.load_state_dict(state["cost_critic"])
        self.lagrange = state["lagrange"]
