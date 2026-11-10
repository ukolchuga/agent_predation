"""PPO-Lagrangian: PPO with a constraint on the expected episode cost (skeleton).

Constrained MDP:   max_pi J_r(pi)   subject to   J_c(pi) <= cost_budget
Lagrangian:        L(pi, lam) = J_r(pi) - lam * (J_c(pi) - cost_budget),   lam >= 0

The policy ascends L (advantage A_r - lam * A_c), the multiplier descends it
(lam grows while the constraint is violated, shrinks towards 0 while it is satisfied).

Differences to plain PPO, everything else is inherited:
  1. a second critic V_c(s) that predicts future cost,
  2. GAE twice: for the reward and for the cost,
  3. combined advantage A_r - lam * A_c for the actor,
  4. after every iteration lam = max(0, lam + lambda_lr * (mean_episode_cost - cost_budget)).

collect_rollout() needs no change: it already records batch["costs"] and batch["ep_cost"],
and the cost critic can evaluate the stored observations at the start of update(),
because no network changes while a rollout is being collected.

Open design decisions (see the discussion notes) that affect this file:
  - which cost: drawdown above cost_start (now) or the same risk measure as the shield,
  - cost keeps accruing after the victim went flat (drawdown is measured from the peak),
  - gamma for the cost: the budget is an undiscounted episode sum, the critic is discounted.
"""
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
        """Dual step after an iteration; returns the new multiplier.
        lam <- max(0, lam + lambda_lr * (mean_episode_cost - cost_budget))
        mean_episode_cost: batch["ep_cost"].mean() of the rollout that was just collected.
        """
        self.lagrange = max(0, self.lagrange + self.cfg.lambda_lr * (mean_episode_cost - self.cfg.cost_budget))
        return self.lagrange

    def update(self, batch: dict) -> dict:
        """batch from collect_rollout() -> training statistics (floats).

        Steps:
          1. cost values V_c(s) for batch["obs"]                      -> (T, B), no grad
          2. reward GAE on batch["rewards"] * reward_scale             -> adv_r, ret_r
             cost   GAE on batch["costs"]   * cost_scale               -> adv_c, ret_c
          3. adv = adv_r - lam * adv_c, then normalise (optionally divide by 1 + lam first,
             so the scale of the advantage does not grow with lam)
          4. the PPO loop of the parent class with
             loss = policy_loss + vf_coef * (value_loss_r + value_loss_c) - ent_coef * entropy
          5. update_lambda(batch["ep_cost"].mean())
          6. stats: parent stats + "lagrange", "cost_value_loss", "ep_cost", "cost_budget"
        """

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
                    "cost_critic": self.cost_critic.state_dict(), "lagrange": self.lagrange,
                    "cfg": asdict(self.cfg)}, path)

    def load(self, path: str):
        state = torch.load(path)
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.cost_critic.load_state_dict(state["cost_critic"])
        self.lagrange = state["lagrange"]
