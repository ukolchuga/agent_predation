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

