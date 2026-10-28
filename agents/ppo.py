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


