import torch
import torch.nn as nn


def build_mlp(input_dim: int, hidden: int = 64) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, 1),
    )


class Actor(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.network = build_mlp(input_dim, hidden)
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, obs):
        mean = self.network(obs).squeeze(-1)
        return torch.distributions.Normal(loc=mean, scale=torch.exp(self.log_std))


class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.network = build_mlp(input_dim, hidden)

    def forward(self, obs):
        return self.network(obs).squeeze(-1)
