"""Vectorised two-agent market (B parallel episodes), same dynamics as market.py.

Victim action a in [-1, 1]  -> target fraction of max position f = (a + 1) / 2.
Predator action in [-1, 1]  -> trade of a * pred_trade shares (0 if no predator).
Equity is marked at LIQUIDATION value: cash + pos*S - gamma/2 * pos^2, so an agent
cannot create paper profit by pushing the price with its own trades.
"""
import numpy as np


class VecMarket:
    K = 10

    def __init__(self, B=64, T=120, s0=100.0, mu=0.0003, sigma=0.004, gamma=1e-3, eta=5e-4,
                 noise_flow=30.0, cash0=100_000.0, max_pos=3000.0, max_trade=600.0,
                 cost_start=0.03, shield_limit=None, pred_trade=350.0, pred_cap=3500.0, seed=0):
        self.__dict__.update(locals()); del self.__dict__["self"]
        self.rng = np.random.default_rng(seed)
        self.obs_dim = self.K + 4

    def _equity(self, cash, pos):
        return cash + pos * self.S - 0.5 * self.gamma * pos ** 2

    def reset(self):
        B = self.B
        self.t = 0
        self.S = np.full(B, self.s0)
        self.cash_v = np.full(B, self.cash0); self.pos_v = np.zeros(B)
        self.cash_p = np.full(B, self.cash0); self.pos_p = np.zeros(B)
        self.peak_v = np.full(B, self.cash0)
        self.rets = np.zeros((B, self.K))
        self.stopped = np.zeros(B, bool)
        self.last_qv = np.zeros(B)
        return self.obs_victim()

    def drawdown(self):
        return 1.0 - self._equity(self.cash_v, self.pos_v) / self.peak_v

    def obs_victim(self):
        e = self._equity(self.cash_v, self.pos_v)
        return np.column_stack([self.rets / self.sigma, self.pos_v / self.max_pos,
                                (e / self.cash0 - 1) * 10, self.drawdown() * 10,
                                np.full(self.B, 1 - self.t / self.T)])

    def step(self, a_v, a_p=None):
        f = (np.clip(a_v, -1, 1) + 1) / 2
        if self.shield_limit is not None:
            self.stopped |= self.drawdown() >= self.shield_limit
            f = np.where(self.stopped, 0.0, f)
        q_v = np.clip(f * self.max_pos - self.pos_v, -self.max_trade, self.max_trade)
        q_p = np.zeros(self.B) if a_p is None else np.clip(
            np.clip(a_p, -1, 1) * self.pred_trade, -self.pred_cap - self.pos_p, self.pred_cap - self.pos_p)

        e_v0, e_p0 = self._equity(self.cash_v, self.pos_v), self._equity(self.cash_p, self.pos_p)
        self.cash_v -= q_v * (self.S + self.eta * q_v); self.pos_v += q_v   # temporary impact
        self.cash_p -= q_p * (self.S + self.eta * q_p); self.pos_p += q_p
        net = q_v + q_p + self.rng.normal(0, self.noise_flow, self.B)
        #  UPWARD DRIFT
        # Constant positive price trend: +s0*mu per step (default +0.03, i.e. +3.6 per
        # episode). This is the reason for the victim to hold the asset at all.
        drift = self.s0 * self.mu
        noise = self.s0 * self.sigma * self.rng.normal(size=self.B)       # random noise
        impact = self.gamma * net                                         # permanent impact
        S_new = np.maximum(self.S + drift + noise + impact, 1e-6)
        self.rets = np.column_stack([self.rets[:, 1:], S_new / self.S - 1])
        self.S = S_new
        self.t += 1; self.last_qv = q_v

        e_v, e_p = self._equity(self.cash_v, self.pos_v), self._equity(self.cash_p, self.pos_p)
        self.peak_v = np.maximum(self.peak_v, e_v)
        r_v = np.log(np.maximum(e_v, 1.0) / np.maximum(e_v0, 1.0))
        r_p = (e_p - e_p0) / self.cash0
        cost = np.maximum(0.0, self.drawdown() - self.cost_start)
        return self.obs_victim(), r_v, cost, r_p, self.t >= self.T
