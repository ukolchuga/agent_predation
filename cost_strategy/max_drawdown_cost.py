import numpy as np

from .cost_strategy import CostStrategy


class MaxDrawdownCost(CostStrategy):
    """Increase of max(0, MDD - cost_start), MDD = maximum drawdown so far in the episode.
    The episode sum is max(0, MDD_T - cost_start): the same risk measure the shield uses,
    and a victim that has sold out accrues no further cost."""

    def reset(self, B):
        self.mdd = np.zeros(B)  # maximum drawdown so far, one value per episode

    def compute_cost(self, dd):
        new_mdd = np.maximum(self.mdd, dd)
        cost = np.maximum(0.0, new_mdd - self.cost_start) - np.maximum(0.0, self.mdd - self.cost_start)
        self.mdd = new_mdd  # remember AFTER computing the cost
        return cost
