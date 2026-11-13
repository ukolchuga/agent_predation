import numpy as np

from .cost_strategy import CostStrategy


class DrawdownCost(CostStrategy):
    """max(0, drawdown - cost_start) every step. No memory.
    Keeps accruing after the victim went flat, because drawdown is measured from the running peak."""

    def compute_cost(self, dd):
        return np.maximum(0.0, dd - self.cost_start)
