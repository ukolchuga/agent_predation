"""Interface of a victim cost (strategy pattern): the market asks for the cost, the strategy decides how."""


class CostStrategy:
    def __init__(self, cost_start):
        self.cost_start = cost_start  # drawdown below this level is free

    def reset(self, B):
        """Start of B new episodes. Strategies with memory (e.g. a running maximum) clear it here."""

    def compute_cost(self, dd):
        """Victim drawdown after this step, shape (B,) -> cost of this step, shape (B,)."""
        raise NotImplementedError
