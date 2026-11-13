from .cost_strategy import CostStrategy
from .drawdown_cost import DrawdownCost
from .max_drawdown_cost import MaxDrawdownCost

# config name (env.cost_mode in the YAML files) -> strategy class
COSTS = {"drawdown": DrawdownCost, "max_drawdown": MaxDrawdownCost}


def make_cost(cost_mode, cost_start) -> CostStrategy:
    """'max_drawdown', 0.03 -> MaxDrawdownCost(0.03). A ready CostStrategy object is returned as is."""
    if isinstance(cost_mode, CostStrategy):
        return cost_mode
    if cost_mode not in COSTS:
        raise ValueError(f"unknown cost_mode {cost_mode!r}, expected one of {list(COSTS)}")
    return COSTS[cost_mode](cost_start)
