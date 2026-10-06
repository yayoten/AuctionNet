import numpy as np
from github.simul_bidding_env.strategy.base_bidding_strategy import BaseBiddingStrategy


class PidBiddingStrategy(BaseBiddingStrategy):
    def __init__(self, budget=100, name="PidBiddingStrategy", cpa=1, category=0, exp_tempral_ratio=np.ones(48),
                 base_action=15, up_factor=1.2, down_factor=0.7, low_threshold=0.7, high_threshold=1.1):
        # base_action 以降は、これまでコードに直書きされていた値（既定値は従来どおり）。params.json から変えられる。
        super().__init__()
        self.budget = budget
        self.remaining_budget = budget
        self.name = name
        self.exp_budget_ratio = exp_tempral_ratio
        self.alpha = None
        self.base_action = base_action
        self.up_factor = up_factor
        self.down_factor = down_factor
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.last_remaining_budget = self.remaining_budget
        self.cpa = cpa
        self.category = category

    def reset(self):
        self.remaining_budget = self.budget
        self.last_remaining_budget = self.budget

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost):
        if timeStepIndex == 0:
            self.alpha = self.base_action
        else:
            last_tick_cost = self.last_remaining_budget - self.remaining_budget
            self.last_remaining_budget -= last_tick_cost
            if last_tick_cost * self.exp_budget_ratio[timeStepIndex:].sum() / self.exp_budget_ratio[timeStepIndex - 1] / self.remaining_budget < self.low_threshold:
                self.alpha *= self.up_factor
            elif last_tick_cost * (48 - timeStepIndex) / self.remaining_budget > self.high_threshold:
                self.alpha *= self.down_factor
        bids = self.alpha * pValues
        return bids

