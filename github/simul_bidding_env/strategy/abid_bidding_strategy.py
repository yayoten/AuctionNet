import numpy as np
from github.simul_bidding_env.strategy.base_bidding_strategy import BaseBiddingStrategy


class AbidBiddingStrategy(BaseBiddingStrategy):
    def __init__(self, budget=100, name="AbidBiddingStrategy", cpa=1 / 1.5, category=0, exp_tempral_ratio=np.ones(48),
                 bid_scale=1.0):
        super().__init__()
        self.budget = budget
        self.remaining_budget = budget
        self.base_actions = exp_tempral_ratio
        self.name = name
        self.cpa = cpa
        self.category = category
        self.bid_scale = bid_scale  # alpha に掛ける倍率（従来は 1 固定）

    def reset(self):
        self.remaining_budget = self.budget

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost):
        alpha = self.bid_scale * self.base_actions[timeStepIndex] * self.cpa / pValues.mean()
        # 記録用（結果には影響しない）。入札額は alpha × 価値の 2 乗
        self.last_internal = dict(alpha=float(alpha), base_action=float(self.base_actions[timeStepIndex]),
                                  pvalue_mean=float(pValues.mean()))
        bids = alpha * pValues * pValues
        bids[bids < 0] = 0
        return bids
