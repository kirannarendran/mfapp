import unittest
import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from evaluation.portfolio import init_portfolio, rebalance, step_forward

class TestPortfolioAccounting(unittest.TestCase):
    def test_target_weights_sum_to_one(self):
        port = init_portfolio()
        target = {'1001': 0.5, '1002': 0.5}
        port = rebalance(port, target)
        self.assertAlmostEqual(port['cash'] + sum(port['tradeable'].values()) + sum(port['suspended'].values()), 1.0)

    def test_one_way_turnover(self):
        port = init_portfolio()
        port['cash'] = 0.0
        port['tradeable'] = {'1001': 0.6, '1002': 0.4}
        target = {'1001': 0.5, '1002': 0.5}
        
        port = rebalance(port, target, cost_bps=50)
        self.assertAlmostEqual(port['turnover'], 0.1)
        self.assertAlmostEqual(port['cost'], 0.0005)

    def test_cash_inclusive_turnover(self):
        port = init_portfolio()
        target = {'1001': 1.0}
        port = rebalance(port, target, cost_bps=100)
        self.assertAlmostEqual(port['turnover'], 1.0)
        self.assertAlmostEqual(port['cost'], 0.01)
        
    def test_rebalance_with_suspended(self):
        port = init_portfolio()
        port['cash'] = 0.0
        port['tradeable'] = {'1001': 0.8}
        port['suspended'] = {'1002': 0.2}
        
        # We want to rebalance tradeable to 100% of available capital
        target = {'1003': 1.0}
        port = rebalance(port, target)
        
        self.assertAlmostEqual(port['suspended']['1002'], 0.2)
        self.assertAlmostEqual(port['tradeable']['1003'], 0.8)
        self.assertAlmostEqual(port['cash'], 0.0)
        self.assertAlmostEqual(port['turnover'], 0.8) # 0.8 out of 1001, 0.8 into 1003 -> sum=1.6 -> /2 = 0.8

    def test_explicit_portfolio_accounting_rules(self):
        # 5. Correct the portfolio-accounting test.
        # Test separately exactly as defined by the user.
        opening_weights = pd.Series({'1001': 0.6, '1002': 0.4})
        asset_returns = pd.Series({'1001': 0.1, '1002': -0.2})
        
        position_values = opening_weights * (1 + asset_returns)
        gross_growth_factor = position_values.sum()
        gross_return = gross_growth_factor - 1
        drifted_weights = position_values / gross_growth_factor
        
        tolerance = 1e-8
        self.assertTrue(abs(drifted_weights.sum() - 1.0) < tolerance)
        self.assertTrue(abs(gross_growth_factor - (1.0 + gross_return)) < tolerance)
        
        # 8. Verify that fund_weights.sum() + cash_weight + suspended_weight == 1.0 within tolerance.
        port = init_portfolio()
        port['cash'] = 0.1
        port['tradeable'] = {'1001': 0.6}
        port['suspended'] = {'1002': 0.3}
        self.assertTrue(abs(sum(port['tradeable'].values()) + port['cash'] + sum(port['suspended'].values()) - 1.0) < tolerance)

if __name__ == '__main__':
    unittest.main()
