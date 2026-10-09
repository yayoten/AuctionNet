"""09: PlayerAgentWrapper（ユーザー戦略を包み、reset に 2 秒のタイムアウトを付ける）。"""
import time

import numpy as np
import pytest
from func_timeout import FunctionTimedOut

from helpers import make_pvalues
from github.simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy
from github.simul_bidding_env.strategy.player_agent_wrapper import PlayerAgentWrapper, custom_timeout


class Dummy:
    def __init__(self):
        self.budget = 10
        self.name = "dummy"
        self.calls = []

    def reset(self):
        self.calls.append("reset")

    def bidding(self, *a):
        self.calls.append("bidding")
        return a[1]

    def action(self, *a):
        return ("action", a)


def test_wrapper_stores_player_agent():
    d = Dummy()
    w = PlayerAgentWrapper(player_agent=d)
    assert w.player_agent is d


def test_attribute_read_is_forwarded():
    w = PlayerAgentWrapper(Dummy())
    assert w.budget == 10 and w.name == "dummy"


def test_attribute_write_is_forwarded_to_player_agent():
    d = Dummy()
    w = PlayerAgentWrapper(d)
    w.budget = 99
    w.new_attr = "x"
    assert d.budget == 99 and d.new_attr == "x"
    assert "budget" not in w.__dict__


def test_player_agent_attribute_is_rebindable():
    d1, d2 = Dummy(), Dummy()
    w = PlayerAgentWrapper(d1)
    w.player_agent = d2
    assert w.player_agent is d2


def test_missing_attribute_raises_attribute_error():
    w = PlayerAgentWrapper(Dummy())
    with pytest.raises(AttributeError):
        w.nope


def test_reset_is_forwarded():
    d = Dummy()
    PlayerAgentWrapper(d).reset()
    assert d.calls == ["reset"]


def test_bidding_is_forwarded_via_getattr():
    d = Dummy()
    w = PlayerAgentWrapper(d)
    p, _ = make_pvalues(3)
    assert np.array_equal(w.bidding(0, p), p)
    assert d.calls == ["bidding"]


def test_action_is_forwarded_with_positional_args():
    w = PlayerAgentWrapper(Dummy())
    out = w.action(1, 2, 3, 4, 5, 6, 7, 8, 9)
    assert out == ("action", (1, 2, 3, 4, 5, 6, 7, 8, 9))


def test_action_on_current_strategy_interface_raises():
    """現行の戦略は bidding() 実装で action() を持たない。ラッパーの action() は旧インタフェースの名残。"""
    w = PlayerAgentWrapper(PidBiddingStrategy())
    with pytest.raises(AttributeError):
        w.action(0, 1, 1, [], [], [], [], [], [])


def test_wrapper_class_has_no_timeout_on_bidding():
    """reset() / action() だけがタイムアウト付き。bidding() は素通しなので、遅い戦略は止められない。"""
    assert "bidding" not in PlayerAgentWrapper.__dict__


def test_reset_times_out_after_two_seconds():
    class Slow(Dummy):
        def reset(self):
            time.sleep(5)

    w = PlayerAgentWrapper(Slow())
    t = time.time()
    with pytest.raises(FunctionTimedOut):
        w.reset()
    assert 1.5 < time.time() - t < 4.0


def test_reset_fast_agent_not_timed_out():
    t = time.time()
    PlayerAgentWrapper(Dummy()).reset()
    assert time.time() - t < 0.5


def test_action_times_out_with_agent_name_in_message():
    class Slow(Dummy):
        def action(self, *a):
            time.sleep(5)

    w = PlayerAgentWrapper(Slow())
    with pytest.raises(FunctionTimedOut) as ei:
        w.action(0, 1, 1, 1, 1, 1, 1, 1, 1)
    assert "action time out" in str(ei.value)


def test_custom_timeout_decorator_passes_through_value():
    @custom_timeout()
    def f(x):
        return x * 2

    assert f(4) == 8


def test_custom_timeout_decorator_reraises_other_exceptions():
    @custom_timeout()
    def f():
        raise KeyError("k")

    with pytest.raises(KeyError):
        f()


def test_wrapped_real_strategy_behaves_like_the_original():
    inner = PidBiddingStrategy()
    w = PlayerAgentWrapper(inner)
    p, sg = make_pvalues(10)
    assert np.allclose(w.bidding(0, p, sg, [], [], [], [], []), 15 * p)
    w.remaining_budget = 5
    assert inner.remaining_budget == 5
    w.reset()
    assert inner.remaining_budget == inner.budget
