"""11: Controller（48 エージェントの生成・予算/CPA/カテゴリの割当・プレイヤーの差し込み）。"""
import numpy as np
import pytest

from conftest import SIM_DIR  # noqa: F401
from github.simul_bidding_env.Controller.Controller import Controller
from github.simul_bidding_env.Environment.BiddingEnv import BiddingEnv
from github.simul_bidding_env.PvGenerator.NeurIPSPvGen import NeurIPSPvGen
from github.simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy
from github.simul_bidding_env.strategy.player_agent_wrapper import PlayerAgentWrapper

KW = dict(num_tick=48, num_agent_category=8, num_category=6, pv_num=1500)


@pytest.fixture
def make():
    def _make(player_index=0, **kw):
        k = dict(KW)
        k.update(kw)
        return Controller(player_index=player_index, player_agent=PidBiddingStrategy(), **k)

    return _make


@pytest.fixture
def c(make):
    return make()


def cls_names(ctrl):
    return [type(a).__name__ for a in ctrl.agent_list]


def test_agent_count_is_48(c):
    assert len(c.agents) == 48 and c.num_agent == 48


def test_components_are_built(c):
    assert isinstance(c.pvGenerator, NeurIPSPvGen)
    assert isinstance(c.biddingEnv, BiddingEnv)


def test_even_category_pool_composition(make):
    names = cls_names(make(player_index=47))  # 末尾を player にして、0 番カテゴリを素のまま見る
    assert names[:8] == ["PidBiddingStrategy", "IqlBiddingStrategy", "TD3_BCBiddingStrategy",
                         "OnlineLpBiddingStrategy", "OnlineLpBiddingStrategy", "CqlBiddingStrategy",
                         "BcBiddingStrategy", "MbrlMopoBiddingStrategy"]


def test_odd_category_pool_composition(make):
    names = cls_names(make(player_index=47))
    assert names[8:16] == ["PidBiddingStrategy", "BcqBiddingStrategy", "MbrlMopoBiddingStrategy",
                           "OnlineLpBiddingStrategy", "OnlineLpBiddingStrategy", "TD3_BCBiddingStrategy",
                           "IqlBiddingStrategy", "MbrlComboMicroBiddingStrategy"]


def test_pool_pattern_alternates_by_category_parity(make):
    names = cls_names(make(player_index=47))  # 47 番はプレイヤー枠なので比較から外す
    for cat in range(6):
        lo, hi = cat * 8, min((cat + 1) * 8, 47)
        ref = names[0:8] if cat % 2 == 0 else names[8:16]
        assert names[lo:hi] == ref[:hi - lo]


def test_every_category_has_one_pid_and_two_onlinelp(make):
    names = cls_names(make(player_index=47))
    for cat in range(6):
        block = names[cat * 8:(cat + 1) * 8]
        assert block.count("PidBiddingStrategy") == 1 and block.count("OnlineLpBiddingStrategy") == 2


def test_budget_list_values(c):
    b = c.budget_list
    assert len(b) == 48 and b[0] == 2900 and b[1] == 4350 and b[47] == 2050
    assert min(b) == 2000 and max(b) == 4850


def test_cpa_constraint_values(c):
    cpa = c.cpa_constraint_list
    assert len(cpa) == 48 and cpa[0] == 100 and cpa[1] == 70 and cpa[47] == 120
    assert set(cpa.tolist()) == {60, 70, 80, 90, 100, 110, 120, 130}


def test_cpa_values_are_balanced_within_each_category(c):
    for cat in range(6):
        block = c.cpa_constraint_list[cat * 8:(cat + 1) * 8]
        assert sorted(block.tolist()) == [60, 70, 80, 90, 100, 110, 120, 130]


def test_category_assignment(c):
    assert c.category.tolist() == [i // 8 for i in range(48)]


def test_every_agent_gets_budget_cpa_category(make):
    c = make(player_index=5)
    for i, a in enumerate(c.agents):
        assert a.budget == c.budget_list[i], i
        assert a.cpa == c.cpa_constraint_list[i], i
        assert a.category == c.category[i], i


def test_all_agents_including_player_start_with_full_budget(c):
    """修正前は、プレイヤー戦略だけ最初の Controller.reset() まで remaining_budget が既定の 100 のままだった。"""
    assert all(a.remaining_budget == a.budget for a in c.agents)
    assert c.player_agent.remaining_budget == c.budget_list[0]


def test_agent_names_get_index_suffix(make):
    c = make(player_index=47)
    assert c.agents[3].name == "OnlineLpBiddingStrategy3"
    assert c.agents[1].name == "Iql-PlayerStrategy1"


@pytest.mark.parametrize("pidx", [0, 1, 7, 8, 23, 47])
def test_player_slot_is_wrapped(make, pidx):
    c = make(player_index=pidx)
    assert isinstance(c.agents[pidx], PlayerAgentWrapper)
    assert sum(isinstance(a, PlayerAgentWrapper) for a in c.agents) == 1


@pytest.mark.parametrize("pidx", [0, 13, 47])
def test_player_agent_receives_budget_cpa_category(make, pidx):
    c = make(player_index=pidx)
    p = c.player_agent
    assert (p.budget, p.cpa, p.category) == (c.budget_list[pidx], c.cpa_constraint_list[pidx], c.category[pidx])


def test_player_agent_is_the_object_passed_in():
    p = PidBiddingStrategy()
    c = Controller(player_index=2, player_agent=p, **KW)
    assert c.agents[2].player_agent is p


def test_wrapper_forwards_attributes_of_player_in_agents_list(make):
    c = make(player_index=4)
    assert c.agents[4].cpa == c.cpa_constraint_list[4]


def test_agents_list_is_agent_list(c):
    assert c.agents is c.agent_list


def test_agents_have_bidding_method(c):
    assert all(callable(getattr(a, "bidding")) for a in c.agents)


def test_player_agent_none_is_rejected():
    with pytest.raises(AttributeError):
        Controller(player_index=0, player_agent=None, **KW)


@pytest.mark.parametrize("pidx", [-1, 48, 100])
def test_player_index_out_of_range_is_rejected_or_wraps(pidx):
    if pidx == -1:
        # 負のインデックスは Python の添字として通ってしまい、末尾のエージェントを黙って置換する
        c = Controller(player_index=pidx, player_agent=PidBiddingStrategy(), **KW)
        assert isinstance(c.agents[47], PlayerAgentWrapper)
    else:
        with pytest.raises(IndexError):
            Controller(player_index=pidx, player_agent=PidBiddingStrategy(), **KW)


def test_reset_resets_env_and_agents(c):
    for a in c.agents:
        a.remaining_budget = 1.0
    c.reset(episode=1)
    assert all(a.remaining_budget == a.budget for a in c.agents)


def test_reset_changes_env_trunc_values(c):
    before = list(c.biddingEnv.advertiser_trunc_values)
    c.reset(episode=3)
    assert c.biddingEnv.advertiser_trunc_values != before


def test_reset_regenerates_pv_for_new_episode(c):
    ref = NeurIPSPvGen(episode=2, num_tick=48, num_agent=48, num_agent_category=8, num_category=6, pv_num=1500)
    c.reset(episode=2)
    assert all(np.array_equal(x, y) for x, y in zip(c.pvGenerator.pv_values, ref.pv_values))


def test_reset_with_player_wrapper_still_works(make):
    c = make(player_index=0)
    c.reset(episode=0)
    c.reset(episode=1)
    assert c.agents[0].remaining_budget == c.budget_list[0]


def test_two_controllers_are_independent(make):
    a, b = make(), make()
    a.agents[3].remaining_budget = 0
    assert b.agents[3].remaining_budget != 0


def test_unknown_pv_generator_type_yields_no_generator():
    c = Controller(player_agent=PidBiddingStrategy(), pv_generator_type="bogus", **KW)
    assert c.pvGenerator is None


def test_pv_num_reaches_generator():
    c = Controller(player_agent=PidBiddingStrategy(), **dict(KW, pv_num=777))
    assert c.pvGenerator.PV_NUM == 777


def test_defaults_are_the_competition_setting():
    c = Controller(player_agent=PidBiddingStrategy(), pv_num=500)
    assert c.num_agent == 48 and c.num_tick == 48 and len(c.cpa_constraint_list) == 48
    assert len(c.agents) == 48


def test_agent_pool_loads_all_official_models_without_error():
    """48 体すべてを生成（学習済みモデルの torch.jit.load を含む）しても落ちない。"""
    c = Controller(player_index=0, player_agent=PidBiddingStrategy(), **KW)
    assert len({type(a).__name__ for a in c.agent_list}) >= 7


def test_importing_controller_does_not_chdir(tmp_path):
    """Controller.py 末尾の `os.chdir('../../')` は `__main__` ブロック内。import では実行されない。"""
    import os
    import subprocess
    import sys

    from conftest import REPO_ROOT

    code = "import os; d=os.getcwd(); import github.simul_bidding_env.Controller.Controller; print(os.getcwd()==d)"
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=tmp_path, capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)), timeout=300)
    assert r.stdout.strip().splitlines()[-1] == "True", r.stderr[-500:]
