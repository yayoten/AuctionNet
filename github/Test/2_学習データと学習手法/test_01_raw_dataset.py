"""01: 公開データ（原本）が、欠けなく・壊れずに取得できていること。

- ファイルそのものの確認（needs_data）：21 日分がそろい、列が仕様どおりで、余計なファイルが混ざっていない。
- 中身の確認：学習データを作るときに書き出した要約（DB/train_data/period-N.json）で見る。
  全行を読み直すと 1 日あたり約 1 分・約 10GB かかるため、原本を読むのは学習データを作るときの 1 回だけにする。
"""
import json

import numpy as np
import pytest

from conftest import DATASET, PERIODS, SUMMARY_DIR

COLUMNS = ["deliveryPeriodIndex", "advertiserNumber", "advertiserCategoryIndex", "budget", "CPAConstraint", "timeStepIndex",
           "remainingBudget", "pvIndex", "pValue", "pValueSigma", "bid", "xi", "adSlot", "cost", "isExposed",
           "conversionAction", "leastWinningCost", "isEnd"]


@pytest.fixture(params=PERIODS)
def period(request):
    return request.param


@pytest.fixture
def summary(period):
    p = SUMMARY_DIR / f"period-{period}.json"
    if not p.exists():
        pytest.skip(f"{p.name} が無い。research/src/make_train_data.py を実行する")
    return json.loads(p.read_text(encoding="utf-8"))


# ---------- ファイル ----------
@pytest.mark.needs_data
def test_raw_file_exists_and_is_large(period):
    f = DATASET / f"period-{period}.csv"
    assert f.is_file(), f"{f.name} が無い"
    assert f.stat().st_size > 3 * 10 ** 9  # 1 日分は約 3.8GB


@pytest.mark.needs_data
def test_raw_header_is_the_documented_18_columns(period):
    with open(DATASET / f"period-{period}.csv", encoding="utf-8") as f:
        header, first = f.readline().strip().split(","), f.readline().strip().split(",")
    assert header == COLUMNS
    assert len(first) == 18 and float(first[0]) == period


@pytest.mark.needs_data
def test_no_stray_files_in_raw_folder():
    """macOS の ._*.csv や、評価が作るキャッシュが混ざると、本家の TrainDataGenerator（*.csv を全部読む）が壊れる。"""
    names = sorted(p.name for p in DATASET.iterdir() if p.is_file())
    assert names == sorted(f"period-{n}.csv" for n in PERIODS)


# ---------- 中身（要約から） ----------
def test_all_21_summaries_exist():
    missing = [n for n in PERIODS if not (SUMMARY_DIR / f"period-{n}.json").exists()]
    assert not missing, f"要約が無い period: {missing}"


def test_summary_columns_and_period_index(summary, period):
    assert summary["columns"] == COLUMNS
    assert summary["delivery_period_values"] == [float(period)]


def test_summary_missing_values_are_only_unwon_bids(summary):
    """欠損は bid 列にだけある（period 9・13・27 の終盤のティック）。その行は落札も支払いもしていない。他の列に欠損はない。"""
    if summary["n_nan"] == 0:
        return
    assert set(summary["nan_by_column"]) == {"bid"}
    assert summary["nan_rows"]["n_won"] == 0 and summary["nan_rows"]["cost"] == 0
    assert min(summary["nan_rows"]["ticks"]) >= 40
    assert summary["n_nan"] / summary["n_rows"] < 0.002


def test_missing_values_appear_in_exactly_three_periods():
    have = sorted(n for n in PERIODS if (SUMMARY_DIR / f"period-{n}.json").exists()
                  and json.loads((SUMMARY_DIR / f"period-{n}.json").read_text(encoding="utf-8"))["n_nan"] > 0)
    assert have == [9, 13, 27]


def test_summary_shape_is_48_advertisers_times_about_500k_pv(summary):
    assert summary["n_advertisers"] == 48 and summary["n_ticks"] == 48
    assert summary["n_rows"] == summary["n_pv"] * 48          # どの PV にも 48 社ぶんの行がある
    assert 450_000 <= summary["n_pv"] <= 550_000
    assert sum(summary["pv_per_tick"].values()) == summary["n_pv"]


def test_summary_three_slots_are_won_for_every_pv(summary):
    assert summary["won_slots_per_pv"] == pytest.approx(3.0)


def test_summary_values_are_in_plausible_ranges(summary):
    assert 0.0003 < summary["pvalue_mean"] < 0.0008           # シミュレータの価値の平均は 0.0005 前後
    assert summary["bid_min"] >= 0 and summary["cost_min"] >= 0
    assert 0 < summary["least_winning_cost_mean"] < 1


def test_summary_advertisers_match_the_simulator_lineup(summary):
    """データの広告主 48 社の予算・CPA 制約・カテゴリは、シミュレータ（Controller）の 48 社と同じ。"""
    from github.simul_bidding_env.Controller.Controller import Controller
    c = Controller.__new__(Controller)
    adv = sorted(summary["advertisers"], key=lambda a: a["advertiser"])
    assert [a["advertiser"] for a in adv] == list(range(48))
    assert [a["budget"] for a in adv] == [float(x) for x in c.calculate_budget()]
    assert [a["cpa_constraint"] for a in adv] == [float(x) for x in c.get_cpa_constraints()]
    assert [a["category"] for a in adv] == [float(i // 8) for i in range(48)]


def test_summary_no_advertiser_overspends(summary):
    for a in summary["advertisers"]:
        assert a["cost"] <= a["budget"] + 1e-6
        assert np.isfinite(a["alpha_mean"]) and a["alpha_mean"] > 0
