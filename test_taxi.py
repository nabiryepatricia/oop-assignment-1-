"""Unit tests for src/taxi.py (run with:  pytest -q)."""
import numpy as np
import pytest

from src.population import Forecaster, LinearTrendForecaster
from src.taxi import (
    ExponentialSmoothingForecaster,
    FleetPlanner,
    MovingAverageForecaster,
    Route,
    SeasonalNaiveForecaster,
    SupplyDemandModel,
    WalkForwardBacktester,
    generate_weekly_demand,
)

NTINDA = [35, 40, 42, 50, 55, 60, 48, 52, 47, 45]


# ------------------------------------------------------------------ Route
def test_route_revenue_and_statistics():
    r = Route("Ntinda", NTINDA, 2000)
    assert len(r) == 10 and list(r.days) == list(range(1, 11))
    assert r.daily_revenue()[0] == pytest.approx(70_000)
    assert r.total_revenue() == pytest.approx(948_000)               # 474 passengers x 2,000
    d = r.describe()
    assert d["mean"] == pytest.approx(47.4) and d["median"] == pytest.approx(47.5)
    assert d["variance"] == pytest.approx(54.2667, rel=1e-4)         # sample variance
    assert d["stdev"] == pytest.approx(d["variance"] ** 0.5)


@pytest.mark.parametrize("passengers,fare", [([5], 1000), ([5, -1], 1000), ([5, float("nan")], 1000),
                                             ([5, 6], 0), ([5, 6], -10)])
def test_route_rejects_invalid_input(passengers, fare):
    with pytest.raises(ValueError):
        Route("X", passengers, fare)


# ------------------------------------------------------------------ supply and demand
def test_equilibrium_by_hand():
    m = SupplyDemandModel()                                          # Qd = 120 - 0.02P, Qs = 10 + 0.03P
    A, rhs = m.system()
    assert A.tolist() == [[1.0, 0.02], [1.0, -0.03]] and rhs.tolist() == [120.0, 10.0]
    price, qty = m.equilibrium()                                     # 0.05P = 110 -> P = 2200, Q = 76
    assert price == pytest.approx(2200.0) and qty == pytest.approx(76.0)
    assert m.demand(price) == pytest.approx(m.supply(price))


def test_current_fare_is_below_equilibrium_with_shortage():
    m = SupplyDemandModel()
    assert m.compare_fare(2000) == "below" and m.compare_fare(2500) == "above" and m.compare_fare(2200) == "at"
    assert m.excess_demand(2000) == pytest.approx(10.0)              # Qd = 80, Qs = 70
    assert m.traded_quantity(2000) == pytest.approx(70.0)            # short side of the market


def test_supply_demand_edge_cases():
    with pytest.raises(ValueError):
        SupplyDemandModel(demand_slope=-0.02)                        # demand must slope down
    with pytest.raises(ValueError):
        SupplyDemandModel(demand_intercept=10, supply_intercept=120).equilibrium()   # negative price


# ------------------------------------------------------------------ forecasters
def test_forecasters_inherit_the_abstract_base():
    for m in (MovingAverageForecaster(), ExponentialSmoothingForecaster(), SeasonalNaiveForecaster(), LinearTrendForecaster()):
        assert isinstance(m, Forecaster)


def test_moving_average_by_hand():
    m = MovingAverageForecaster(3).fit([1, 2, 3, 4], [10, 20, 30, 50])
    assert m.predict(2) == pytest.approx([100 / 3, 100 / 3])         # mean(20, 30, 50), flat
    assert m.fitted_values()[3] == pytest.approx(20.0)               # mean(10, 20, 30)
    assert np.isnan(m.fitted_values()[:3]).all()


def test_ses_by_hand_and_limits():
    m = ExponentialSmoothingForecaster(0.5).fit([1, 2], [10, 20])
    assert m.predict(1)[0] == pytest.approx(15.0)                    # 0.5*20 + 0.5*10
    naive = ExponentialSmoothingForecaster(1.0).fit([1, 2, 3], [10, 20, 33])
    assert naive.predict(1)[0] == pytest.approx(33.0)                # alpha = 1 is the naive forecast
    slow = ExponentialSmoothingForecaster(0.01).fit([1, 2, 3], [10, 20, 33])
    assert slow.predict(1)[0] == pytest.approx(10.0, abs=0.5)        # alpha ~ 0 barely moves


def test_seasonal_naive_repeats_the_weekly_pattern():
    week = [50, 52, 49, 55, 80, 60, 30]
    m = SeasonalNaiveForecaster(7).fit(range(1, 15), week * 2)
    assert m.predict(7).tolist() == week
    assert m.predict(9).tolist() == week + week[:2]                  # wraps around


def test_forecaster_parameter_and_data_validation():
    with pytest.raises(ValueError):
        MovingAverageForecaster(0)
    with pytest.raises(ValueError):
        MovingAverageForecaster(5).fit([1, 2, 3], [1, 2, 3])         # fewer points than the window
    for bad_alpha in (0, -0.1, 1.5):
        with pytest.raises(ValueError):
            ExponentialSmoothingForecaster(bad_alpha)
    with pytest.raises(ValueError):
        SeasonalNaiveForecaster(7).fit(range(1, 6), [1, 2, 3, 4, 5])  # less than one season


# ------------------------------------------------------------------ walk-forward backtesting
def test_backtest_target_count_and_hand_calculation():
    days, y = [1, 2, 3, 4, 5], [10, 20, 30, 40, 50]
    res = WalkForwardBacktester(first_target=4).evaluate(MovingAverageForecaster(3), days, y)
    assert res.targets.tolist() == [4, 5]
    assert res.forecast.tolist() == pytest.approx([20.0, 30.0])      # mean(10,20,30), mean(20,30,40)
    assert res.mae == pytest.approx(20.0)
    assert len(WalkForwardBacktester(4).evaluate(LinearTrendForecaster(), range(1, 11), range(10, 20)).targets) == 7


def test_backtest_never_uses_future_data():
    days = list(range(1, 11))
    a = [35, 40, 42, 50, 55, 60, 48, 52, 47, 45]
    b = a[:-3] + [999, 999, 999]                                     # change only the last 3 days
    bt = WalkForwardBacktester(4)
    ra = bt.evaluate(ExponentialSmoothingForecaster(0.5), days, a)
    rb = bt.evaluate(ExponentialSmoothingForecaster(0.5), days, b)
    assert ra.forecast[:4] == pytest.approx(rb.forecast[:4])         # forecasts of days 4-7 unchanged


def test_linear_trend_is_perfect_on_a_line():
    res = WalkForwardBacktester(4).evaluate(LinearTrendForecaster(), range(1, 11), [5 + 3 * t for t in range(1, 11)])
    assert res.mae == pytest.approx(0.0, abs=1e-9)


def test_alpha_grid_search_behaviour():
    days = list(range(1, 11))
    best, table = WalkForwardBacktester(4).tune_alpha(days, [10 * t for t in days])   # steady trend
    assert best == 1.0 and table[1.0] < table[0.5]                    # react fully to the latest value
    best_flat, _ = WalkForwardBacktester(4).tune_alpha(days, [7] * 10)                # constant: all tie
    assert best_flat == 0.05                                           # ties go to the smaller alpha


def test_backtester_validation():
    with pytest.raises(ValueError):
        WalkForwardBacktester(first_target=1)
    with pytest.raises(ValueError):
        WalkForwardBacktester(4, 20).evaluate(LinearTrendForecaster(), range(1, 11), range(10, 20))


# ------------------------------------------------------------------ fleet planner
def test_fleet_arithmetic_and_rounding():
    f = FleetPlanner()                                               # 8 trips x 14 seats = 112, buffer 15%
    assert f.capacity_per_vehicle == 112
    assert f.vehicles_needed(97) == 1                                # 97 x 1.15 = 111.55 <= 112
    assert f.vehicles_needed(98) == 2                                # 98 x 1.15 = 112.7 > 112 -> round up
    assert f.vehicles_needed(45) == 1 and f.vehicles_needed(0) == 1  # at least one vehicle on an open route
    assert f.utilisation(56, 1) == pytest.approx(0.5)


def test_fleet_boundary_is_not_corrupted_by_float_noise():
    f = FleetPlanner(buffer=0.0)
    assert f.vehicles_needed(112) == 1 and f.vehicles_needed(112.0001) == 2
    assert FleetPlanner(min_vehicles=0).vehicles_needed(0) == 0


def test_fleet_validation():
    f = FleetPlanner()
    for bad in (-1, float("nan")):
        with pytest.raises(ValueError):
            f.vehicles_needed(bad)
    with pytest.raises(ValueError):
        f.utilisation(10, 0)
    with pytest.raises(ValueError):
        FleetPlanner(buffer=-0.1)
    with pytest.raises(ValueError):
        FleetPlanner(seats=0)


# ------------------------------------------------------------------ extension: weekly data
def test_weekly_generator_reproducible_with_friday_peak_and_sunday_dip():
    a, b = generate_weekly_demand(140, seed=1), generate_weekly_demand(140, seed=1)
    assert np.array_equal(a, b) and len(a) == 140 and (a >= 0).all()
    assert a[4::7].mean() > a[0::7].mean() > a[6::7].mean()           # Friday > Monday > Sunday


def test_seasonal_naive_beats_moving_average_on_weekly_data():
    y = generate_weekly_demand(60, seed=42)
    days, bt = np.arange(1, 61), WalkForwardBacktester(15, 60)
    assert bt.evaluate(SeasonalNaiveForecaster(7), days, y).mae < bt.evaluate(MovingAverageForecaster(3), days, y).mae


def test_weekly_generator_validation():
    with pytest.raises(ValueError):
        generate_weekly_demand(0)
    with pytest.raises(ValueError):
        generate_weekly_demand(10, base=-5)
