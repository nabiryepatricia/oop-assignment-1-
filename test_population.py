"""Unit tests for src/population.py (run with:  pytest -q)."""
import statistics

import numpy as np
import pytest

from src.population import (
    CAGRForecaster,
    ClassroomPlanner,
    DistrictPopulation,
    FibonacciRatioForecaster,
    Forecaster,
    LinearTrendForecaster,
    ModelSelector,
    fibonacci_numbers,
    mae,
    mape,
    rmse,
)

YEARS = list(range(2015, 2025))
KAMPALA = [1200, 1250, 1300, 1350, 1420, 1500, 1580, 1650, 1720, 1800]


def kampala() -> DistrictPopulation:
    return DistrictPopulation("Kampala", YEARS, KAMPALA)


# ------------------------------------------------------------ DistrictPopulation
def test_district_basics_repr_len():
    d = kampala()
    assert len(d) == 10
    assert "Kampala" in repr(d) and "2015-2024" in repr(d)


@pytest.mark.parametrize("years,pop", [
    ([2015, 2016], [1.0]),                         # unequal lengths
    ([2015, 2016], [1.0, -2.0]),                   # negative population
    ([2015, 2016], [1.0, float("nan")]),           # NaN
    ([2015], [1.0]),                               # too few observations
    ([2016, 2015], [1.0, 2.0]),                    # years not increasing
    ([2015.5, 2016.5], [1.0, 2.0]),                # non-integer years
])
def test_district_rejects_invalid_input(years, pop):
    with pytest.raises(ValueError):
        DistrictPopulation("X", years, pop)


def test_district_rejects_blank_name():
    with pytest.raises(ValueError):
        DistrictPopulation("  ", YEARS, KAMPALA)


def test_growth_rates_and_cagr_by_hand():
    d = kampala()
    assert d.yoy_growth()[0] == pytest.approx(50 / 1200)
    assert len(d.yoy_growth()) == 9
    assert d.cagr() == pytest.approx(1.5 ** (1 / 9) - 1)          # 1800 / 1200 over 9 years
    assert DistrictPopulation("flat", [2020, 2021, 2022], [5, 5, 5]).cagr() == pytest.approx(0.0)


def test_growth_edge_cases_zero_population():
    d = DistrictPopulation("Z", [2020, 2021, 2022], [0, 5, 10])
    with pytest.raises(ValueError):
        d.cagr()
    with pytest.raises(ValueError):
        d.yoy_growth()


def test_sample_vs_population_variance():
    d = DistrictPopulation("T", [2020, 2021, 2022, 2023], [1, 2, 3, 4])
    s, n0, n1 = d.stats_statistics(), d.stats_numpy(0), d.stats_numpy(1)
    assert s["variance"] == pytest.approx(5 / 3)                   # sample: divide by n-1
    assert n0["variance"] == pytest.approx(1.25)                   # population: divide by n
    assert n1["variance"] == pytest.approx(s["variance"])          # ddof=1 reproduces statistics.variance
    assert s["pvariance"] == pytest.approx(n0["variance"])
    assert s["variance"] != pytest.approx(n0["variance"])


def test_numpy_and_statistics_agree_on_mean_median_and_real_data():
    d = kampala()
    assert d.stats_statistics()["mean"] == pytest.approx(d.stats_numpy()["mean"])
    assert d.stats_statistics()["median"] == pytest.approx(d.stats_numpy()["median"])
    assert d.stats_statistics()["variance"] == pytest.approx(statistics.variance(KAMPALA))
    with pytest.raises(ValueError):
        d.stats_numpy(ddof=10)


def test_split_by_year_and_invalid_split():
    train, test = kampala().split(2021)
    assert list(train.years) == list(range(2015, 2022)) and list(test.years) == [2022, 2023, 2024]
    with pytest.raises(ValueError):
        kampala().split(2024)                                      # nothing left to test on
    with pytest.raises(ValueError):
        kampala().split(2015)                                      # fewer than 2 training years


# ------------------------------------------------------------ forecasters
def test_forecaster_is_abstract():
    with pytest.raises(TypeError):
        Forecaster()


def test_linear_trend_exact_on_linear_data():
    m = LinearTrendForecaster().fit([2020, 2021, 2022, 2023], [100, 110, 120, 130])
    assert m.predict(3) == pytest.approx([140, 150, 160])
    assert m.slope_ == pytest.approx(10.0)
    assert m.fitted_values() == pytest.approx([100, 110, 120, 130])


def test_cagr_exact_on_geometric_data():
    vals = [1000 * 1.05 ** t for t in range(5)]
    m = CAGRForecaster().fit(range(2020, 2025), vals)
    assert m.rate_ == pytest.approx(0.05)
    assert m.predict(2) == pytest.approx([1000 * 1.05 ** 5, 1000 * 1.05 ** 6])
    assert m.fitted_values() == pytest.approx(vals)


def test_fibonacci_ratios_and_forecast():
    m = FibonacciRatioForecaster().fit([2020, 2021], [100.0, 200.0])
    assert m.ratios(5) == pytest.approx([1, 2, 1.5, 5 / 3, 1.6])
    assert m.predict(3) == pytest.approx([200, 400, 300])           # last * ratio_h
    assert FibonacciRatioForecaster(chain=True).fit([2020, 2021], [100.0, 200.0]).predict(3) == pytest.approx([200, 400, 600])
    assert fibonacci_numbers(8) == [1, 1, 2, 3, 5, 8, 13, 21]
    assert FibonacciRatioForecaster(start_index=20).ratios(1)[0] == pytest.approx((1 + 5 ** 0.5) / 2, abs=1e-6)


def test_predict_requires_fit_and_valid_horizon():
    with pytest.raises(RuntimeError):
        LinearTrendForecaster().predict(3)
    m = LinearTrendForecaster().fit([2020, 2021, 2022], [1, 2, 3])
    for bad in (0, -1, 2.5):
        with pytest.raises(ValueError):
            m.predict(bad)


def test_fit_validates_inputs():
    with pytest.raises(ValueError):
        LinearTrendForecaster().fit([2020, 2021], [1, 2, 3])
    with pytest.raises(ValueError):
        CAGRForecaster().fit([2020, 2021], [1, -2])
    with pytest.raises(ValueError):
        CAGRForecaster().fit([2020, 2021], [0, 5])                  # CAGR needs positive start
    with pytest.raises(ValueError):
        FibonacciRatioForecaster(start_index=0)


def test_clone_is_unfitted_and_independent():
    m = LinearTrendForecaster().fit([2020, 2021, 2022], [1, 2, 3])
    c = m.clone()
    assert c.years_ is None and m.years_ is not None
    with pytest.raises(RuntimeError):
        c.predict(1)


# ------------------------------------------------------------ bootstrap intervals
def test_bootstrap_reproducible_and_brackets_point_forecast():
    d = kampala()
    m = LinearTrendForecaster().fit_district(d)
    a, b = m.bootstrap_interval(5, n_boot=1000, seed=3), m.bootstrap_interval(5, n_boot=1000, seed=3)
    assert np.array_equal(a["lower"], b["lower"]) and np.array_equal(a["upper"], b["upper"])
    assert (a["lower"] <= a["point"]).all() and (a["point"] <= a["upper"]).all()
    assert (a["upper"] - a["lower"] > 0).all()


def test_bootstrap_zero_noise_gives_zero_width():
    m = LinearTrendForecaster().fit(range(2015, 2025), [100 + 10 * t for t in range(10)])
    iv = m.bootstrap_interval(3, n_boot=200)
    assert iv["upper"] - iv["lower"] == pytest.approx([0, 0, 0], abs=1e-6)


def test_bootstrap_cagr_interval_is_centred_on_forecast():
    iv = CAGRForecaster().fit_district(kampala()).bootstrap_interval(5, n_boot=1000)
    mid = (iv["lower"] + iv["upper"]) / 2
    assert np.abs(mid / iv["point"] - 1).max() < 0.01              # within 1% of the point forecast


def test_bootstrap_edge_cases():
    fib = FibonacciRatioForecaster().fit([2020, 2021], [1.0, 2.0])
    with pytest.raises(NotImplementedError):
        fib.bootstrap_interval(2)                                   # no in-sample fit -> no residuals
    m = LinearTrendForecaster().fit_district(kampala())
    with pytest.raises(ValueError):
        m.bootstrap_interval(3, level=1.5)
    with pytest.raises(ValueError):
        m.bootstrap_interval(3, n_boot=0)


# ------------------------------------------------------------ metrics and selection
def test_metrics_by_hand():
    a, f = [100, 200], [110, 180]
    assert mae(a, f) == pytest.approx(15.0)
    assert rmse(a, f) == pytest.approx(np.sqrt((100 + 400) / 2))
    assert mape(a, f) == pytest.approx(10.0)


def test_metric_edge_cases():
    with pytest.raises(ValueError):
        mape([0, 5], [1, 5])                                        # zero actual
    with pytest.raises(ValueError):
        mae([1, 2], [1])                                            # length mismatch
    with pytest.raises(ValueError):
        rmse([], [])


def test_selector_picks_model_matching_the_data():
    models = [LinearTrendForecaster(), CAGRForecaster(), FibonacciRatioForecaster()]
    sel = ModelSelector(models, last_train_year=2021)
    geometric = DistrictPopulation("G", YEARS, [1000 * 1.05 ** t for t in range(10)])
    linear = DistrictPopulation("L", YEARS, [500 + 25 * t for t in range(10)])
    assert sel.best(sel.evaluate(geometric)).model == "CAGR (exponential)"
    assert sel.best(sel.evaluate(linear)).model == "Linear trend"
    assert sel.select(linear).years_ is None                       # returns an unfitted clone


def test_selector_fibonacci_is_worst_on_real_data():
    scores = ModelSelector([LinearTrendForecaster(), CAGRForecaster(), FibonacciRatioForecaster()]).evaluate(kampala())
    worst = max(scores, key=lambda s: s.rmse)
    assert worst.model == "Fibonacci ratio" and worst.mape > 20


def test_selector_validation():
    with pytest.raises(ValueError):
        ModelSelector([])
    with pytest.raises(ValueError):
        ModelSelector([LinearTrendForecaster()], criterion="r2")


# ------------------------------------------------------------ planner
def test_classroom_arithmetic_rounds_up():
    p = ClassroomPlanner(primary_share=0.5, pupils_per_classroom=50)
    assert p.classrooms_needed(100) == 1000                         # 100k * 0.5 = 50,000 pupils / 50
    assert p.classrooms_needed(100.001) == 1001                     # partial classroom -> round up
    assert p.additional_classrooms(100, 110) == 100


def test_default_planner_matches_hand_calculation():
    p = ClassroomPlanner()
    assert p.pupils(1800) == pytest.approx(324_000)
    assert p.classrooms_needed(1800) == 6114                        # ceil(324000 / 53) = ceil(6113.2)


def test_planner_edge_cases():
    p = ClassroomPlanner()
    assert p.additional_classrooms(500, 400) == 0                   # shrinking population: never negative
    assert p.classrooms_needed(0) == 0
    with pytest.raises(ValueError):
        p.pupils(-1)
    with pytest.raises(ValueError):
        ClassroomPlanner(primary_share=1.2)
    with pytest.raises(ValueError):
        ClassroomPlanner(pupils_per_classroom=0)
