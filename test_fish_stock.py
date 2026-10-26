"""Unit tests for src/fish_stock.py (run with:  pytest -q)."""
import numpy as np
import pytest

from src.fish_stock import (
    FishStock,
    Fishery,
    PriceModel,
    RiskAssessor,
    fibonacci,
    weekly_revenue,
)


# ---------------------------------------------------------------- normal cases
def test_fibonacci_first_15():
    fib = fibonacci(15)
    assert len(fib) == 15
    assert fib[:6] == [1, 1, 2, 3, 5, 8] and fib[-1] == 610


def test_one_logistic_step_by_hand():
    # N1 = 4000 + 0.4*4000*(1-0.4) - 0.1*4000 = 4000 + 960 - 400 = 4560
    path = FishStock(r=0.4, K=10_000, n0=4_000, h=0.10).simulate(weeks=1)
    assert path.stock[1] == pytest.approx(4560.0)
    assert path.harvest_tonnes[0] == pytest.approx(400.0)


def test_stock_converges_to_equilibrium():
    fs = FishStock(h=0.10)
    assert fs.equilibrium_stock == pytest.approx(7500.0)
    assert fs.simulate(weeks=200).stock[-1] == pytest.approx(7500.0, abs=1e-3)


def test_msy_and_yield_at_msy_harvest_rate():
    fs = FishStock(r=0.4, K=10_000, h=0.2)
    assert fs.msy == pytest.approx(1000.0)                  # rK/4
    assert fs.h_msy == pytest.approx(0.2)
    assert fs.equilibrium_yield == pytest.approx(fs.msy)    # harvesting at r/2 attains MSY


def test_harvest_status_labels():
    assert FishStock(h=0.10).harvest_status == "sustainable"
    assert FishStock(h=0.30).harvest_status == "overfishing"
    assert FishStock(h=0.40).harvest_status == "collapse"


def test_price_walk_is_bounded_and_reproducible():
    pm = PriceModel()
    a = pm.simulate(52, seed=1, n_paths=500)
    b = pm.simulate(52, seed=1, n_paths=500)
    assert np.array_equal(a, b)
    assert a.shape == (500, 52) and (a[:, 0] == 12_000).all()
    assert a.min() >= 9_000 and a.max() <= 16_000


def test_weekly_revenue_units():
    # 400 tonnes = 400,000 kg at UGX 12,000/kg = UGX 4.8 billion
    assert weekly_revenue(np.array([400.0]), np.array([12_000.0]))[0] == pytest.approx(4.8e9)


def test_describe_and_cv():
    d = RiskAssessor.describe([10, 20, 30, 40])
    assert d["mean"] == pytest.approx(25) and d["median"] == pytest.approx(25)
    assert d["variance"] == pytest.approx(166.6667, rel=1e-5)        # sample variance
    assert d["cv"] == pytest.approx(d["stdev"] / d["mean"])


def test_cv_is_scale_free_but_variance_is_not():
    x = np.array([10.0, 20.0, 30.0, 40.0])
    a, b = RiskAssessor.describe(x), RiskAssessor.describe(x * 1_000)
    assert b["variance"] == pytest.approx(a["variance"] * 1e6)      # units matter
    assert b["cv"] == pytest.approx(a["cv"])                         # CV does not


def test_classify_thresholds():
    ra = RiskAssessor(low_cv=0.05, high_cv=0.15)
    assert [ra.classify(c) for c in (0.0, 0.049, 0.05, 0.149, 0.15, 0.6)] == [
        "Low", "Low", "Moderate", "Moderate", "High", "High"]


def test_value_at_risk_known_distribution():
    rev = np.arange(1, 101, dtype=float)                 # mean 50.5
    v = RiskAssessor(alpha=0.05).value_at_risk(rev)
    assert v["percentile"] == pytest.approx(np.percentile(rev, 5))
    assert v["var"] == pytest.approx(50.5 - np.percentile(rev, 5))


def test_monte_carlo_reproducible_and_sized():
    f = Fishery(FishStock(h=0.1), PriceModel())
    ra = RiskAssessor()
    a = ra.annual_revenue_mc(f, n_paths=1_000, seed=3)
    b = ra.annual_revenue_mc(f, n_paths=1_000, seed=3)
    assert a.shape == (1_000,) and np.array_equal(a, b)


def test_closed_season_stops_harvest_and_helps_overfished_stock():
    fs = FishStock(h=0.30)
    closed = range(20, 28)                                # 8 weeks
    path = fs.simulate(weeks=52, closed_weeks=closed)
    assert (path.harvest_tonnes[20:28] == 0).all()
    assert fs.simulate(weeks=260, closed_weeks=closed).stock[-1] > fs.simulate(weeks=260).stock[-1]


# ------------------------------------------------------------------ edge cases
def test_zero_harvest_fixed_points():
    assert FishStock(n0=10_000, h=0.0).simulate(10).stock[-1] == pytest.approx(10_000)   # at K
    assert FishStock(n0=0.0, h=0.1).simulate(10).stock[-1] == 0.0                         # extinct stays extinct


def test_harvest_above_growth_rate_collapses_and_never_negative():
    path = FishStock(r=0.4, h=0.9).simulate(weeks=60)
    assert (path.stock >= 0).all() and path.stock[-1] < 1.0


@pytest.mark.parametrize("kwargs", [{"h": -0.1}, {"h": 1.5}, {"r": 0}, {"K": -5}, {"n0": -1}])
def test_fishstock_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        FishStock(**kwargs)


def test_price_model_rejects_bad_bounds_and_sizes():
    with pytest.raises(ValueError):
        PriceModel(start=8_000)                          # start below lower bound
    with pytest.raises(ValueError):
        PriceModel(low=16_000, high=9_000)
    with pytest.raises(ValueError):
        PriceModel().simulate(0)


def test_fibonacci_edge_cases():
    assert fibonacci(0) == []
    with pytest.raises(ValueError):
        fibonacci(-1)


def test_risk_assessor_input_validation():
    with pytest.raises(ValueError):
        RiskAssessor.describe([5.0])                     # variance needs 2+ points
    with pytest.raises(ValueError):
        RiskAssessor().value_at_risk([])
    with pytest.raises(ValueError):
        RiskAssessor().classify(-0.1)
    with pytest.raises(ValueError):
        RiskAssessor(low_cv=0.3, high_cv=0.1)
    assert np.isnan(RiskAssessor.describe([-1.0, 1.0])["cv"])     # zero mean -> CV undefined
