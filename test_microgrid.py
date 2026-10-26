"""Unit tests for src/microgrid.py (run with:  pytest -q)."""
import numpy as np
import pytest

from src.microgrid import (
    HybridMicroGrid,
    MicroGrid,
    SingularSystemError,
    generate_demand_csv,
    load_demand_csv,
    read_positive_float,
)


# ---------------------------------------------------------------- normal cases
def test_default_system_is_well_posed():
    g = MicroGrid()
    assert g.determinant == pytest.approx(-5.0)
    assert 1.0 <= g.condition_number < 10.0   # well conditioned


def test_solve_day_known_answer_and_residual():
    g = MicroGrid()
    x = g.solve_day(60, 55)                   # hand calc: x=(2*55-60)/5=10, y=(4*60-3*55)/5=15
    assert x == pytest.approx([10.0, 15.0])
    assert g.A @ x == pytest.approx([60.0, 55.0])


def test_vectorised_equals_loop():
    g = MicroGrid()
    D = np.random.default_rng(0).uniform(40, 80, size=(2, 30))
    assert np.allclose(g.solve_loop(D), g.solve_vectorised(D))


def test_daily_cost_and_monthly_cost():
    g = MicroGrid()
    X = np.array([[10.0, 20.0], [15.0, 5.0]])   # day1: 10 solar, 15 battery
    assert g.daily_cost(X) == pytest.approx([150 * 10 + 450 * 15, 150 * 20 + 450 * 5])
    assert g.monthly_cost(X) == pytest.approx(8250 + 5250)


def test_infeasible_day_detected_and_nnls_is_nonnegative():
    g = MicroGrid()
    D = np.array([[60.0, 40.0], [55.0, 60.0]])  # day 2: y = (160-180)/5 < 0
    plan = g.plan(D, strategy="nnls")
    assert plan.infeasible.tolist() == [False, True]
    assert (plan.dispatch >= 0).all()
    assert plan.unmet[0] == pytest.approx(0.0, abs=1e-9)   # feasible day untouched
    assert plan.unmet[1] > 0                                 # unmet demand reported


def test_clip_strategy_zeroes_negative_entries():
    g = MicroGrid()
    plan = g.plan(np.array([[40.0], [60.0]]), strategy="clip")
    assert plan.raw[1, 0] < 0 and plan.dispatch[1, 0] == 0.0


def test_usage_stats_sample_variance():
    g = MicroGrid()
    X = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])
    s = g.usage_stats(X)
    assert s["solar"]["variance"] == pytest.approx(1.0)   # sample variance (n-1)
    assert s["battery"]["variance"] == pytest.approx(0.0)


def test_csv_is_reproducible_and_loads(tmp_path):
    a = generate_demand_csv(tmp_path / "a.csv", seed=7)
    b = generate_demand_csv(tmp_path / "b.csv", seed=7)
    assert a.read_text() == b.read_text()
    dates, D = load_demand_csv(a)
    assert D.shape == (2, 30) and len(dates) == 30


def test_hybrid_solves_3x3():
    h = HybridMicroGrid()
    x = h.solve_day(100, 90, 40)
    assert h.A @ x == pytest.approx([100, 90, 40])


def test_sensitivity_bounded_by_condition_number():
    g = MicroGrid()
    r = g.sensitivity([60, 55], rel=0.05, n_draws=1000, seed=1)
    assert r["max_amplification"] <= r["condition_number"] + 1e-9
    assert r["max_rel_input_change"] <= 0.05 * 1.01


# ------------------------------------------------------------------ edge cases
def test_singular_matrix_rejected():
    with pytest.raises(SingularSystemError):
        MicroGrid(A=[[1, 2], [2, 4]])


def test_hybrid_with_dependent_row_rejected():
    A = [[3, 2, 1], [4, 1, 2], [7, 3, 3]]       # row3 = row1 + row2
    with pytest.raises(SingularSystemError):
        HybridMicroGrid(A=A)


@pytest.mark.parametrize("bad", [(-1, 50), (float("nan"), 50), (float("inf"), 5)])
def test_solve_day_rejects_invalid_demand(bad):
    with pytest.raises(ValueError):
        MicroGrid().solve_day(*bad)


def test_solve_day_wrong_arity_and_empty_batch():
    g = MicroGrid()
    with pytest.raises(ValueError):
        g.solve_day(1.0)
    with pytest.raises(ValueError):
        g.solve_vectorised(np.empty((2, 0)))


def test_usage_stats_needs_two_days():
    with pytest.raises(ValueError):
        MicroGrid().usage_stats(np.array([[1.0], [2.0]]))


def test_read_positive_float_reprompts_until_valid():
    answers = iter(["", "abc", "-5", "nan", "  12.5 "])
    msgs = []
    value = read_positive_float("d1: ", input_fn=lambda _: next(answers), output_fn=msgs.append)
    assert value == 12.5
    assert len(msgs) == 4                       # one complaint per bad answer


def test_read_positive_float_gives_up():
    with pytest.raises(ValueError):
        read_positive_float("x", input_fn=lambda _: "", output_fn=lambda _: None, max_attempts=3)


def test_load_csv_rejects_bad_rows(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("date,d1_kwh,d2_kwh\n2026-09-01,abc,50\n")
    with pytest.raises(ValueError):
        load_demand_csv(p)
    p.write_text("date,d1_kwh,d2_kwh\n")
    with pytest.raises(ValueError):
        load_demand_csv(p)
