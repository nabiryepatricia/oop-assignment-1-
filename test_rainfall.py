"""Unit tests for src/rainfall.py (run with:  pytest -q)."""
import numpy as np
import pytest
from scipy.spatial import distance

from src.rainfall import (
    CropRule,
    Region,
    RegionComparator,
    Status,
    Window,
    cosine_similarity,
    default_crops,
    euclidean_distance,
    pearson_correlation,
    simulate_years,
    suitability_reliability,
)

KAMPALA = [120, 140, 180, 200, 220, 180, 90, 70, 60, 100, 110, 130]
GULU = [8, 25, 75, 160, 190, 145, 170, 215, 175, 150, 60, 15]
MBARARA = [70, 85, 120, 140, 90, 25, 20, 55, 100, 125, 120, 90]


# ---------------------------------------------------------------- Region
def test_region_statistics_known_values():
    r = Region("Kampala", KAMPALA)
    assert len(r) == 12
    assert r.annual_total == pytest.approx(1600.0)          # hand sum
    assert r.mean == pytest.approx(1600 / 12)
    assert r.wettest_month == ("May", 220.0)
    assert r.driest_month == ("Sep", 60.0)
    assert r.cv == pytest.approx(np.std(KAMPALA) / np.mean(KAMPALA))


@pytest.mark.parametrize("bad", [[1.0] * 11, [1.0] * 13, [-1.0] + [1.0] * 11, [float("nan")] + [1.0] * 11])
def test_region_rejects_invalid_rainfall(bad):
    with pytest.raises(ValueError):
        Region("X", bad)


def test_region_rejects_blank_name():
    with pytest.raises(ValueError):
        Region("  ", KAMPALA)


# ---------------------------------------------------------------- CropRule
def test_croprule_boundaries_are_inclusive():
    rule = CropRule("maize", 125, 200)
    assert rule.classify(124.9) is Status.DROUGHT_RISK
    assert rule.classify(125) is Status.GOOD
    assert rule.classify(200) is Status.GOOD
    assert rule.classify(200.1) is Status.WATERLOGGING_RISK
    assert rule.label(60) == "Drought risk (maize)" and rule.label(150) == "Good for maize"


def test_from_seasonal_and_defaults():
    r = CropRule.from_seasonal("maize", 500, 800, cycle_months=4, min_run=4)
    assert (r.min_mm, r.max_mm) == (pytest.approx(125.0), pytest.approx(200.0))
    names = [c.name for c in default_crops()]
    assert names == ["maize", "beans", "coffee"]
    coffee = default_crops()[2]
    assert coffee.min_mm == pytest.approx(1100 / 12) and coffee.max_mm == pytest.approx(1650 / 12)


def test_croprule_vectorised_matches_scalar():
    rule = CropRule("beans", 85, 145)
    rain = np.array(MBARARA, dtype=float)
    assert [rule.classify(x) for x in rain] == list(rule.classify_array(rain))


def test_croprule_rejects_bad_parameters_and_inputs():
    with pytest.raises(ValueError):
        CropRule("x", 200, 100)                  # min >= max
    with pytest.raises(ValueError):
        CropRule("x", -5, 100)
    with pytest.raises(ValueError):
        CropRule("x", 10, 20).classify(-1)
    with pytest.raises(ValueError):
        CropRule("x", 10, 20).classify(float("nan"))
    with pytest.raises(ValueError):
        CropRule("x", 10, 20).scaled(0)


# ---------------------------------------------------------------- similarity measures
def test_cosine_matches_scipy():
    a, b = np.array(KAMPALA, float), np.array(GULU, float)
    assert cosine_similarity(a, b) == pytest.approx(1 - distance.cosine(a, b))


def test_cosine_is_scale_invariant_but_euclidean_is_not():
    a = np.array(KAMPALA, float)
    assert cosine_similarity(a, 5 * a) == pytest.approx(1.0)       # "identical" shape
    assert euclidean_distance(a, 5 * a) > 1000                       # very different amounts


def test_cosine_orthogonal_and_euclidean_345():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert euclidean_distance([0, 0], [3, 4]) == pytest.approx(5.0)


def test_pearson_known_cases_and_numpy():
    a = np.array(KAMPALA, float)
    assert pearson_correlation(a, 2 * a + 7) == pytest.approx(1.0)
    assert pearson_correlation(a, -a) == pytest.approx(-1.0)
    b = np.array(GULU, float)
    assert pearson_correlation(a, b) == pytest.approx(np.corrcoef(a, b)[0, 1])


def test_similarity_edge_cases():
    with pytest.raises(ValueError):
        cosine_similarity([0, 0, 0], [1, 2, 3])                      # zero vector
    with pytest.raises(ValueError):
        cosine_similarity([1, 2], [1, 2, 3])                         # length mismatch
    with pytest.raises(ValueError):
        pearson_correlation([5, 5, 5], [1, 2, 3])                    # constant series
    with pytest.raises(ValueError):
        euclidean_distance([], [])


def test_comparator_matrices_symmetric_with_correct_diagonals():
    regs = [Region("K", KAMPALA), Region("G", GULU), Region("M", MBARARA)]
    mats = RegionComparator(regs).all_matrices()
    for name, M in mats.items():
        assert np.allclose(M, M.T), name
    assert np.allclose(np.diag(mats["cosine similarity"]), 1.0)
    assert np.allclose(np.diag(mats["Euclidean distance (mm)"]), 0.0)
    with pytest.raises(ValueError):
        RegionComparator(regs[:1])


# ---------------------------------------------------------------- rainy seasons
def test_modality_of_the_three_regions():
    assert Region("Gulu", GULU).modality() == "unimodal"
    mb = Region("Mbarara", MBARARA)
    assert mb.modality() == "bimodal"
    assert [m for m in mb.rainy_season_peaks()] == [3, 9]            # Apr, Oct


def test_peak_at_year_boundary_is_not_missed():
    r = Region("Wrap", [190] + [10] * 10 + [200])                    # wet Dec-Jan, dry rest
    assert r.rainy_season_peaks() == [11]                            # Dec peak found
    assert r.modality() == "unimodal"


def test_flat_series_is_aseasonal_and_bad_prominence_rejected():
    flat = Region("Flat", [100] * 12)
    assert flat.rainy_season_peaks() == [] and flat.modality() == "aseasonal"
    with pytest.raises(ValueError):
        Region("G", GULU).rainy_season_peaks(min_prominence_frac=0)


def test_prominence_threshold_changes_the_answer():
    k = Region("Kampala", KAMPALA)
    assert k.modality(0.25) == "unimodal"
    assert k.modality(0.05) == "bimodal"                             # weak Dec bump now counts


# ---------------------------------------------------------------- suitability / planting windows
def test_monthly_labels_for_known_months():
    labels = Region("Mbarara", MBARARA).monthly_labels(default_crops())
    assert labels[5] == "Drought risk"                               # Jun, 25 mm
    assert labels[3] == "Good for maize, beans"                      # Apr, 140 mm


def test_planting_window_wraps_over_new_year():
    rain = [150, 150, 10, 10, 10, 10, 10, 10, 10, 10, 150, 150]      # good: Nov, Dec, Jan, Feb
    rule = CropRule("beans", 100, 200, min_run=3)
    assert Region("W", rain).planting_windows(rule) == [Window(10, 4)]
    assert Window(10, 4).months == ["Nov", "Dec", "Jan", "Feb"] and str(Window(10, 4)) == "Nov-Feb (4 months)"


def test_planting_window_too_short_or_all_good():
    rain = [150, 150, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    assert Region("S", rain).planting_windows(CropRule("beans", 100, 200, min_run=3)) == []
    assert Region("A", [150] * 12).planting_windows(CropRule("c", 100, 200)) == [Window(0, 12)]


def test_mbarara_beans_have_two_windows():
    beans = default_crops()[1]
    wins = Region("Mbarara", MBARARA).planting_windows(beans)
    assert [str(w) for w in wins] == ["Mar-May (3 months)", "Sep-Dec (4 months)"]


# ---------------------------------------------------------------- extension helpers
def test_simulate_years_shape_reproducible_and_nonnegative():
    reg = Region("Mbarara", MBARARA)
    a, b = simulate_years(reg, n_years=200, seed=5), simulate_years(reg, n_years=200, seed=5)
    assert a.shape == (200, 12) and np.array_equal(a, b) and (a >= 0).all()
    assert a.mean(axis=0) == pytest.approx(reg.rainfall, rel=0.2)    # centred on the climatology


def test_reliability_is_a_fraction_and_matches_hand_case():
    years = np.array([[150.0] * 12, [10.0] * 12])                    # one good year, one dry year
    rel = suitability_reliability(years, CropRule("x", 100, 200))
    assert rel.shape == (12,) and (rel == 0.5).all()


def test_extension_helpers_validate_inputs():
    with pytest.raises(ValueError):
        simulate_years(Region("G", GULU), n_years=0)
    with pytest.raises(ValueError):
        simulate_years(Region("G", GULU), cv=0)
    with pytest.raises(ValueError):
        suitability_reliability(np.ones((5, 11)), CropRule("x", 1, 2))
