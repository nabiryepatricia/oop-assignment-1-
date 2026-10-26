"""UBOS district population forecaster (Mini-Project 1).

Classes
-------
DistrictPopulation        Validated yearly population series for one district.
Forecaster (abstract)     fit() / predict(horizon) interface, shared validation
                          and residual-bootstrap prediction intervals.
  LinearTrendForecaster     least-squares line via ``np.polyfit``.
  CAGRForecaster            exponential growth at the compound annual growth rate.
  FibonacciRatioForecaster  last value scaled by successive Fibonacci ratios.
ModelSelector             train/test validation (MAE, RMSE, MAPE) and model choice.
ClassroomPlanner          primary-school classrooms implied by a population.

Populations are in thousands, as in the assignment.
"""
from __future__ import annotations

import copy
import math
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# --------------------------------------------------------------------------
# Data container
# --------------------------------------------------------------------------
class DistrictPopulation:
    """Yearly population estimates (thousands) for one district.

    Validation: years and populations must have equal length (at least two
    observations), be finite, populations non-negative, and years strictly
    increasing whole numbers.
    """

    def __init__(self, name: str, years: Sequence[int], population: Sequence[float]) -> None:
        if not name or not name.strip():
            raise ValueError("name cannot be empty")
        y = np.asarray(years, dtype=float)
        p = np.asarray(population, dtype=float)
        if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
            raise ValueError("years and population must be 1-D sequences of equal length")
        if len(y) < 2:
            raise ValueError("need at least two observations")
        if not (np.isfinite(y).all() and np.isfinite(p).all()):
            raise ValueError("years and population must be finite")
        if (p < 0).any():
            raise ValueError("population cannot be negative")
        if not np.array_equal(y, np.round(y)) or (np.diff(y) <= 0).any():
            raise ValueError("years must be strictly increasing whole numbers")
        self.name = name
        self.years: np.ndarray = y.astype(int)
        self.population: np.ndarray = p

    def __repr__(self) -> str:
        return (f"DistrictPopulation({self.name!r}, {self.years[0]}-{self.years[-1]}, "
                f"n={len(self)}, last={self.population[-1]:,.0f}k)")

    def __len__(self) -> int:
        return len(self.years)

    # ---- growth ---------------------------------------------------------
    def yoy_growth(self) -> np.ndarray:
        """Year-on-year growth as fractions: (P_t - P_{t-1}) / P_{t-1}."""
        if (self.population[:-1] == 0).any():
            raise ValueError("growth is undefined after a zero population")
        return np.diff(self.population) / self.population[:-1]

    def cagr(self) -> float:
        """Compound annual growth rate: (P_last / P_first) ** (1 / years_elapsed) - 1."""
        first, last = self.population[0], self.population[-1]
        if first <= 0:
            raise ValueError("CAGR needs a positive starting population")
        return float((last / first) ** (1.0 / (self.years[-1] - self.years[0])) - 1.0)

    # ---- descriptive statistics ----------------------------------------
    def stats_statistics(self) -> dict[str, float]:
        """Mean, median, variance and st.dev with the ``statistics`` module.

        ``variance``/``stdev`` are *sample* statistics (divide by n - 1);
        ``pvariance``/``pstdev`` are *population* statistics (divide by n).
        """
        v = self.population.tolist()
        return {
            "mean": statistics.mean(v), "median": statistics.median(v),
            "variance": statistics.variance(v), "stdev": statistics.stdev(v),
            "pvariance": statistics.pvariance(v), "pstdev": statistics.pstdev(v),
        }

    def stats_numpy(self, ddof: int = 0) -> dict[str, float]:
        """Mean, median, variance and st.dev with NumPy.

        ``ddof`` ("delta degrees of freedom") sets the divisor to ``n - ddof``.
        NumPy's default is ``ddof = 0`` (population); ``ddof = 1`` gives the
        sample statistics that ``statistics.variance`` returns.
        """
        if not 0 <= ddof < len(self):
            raise ValueError("ddof must satisfy 0 <= ddof < n")
        p = self.population
        return {"mean": float(np.mean(p)), "median": float(np.median(p)),
                "variance": float(np.var(p, ddof=ddof)), "stdev": float(np.std(p, ddof=ddof))}

    # ---- splitting ------------------------------------------------------
    def split(self, last_train_year: int) -> tuple["DistrictPopulation", "DistrictPopulation"]:
        """Split into (train, test); train ends at ``last_train_year`` inclusive."""
        mask = self.years <= last_train_year
        if mask.sum() < 2 or (~mask).sum() < 1:
            raise ValueError("split must leave at least 2 training years and 1 test year")
        return (DistrictPopulation(self.name, self.years[mask], self.population[mask]),
                DistrictPopulation(self.name, self.years[~mask], self.population[~mask]))


# --------------------------------------------------------------------------
# Forecasters
# --------------------------------------------------------------------------
class Forecaster(ABC):
    """Abstract forecaster: ``fit(years, values)`` then ``predict(horizon)``.

    ``fit`` and ``predict`` are shared template methods that validate their
    inputs; subclasses only implement the model-specific ``_fit`` and
    ``_predict_years``, so validation logic is not duplicated.
    """

    name: str = "Forecaster"

    def __init__(self) -> None:
        self.years_: np.ndarray | None = None
        self.values_: np.ndarray | None = None

    # ---- public interface ----------------------------------------------
    def fit(self, years: Sequence[int], values: Sequence[float]) -> "Forecaster":
        y, v = np.asarray(years, dtype=float), np.asarray(values, dtype=float)
        if y.ndim != 1 or y.shape != v.shape or len(y) < 2:
            raise ValueError("years and values must be 1-D, equal length, with at least 2 points")
        if not (np.isfinite(y).all() and np.isfinite(v).all()) or (v < 0).any():
            raise ValueError("values must be finite and non-negative")
        self.years_, self.values_ = y.astype(int), v
        self._fit()
        return self

    def fit_district(self, district: DistrictPopulation) -> "Forecaster":
        """Convenience: fit on a :class:`DistrictPopulation`."""
        return self.fit(district.years, district.population)

    def predict(self, horizon: int) -> np.ndarray:
        """Forecast the ``horizon`` years after the last training year."""
        return self._predict_years(self.forecast_years(horizon))

    def forecast_years(self, horizon: int) -> np.ndarray:
        self._check_fitted()
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        return self.years_[-1] + np.arange(1, horizon + 1)

    def fitted_values(self) -> np.ndarray:
        """In-sample fitted values (NaN where the model has no in-sample fit)."""
        self._check_fitted()
        return self._fitted()

    def clone(self) -> "Forecaster":
        """An unfitted copy with the same hyper-parameters."""
        new = copy.copy(self)
        new.years_ = new.values_ = None
        return new

    def bootstrap_interval(self, horizon: int, n_boot: int = 2000, level: float = 0.95,
                           seed: int = 42) -> dict[str, np.ndarray]:
        """Prediction interval by residual bootstrap.

        1. residuals e_t = y_t - fitted_t, centred to mean zero (the CAGR curve is
           pinned to the first and last points, so its raw residuals are biased);
        2. repeat ``n_boot`` times: resample the residuals with replacement,
           build a pseudo-series fitted_t + e*_t, refit, forecast, and add a
           resampled residual to each forecast (future noise);
        3. take the (1-level)/2 and (1+level)/2 percentiles per forecast year.
        Refitting captures parameter uncertainty; the added residual captures
        irreducible noise.
        """
        self._check_fitted()
        if n_boot < 1 or not 0 < level < 1:
            raise ValueError("n_boot must be >= 1 and level in (0, 1)")
        fitted = self.fitted_values()
        if np.isnan(fitted).any():
            raise NotImplementedError(f"{self.name} has no in-sample fit, so residuals are undefined")
        residuals = self.values_ - fitted
        residuals = residuals - residuals.mean()      # centre: OLS residuals already average 0, the CAGR curve's do not
        rng = np.random.default_rng(seed)
        sims = np.empty((n_boot, horizon))
        for b in range(n_boot):
            pseudo = np.maximum(fitted + rng.choice(residuals, size=len(residuals), replace=True), 1e-9)
            model = self.clone().fit(self.years_, pseudo)
            sims[b] = model.predict(horizon) + rng.choice(residuals, size=horizon, replace=True)
        lo, hi = np.percentile(sims, [100 * (1 - level) / 2, 100 * (1 + level) / 2], axis=0)
        return {"point": self.predict(horizon), "lower": lo, "upper": hi}

    # ---- model-specific parts ------------------------------------------
    @abstractmethod
    def _fit(self) -> None: ...

    @abstractmethod
    def _predict_years(self, future_years: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def _fitted(self) -> np.ndarray: ...

    def _check_fitted(self) -> None:
        if self.years_ is None:
            raise RuntimeError("call fit() before predict()")

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class LinearTrendForecaster(Forecaster):
    """Straight-line trend ``P = slope * year + intercept`` fitted with ``np.polyfit``."""

    name = "Linear trend"

    def _fit(self) -> None:
        self.slope_, self.intercept_ = np.polyfit(self.years_, self.values_, 1)

    def _predict_years(self, future_years: np.ndarray) -> np.ndarray:
        return self.slope_ * future_years + self.intercept_

    def _fitted(self) -> np.ndarray:
        return self.slope_ * self.years_ + self.intercept_


class CAGRForecaster(Forecaster):
    """Exponential growth at the training-period CAGR, anchored on the last observation.

    ``P(last + h) = P(last) * (1 + CAGR) ** h``
    """

    name = "CAGR (exponential)"

    def _fit(self) -> None:
        first, last = self.values_[0], self.values_[-1]
        if first <= 0:
            raise ValueError("CAGR needs a positive first value")
        self.rate_ = float((last / first) ** (1.0 / (self.years_[-1] - self.years_[0])) - 1.0)

    def _predict_years(self, future_years: np.ndarray) -> np.ndarray:
        return self.values_[-1] * (1.0 + self.rate_) ** (future_years - self.years_[-1])

    def _fitted(self) -> np.ndarray:
        return self.values_[0] * (1.0 + self.rate_) ** (self.years_ - self.years_[0])


def fibonacci_numbers(n: int) -> list[int]:
    """First ``n`` Fibonacci numbers F1..Fn = 1, 1, 2, 3, 5, ..."""
    if n < 0:
        raise ValueError("n must be non-negative")
    out: list[int] = []
    a, b = 1, 1
    for _ in range(n):
        out.append(a)
        a, b = b, a + b
    return out


class FibonacciRatioForecaster(Forecaster):
    """Scale the last observed value by successive Fibonacci ratios.

    The h-th ratio is ``F(k+h) / F(k+h-1)`` with ``k = start_index``.  With the
    default ``start_index = 1`` the ratios are 1/1, 2/1, 3/2, 5/3, 8/5, ... and

    * ``chain=False`` (default, the literal reading): ``forecast_h = last * ratio_h``
    * ``chain=True``  : ``forecast_h = last * ratio_1 * ... * ratio_h`` (= last * F(h+1))

    The model has no parameters estimated from the data, so there is no
    in-sample fit; it exists as the baseline to be critiqued.
    """

    name = "Fibonacci ratio"

    def __init__(self, start_index: int = 1, chain: bool = False) -> None:
        super().__init__()
        if start_index < 1:
            raise ValueError("start_index must be at least 1")
        self.start_index, self.chain = start_index, chain

    def ratios(self, horizon: int) -> np.ndarray:
        """The first ``horizon`` ratios F(k+h)/F(k+h-1), h = 1..horizon."""
        fib = fibonacci_numbers(self.start_index + horizon)
        return np.array([fib[self.start_index + h - 1] / fib[self.start_index + h - 2]
                         for h in range(1, horizon + 1)])

    def _fit(self) -> None:
        self.last_ = float(self.values_[-1])

    def _predict_years(self, future_years: np.ndarray) -> np.ndarray:
        r = self.ratios(len(future_years))
        return self.last_ * (np.cumprod(r) if self.chain else r)

    def _fitted(self) -> np.ndarray:
        return np.full(len(self.years_), np.nan)


# --------------------------------------------------------------------------
# Metrics and model selection
# --------------------------------------------------------------------------
def _pair(actual, forecast) -> tuple[np.ndarray, np.ndarray]:
    a, f = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    if a.ndim != 1 or a.shape != f.shape or a.size == 0:
        raise ValueError("actual and forecast must be non-empty 1-D arrays of equal length")
    return a, f


def mae(actual, forecast) -> float:
    """Mean absolute error (same units as the data)."""
    a, f = _pair(actual, forecast)
    return float(np.mean(np.abs(a - f)))


def rmse(actual, forecast) -> float:
    """Root mean squared error (same units; penalises large misses more)."""
    a, f = _pair(actual, forecast)
    return float(np.sqrt(np.mean((a - f) ** 2)))


def mape(actual, forecast) -> float:
    """Mean absolute percentage error, in percent; undefined if an actual value is 0."""
    a, f = _pair(actual, forecast)
    if (a == 0).any():
        raise ValueError("MAPE is undefined when an actual value is zero")
    return float(np.mean(np.abs((a - f) / a)) * 100.0)


@dataclass(frozen=True)
class ModelScore:
    """Test-set accuracy of one model on one district."""

    model: str
    mae: float
    rmse: float
    mape: float
    forecast: np.ndarray


class ModelSelector:
    """Train each model on the early years, score it on the held-out years, pick the best."""

    CRITERIA = ("mae", "rmse", "mape")

    def __init__(self, models: Sequence[Forecaster], last_train_year: int = 2021, criterion: str = "rmse") -> None:
        if not models:
            raise ValueError("need at least one model")
        if criterion not in self.CRITERIA:
            raise ValueError(f"criterion must be one of {self.CRITERIA}")
        self.models = list(models)
        self.last_train_year = last_train_year
        self.criterion = criterion

    def evaluate(self, district: DistrictPopulation) -> list[ModelScore]:
        """Fit every model on the training years and score its forecast of the test years."""
        train, test = district.split(self.last_train_year)
        scores = []
        for model in self.models:
            pred = model.clone().fit_district(train).predict(len(test))
            scores.append(ModelScore(model.name, mae(test.population, pred),
                                     rmse(test.population, pred), mape(test.population, pred), pred))
        return scores

    def best(self, scores: Sequence[ModelScore]) -> ModelScore:
        """Lowest value of the selection criterion."""
        return min(scores, key=lambda s: getattr(s, self.criterion))

    def select(self, district: DistrictPopulation) -> Forecaster:
        """Return an unfitted clone of the winning model."""
        winner = self.best(self.evaluate(district)).model
        return next(m for m in self.models if m.name == winner).clone()


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------
class ClassroomPlanner:
    """Primary-school classrooms implied by a population (given in thousands)."""

    def __init__(self, primary_share: float = 0.18, pupils_per_classroom: int = 53) -> None:
        if not 0 < primary_share < 1:
            raise ValueError("primary_share must lie in (0, 1)")
        if pupils_per_classroom < 1:
            raise ValueError("pupils_per_classroom must be at least 1")
        self.primary_share, self.pupils_per_classroom = primary_share, pupils_per_classroom

    def pupils(self, population_thousands: float) -> float:
        """Primary-school-age children."""
        if population_thousands < 0:
            raise ValueError("population cannot be negative")
        return population_thousands * 1000.0 * self.primary_share

    def classrooms_needed(self, population_thousands: float) -> int:
        """Classrooms to seat all pupils (rounded up: no partial classrooms)."""
        return math.ceil(self.pupils(population_thousands) / self.pupils_per_classroom)

    def additional_classrooms(self, current_thousands: float, future_thousands: float) -> int:
        """New classrooms needed, assuming today's stock exactly meets today's need (never negative)."""
        return max(0, self.classrooms_needed(future_thousands) - self.classrooms_needed(current_thousands))


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
@dataclass
class DistrictForecast:
    """Everything the combined plot needs for one district."""

    district: DistrictPopulation
    model_name: str
    split_year: int
    train_fitted: np.ndarray | None      # fit on the training years (None/NaN if the model has none)
    test_forecast: np.ndarray            # forecast of the held-out years from the training-only fit
    future_years: np.ndarray
    future_values: np.ndarray            # final forecast (model refitted on all data)
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None


def plot_district_forecasts(results: Sequence[DistrictForecast], ncols: int = 3):
    """One subplot per district: actual, fitted, test forecast, final forecast, 95% interval, split marked."""
    import matplotlib.pyplot as plt

    n = len(results)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)
    for ax, res in zip(axes.ravel(), results):
        d = res.district
        ax.axvspan(res.split_year + 0.5, d.years[-1] + 0.5, color="#f2d7a6", alpha=0.35,
                   label="test period")
        ax.axvline(res.split_year + 0.5, color="grey", ls="--", lw=1, label="train/test split")
        ax.plot(d.years, d.population, "ko-", ms=4, label="actual")
        if res.train_fitted is not None and not np.isnan(res.train_fitted).all():
            n_train = len(res.train_fitted)
            ax.plot(d.years[:n_train], res.train_fitted, color="#1f77b4", lw=2, label="fitted (train)")
        test_years = d.years[d.years > res.split_year]
        ax.plot(test_years, res.test_forecast, "s--", color="#d62728", ms=4, label="test forecast")
        ax.plot(res.future_years, res.future_values, "^-", color="#2ca02c", ms=4, label="forecast 2025-29")
        if res.lower is not None:
            ax.fill_between(res.future_years, res.lower, res.upper, color="#2ca02c", alpha=0.2,
                            label="95% prediction interval")
        ax.set_title(f"{d.name} ({res.model_name})", fontsize=10)
        ax.set_xlabel("Year")
        ax.set_ylabel("Population (thousands)")
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.97, 0.06), fontsize=8)
    fig.tight_layout()
    return fig
