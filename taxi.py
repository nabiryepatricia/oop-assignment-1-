"""Taxi (matatu) route revenue, pricing and fleet planner (Mini-Project 5).

Classes
-------
Route                          Daily passenger counts + fare: revenue and statistics.
SupplyDemandModel              Linear demand/supply curves solved as a 2x2 system.
MovingAverageForecaster        Mean of the last k days.
ExponentialSmoothingForecaster Simple exponential smoothing with tunable alpha.
SeasonalNaiveForecaster        Same weekday last week (extension).
WalkForwardBacktester          Rolling-origin evaluation and alpha grid search.
FleetPlanner                   Vehicles needed for a forecast, with a buffer.

The forecasters subclass the abstract ``Forecaster`` from ``src.population``
(Mini-Project 1), and the linear-trend model is reused from there unchanged:
"days" play the role of "years".
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import linalg

from src.population import Forecaster, LinearTrendForecaster  # noqa: F401  (re-exported for the notebook)

WEEKLY_PATTERN = (1.00, 1.00, 1.00, 1.05, 1.30, 1.10, 0.70)  # Mon..Sun multipliers: busy Friday, quiet Sunday


# --------------------------------------------------------------------------
# Route
# --------------------------------------------------------------------------
class Route:
    """One taxi route: daily passenger counts (NumPy array) and a fare in UGX."""

    def __init__(self, name: str, passengers: Sequence[float], fare: float) -> None:
        p = np.asarray(passengers, dtype=float)
        if not name or not name.strip():
            raise ValueError("name cannot be empty")
        if p.ndim != 1 or len(p) < 2:
            raise ValueError("passengers must be a 1-D sequence with at least 2 days")
        if not np.isfinite(p).all() or (p < 0).any():
            raise ValueError("passenger counts must be finite and non-negative")
        if not math.isfinite(fare) or fare <= 0:
            raise ValueError("fare must be a positive number")
        self.name, self.passengers, self.fare = name, p, float(fare)

    def __repr__(self) -> str:
        return f"Route({self.name!r}, days={len(self)}, fare=UGX {self.fare:,.0f})"

    def __len__(self) -> int:
        return len(self.passengers)

    @property
    def days(self) -> np.ndarray:
        """Day numbers 1..n."""
        return np.arange(1, len(self) + 1)

    def daily_revenue(self) -> np.ndarray:
        """UGX per day = passengers x fare."""
        return self.passengers * self.fare

    def total_revenue(self) -> float:
        return float(self.daily_revenue().sum())

    def describe(self) -> dict[str, float]:
        """Mean, median, sample variance and st.dev of daily passengers (``statistics`` module)."""
        v = self.passengers.tolist()
        return {"mean": statistics.mean(v), "median": statistics.median(v),
                "variance": statistics.variance(v), "stdev": statistics.stdev(v),
                "min": min(v), "max": max(v)}


# --------------------------------------------------------------------------
# Supply and demand
# --------------------------------------------------------------------------
class SupplyDemandModel:
    """Linear demand ``Qd = a - b*P`` and supply ``Qs = c + d*P`` (P = fare in UGX, Q = passengers per trip-hour).

    Setting Qd = Qs = Q gives a 2x2 linear system in the unknowns (Q, P):

        Q + b*P = a
        Q - d*P = c
    """

    def __init__(self, demand_intercept: float = 120.0, demand_slope: float = 0.02,
                 supply_intercept: float = 10.0, supply_slope: float = 0.03) -> None:
        if demand_slope <= 0 or supply_slope <= 0:
            raise ValueError("demand must slope down (b > 0) and supply up (d > 0)")
        self.a, self.b, self.c, self.d = demand_intercept, demand_slope, supply_intercept, supply_slope

    def system(self) -> tuple[np.ndarray, np.ndarray]:
        """Coefficient matrix A and right-hand side for ``A @ [Q, P] = rhs``."""
        return np.array([[1.0, self.b], [1.0, -self.d]]), np.array([self.a, self.c])

    def equilibrium(self) -> tuple[float, float]:
        """Solve the system with ``scipy.linalg.solve``; returns ``(price, quantity)``."""
        A, rhs = self.system()
        q, p = linalg.solve(A, rhs)
        if p <= 0 or q <= 0:
            raise ValueError("no economically meaningful equilibrium (price or quantity <= 0)")
        return float(p), float(q)

    def demand(self, price: float) -> float:
        return self.a - self.b * price

    def supply(self, price: float) -> float:
        return self.c + self.d * price

    def excess_demand(self, price: float) -> float:
        """Qd - Qs at ``price``: positive = shortage (queues), negative = surplus (empty seats)."""
        return self.demand(price) - self.supply(price)

    def compare_fare(self, fare: float) -> str:
        """'below', 'above' or 'at' the equilibrium price."""
        p_star, _ = self.equilibrium()
        if math.isclose(fare, p_star, rel_tol=1e-9):
            return "at"
        return "below" if fare < p_star else "above"

    def traded_quantity(self, price: float) -> float:
        """Passengers actually served at a fixed fare: the short side of the market."""
        return max(0.0, min(self.demand(price), self.supply(price)))


# --------------------------------------------------------------------------
# Forecasters (subclasses of the abstract Forecaster from Project 1)
# --------------------------------------------------------------------------
class MovingAverageForecaster(Forecaster):
    """Forecast = mean of the last ``window`` observations (flat for any horizon)."""

    name = "Moving average"

    def __init__(self, window: int = 3) -> None:
        super().__init__()
        if window < 1:
            raise ValueError("window must be at least 1")
        self.window = window
        self.name = f"{window}-day moving average"

    def _fit(self) -> None:
        if len(self.values_) < self.window:
            raise ValueError(f"need at least {self.window} observations")
        self.level_ = float(self.values_[-self.window:].mean())

    def _predict_years(self, future_years: np.ndarray) -> np.ndarray:
        return np.full(len(future_years), self.level_)

    def _fitted(self) -> np.ndarray:          # one-step-ahead: mean of the previous `window` values
        out = np.full(len(self.values_), np.nan)
        for t in range(self.window, len(self.values_)):
            out[t] = self.values_[t - self.window:t].mean()
        return out


class ExponentialSmoothingForecaster(Forecaster):
    """Simple exponential smoothing: ``level_t = alpha*y_t + (1-alpha)*level_{t-1}``.

    alpha near 1 reacts quickly to new data; near 0 it averages over a long
    history.  The forecast is the final level (flat for any horizon).
    """

    name = "Exponential smoothing"

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        if not 0 < alpha <= 1:
            raise ValueError("alpha must lie in (0, 1]")
        self.alpha = alpha
        self.name = f"SES (alpha={alpha:.2f})"

    def _levels(self) -> np.ndarray:
        levels = np.empty(len(self.values_))
        levels[0] = self.values_[0]
        for t in range(1, len(self.values_)):
            levels[t] = self.alpha * self.values_[t] + (1 - self.alpha) * levels[t - 1]
        return levels

    def _fit(self) -> None:
        self.level_ = float(self._levels()[-1])

    def _predict_years(self, future_years: np.ndarray) -> np.ndarray:
        return np.full(len(future_years), self.level_)

    def _fitted(self) -> np.ndarray:          # one-step-ahead: the level before seeing y_t
        out = np.full(len(self.values_), np.nan)
        out[1:] = self._levels()[:-1]
        return out


class SeasonalNaiveForecaster(Forecaster):
    """Forecast = the value observed one season (``period`` days) earlier."""

    name = "Seasonal naive"

    def __init__(self, period: int = 7) -> None:
        super().__init__()
        if period < 1:
            raise ValueError("period must be at least 1")
        self.period = period
        self.name = f"Seasonal naive (period {period})"

    def _fit(self) -> None:
        if len(self.values_) < self.period:
            raise ValueError(f"need at least one full season ({self.period} observations)")

    def _predict_years(self, future_years: np.ndarray) -> np.ndarray:
        h = np.arange(len(future_years))
        return self.values_[len(self.values_) - self.period + (h % self.period)].astype(float)

    def _fitted(self) -> np.ndarray:
        out = np.full(len(self.values_), np.nan)
        out[self.period:] = self.values_[:-self.period]
        return out


# --------------------------------------------------------------------------
# Walk-forward backtesting
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class BacktestResult:
    """One-step-ahead results over the target days."""

    model: str
    targets: np.ndarray
    actual: np.ndarray
    forecast: np.ndarray

    @property
    def errors(self) -> np.ndarray:
        return self.actual - self.forecast

    @property
    def mae(self) -> float:
        return float(np.mean(np.abs(self.errors)))

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.errors ** 2)))


class WalkForwardBacktester:
    """Rolling-origin evaluation: for each target day t, fit on days 1..t-1 only, forecast day t.

    ``first_target`` / ``last_target`` are day numbers (1-based, inclusive).
    No future data is ever used when forecasting a day.
    """

    def __init__(self, first_target: int = 4, last_target: int | None = None) -> None:
        if first_target < 2:
            raise ValueError("first_target must be at least 2 (need history)")
        self.first_target, self.last_target = first_target, last_target

    def evaluate(self, model: Forecaster, days: Sequence[int], values: Sequence[float]) -> BacktestResult:
        days, values = np.asarray(days), np.asarray(values, dtype=float)
        last = len(values) if self.last_target is None else self.last_target
        if last > len(values) or last < self.first_target:
            raise ValueError("target range is outside the data")
        targets, actual, forecast = [], [], []
        for t in range(self.first_target, last + 1):
            fitted = model.clone().fit(days[: t - 1], values[: t - 1])
            targets.append(t)
            actual.append(values[t - 1])
            forecast.append(float(fitted.predict(1)[0]))
        return BacktestResult(model.name, np.array(targets), np.array(actual), np.array(forecast))

    def evaluate_route(self, model: Forecaster, route: Route) -> BacktestResult:
        return self.evaluate(model, route.days, route.passengers)

    def tune_alpha(self, days: Sequence[int], values: Sequence[float],
                   grid: Sequence[float] | None = None) -> tuple[float, dict[float, float]]:
        """Grid search for the SES alpha with the lowest walk-forward MAE.

        Returns ``(best_alpha, {alpha: mae})``; ties go to the smaller alpha.
        """
        grid = np.round(np.arange(0.05, 1.0001, 0.05), 2) if grid is None else grid
        table = {float(a): self.evaluate(ExponentialSmoothingForecaster(float(a)), days, values).mae for a in grid}
        best = min(table, key=lambda a: (round(table[a], 9), a))   # rounding stops float noise breaking true ties
        return best, table


# --------------------------------------------------------------------------
# Fleet planning
# --------------------------------------------------------------------------
class FleetPlanner:
    """Vehicles needed to carry a forecast number of daily passengers."""

    def __init__(self, trips_per_vehicle: int = 8, seats: int = 14, buffer: float = 0.15, min_vehicles: int = 1) -> None:
        if trips_per_vehicle < 1 or seats < 1:
            raise ValueError("trips_per_vehicle and seats must be at least 1")
        if buffer < 0:
            raise ValueError("buffer cannot be negative")
        if min_vehicles < 0:
            raise ValueError("min_vehicles cannot be negative")
        self.trips, self.seats, self.buffer, self.min_vehicles = trips_per_vehicle, seats, buffer, min_vehicles

    @property
    def capacity_per_vehicle(self) -> int:
        """Passengers one vehicle can carry per day (trips x seats)."""
        return self.trips * self.seats

    def vehicles_needed(self, passengers: float) -> int:
        """ceil(passengers x (1 + buffer) / capacity), never below ``min_vehicles``.

        Always rounded *up*: rounding down would leave the route short of seats.
        """
        if passengers < 0 or not math.isfinite(passengers):
            raise ValueError("passengers must be a finite, non-negative number")
        required = round(passengers * (1.0 + self.buffer) / self.capacity_per_vehicle, 9)  # guard float noise
        return max(self.min_vehicles, math.ceil(required))

    def utilisation(self, passengers: float, vehicles: int) -> float:
        """Share of seat-trips filled by ``passengers`` with ``vehicles`` vehicles."""
        if vehicles < 1:
            raise ValueError("vehicles must be at least 1")
        return passengers / (vehicles * self.capacity_per_vehicle)


# --------------------------------------------------------------------------
# Extension: simulated data with a weekly pattern
# --------------------------------------------------------------------------
def generate_weekly_demand(n_days: int = 60, base: float = 50.0, seed: int = 42,
                           noise_sd: float = 3.0) -> np.ndarray:
    """Seeded daily passengers with a weekly pattern (day 1 = Monday): busiest Friday, quietest Sunday."""
    if n_days < 1 or base <= 0 or noise_sd < 0:
        raise ValueError("n_days >= 1, base > 0 and noise_sd >= 0 are required")
    rng = np.random.default_rng(seed)
    pattern = np.array(WEEKLY_PATTERN)[np.arange(n_days) % 7]
    return np.maximum(np.round(base * pattern + rng.normal(0, noise_sd, n_days)), 0.0)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot_revenue_forecasts(routes: Sequence[Route], backtests: dict[str, BacktestResult],
                           day11_forecast: dict[str, float]):
    """Per route: actual daily revenue, one-step-ahead forecasts of the best model, and the day-11 forecast."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(routes), figsize=(5.2 * len(routes), 4.2), squeeze=False)
    for ax, route in zip(axes[0], routes):
        rev = route.daily_revenue() / 1000.0
        ax.plot(route.days, rev, "ko-", label="actual revenue")
        bt = backtests[route.name]
        ax.plot(bt.targets, bt.forecast * route.fare / 1000.0, "s--", color="#d62728", ms=4,
                label=f"one-step forecasts ({bt.model})")
        f11 = day11_forecast[route.name] * route.fare / 1000.0
        ax.plot([len(route) + 1], [f11], "*", color="#2ca02c", ms=14, label=f"day {len(route) + 1} forecast")
        ax.set_title(f"{route.name} (fare UGX {route.fare:,.0f})", fontsize=10)
        ax.set_xlabel("Day")
        ax.set_ylabel("Revenue (UGX thousand)")
        ax.set_xticks(range(1, len(route) + 2))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig


def plot_seasonal_comparison(values: np.ndarray, results: dict[str, BacktestResult], start_day: int):
    """Simulated weekly demand with each model's one-step forecasts overlaid."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4.3))
    days = np.arange(1, len(values) + 1)
    ax.plot(days, values, "ko-", ms=3, lw=1, label="simulated passengers")
    for name, res in results.items():
        ax.plot(res.targets, res.forecast, "--", lw=1.4, label=f"{name} (MAE {res.mae:.1f})")
    ax.axvline(start_day - 0.5, color="grey", ls=":")
    ax.set_ylim(values.min() * 0.9, values.max() * 1.22)   # headroom so the legend does not hide peaks
    ax.set_xlabel("Day (day 1 = Monday)")
    ax.set_ylabel("Passengers per day")
    ax.set_title("Simulated 60 days with a weekly pattern: one-step-ahead forecasts")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
