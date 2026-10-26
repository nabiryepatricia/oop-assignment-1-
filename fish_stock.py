"""Lake Victoria fish stock and export-risk model (Mini-Project 3).

Classes
-------
FishStock     Discrete logistic growth with proportional harvesting.
PriceModel    Seeded, bounded random walk for the weekly price (UGX/kg).
Fishery       Composes a FishStock and a PriceModel into weekly revenue.
RiskAssessor  Descriptive statistics, CV-based risk class, Monte Carlo VaR.

Units: stock and harvest in tonnes, price in UGX/kg, revenue in UGX.
One model time-step is one week, as the assignment specifies.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

KG_PER_TONNE = 1_000.0


# --------------------------------------------------------------------------
# Baseline: Fibonacci "stock"
# --------------------------------------------------------------------------
def fibonacci(n: int) -> list[int]:
    """Return the first ``n`` Fibonacci numbers (1, 1, 2, 3, 5, ...)."""
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer")
    seq: list[int] = []
    a, b = 1, 1
    for _ in range(n):
        seq.append(a)
        a, b = b, a + b
    return seq


# --------------------------------------------------------------------------
# Biology
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class StockPath:
    """Result of a stock simulation.

    ``stock`` has length ``weeks + 1`` (N(0) .. N(weeks)); ``harvest_tonnes``
    has length ``weeks`` (the harvest h*N(t) taken in week t).
    """

    stock: np.ndarray
    harvest_tonnes: np.ndarray


class FishStock:
    """Discrete logistic growth with harvesting.

    ``N(t+1) = N(t) + r*N(t)*(1 - N(t)/K) - h*N(t)``

    Parameters
    ----------
    r : intrinsic growth rate per week (> 0)
    K : carrying capacity in tonnes (> 0)
    n0 : initial stock in tonnes (>= 0)
    h : fraction of the stock harvested each week, 0 <= h <= 1
    """

    def __init__(self, r: float = 0.4, K: float = 10_000.0, n0: float = 4_000.0, h: float = 0.10) -> None:
        if r <= 0 or K <= 0:
            raise ValueError("r and K must be positive")
        if n0 < 0:
            raise ValueError("n0 cannot be negative")
        if not 0.0 <= h <= 1.0:
            raise ValueError("h must lie in [0, 1]")
        self.r, self.K, self.n0, self.h = float(r), float(K), float(n0), float(h)

    def __repr__(self) -> str:
        return f"FishStock(r={self.r}, K={self.K:,.0f}, n0={self.n0:,.0f}, h={self.h})"

    def with_harvest(self, h: float) -> "FishStock":
        """Copy of this stock with a different harvest rate (for scenario runs)."""
        return FishStock(self.r, self.K, self.n0, h)

    # ---- theory ---------------------------------------------------------
    @property
    def msy(self) -> float:
        """Maximum sustainable yield of the continuous logistic model: rK/4 tonnes/week."""
        return self.r * self.K / 4.0

    @property
    def h_msy(self) -> float:
        """Harvest rate that attains MSY (= r/2)."""
        return self.r / 2.0

    @property
    def equilibrium_stock(self) -> float:
        """Non-zero fixed point K(1 - h/r); 0 when h >= r (stock cannot persist)."""
        return max(self.K * (1.0 - self.h / self.r), 0.0)

    @property
    def equilibrium_yield(self) -> float:
        """Long-run weekly harvest h * N* in tonnes."""
        return self.h * self.equilibrium_stock

    @property
    def harvest_status(self) -> str:
        """Biological verdict on the harvest rate, from the logistic equilibrium.

        * ``h <= r/2``  : "sustainable" (equilibrium stock >= K/2, yield <= MSY)
        * ``r/2 < h < r``: "overfishing" (stock settles below K/2; yield falls)
        * ``h >= r``     : "collapse" (no positive equilibrium; stock heads to zero)
        """
        if self.h >= self.r:
            return "collapse"
        return "sustainable" if self.h <= self.h_msy else "overfishing"

    # ---- simulation -----------------------------------------------------
    def simulate(
        self,
        weeks: int = 52,
        closed_weeks: Iterable[int] | None = None,
        year_length: int = 52,
    ) -> StockPath:
        """Iterate the model for ``weeks`` steps.

        ``closed_weeks`` lists week-of-year indices (0..year_length-1) with no
        harvesting; the closure repeats every year.
        """
        if weeks <= 0:
            raise ValueError("weeks must be positive")
        closed = {int(w) % year_length for w in (closed_weeks or [])}
        stock = np.empty(weeks + 1)
        harvest = np.empty(weeks)
        stock[0] = self.n0
        for t in range(weeks):
            h_t = 0.0 if (t % year_length) in closed else self.h
            harvest[t] = h_t * stock[t]
            growth = self.r * stock[t] * (1.0 - stock[t] / self.K)
            stock[t + 1] = max(stock[t] + growth - harvest[t], 0.0)
        return StockPath(stock, harvest)


# --------------------------------------------------------------------------
# Economics
# --------------------------------------------------------------------------
class PriceModel:
    """Bounded random walk for the weekly price in UGX/kg.

    ``P(t+1) = clip(P(t) + eps, low, high)``, ``eps ~ N(0, sigma^2)``.
    """

    def __init__(self, start: float = 12_000.0, sigma: float = 350.0, low: float = 9_000.0, high: float = 16_000.0) -> None:
        if not low < high:
            raise ValueError("low must be smaller than high")
        if not low <= start <= high:
            raise ValueError("start must lie inside [low, high]")
        if sigma < 0:
            raise ValueError("sigma cannot be negative")
        self.start, self.sigma, self.low, self.high = float(start), float(sigma), float(low), float(high)

    def __repr__(self) -> str:
        return f"PriceModel(start={self.start:,.0f}, sigma={self.sigma:,.0f}, bounds=[{self.low:,.0f}, {self.high:,.0f}])"

    def simulate(self, n_weeks: int, seed: int | np.random.Generator | None = 0, n_paths: int = 1) -> np.ndarray:
        """Return prices of shape ``(n_paths, n_weeks)``; week 0 equals ``start``."""
        if n_weeks <= 0 or n_paths <= 0:
            raise ValueError("n_weeks and n_paths must be positive")
        rng = np.random.default_rng(seed)
        steps = rng.normal(0.0, self.sigma, size=(n_paths, n_weeks))
        prices = np.empty((n_paths, n_weeks))
        prices[:, 0] = self.start
        for t in range(1, n_weeks):               # sequential clipping keeps the walk bounded
            prices[:, t] = np.clip(prices[:, t - 1] + steps[:, t], self.low, self.high)
        return prices


def weekly_revenue(harvest_tonnes: np.ndarray, price_per_kg: np.ndarray) -> np.ndarray:
    """Revenue in UGX = harvest (kg) x price (UGX/kg)."""
    return np.asarray(harvest_tonnes, dtype=float) * KG_PER_TONNE * np.asarray(price_per_kg, dtype=float)


class Fishery:
    """A stock plus a price process: produces revenue paths."""

    def __init__(self, stock: FishStock, prices: PriceModel) -> None:
        self.stock = stock
        self.prices = prices

    def revenue(
        self,
        weeks: int = 52,
        n_paths: int = 1,
        seed: int | None = 0,
        closed_weeks: Iterable[int] | None = None,
    ) -> np.ndarray:
        """Weekly revenue, shape ``(n_paths, weeks)``.

        The stock path is deterministic; only the price is random.  The same
        ``seed`` gives the same price draws for every harvest rate, so
        scenarios are compared on identical price paths.
        """
        path = self.stock.simulate(weeks, closed_weeks)
        prices = self.prices.simulate(weeks, seed, n_paths)
        return weekly_revenue(path.harvest_tonnes, prices)


# --------------------------------------------------------------------------
# Risk
# --------------------------------------------------------------------------
class RiskAssessor:
    """Risk classification by coefficient of variation plus Monte Carlo VaR.

    The rule is applied to the CV of *annual* revenue across Monte Carlo price
    paths.  The stock path is deterministic, so this CV isolates genuine
    uncertainty (price) and is not inflated by the predictable rise or fall of
    the stock.  It is dimensionless, so it does not depend on whether revenue
    is measured in UGX, UGX million or USD.

        CV <  low_cv            -> "Low"
        low_cv <= CV < high_cv  -> "Moderate"
        CV >= high_cv           -> "High"

    Defaults 0.05 / 0.15 are a judgement call, stated openly.  Assumption: a
    one-standard-deviation bad year (revenue 5% below average) is absorbable,
    while one 15% below average would likely exceed a thin export margin of
    roughly 10%.  Change them to match the co-operative's real margin.
    """

    def __init__(self, low_cv: float = 0.05, high_cv: float = 0.15, alpha: float = 0.05) -> None:
        if not 0 < low_cv < high_cv:
            raise ValueError("require 0 < low_cv < high_cv")
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        self.low_cv, self.high_cv, self.alpha = low_cv, high_cv, alpha

    @staticmethod
    def describe(revenue: Iterable[float]) -> dict[str, float]:
        """Mean, median, sample variance, st.dev and CV via the ``statistics`` module."""
        values = [float(v) for v in revenue]
        if len(values) < 2:
            raise ValueError("Need at least two observations")
        mean = statistics.mean(values)
        stdev = statistics.stdev(values)
        return {
            "mean": mean,
            "median": statistics.median(values),
            "variance": statistics.variance(values),
            "stdev": stdev,
            "cv": stdev / mean if mean != 0 else float("nan"),
        }

    def classify(self, cv: float) -> str:
        """Map a coefficient of variation to Low / Moderate / High."""
        if math.isnan(cv) or cv < 0:
            raise ValueError("cv must be a non-negative number")
        if cv < self.low_cv:
            return "Low"
        return "Moderate" if cv < self.high_cv else "High"

    def annual_revenue_mc(
        self,
        fishery: Fishery,
        n_paths: int = 2_000,
        seed: int = 42,
        weeks: int = 52,
        closed_weeks: Iterable[int] | None = None,
    ) -> np.ndarray:
        """Monte Carlo: total revenue over ``weeks`` for each simulated price path."""
        return fishery.revenue(weeks, n_paths, seed, closed_weeks).sum(axis=1)

    def value_at_risk(self, annual_revenue: Iterable[float]) -> dict[str, float]:
        """Revenue-shortfall VaR at level ``alpha``.

        ``percentile``: the revenue exceeded in (1 - alpha) of simulated years.
        ``var``: ``expected - percentile``, the shortfall below expectation that
        is not exceeded with probability (1 - alpha).
        """
        values = np.asarray(list(annual_revenue), dtype=float)
        if values.size == 0:
            raise ValueError("No simulated revenues supplied")
        pct = float(np.percentile(values, 100 * self.alpha))
        expected = float(values.mean())
        return {"expected": expected, "percentile": pct, "var": expected - pct}


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot_stock_paths(paths: Mapping[float, StockPath], K: float, r: float):
    """Stock trajectories for several harvest rates on one chart."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for h, p in paths.items():
        ax.plot(p.stock, label=f"h = {h:.2f}")
    ax.axhline(K, color="grey", ls=":", lw=1)
    ax.axhline(K / 2, color="grey", ls="--", lw=1)
    ax.text(1, K * 1.01, "carrying capacity K", color="grey", fontsize=8)
    ax.text(26, K / 2 * 1.015, "K/2 (stock level at MSY)", color="grey", fontsize=8)
    ax.set_xlabel("Week")
    ax.set_ylabel("Fish stock (tonnes)")
    ax.set_title(f"Stock trajectories by harvest rate (r = {r}, K = {K:,.0f} t)")
    ax.legend(title="harvest rate")
    fig.tight_layout()
    return fig


def plot_revenue_hist(results: Mapping[str, tuple[np.ndarray, dict[str, float]]]):
    """Histograms of simulated annual revenue (UGX billion) with expected value and VaR level marked."""
    import matplotlib.pyplot as plt

    n = len(results)
    cols = 2 if n > 1 else 1
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.6 * cols, 3.6 * rows), squeeze=False)
    for ax, (label, (rev, var)) in zip(axes.ravel(), results.items()):
        ax.hist(rev / 1e9, bins=40, color="#4c78a8", alpha=0.85)
        ax.axvline(var["expected"] / 1e9, color="black", ls="--", label="expected")
        ax.axvline(var["percentile"] / 1e9, color="red", label=f"5th pct (VaR = {var['var'] / 1e9:.1f} bn)")
        ax.set_title(label)
        ax.set_xlabel("Annual revenue (UGX billion)")
        ax.set_ylabel("Simulated years")
        ax.legend(fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig
