"""Solar micro-grid dispatch planner (Mini-Project 2).

A rural health centre draws energy from several sources.  Each daily demand is
a linear equation in the source quantities, so the dispatch problem is the
linear system ``A @ x = d`` where ``x`` holds kWh from each source.

Classes
-------
MicroGrid        2 sources (solar ``x``, battery ``y``), 2 constraints.
HybridMicroGrid  3 sources (adds diesel ``z``), 3 constraints (subclass).
"""
from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy import linalg, optimize


class SingularSystemError(ValueError):
    """Raised when the coefficient matrix is singular or too ill-conditioned."""


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------
def read_positive_float(
    prompt: str,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    max_attempts: int | None = None,
) -> float:
    """Prompt until the user types a finite, non-negative number.

    Rejects empty input, non-numeric text, NaN/inf and negative values and
    re-prompts.  ``input_fn``/``output_fn`` are injectable so the function can
    be unit-tested without a keyboard.  ``max_attempts`` (None = unlimited)
    raises ``ValueError`` once exhausted.
    """
    attempts = 0
    while max_attempts is None or attempts < max_attempts:
        attempts += 1
        raw = input_fn(prompt).strip()
        if raw == "":
            output_fn("Input is empty. Please enter a number.")
            continue
        try:
            value = float(raw)
        except ValueError:
            output_fn(f"'{raw}' is not a number. Please try again.")
            continue
        if not math.isfinite(value):
            output_fn("Value must be finite. Please try again.")
            continue
        if value < 0:
            output_fn("Demand cannot be negative. Please try again.")
            continue
        return value
    raise ValueError("Too many invalid attempts.")


def generate_demand_csv(
    path: str | Path,
    n_days: int = 30,
    seed: int = 42,
    start: date = date(2026, 9, 1),
) -> Path:
    """Write a seeded, realistic demand file with a weekly pattern and noise.

    * ``d1_kwh`` (daytime load): ~70 kWh on weekdays, 80 % of that on Saturday
      and 60 % on Sunday (the clinic is quieter at weekends), plus Gaussian noise.
    * ``d2_kwh`` (critical equipment: fridges, oxygen, lights): roughly
      constant at ~55 kWh with small noise.
    """
    if n_days <= 0:
        raise ValueError("n_days must be positive")
    rng = np.random.default_rng(seed)
    weekly = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6])  # Mon..Sun
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["date", "d1_kwh", "d2_kwh"])
        for i in range(n_days):
            day = start + timedelta(days=i)
            d1 = 70.0 * weekly[day.weekday()] + rng.normal(0, 4)
            d2 = 55.0 + rng.normal(0, 5)
            writer.writerow([day.isoformat(), round(max(d1, 1.0), 1), round(max(d2, 1.0), 1)])
    return path


def load_demand_csv(
    path: str | Path, columns: Sequence[str] = ("d1_kwh", "d2_kwh")
) -> tuple[list[str], np.ndarray]:
    """Load demands from CSV.

    Returns ``(dates, D)`` where ``D`` has shape ``(len(columns), n_days)`` so
    it can be passed straight to :meth:`MicroGrid.solve_vectorised`.
    """
    with Path(path).open(newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in ("date", *columns) if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV contains no data rows")
    try:
        D = np.array([[float(r[c]) for r in rows] for c in columns])
    except (TypeError, ValueError) as exc:
        raise ValueError("CSV contains a non-numeric or empty demand value") from exc
    if not np.isfinite(D).all() or (D < 0).any():
        raise ValueError("Demands must be finite and non-negative")
    return [r["date"] for r in rows], D


# --------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DispatchPlan:
    """Outcome of a feasibility-aware solve over many days."""

    raw: np.ndarray            # exact solution of A x = d (may contain negatives)
    dispatch: np.ndarray       # feasible (non-negative) dispatch actually used
    infeasible: np.ndarray     # bool mask, True where raw had a negative entry
    unmet: np.ndarray          # 2-norm of (A @ dispatch - d) per day, kWh
    strategy: str


# --------------------------------------------------------------------------
# Domain classes
# --------------------------------------------------------------------------
class MicroGrid:
    """Square linear dispatch model ``A @ x = d`` with source prices.

    Parameters
    ----------
    A : array-like, optional
        Coefficient matrix.  Defaults to ``[[3, 2], [4, 1]]``
        (3x + 2y = D1, 4x + y = D2).
    prices : sequence of float
        UGX per kWh for each source (illustrative defaults: solar 150,
        battery 450).
    labels : sequence of str
        Names of the sources, in the same order as the columns of ``A``.
    max_condition : float
        Systems with a larger condition number are rejected as ill-posed.
    """

    DEFAULT_A = ((3.0, 2.0), (4.0, 1.0))
    DEFAULT_PRICES = (150.0, 450.0)
    DEFAULT_LABELS = ("solar", "battery")

    def __init__(
        self,
        A: Sequence[Sequence[float]] | np.ndarray | None = None,
        prices: Sequence[float] | None = None,
        labels: Sequence[str] | None = None,
        max_condition: float = 1e10,
    ) -> None:
        self.A = np.array(self.DEFAULT_A if A is None else A, dtype=float)
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A must be a square 2-D matrix")
        if not np.isfinite(self.A).all():
            raise ValueError("A must contain only finite numbers")
        n = self.A.shape[0]
        self.prices = np.array(self.DEFAULT_PRICES if prices is None else prices, dtype=float)
        self.labels = tuple(self.DEFAULT_LABELS if labels is None else labels)
        if len(self.prices) != n or len(self.labels) != n:
            raise ValueError(f"prices and labels must each have length {n}")
        if (self.prices < 0).any():
            raise ValueError("prices cannot be negative")
        self.max_condition = float(max_condition)
        self.check_well_posed()

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(sources={self.labels}, "
                f"det={self.determinant:.3g}, cond={self.condition_number:.3g})")

    # ---- well-posedness -------------------------------------------------
    @property
    def n_sources(self) -> int:
        return self.A.shape[0]

    @property
    def determinant(self) -> float:
        """det(A).  Zero means no unique solution (dependent equations)."""
        return float(np.linalg.det(self.A))

    @property
    def condition_number(self) -> float:
        """2-norm condition number: worst-case amplification of relative error."""
        return float(np.linalg.cond(self.A))

    def check_well_posed(self) -> None:
        """Raise :class:`SingularSystemError` if A is singular / ill-conditioned."""
        cond = self.condition_number
        if math.isclose(self.determinant, 0.0, abs_tol=1e-12) or cond > self.max_condition:
            raise SingularSystemError(
                f"System is not well-posed (det={self.determinant:.3g}, cond={cond:.3g})."
            )

    # ---- demand validation ---------------------------------------------
    def _check_demands(self, D: np.ndarray | Sequence) -> np.ndarray:
        D = np.asarray(D, dtype=float)
        if D.ndim != 2 or D.shape[0] != self.n_sources:
            raise ValueError(f"D must have shape ({self.n_sources}, n_days)")
        if D.shape[1] == 0:
            raise ValueError("D contains no days")
        if not np.isfinite(D).all() or (D < 0).any():
            raise ValueError("Demands must be finite and non-negative")
        return D

    # ---- solving --------------------------------------------------------
    def solve_day(self, *demands: float) -> np.ndarray:
        """Solve one day.  ``solve_day(d1, d2)`` returns ``[solar, battery]``."""
        if len(demands) != self.n_sources:
            raise ValueError(f"Expected {self.n_sources} demands, got {len(demands)}")
        D = self._check_demands(np.array(demands, dtype=float).reshape(-1, 1))
        return linalg.solve(self.A, D[:, 0])

    def solve_loop(self, D: np.ndarray) -> np.ndarray:
        """Solve every day with a Python loop (one ``linalg.solve`` per day)."""
        D = self._check_demands(D)
        return np.column_stack([linalg.solve(self.A, D[:, j]) for j in range(D.shape[1])])

    def solve_vectorised(self, D: np.ndarray) -> np.ndarray:
        """Solve every day in a single call (right-hand side is n x n_days)."""
        D = self._check_demands(D)
        return linalg.solve(self.A, D)

    # ---- feasibility ----------------------------------------------------
    @staticmethod
    def infeasible_mask(X: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        """True for days where any source quantity is (meaningfully) negative."""
        return (np.asarray(X) < -tol).any(axis=0)

    def solve_nnls(self, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Non-negative least squares per day: min ||A x - d|| s.t. x >= 0."""
        D = self._check_demands(D)
        cols, res = [], []
        for j in range(D.shape[1]):
            x, rnorm = optimize.nnls(self.A, D[:, j])
            cols.append(x)
            res.append(rnorm)
        return np.column_stack(cols), np.array(res)

    def plan(self, D: np.ndarray, strategy: str = "nnls") -> DispatchPlan:
        """Solve all days and repair infeasible ones.

        strategy="clip": set negative quantities to zero (simple, but the
                         demand equations are then violated by an arbitrary amount).
        strategy="nnls": best non-negative fit in the least-squares sense; keeps
                         the unmet-demand residual as small as possible.
        Feasible days are untouched by either strategy.
        """
        if strategy not in {"clip", "nnls"}:
            raise ValueError("strategy must be 'clip' or 'nnls'")
        D = self._check_demands(D)
        raw = self.solve_vectorised(D)
        bad = self.infeasible_mask(raw)
        dispatch = raw.copy()
        if strategy == "clip":
            dispatch[:, bad] = np.clip(raw[:, bad], 0.0, None)
        elif bad.any():
            dispatch[:, bad], _ = self.solve_nnls(D[:, bad])
        unmet = np.linalg.norm(self.A @ dispatch - D, axis=0)
        return DispatchPlan(raw, dispatch, bad, unmet, strategy)

    # ---- statistics and cost -------------------------------------------
    def usage_stats(self, X: np.ndarray) -> dict[str, dict[str, float]]:
        """Mean, sample variance, st.dev and CV for each source (``statistics``)."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] != self.n_sources:
            raise ValueError("X has the wrong shape")
        if X.shape[1] < 2:
            raise ValueError("Need at least two days to compute variance")
        out: dict[str, dict[str, float]] = {}
        for label, row in zip(self.labels, X):
            vals = row.tolist()
            mean = statistics.mean(vals)
            out[label] = {
                "mean": mean,
                "variance": statistics.variance(vals),
                "stdev": statistics.stdev(vals),
                "cv": statistics.stdev(vals) / mean if mean else float("nan"),
            }
        return out

    def daily_cost(self, X: np.ndarray) -> np.ndarray:
        """UGX per day = sum over sources of price * kWh."""
        return self.prices @ np.asarray(X, dtype=float)

    def monthly_cost(self, X: np.ndarray) -> float:
        """Total UGX over all days in ``X``."""
        return float(self.daily_cost(X).sum())

    # ---- sensitivity ----------------------------------------------------
    def sensitivity(
        self, demands: Sequence[float], rel: float = 0.05, n_draws: int = 1000, seed: int = 0
    ) -> dict[str, np.ndarray | float]:
        """Monte Carlo: perturb every demand by U(-rel, +rel) and re-solve.

        Returns the draws plus the worst relative change in the solution, the
        worst relative change in the input, and the amplification ratio, which
        is bounded by ``condition_number``.
        """
        b = np.asarray(demands, dtype=float)
        if b.shape != (self.n_sources,) or (b < 0).any():
            raise ValueError("demands must be a non-negative vector of length n_sources")
        if rel < 0 or n_draws <= 1:
            raise ValueError("rel must be >= 0 and n_draws > 1")
        rng = np.random.default_rng(seed)
        mult = rng.uniform(1 - rel, 1 + rel, size=(self.n_sources, n_draws))
        B = b[:, None] * mult
        X = linalg.solve(self.A, B)
        x0 = linalg.solve(self.A, b)
        dx = np.linalg.norm(X - x0[:, None], axis=0) / np.linalg.norm(x0)
        db = np.linalg.norm(B - b[:, None], axis=0) / np.linalg.norm(b)
        with np.errstate(divide="ignore", invalid="ignore"):
            amp = np.where(db > 0, dx / db, 0.0)
        return {
            "draws": X,
            "baseline": x0,
            "std": X.std(axis=1, ddof=1),
            "max_rel_output_change": float(dx.max()),
            "max_rel_input_change": float(db.max()),
            "max_amplification": float(amp.max()),
            "condition_number": self.condition_number,
        }


class HybridMicroGrid(MicroGrid):
    """Solar + battery + diesel generator: a 3 x 3 system.

    Default equations (illustrative):
        3x + 2y + 1z = D1   daytime load
        4x + 1y + 2z = D2   critical load
        1x + 1y + 1z = D3   total energy balance
    If the third row were a combination of the first two, ``det(A) = 0`` and
    construction raises :class:`SingularSystemError`: there is then either no
    solution or infinitely many (a free choice of generator output).
    """

    DEFAULT_A = ((3.0, 2.0, 1.0), (4.0, 1.0, 2.0), (1.0, 1.0, 1.0))
    DEFAULT_PRICES = (150.0, 450.0, 900.0)
    DEFAULT_LABELS = ("solar", "battery", "diesel")


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot_dispatch(X, daily_cost, labels, day_labels=None):
    """Stacked bars of kWh per source with daily cost on a secondary axis."""
    import matplotlib.pyplot as plt

    X = np.asarray(X)
    days = np.arange(X.shape[1])
    fig, ax = plt.subplots(figsize=(11, 4.5))
    bottom = np.zeros(X.shape[1])
    for label, row in zip(labels, X):
        ax.bar(days, row, bottom=bottom, label=f"{label} (kWh)")
        bottom += row
    ax.set_ylim(0, bottom.max() * 1.25)   # headroom so the legend does not hide bars
    ax.set_xlabel("Day of month" if day_labels is None else "Date")
    ax.set_ylabel("Energy drawn (kWh)")
    if day_labels is not None:
        ax.set_xticks(days[::3])
        ax.set_xticklabels([day_labels[i][5:] for i in days[::3]], rotation=45)
    ax2 = ax.twinx()
    ax2.plot(days, daily_cost, color="black", marker="o", ms=3, label="Daily cost (UGX)")
    ax2.set_ylabel("Daily cost (UGX)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", ncol=3)
    ax.set_title("Kasese health centre: daily solar/battery dispatch and cost")
    fig.tight_layout()
    return fig
