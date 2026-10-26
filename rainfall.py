"""Rainfall pattern and crop suitability analyser (Mini-Project 4).

Classes
-------
Region            Monthly rainfall for one region: statistics, rainy-season
                  detection (scipy.signal.find_peaks), crop classification.
CropRule          A suitable monthly rainfall band for one crop, with its source.
RegionComparator  Cosine similarity, Pearson correlation and Euclidean distance
                  matrices between regions.

Functions ``cosine_similarity``, ``pearson_correlation`` and
``euclidean_distance`` are implemented directly with NumPy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, NamedTuple, Sequence

import numpy as np
from scipy.signal import find_peaks

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


class Status(IntEnum):
    """Outcome of comparing one month's rainfall with a crop's suitable band."""

    DROUGHT_RISK = -1
    GOOD = 0
    WATERLOGGING_RISK = 1


class Window(NamedTuple):
    """A run of consecutive suitable months (circular: may wrap Dec -> Jan)."""

    start: int      # month index 0..11
    length: int     # number of months

    @property
    def months(self) -> list[str]:
        return [MONTHS[(self.start + i) % 12] for i in range(self.length)]

    def __str__(self) -> str:  # e.g. "Sep-Nov (3 months)"
        m = self.months
        return f"{m[0]}-{m[-1]} ({self.length} months)" if self.length > 1 else f"{m[0]} (1 month)"


# --------------------------------------------------------------------------
# Crop rules
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CropRule:
    """Suitable monthly rainfall band for a crop.

    ``min_mm <= rain <= max_mm``  -> GOOD
    ``rain < min_mm``              -> DROUGHT_RISK
    ``rain > max_mm``              -> WATERLOGGING_RISK

    ``min_run`` is the number of consecutive GOOD months needed to carry the
    crop from planting to harvest (used by :meth:`Region.planting_windows`).
    ``source`` records where the thresholds come from.
    """

    name: str
    min_mm: float
    max_mm: float
    min_run: int = 1
    source: str = ""

    def __post_init__(self) -> None:
        if not (0 <= self.min_mm < self.max_mm):
            raise ValueError("require 0 <= min_mm < max_mm")
        if self.min_run < 1:
            raise ValueError("min_run must be at least 1")

    @classmethod
    def from_seasonal(
        cls, name: str, season_min_mm: float, season_max_mm: float,
        cycle_months: float, min_run: int, source: str = "",
    ) -> "CropRule":
        """Convert a whole-season water requirement into a monthly band.

        monthly band = seasonal requirement / length of the crop cycle in months.
        """
        if cycle_months <= 0:
            raise ValueError("cycle_months must be positive")
        return cls(name, season_min_mm / cycle_months, season_max_mm / cycle_months, min_run, source)

    @classmethod
    def from_annual(
        cls, name: str, annual_min_mm: float, annual_max_mm: float, min_run: int = 1, source: str = "",
    ) -> "CropRule":
        """Monthly band for a perennial crop from its annual rainfall range (/12)."""
        return cls(name, annual_min_mm / 12.0, annual_max_mm / 12.0, min_run, source)

    def scaled(self, factor: float) -> "CropRule":
        """Copy with both thresholds multiplied by ``factor`` (sensitivity tests)."""
        if factor <= 0:
            raise ValueError("factor must be positive")
        return CropRule(self.name, self.min_mm * factor, self.max_mm * factor, self.min_run, self.source)

    def classify(self, rain_mm: float) -> Status:
        """Classify one month's rainfall."""
        if not math.isfinite(rain_mm) or rain_mm < 0:
            raise ValueError("rainfall must be a finite, non-negative number")
        if rain_mm < self.min_mm:
            return Status.DROUGHT_RISK
        return Status.GOOD if rain_mm <= self.max_mm else Status.WATERLOGGING_RISK

    def classify_array(self, rain_mm: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`classify`; returns integer codes -1 / 0 / +1."""
        r = np.asarray(rain_mm, dtype=float)
        return np.where(r < self.min_mm, Status.DROUGHT_RISK,
                        np.where(r > self.max_mm, Status.WATERLOGGING_RISK, Status.GOOD)).astype(int)

    def label(self, rain_mm: float) -> str:
        """Human-readable verdict, e.g. 'Good for maize'."""
        s = self.classify(rain_mm)
        return {Status.GOOD: f"Good for {self.name}",
                Status.DROUGHT_RISK: f"Drought risk ({self.name})",
                Status.WATERLOGGING_RISK: f"Waterlogging risk ({self.name})"}[s]


def default_crops() -> list[CropRule]:
    """Maize, beans and coffee, thresholds derived from the cited sources.

    Maize  : FAO crop information: a medium-maturity grain crop needs 500-800 mm
             per season; FAO agro-ecological zones: lowland maize cycle 110-130
             days (~4 months)  ->  125-200 mm/month, 4 consecutive months.
    Beans  : FAO crop information: a 60-120-day bean crop needs 300-500 mm; dry
             bean cycle 90-120 days (~3.5 months) -> ~86-143 mm/month, 3 months.
    Coffee : perennial; UCDA 'Uganda Coffee Profiles' give annual rainfall of
             ~1,200-1,450 mm for Robusta zones and 920-1,650 mm for Arabica
             highlands, and Nakyagaba et al. (2024, Int. J. Agronomy) found
             Robusta yields highest at 1,100-1,200 mm/yr -> band 1,100-1,650
             mm/yr, i.e. ~92-138 mm/month.
    The FAO figures are crop water *requirements*: the lower bound (drought) is
    better supported than the upper bound (waterlogging), which also depends on
    soil, drainage and evapotranspiration (not modelled).
    """
    return [
        CropRule.from_seasonal("maize", 500, 800, cycle_months=4.0, min_run=4,
                               source="FAO crop information (maize); FAO AEZ growth cycle 110-130 d"),
        CropRule.from_seasonal("beans", 300, 500, cycle_months=3.5, min_run=3,
                               source="FAO crop information (bean); FAO AEZ growth cycle 90-120 d"),
        CropRule.from_annual("coffee", 1100, 1650, min_run=1,
                             source="UCDA Uganda Coffee Profiles; Nakyagaba et al. 2024"),
    ]


# --------------------------------------------------------------------------
# Region
# --------------------------------------------------------------------------
class Region:
    """Mean monthly rainfall (mm, Jan..Dec) for one region."""

    def __init__(self, name: str, rainfall_mm: Sequence[float] | np.ndarray) -> None:
        r = np.asarray(rainfall_mm, dtype=float)
        if r.shape != (12,):
            raise ValueError("rainfall_mm must contain exactly 12 monthly values")
        if not np.isfinite(r).all() or (r < 0).any():
            raise ValueError("rainfall must be finite and non-negative")
        if not name.strip():
            raise ValueError("name cannot be empty")
        self.name = name
        self.rainfall = r

    def __repr__(self) -> str:
        return f"Region({self.name!r}, annual={self.annual_total:,.0f} mm, wettest={self.wettest_month[0]})"

    def __len__(self) -> int:
        return len(self.rainfall)

    # ---- descriptive statistics ----------------------------------------
    @property
    def annual_total(self) -> float:
        return float(self.rainfall.sum())

    @property
    def mean(self) -> float:
        return float(self.rainfall.mean())

    @property
    def wettest_month(self) -> tuple[str, float]:
        i = int(self.rainfall.argmax())
        return MONTHS[i], float(self.rainfall[i])

    @property
    def driest_month(self) -> tuple[str, float]:
        i = int(self.rainfall.argmin())
        return MONTHS[i], float(self.rainfall[i])

    @property
    def cv(self) -> float:
        """Coefficient of variation of the 12 monthly values (population st.dev / mean).

        The 12 months are the complete annual cycle, not a sample, hence ddof = 0.
        High CV = strongly seasonal; low CV = evenly spread rain.
        """
        return float(self.rainfall.std() / self.rainfall.mean()) if self.rainfall.mean() else float("nan")

    # ---- rainy seasons --------------------------------------------------
    def rainy_season_peaks(self, min_prominence_frac: float = 0.25) -> list[int]:
        """Month indices of distinct rainy-season peaks, via ``find_peaks``.

        Rainfall is cyclical (December is next to January), so the series is
        rotated to start at the driest month before searching; a peak at the
        year boundary is then not missed.  A peak counts as its own season only
        if it stands out by at least ``min_prominence_frac`` of the annual range
        (max - min) above its surrounding troughs.
        """
        if not 0 < min_prominence_frac < 1:
            raise ValueError("min_prominence_frac must lie in (0, 1)")
        r = self.rainfall
        amplitude = r.max() - r.min()
        if amplitude == 0:
            return []
        start = int(r.argmin())
        rotated = np.roll(r, -start)
        extended = np.append(rotated, rotated[0])   # close the cycle so the last month can be a peak
        peaks, _ = find_peaks(extended, prominence=min_prominence_frac * amplitude)
        return sorted(int((p + start) % 12) for p in peaks)

    def modality(self, min_prominence_frac: float = 0.25) -> str:
        """'aseasonal', 'unimodal', 'bimodal' or 'multimodal'."""
        n = len(self.rainy_season_peaks(min_prominence_frac))
        return {0: "aseasonal", 1: "unimodal", 2: "bimodal"}.get(n, "multimodal")

    # ---- crop classification -------------------------------------------
    def suitability(self, rule: CropRule) -> np.ndarray:
        """Status codes (-1 drought, 0 good, +1 waterlogging) for the 12 months."""
        return rule.classify_array(self.rainfall)

    def monthly_labels(self, rules: Sequence[CropRule]) -> list[str]:
        """One verdict per month across all crops, e.g. 'Good for maize, beans'."""
        out = []
        for i, rain in enumerate(self.rainfall):
            statuses = {rule.name: rule.classify(rain) for rule in rules}
            good = [n for n, s in statuses.items() if s is Status.GOOD]
            if good:
                out.append("Good for " + ", ".join(good))
            elif all(s is Status.DROUGHT_RISK for s in statuses.values()):
                out.append("Drought risk")
            elif all(s is Status.WATERLOGGING_RISK for s in statuses.values()):
                out.append("Waterlogging risk")
            else:
                dry = [n for n, s in statuses.items() if s is Status.DROUGHT_RISK]
                wet = [n for n, s in statuses.items() if s is Status.WATERLOGGING_RISK]
                out.append(f"Mixed (drought: {', '.join(dry) or '-'}; waterlogging: {', '.join(wet) or '-'})")
        return out

    def planting_windows(self, rule: CropRule) -> list[Window]:
        """Runs of consecutive GOOD months (wrapping Dec -> Jan) at least ``rule.min_run`` long."""
        good = self.suitability(rule) == Status.GOOD
        if good.all():
            return [Window(0, 12)]
        windows = []
        for i in range(12):
            if good[i] and not good[i - 1]:             # start of a run (i-1 wraps to Dec for i = 0)
                n = 0
                while good[(i + n) % 12]:
                    n += 1
                if n >= rule.min_run:
                    windows.append(Window(i, n))
        return windows


# --------------------------------------------------------------------------
# Similarity and distance measures (NumPy implementations)
# --------------------------------------------------------------------------
def _as_pair(a, b) -> tuple[np.ndarray, np.ndarray]:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or a.size == 0:
        raise ValueError("a and b must be non-empty 1-D arrays of equal length")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("inputs must be finite")
    return a, b


def cosine_similarity(a, b) -> float:
    """cos(theta) = (a . b) / (||a|| ||b||): angle between the two *vectors*.

    This is NOT ``math.cos(x)``, which is the cosine of a single angle in
    radians and takes no vectors at all.
    """
    a, b = _as_pair(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return float(np.dot(a, b) / (na * nb))


def pearson_correlation(a, b) -> float:
    """Pearson r = cosine similarity of the mean-centred vectors."""
    a, b = _as_pair(a, b)
    ac, bc = a - a.mean(), b - b.mean()
    if not ac.any() or not bc.any():
        raise ValueError("Pearson correlation is undefined for a constant series")
    return cosine_similarity(ac, bc)


def euclidean_distance(a, b) -> float:
    """Straight-line distance sqrt(sum (a_i - b_i)^2), in mm for rainfall vectors."""
    a, b = _as_pair(a, b)
    return float(np.linalg.norm(a - b))


class RegionComparator:
    """Pairwise comparison matrices for a set of regions."""

    MEASURES: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
        "cosine similarity": cosine_similarity,
        "Pearson correlation": pearson_correlation,
        "Euclidean distance (mm)": euclidean_distance,
    }

    def __init__(self, regions: Sequence[Region]) -> None:
        if len(regions) < 2:
            raise ValueError("need at least two regions to compare")
        self.regions = list(regions)
        self.names = [r.name for r in self.regions]

    def matrix(self, measure: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
        """Symmetric n x n matrix of ``measure`` over every pair of regions."""
        n = len(self.regions)
        M = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                M[i, j] = measure(self.regions[i].rainfall, self.regions[j].rainfall)
        return M

    def all_matrices(self) -> dict[str, np.ndarray]:
        return {name: self.matrix(fn) for name, fn in self.MEASURES.items()}


# --------------------------------------------------------------------------
# Year-to-year variability (extension)
# --------------------------------------------------------------------------
def simulate_years(region: Region, n_years: int = 10, cv: float = 0.35, seed: int = 42) -> np.ndarray:
    """SYNTHETIC multi-year rainfall, shape ``(n_years, 12)``, for demonstrating the pipeline.

    Each month is drawn from a gamma distribution whose mean is the region's
    climatological value and whose coefficient of variation is ``cv`` (an
    assumption, not an estimate).  Replace with real NASA POWER / UNMA data by
    passing a real ``(years, 12)`` array to the functions below.
    """
    if n_years < 1 or cv <= 0:
        raise ValueError("n_years must be >= 1 and cv > 0")
    rng = np.random.default_rng(seed)
    shape = 1.0 / cv**2
    mean = np.maximum(region.rainfall, 1e-9)
    return rng.gamma(shape, mean / shape, size=(n_years, 12))


def suitability_reliability(years: np.ndarray, rule: CropRule) -> np.ndarray:
    """Fraction of years in which each month is GOOD for ``rule`` (length-12 array)."""
    y = np.asarray(years, dtype=float)
    if y.ndim != 2 or y.shape[1] != 12 or y.shape[0] < 1:
        raise ValueError("years must have shape (n_years, 12)")
    return (rule.classify_array(y) == Status.GOOD).mean(axis=0)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot_rainfall_lines(regions: Sequence[Region], peaks: dict[str, list[int]] | None = None):
    """All regions' monthly rainfall on one chart; detected rainy-season peaks marked."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.6))
    for reg in regions:
        (line,) = ax.plot(MONTHS, reg.rainfall, marker="o", label=reg.name)
        if peaks and reg.name in peaks:
            idx = peaks[reg.name]
            ax.scatter([MONTHS[i] for i in idx], reg.rainfall[idx], s=140, facecolors="none",
                       edgecolors=line.get_color(), linewidths=2)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean monthly rainfall (mm)")
    ax.set_title("Monthly rainfall by region (circles = detected rainy-season peaks)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_suitability_heatmap(regions: Sequence[Region], rules: Sequence[CropRule]):
    """One panel per crop: regions (rows) x months (columns), coloured by status, rainfall annotated."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    cmap = ListedColormap(["#e0b34a", "#5aa469", "#4a7fb5"])   # drought, good, waterlogging
    fig, axes = plt.subplots(len(rules), 1, figsize=(10, 2.1 * len(rules) + 1), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, rule in zip(axes, rules):
        grid = np.array([reg.suitability(rule) for reg in regions]) + 1      # 0, 1, 2 for the colormap
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, aspect="auto")
        for i, reg in enumerate(regions):
            for j, v in enumerate(reg.rainfall):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7)
        ax.set_yticks(range(len(regions)), [r.name for r in regions])
        ax.set_title(f"{rule.name}: good band {rule.min_mm:.0f}-{rule.max_mm:.0f} mm/month", fontsize=10)
    axes[-1].set_xticks(range(12), MONTHS)
    axes[-1].set_xlabel("Month (numbers show rainfall in mm)")
    fig.legend(handles=[Patch(color="#e0b34a", label="drought risk"), Patch(color="#5aa469", label="good"),
                        Patch(color="#4a7fb5", label="waterlogging risk")], loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def plot_monthly_boxplots(years: np.ndarray, title: str, rule: CropRule | None = None):
    """Box plot of rainfall per calendar month across years; optional crop band shaded."""
    import matplotlib.pyplot as plt

    y = np.asarray(years, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    if rule is not None:
        ax.axhspan(rule.min_mm, rule.max_mm, color="#5aa469", alpha=0.18, label=f"good band for {rule.name}")
    ax.boxplot([y[:, m] for m in range(12)], tick_labels=MONTHS)
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly rainfall (mm)")
    ax.set_title(title)
    if rule is not None:
        ax.legend(loc="upper right")
    fig.tight_layout()
    return fig
