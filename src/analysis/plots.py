"""Static report figures (matplotlib), following one consistent visual system.

Palette (validated for colour-vision deficiency, light surface):
* slot 1 blue ``#2a78d6`` - the default single series; also the "RT below DA" pole
* slot 2 orange ``#eb6834`` - second series where one is needed (e.g. mean vs median)
* red ``#e34948`` - the "RT above DA" pole; blue<->red is the diverging pair
* sequential: one-hue blue ramp for heatmaps
Thin marks, hairline solid grid, muted axes, selective direct labels, no dual axes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#e34948"
BLUE_LIGHT = "#cde2fb"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)

SOURCE_NOTE = "Source: PJM Data Miner 2 (Western Hub, RTO load). Jan 2023 – Jun 2025."


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.major.size": 0, "ytick.major.size": 0, "lines.linewidth": 2, "legend.frameon": False,
        "legend.fontsize": 9, "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "text.parse_math": False,          # "$" is a currency sign here, never math
    })


def money(v: float, decimals: int = 2, sign: bool = False) -> str:
    """Format dollars with a true minus sign before the $: −$2.58, +$76.50."""
    s = f"${abs(v):,.{decimals}f}"
    if v < 0:
        return "−" + s
    return ("+" + s) if sign and v > 0 else s


def _finish(fig, ax_or_axes, title: str | None, subtitle: str | None, path: Path | None, note: str = SOURCE_NOTE):
    """Place title, subtitle and source note from the measured layout, so they never collide."""
    for ax in np.atleast_1d(ax_or_axes):
        ax.tick_params(length=0)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    H = fig.bbox.height
    pt = fig.dpi / 72 / H                       # one typographic point as a figure fraction
    boxes = [a.get_tightbbox(r) for a in fig.axes]
    top, bottom = max(b.y1 for b in boxes) / H, min(b.y0 for b in boxes) / H
    y = top + 10 * pt
    if subtitle:
        st = fig.text(0.01, y, subtitle, ha="left", va="bottom", fontsize=9.5, color=INK_2)
        y += st.get_window_extent(r).height / H + 4 * pt
    if title:
        fig.text(0.01, y, title, ha="left", va="bottom", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, bottom - 8 * pt, note, ha="left", va="top", fontsize=8, color=MUTED)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    return fig


def _zero_line(ax, axis: str = "y"):
    (ax.axhline if axis == "y" else ax.axvline)(0, color=AXIS, linewidth=1, zorder=1)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def spread_histogram(s: pd.Series, path=None, clip: float = 60, binw: float = 1.0):
    s = s.dropna()
    inside = s[(s >= -clip) & (s <= clip)]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bins = np.arange(-clip, clip + binw, binw)
    ax.hist(inside, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.6)
    _zero_line(ax, "x")
    med, mean = s.median(), s.mean()
    ymax = ax.get_ylim()[1]
    ax.axvline(med, color=INK, linewidth=1.2)
    ax.text(med - 1, ymax * 0.97, f"median {money(med)}", ha="right", va="top", color=INK, fontsize=9)
    ax.axvline(mean, color=ORANGE, linewidth=1.2)
    ax.text(mean + 1, ymax * 0.97, f"mean {money(mean)}", ha="left", va="top", color=INK, fontsize=9)
    lo, hi = int((s < -clip).sum()), int((s > clip).sum())
    ax.text(-clip + 0.5, ymax * 0.55, f"{lo} hours\nbelow −${clip:.0f}\n(min {money(s.min(), 0)})", ha="left", color=INK_2, fontsize=8.5)
    ax.text(clip - 0.5, ymax * 0.55, f"{hi} hours\nabove +${clip:.0f}\n(max {money(s.max(), 0)})", ha="right", color=INK_2, fontsize=8.5)
    ax.set_xlabel("Hourly spread, RT − DA ($/MWh)")
    ax.set_ylabel("Hours")
    ax.set_xlim(-clip, clip)
    pos = (s > 0).mean()
    return _finish(fig, ax, "Real-time usually clears slightly below day-ahead",
                   f"PJM Western Hub, hourly RT − DA spread. RT > DA in {pos:.0%} of hours; the long positive tail pulls the mean toward zero.", path)


def spread_cdf(s: pd.Series, path=None):
    s = np.sort(s.dropna().to_numpy())
    y = np.arange(1, len(s) + 1) / len(s)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(s, y, color=BLUE)
    ax.set_xscale("symlog", linthresh=10)
    _zero_line(ax, "x")
    for q in (0.05, 0.5, 0.95, 0.99):
        v = np.quantile(s, q)
        ax.plot([v], [q], "o", color=BLUE, markersize=6, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        offset = {0.05: (-10, 6), 0.5: (10, -4), 0.95: (8, -14), 0.99: (8, -14)}[q]
        ax.annotate(f"{q:.0%} of hours ≤ {money(v, 1)}", (v, q), xytext=offset, textcoords="offset points",
                    ha="right" if q == 0.05 else "left", fontsize=8.5, color=INK_2)
    ax.set_xticks([-200, -50, -10, 0, 10, 50, 200, 1000])
    ax.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Hourly spread, RT − DA ($/MWh) — symmetric log scale beyond ±$10")
    ax.set_ylabel("Share of hours at or below")
    return _finish(fig, ax, "Spread distribution: tight body, very long right tail",
                   "Empirical cumulative distribution of the hourly spread.", path)


def profile_band(stats: pd.DataFrame, xlabel: str, title: str, subtitle: str, path=None, xticklabels=None):
    """Median line + 5–95% band + mean line for a grouped profile (one axis, two series)."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = stats.index.to_numpy()
    ax.fill_between(x, stats["p05"], stats["p95"], color=BLUE_LIGHT, linewidth=0, label="5th–95th percentile")
    ax.plot(x, stats["median"], color=BLUE, label="Median")
    ax.plot(x, stats["mean"], color=ORANGE, label="Mean")
    _zero_line(ax)
    ax.set_xticks(x)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    ax.set_xlim(x.min() - 0.3, x.max() + 0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("RT − DA ($/MWh)")
    ax.legend(loc="upper left", ncols=3, bbox_to_anchor=(0, 1.0))
    ax.set_ylim(min(stats["p05"].min(), 0) * 1.15, stats["p95"].max() * 1.3)
    return _finish(fig, ax, title, subtitle, path)


def volatility_heatmap(pivot: pd.DataFrame, title: str, subtitle: str, path=None, fmt="{:.0f}"):
    """Sequential one-hue heatmap of spread standard deviation (rows x columns)."""
    fig, ax = plt.subplots(figsize=(10, 3.4))
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap=SEQ_BLUE, vmin=0, vmax=np.nanmax(data))
    ax.set_xticks(range(pivot.shape[1]), [str(c) for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]), [str(r) for r in pivot.index])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # label only the cells that stand out (top decile), keeping the rest clean
    cut = np.nanquantile(data, 0.9)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] >= cut:
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=8,
                        color="#ffffff" if data[i, j] > 0.55 * np.nanmax(data) else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelcolor=INK_2)
    cb.set_label("Std of spread ($/MWh)", color=INK_2)
    ax.set_xlabel("Hour ending (EPT)")
    return _finish(fig, ax, title, subtitle, path)


def range_dots(stats: pd.DataFrame, title: str, subtitle: str, path=None, xlabel="", label_col="std"):
    """Per-category 5–95% range bar with the median dot; std written beside each category."""
    fig, ax = plt.subplots(figsize=(9, 4.4))
    cats = [str(c) for c in stats.index]
    y = np.arange(len(cats))[::-1]
    ax.hlines(y, stats["p05"], stats["p95"], color=BLUE, linewidth=6, alpha=0.35, capstyle="round")
    ax.plot(stats["median"], y, "o", color=BLUE, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    _zero_line(ax, "x")
    ax.set_yticks(y, cats)
    xmax = stats["p95"].max()
    for yi, (_, r) in zip(y, stats.iterrows()):
        ax.text(xmax * 1.08, yi, f"median {money(r['median'], 1)} · std {money(r[label_col], 0)} · {int(r['hours']):,} h",
                va="center", fontsize=8.5, color=INK_2)
    ax.set_xlim(stats["p05"].min() * 1.15, xmax * 2.3)
    ax.set_xticks(np.arange(np.floor(stats["p05"].min() / 50) * 50, xmax + 1, 50))   # no ticks/grid behind the labels
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    return _finish(fig, ax, title, subtitle, path)


def error_bucket_panels(tbl: pd.DataFrame, labels: list[str], title: str, subtitle: str, path=None):
    """Two panels (no dual axis): mean spread ± 95% CI, and share of hours with RT > DA, per bucket."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(len(tbl))
    ax = axes[0]
    colors = [RED if m > 0 else BLUE for m in tbl["mean_spread"]]
    ax.bar(x, tbl["mean_spread"], color=colors, width=0.72, edgecolor=SURFACE, linewidth=2)
    ax.errorbar(x, tbl["mean_spread"], yerr=1.96 * tbl["se_mean"], fmt="none", ecolor=INK_2, elinewidth=1, capsize=3)
    _zero_line(ax)
    ax.set_xticks(x, labels, rotation=0, fontsize=8.5)
    ax.set_ylabel("Mean RT − DA ($/MWh)")
    ax.set_title("Mean spread (bars: 95% CI)", fontsize=10.5)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    ax.plot(x, tbl["share_rt_above_da"], "o-", color=BLUE, markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.axhline(0.5, color=AXIS, linewidth=1)
    for xi, v in zip(x[[0, -1]], tbl["share_rt_above_da"].iloc[[0, -1]]):
        ax.annotate(f"{v:.0%}", (xi, v), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_ylim(0, 0.8)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.set_title("Share of hours with RT > DA", fontsize=10.5)
    for a in axes:
        a.set_xlabel("Load forecast error: (actual − D-1 forecast) / forecast")
    fig.tight_layout()
    return _finish(fig, axes, title, subtitle, path)


def risk_decile_bars(tbl: pd.DataFrame, title: str, subtitle: str, path=None, note: str = SOURCE_NOTE):
    """Realized mean |spread| by decile of predicted risk (one series, ordered categories)."""
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = tbl["decile"].astype(int).to_numpy()
    ax.bar(x, tbl["mean_abs_spread"], color=BLUE, width=0.72, edgecolor=SURFACE, linewidth=2)
    ax.plot(x, tbl["median_abs_spread"], "o", color=INK_2, markersize=6, markeredgecolor=SURFACE,
            markeredgewidth=2, label="Median |spread|")
    for xi in (x[0], x[-1]):
        v = tbl.loc[tbl["decile"].astype(int) == xi, "mean_abs_spread"].iloc[0]
        ax.text(xi, v + 1, money(v, 0), ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xlabel("Predicted-risk decile (1 = calmest hours, 10 = riskiest)")
    ax.set_ylabel("Realized |RT − DA| ($/MWh)")
    ax.legend(handles=[mpl.patches.Patch(color=BLUE, label="Mean |spread|"), ax.lines[0]], loc="upper left")
    ax.grid(axis="x", visible=False)
    return _finish(fig, ax, title, subtitle, path, note)


def reliability_plot(tbl: pd.DataFrame, title: str, subtitle: str, path=None, note: str = SOURCE_NOTE, base_rate=None):
    """Predicted vs observed P(RT > DA) by predicted-probability decile; diagonal = perfect calibration."""
    fig, ax = plt.subplots(figsize=(6.4, 5))
    lim = (0, max(0.6, tbl[["mean_predicted", "observed_rate"]].to_numpy().max() * 1.15))
    ax.plot(lim, lim, color=AXIS, linewidth=1)
    ax.text(lim[1] * 0.97, lim[1] * 0.93, "perfect calibration", ha="right", fontsize=8.5, color=MUTED, rotation=38)
    if base_rate is not None:
        ax.axhline(base_rate, color=ORANGE, linewidth=1.2)
        ax.text(lim[0] + 0.01, base_rate + 0.01, f"observed base rate {base_rate:.0%}", fontsize=8.5, color=INK_2)
    ax.plot(tbl["mean_predicted"], tbl["observed_rate"], "o-", color=BLUE, markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Predicted P(RT > DA), decile mean")
    ax.set_ylabel("Observed share of hours with RT > DA")
    return _finish(fig, ax, title, subtitle, path, note)


def fold_stability(tbl: pd.DataFrame, title: str, subtitle: str, path=None, note: str = SOURCE_NOTE):
    """Two panels: direction AUC and risk Spearman per evaluation block (no dual axis)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(tbl))
    for ax, col, ref, ttl, lo in ((axes[0], "direction_auc", 0.5, "Direction model: ROC-AUC", 0.4),
                                  (axes[1], "risk_spearman", 0.0, "Risk model: rank correlation with |spread|", 0.0)):
        colors = [ORANGE if lbl.startswith("Validation") else BLUE for lbl in tbl["block"]]
        ax.bar(x, tbl[col] - lo, bottom=lo, color=colors, width=0.62, edgecolor=SURFACE, linewidth=2)
        ax.axhline(ref, color=INK_2, linewidth=1)
        for xi, v in zip(x, tbl[col]):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_xticks(x, tbl["block"], fontsize=8.5)
        ax.set_title(ttl, fontsize=10.5)
        ax.set_ylim(lo, max(0.75, tbl[col].max() + 0.08))
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return _finish(fig, axes, title, subtitle, path, note)


def importance_panels(panels: dict[str, pd.DataFrame], title: str, subtitle: str, path=None, note: str = SOURCE_NOTE):
    """Grouped permutation importance, one panel per model; error bars = std over repeats."""
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 3.8))
    for ax, (name, tbl) in zip(np.atleast_1d(axes), panels.items()):
        t = tbl.sort_values("importance")
        y = np.arange(len(t))
        ax.barh(y, t["importance"], xerr=t["std"], color=BLUE, height=0.6, edgecolor=SURFACE, linewidth=2,
                error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3})
        ax.axvline(0, color=INK_2, linewidth=1)
        ax.set_yticks(y, [f"{g} ({n})" for g, n in zip(t["group"], t["n_features"])])
        ax.set_title(name, fontsize=10.5)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _finish(fig, axes, title, subtitle, path, note)


def cumulative_pnl(daily: dict[str, pd.Series], highlight: dict[str, str], title: str, subtitle: str,
                   path=None, note: str = SOURCE_NOTE, labels: dict[str, str] | None = None):
    """Cumulative daily P&L: highlighted series in colour, the rest recessive gray, all direct-labelled."""
    labels = labels or {}
    fig, ax = plt.subplots(figsize=(10, 5))
    order = [k for k in daily if k not in highlight] + [k for k in daily if k in highlight]
    ends = []
    for k in order:
        cum = daily[k].cumsum()
        hl = k in highlight
        ax.plot(cum.index, cum.to_numpy(), color=highlight.get(k, "#c3c2b7"), linewidth=2.2 if hl else 1.2,
                zorder=3 if hl else 2)
        ends.append((cum.iloc[-1], labels.get(k, k), hl))
    _zero_line(ax)
    # direct labels at line ends, nudged apart so they don't overlap
    ends.sort(key=lambda t: t[0])
    span = max(abs(e[0]) for e in ends) or 1
    placed = []
    for v, lab, hl in ends:
        y = v
        while placed and y - placed[-1] < span * 0.07:
            y = placed[-1] + span * 0.07
        placed.append(y)
        ax.annotate(f"{lab}  {money(v, 0)}", (ax.get_xlim()[1], y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK if hl else INK_2, fontweight="bold" if hl else "normal",
                    annotation_clip=False)
    ax.set_ylabel("Cumulative net P&L ($, 1 MW per hour)")
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: money(v, 0)))
    fig.autofmt_xdate(rotation=0, ha="center")
    return _finish(fig, ax, title, subtitle, path, note)


def drawdown_plot(daily: dict[str, pd.Series], colors: dict[str, str], title: str, subtitle: str,
                  path=None, note: str = SOURCE_NOTE):
    """Underwater curves: cumulative P&L minus its running peak."""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for k, d in daily.items():
        cum = d.cumsum()
        dd = cum - cum.cummax()
        ax.plot(dd.index, dd.to_numpy(), color=colors[k], linewidth=1.6, label=f"{k} (max {money(dd.min(), 0)})")
    _zero_line(ax)
    ax.set_ylabel("Drawdown ($)")
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: money(v, 0)))
    ax.legend(loc="lower left")
    return _finish(fig, ax, title, subtitle, path, note)


def grouped_bars(tbl: pd.DataFrame, colors: dict[str, str], ylabel: str, title: str, subtitle: str,
                 path=None, note: str = SOURCE_NOTE, fmt=lambda v: money(v, 0)):
    """Categories on x, one bar per series (columns of ``tbl``), 2px surface gap between bars."""
    fig, ax = plt.subplots(figsize=(10, 4.4))
    n = len(tbl.columns)
    w = 0.8 / n
    x = np.arange(len(tbl))
    for i, col in enumerate(tbl.columns):
        ax.bar(x + (i - (n - 1) / 2) * w, tbl[col], width=w, color=colors[col], edgecolor=SURFACE, linewidth=2, label=col)
    _zero_line(ax)
    ax.set_xticks(x, [str(i) for i in tbl.index])
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: fmt(v)))
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    return _finish(fig, ax, title, subtitle, path, note)


def lift_panels(panels: dict[str, pd.DataFrame], title: str, subtitle: str, path=None):
    """Small multiples: lift of extreme RT>DA (red) and RT<DA (blue) spread hours per category."""
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 4.4), sharey=True)
    for ax, (name, tbl) in zip(np.atleast_1d(axes), panels.items()):
        x = np.arange(len(tbl))
        w = 0.38
        ax.bar(x - w / 2 - 0.01, tbl["pos"], width=w, color=RED, edgecolor=SURFACE, linewidth=2, label="Extreme RT > DA")
        ax.bar(x + w / 2 + 0.01, tbl["neg"], width=w, color=BLUE, edgecolor=SURFACE, linewidth=2, label="Extreme RT < DA")
        ax.axhline(1, color=INK_2, linewidth=1)
        ax.set_xticks(x, [str(i) for i in tbl.index], fontsize=8.5)
        ax.set_title(name, fontsize=10.5)
        ax.grid(axis="x", visible=False)
        # label only each series' peak category in this panel
        for off, col in ((-w / 2 - 0.01, "pos"), (w / 2 + 0.01, "neg")):
            ti = int(np.argmax(tbl[col].to_numpy()))
            ax.text(ti + off, tbl[col].iloc[ti] + 0.2, f"{tbl[col].iloc[ti]:.1f}×", ha="center", fontsize=8.5, color=INK)
    np.atleast_1d(axes)[0].set_ylabel("Lift vs share of all hours (1 = no over-representation)")
    np.atleast_1d(axes)[0].legend(loc="upper left")
    fig.tight_layout()
    return _finish(fig, axes, title, subtitle, path)
