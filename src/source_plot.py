"""
source_plot.py
==============
Visualization module for Source objects.

Public API
----------
draw_energy(source, **kwargs)           -> (fig, ax)    — energy PDF + CDF
draw_spatial(source, geometry, **kwargs)-> (fig, ax)    — weight per region on x-axis
draw_source(source, geometry, **kwargs) -> (fig, axes)  — full dashboard
SourcePlotter(source, geometry)         -> object with .draw() / .save()

Integrates with geometry_plot.py: the spatial panel reuses the same x-axis
layout as geometry_plot.draw() so panels can be stacked together.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import lognorm
from src.geometry_plot import _sorted_regions, _boundary_xs, _subsample_ticks
from src.geometry_plot import draw as geom_draw
from src.geometry_plot import _sorted_regions
# ─────────────────────────────────────────────────────────────────────────────
# Colours
# ─────────────────────────────────────────────────────────────────────────────

_SOURCE_COLOUR  = "#E8A838"   # amber — source weight bars
_PDF_COLOUR     = "#2471A3"   # steel blue — PDF curve
_CDF_COLOUR     = "#1E8449"   # green — CDF curve
_CUTOFF_COLOUR  = "#C0392B"   # red — E_max cutoff line
_MODE_COLOUR    = "#7D3C98"   # purple — mode marker


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_source_regions(source) -> list:
    return list(source.source_regions)


def _norm_weights(source) -> dict[str, float]:
    """Return {region_name: normalised_weight} from the source."""
    srs   = _get_source_regions(source)
    total = sum(sr.weight for sr in srs)
    return {sr.region_name: sr.weight / total for sr in srs}


def _lognormal_pdf_cdf(
    mu: float,
    sigma: float,
    E_min: float,
    E_max: float,
    n_points: int = 800,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (E, pdf, cdf) arrays for a log-normal distribution
    with parameters (mu, sigma) in log(eV) space, evaluated on a
    log-spaced grid between E_min and E_max.

    scipy parametrisation:  X = exp(mu + sigma * Z),  Z ~ N(0,1)
    so  s=sigma,  scale=exp(mu)
    """
    E  = np.geomspace(max(E_min, 1e-3), E_max, n_points)
    rv = lognorm(s=sigma, scale=math.exp(mu))
    pdf = rv.pdf(E)
    cdf = rv.cdf(E)
    return E, pdf, cdf


def _lognormal_mode(mu: float, sigma: float) -> float:
    return math.exp(mu - sigma ** 2)


def _lognormal_mean(mu: float, sigma: float) -> float:
    return math.exp(mu + sigma ** 2 / 2)


def _lognormal_median(mu: float, sigma: float) -> float:
    return math.exp(mu)


def _acceptance_rate(mu: float, sigma: float, E_min: float, E_max: float) -> float:
    """Fraction of raw log-normal samples that fall within [E_min, E_max]."""
    rv  = lognorm(s=sigma, scale=math.exp(mu))
    return float(rv.cdf(E_max) - rv.cdf(E_min))


# ─────────────────────────────────────────────────────────────────────────────
# Energy distribution plot
# ─────────────────────────────────────────────────────────────────────────────

def draw_energy(
    source,
    *,
    figsize:      tuple = (10, 4.5),
    show_cdf:     bool  = True,
    show_markers: bool  = True,
    title:        str | None = None,
    ax_pdf:       matplotlib.axes.Axes | None = None,
    ax_cdf:       matplotlib.axes.Axes | None = None,
) -> tuple[matplotlib.figure.Figure, dict]:
    """
    Plot the energy distribution of the source.

    Handles:
      - log_normal  : full PDF + CDF with mode / mean / median markers
      - flat        : uniform rectangle between E_min and E_max
      - mono        : vertical line at E_min

    Returns
    -------
    fig, axes  — axes dict has keys 'pdf' and optionally 'cdf'
    """
    srs = _get_source_regions(source)
    if not srs:
        raise ValueError("Source has no SourceRegions.")

    # All regions must share the same energy distribution (typical case).
    # If they differ we plot the first one and warn.
    sr0 = srs[0]
    if len({sr.energy_dist for sr in srs}) > 1:
        warnings.warn(
            "[source_plot] Source regions have mixed energy distributions — "
            "plotting the first region only."
        )

    E_min, E_max = sr0.energy_range
    dist  = sr0.energy_dist

    # ── build figure ─────────────────────────────────────────────────────────
    if ax_pdf is None:
        if show_cdf and dist == "log_normal":
            fig, (ax_l, ax_r) = plt.subplots(
                1, 2, figsize=figsize, facecolor="white",
                gridspec_kw={"width_ratios": [3, 1], "wspace": 0.08},
            )
        else:
            fig, ax_l = plt.subplots(figsize=figsize, facecolor="white")
            ax_r = None
    else:
        fig   = ax_pdf.figure
        ax_l  = ax_pdf
        ax_r  = ax_cdf

    axes_out = {"pdf": ax_l, "cdf": ax_r}

    # ── PDF panel ────────────────────────────────────────────────────────────
    ax = ax_l
    ax.set_facecolor("white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    if dist == "log_normal":
        mu, sigma = sr0.mu, sr0.sigma
        E, pdf, cdf = _lognormal_pdf_cdf(mu, sigma, E_min, E_max)

        # Truncated PDF: zero outside [E_min, E_max], renormalise
        accept = _acceptance_rate(mu, sigma, E_min, E_max)
        pdf_trunc = pdf / accept

        ax.semilogx(E, pdf_trunc, color=_PDF_COLOUR, lw=2.0, label="PDF (truncated)")
        ax.fill_between(E, pdf_trunc, alpha=0.15, color=_PDF_COLOUR)

        # E_max cutoff
        ax.axvline(E_max, color=_CUTOFF_COLOUR, lw=1.5, ls="--",
                   label=f"$E_{{\\rm max}}$ = {E_max:.0f} eV")

        if show_markers:
            mode   = _lognormal_mode(mu, sigma)
            mean_e = _lognormal_mean(mu, sigma)
            median = _lognormal_median(mu, sigma)

            for xv, lbl, col, ls in [
                (mode,   f"mode = {mode:.0f} eV",   _MODE_COLOUR, "-"),
                (mean_e, f"mean = {mean_e:.0f} eV",  "#E67E22",    "-."),
                (median, f"median = {median:.0f} eV", "#16A085",   ":"),
            ]:
                if E_min <= xv <= E_max:
                    ax.axvline(xv, color=col, lw=1.2, ls=ls, alpha=0.85, label=lbl)

        ax.set_xlabel("Energy  [eV]", fontsize=10)
        ax.set_ylabel("Probability density  [eV⁻¹]", fontsize=10)

        # acceptance rate annotation
        ax.text(
            0.97, 0.95,
            f"Acceptance rate: {100*accept:.1f}%\n"
            f"μ = {mu:.3f},  σ = {sigma:.3f}",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.85),
        )

    elif dist == "flat":
        ax.axhspan(0, 1, xmin=0, xmax=1, alpha=0)   # invisible — just set limits
        ax.fill_betweenx([0, 1], E_min, E_max,
                         color=_PDF_COLOUR, alpha=0.35, label="Uniform")
        ax.set_xscale("log")
        ax.set_xlim(E_min * 0.5, E_max * 2)
        ax.set_ylim(0, 1.2)
        ax.set_xlabel("Energy  [eV]", fontsize=10)
        ax.set_ylabel("Relative probability", fontsize=10)

    elif dist == "mono":
        E0 = E_min
        ax.axvline(E0, color=_PDF_COLOUR, lw=2.5,
                   label=f"Mono-energetic: {E0:.3e} eV")
        ax.set_xscale("log")
        ax.set_xlim(E0 * 0.1, E0 * 10)
        ax.set_xlabel("Energy  [eV]", fontsize=10)
        ax.set_ylabel("", fontsize=10)

    ax.legend(fontsize=8, framealpha=0.8, edgecolor="#cccccc")
    ax.tick_params(labelsize=8)

    # ── CDF panel (log-normal only) ───────────────────────────────────────────
    if ax_r is not None and dist == "log_normal":
        ax_r.set_facecolor("white")
        for sp in ("top", "right"):
            ax_r.spines[sp].set_visible(False)

        ax_r.semilogx(E, cdf, color=_CDF_COLOUR, lw=2.0)
        ax_r.fill_between(E, cdf, alpha=0.12, color=_CDF_COLOUR)
        ax_r.axvline(E_max, color=_CUTOFF_COLOUR, lw=1.5, ls="--")
        ax_r.set_xlabel("Energy  [eV]", fontsize=10)
        ax_r.set_ylabel("CDF", fontsize=10)
        ax_r.set_ylim(0, 1.05)
        ax_r.yaxis.set_label_position("right")
        ax_r.yaxis.tick_right()
        ax_r.tick_params(labelsize=8)

        # shade truncated tail
        ax_r.axhspan(float(lognorm(s=sigma, scale=math.exp(mu)).cdf(E_max)),
                     1.0, color=_CUTOFF_COLOUR, alpha=0.08,
                     label="rejected tail")

    # ── title ─────────────────────────────────────────────────────────────────
    dist_label = {"log_normal": "Log-normal", "flat": "Uniform", "mono": "Mono-energetic"}
    fig.suptitle(
        title or f"Source energy distribution — {dist_label.get(dist, dist)}",
        fontsize=11, y=1.01,
    )
    try:
        fig.tight_layout()
    except Exception:
        pass

    return fig, axes_out


# ─────────────────────────────────────────────────────────────────────────────
# Spatial distribution plot
# ─────────────────────────────────────────────────────────────────────────────

def draw_spatial(
    source,
    geometry,
    *,
    figsize:    tuple = (14, 2.5),
    title:      str | None = None,
    ax:         matplotlib.axes.Axes | None = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Draw a strip showing the normalised source weight per region on the
    geometry x-axis.  Regions not in the source have zero weight and are
    drawn in light grey.  The bar height encodes the weight, so non-uniform
    spatial distributions are immediately visible.
    """


    sr_map    = _norm_weights(source)   # {region_name: norm_weight}
    sorted_r  = _sorted_regions(geometry)
    x0        = sorted_r[0].x_min
    x1        = sorted_r[-1].x_max
    span      = x1 - x0

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    else:
        fig = ax.figure

    ax.set_facecolor("white")
    ax.set_xlim(x0 - 0.015 * span, x1 + 0.015 * span)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1"], fontsize=7)
    ax.set_ylabel("norm.\nweight", fontsize=8, rotation=0,
                  labelpad=32, va="center", color="#555555")
    ax.set_xlabel("x  [cm]", fontsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    max_w = max(sr_map.values()) if sr_map else 1.0

    for r in sorted_r:
        w       = r.x_max - r.x_min
        weight  = sr_map.get(r.name, 0.0)
        h       = weight / max_w   # normalise to [0,1] so tallest bar = 1

        if weight > 0:
            col   = _SOURCE_COLOUR
            alpha = 0.75
            ec    = "#8B5E1A"
        else:
            col   = "#DDDDDD"
            alpha = 0.45
            ec    = "#AAAAAA"

        rect = mpatches.FancyBboxPatch(
            (r.x_min, 0.0), w, max(h, 0.03),
            boxstyle="square,pad=0",
            facecolor=col, edgecolor=ec,
            linewidth=0.3, alpha=alpha,
        )
        ax.add_patch(rect)

    # weight scale reference line
    ax.axhline(1.0, color="#AAAAAA", lw=0.6, ls=":", zorder=0)

    # legend
    handles = [
        mpatches.Patch(facecolor=_SOURCE_COLOUR, edgecolor="#8B5E1A",
                       alpha=0.75, label="Source region"),
        mpatches.Patch(facecolor="#DDDDDD", edgecolor="#AAAAAA",
                       alpha=0.45, label="Non-source region"),
    ]
    ax.legend(handles=handles, loc="upper right",
              fontsize=8, framealpha=0.8, edgecolor="#cccccc")

    # x ticks
    bxs   = _boundary_xs(geometry)
    ticks = _subsample_ticks(bxs, span, max_ticks=20)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="x", labelsize=7.5, rotation=45, length=3)

    n_source = sum(1 for r in sorted_r if r.name in sr_map)
    ax.set_title(
        title or f"Source spatial distribution  —  {n_source} source regions  "
                 f"/ {len(sorted_r)} total",
        fontsize=10, pad=6,
    )

    try:
        fig.tight_layout()
    except Exception:
        pass

    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Full dashboard
# ─────────────────────────────────────────────────────────────────────────────

def draw_source(
    source,
    geometry,
    *,
    figsize: tuple = (15, 9),
    title:   str | None = None,
) -> tuple[matplotlib.figure.Figure, dict]:
    """
    Three-panel dashboard:

        ┌──────────────────────────────────────────┐
        │  Geometry strip  (from geometry_plot)    │  row 0
        ├──────────────────────────────────────────┤
        │  Spatial weight strip                    │  row 1
        ├──────────────────────────────────────────┤
        │  Energy PDF          │  CDF              │  row 2
        └──────────────────────────────────────────┘

    Returns
    -------
    fig, axes  — dict with keys 'geometry', 'spatial', 'pdf', 'cdf'
    """


    fig = plt.figure(figsize=figsize, facecolor="white")
    gs  = gridspec.GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[2.5, 1.2, 3],
        width_ratios=[3, 1],
        hspace=0.52,
        wspace=0.08,
    )

    ax_geom    = fig.add_subplot(gs[0, :])   # geometry strip spans both columns
    ax_spatial = fig.add_subplot(gs[1, :])   # spatial strip spans both columns
    ax_pdf     = fig.add_subplot(gs[2, 0])   # energy PDF
    ax_cdf     = fig.add_subplot(gs[2, 1])   # CDF

    # ── geometry strip ────────────────────────────────────────────────────────
    geom_draw(
        geometry,
        ax_mat=ax_geom,
        show_temp_strip=False,
        show_temp_colorbar=False,
        show_legend=True,
        title="",
    )
    # Overlay source region boundaries in amber
    sr_map   = _norm_weights(source)

    sorted_r = _sorted_regions(geometry)
    x0       = sorted_r[0].x_min
    x1       = sorted_r[-1].x_max
    span     = x1 - x0

    for r in sorted_r:
        if r.name in sr_map:
            ax_geom.axvspan(r.x_min, r.x_max,
                            color=_SOURCE_COLOUR, alpha=0.22, zorder=2)

    # amber patch legend entry
    geom_legend = ax_geom.get_legend()
    extra = mpatches.Patch(facecolor=_SOURCE_COLOUR, alpha=0.5,
                           edgecolor="#8B5E1A", label="Source")
    if geom_legend:
        handles = geom_legend.legend_handles + [extra]
        ax_geom.legend(handles=handles, loc="upper right",
                       fontsize=8, framealpha=0.85, edgecolor="#cccccc",
                       title="Material class", title_fontsize=8)

    # ── spatial strip ─────────────────────────────────────────────────────────
    draw_spatial(source, geometry, ax=ax_spatial, title="")
    ax_spatial.set_xlabel("")   # shared x-axis — label only on bottom panel

    # ── energy panels ─────────────────────────────────────────────────────────
    draw_energy(source, ax_pdf=ax_pdf, ax_cdf=ax_cdf, show_cdf=True)

    # ── overall title ─────────────────────────────────────────────────────────
    srs   = _get_source_regions(source)
    dist  = srs[0].energy_dist if srs else "?"
    E_min, E_max = srs[0].energy_range if srs else (0, 0)
    fig.suptitle(
        title or (
            f"Source dashboard  ·  {source.neutron_nbr:,} neutrons  ·  "
            f"{len(srs)} source regions  ·  {dist}  "
            f"[{E_min:.1f}, {E_max:.0f}] eV"
        ),
        fontsize=12, y=1.01,
    )

    axes_out = {
        "geometry": ax_geom,
        "spatial":  ax_spatial,
        "pdf":      ax_pdf,
        "cdf":      ax_cdf,
    }
    return fig, axes_out


# ─────────────────────────────────────────────────────────────────────────────
# High-level plotter object
# ─────────────────────────────────────────────────────────────────────────────

class SourcePlotter:
    """
    Convenience wrapper.

    Usage
    -----
    >>> p = SourcePlotter(source, geometry)
    >>> p.draw()               # full dashboard
    >>> p.draw_energy()        # energy distribution only
    >>> p.draw_spatial()       # spatial weight strip only
    >>> p.save("source.png")
    """

    def __init__(self, source, geometry):
        self.source   = source
        self.geometry = geometry
        self._last_fig: matplotlib.figure.Figure | None = None

    def draw(self, **kwargs):
        fig, axes = draw_source(self.source, self.geometry, **kwargs)
        self._last_fig = fig
        plt.show()
        return fig, axes

    def draw_energy(self, **kwargs):
        fig, axes = draw_energy(self.source, **kwargs)
        self._last_fig = fig
        plt.show()
        return fig, axes

    def draw_spatial(self, **kwargs):
        fig, ax = draw_spatial(self.source, self.geometry, **kwargs)
        self._last_fig = fig
        plt.show()
        return fig, ax

    def save(self, path: str, dpi: int = 150, **kwargs):
        if self._last_fig is None:
            warnings.warn("[SourcePlotter] No figure to save — call draw() first.")
            return
        self._last_fig.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)
        print(f"[SourcePlotter] Saved → {path}")

    def __repr__(self):
        n = len(_get_source_regions(self.source))
        return f"<SourcePlotter  {n} source regions>"