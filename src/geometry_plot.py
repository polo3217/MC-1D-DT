"""
geometry_plot.py
================
Visualization module for 1-D Monte Carlo geometry objects.

Public API
----------
draw(geometry, **kwargs)            -> (fig, axes)   — main entry point
draw_temperature_profile(geometry)  -> (fig, ax)     — T vs x line plot
draw_xs_overview(geometry)          -> (fig, ax)     — majorant XS ribbon (if available)
GeometryPlotter(geometry)           -> object with .draw() / .save() / .summary()

All functions also work as bound methods when attached to a Geometry class:
    Geometry.draw = lambda self, **kw: draw(self, **kw)
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LogNorm, BoundaryNorm
from matplotlib.cm import ScalarMappable


# ─────────────────────────────────────────────────────────────────────────────
# Internal constants
# ─────────────────────────────────────────────────────────────────────────────

# Colour palette for material classes
_CLASS_PALETTE: dict[str, str] = {
    "fuel":    "#D4622A",   # burnt sienna
    "water":   "#3A7FBF",   # steel blue
    "clad":    "#8C9E9E",   # slate grey
    "void":    "#E8E8E8",   # light grey
    "poison":  "#6A3FA0",   # purple
    "other":   "#BFA060",   # tan
}

# Hatching patterns per class (for accessibility / B&W printing)
_CLASS_HATCH: dict[str, str] = {
    "fuel":    "",
    "water":   "//",
    "clad":    "\\\\",
    "void":    "..",
    "poison":  "xx",
    "other":   "--",
}

_BC_META: dict[str, tuple[str, str]] = {
    "reflective": ("↔", "#2471A3"),
    "vacuum":     ("✕", "#C0392B"),
    "periodic":   ("⇄", "#1E8449"),
}

# Temperature colormaps to choose from
_TEMP_CMAPS = ("plasma", "inferno", "magma", "hot", "RdYlBu_r")


# ─────────────────────────────────────────────────────────────────────────────
# Region classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_region(region) -> str:
    """
    Infer a coarse material class from the region or material name.
    Extend `_CLASS_PALETTE` to add new classes.
    """
    raw = (region.name + " " + region.material.name).lower()
    tokens = set(raw.replace("-", " ").replace("_", " ").split())
    if tokens & {"fuel", "uo2", "u235", "u238", "pellet", "pin"}:
        return "fuel"
    if tokens & {"water", "h2o", "coolant", "moderator", "h1", "h-1"}:
        return "water"
    if tokens & {"clad", "cladding", "zr", "zircaloy", "zr4", "m5"}:
        return "clad"
    if tokens & {"void", "air", "gap", "vacuum"}:
        return "void"
    if tokens & {"poison", "b4c", "boron", "gd", "gadolinium", "hafnium"}:
        return "poison"
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sorted_regions(geometry) -> list:
    return sorted(geometry._regions, key=lambda r: r.x_min)


def _geometry_span(geometry) -> tuple[float, float]:
    sr = _sorted_regions(geometry)
    return sr[0].x_min, sr[-1].x_max


def _all_temps(geometry) -> list[float]:
    return [r.material.T for r in geometry._regions]


def _boundary_xs(geometry) -> list[float]:
    xs = set()
    for r in geometry._regions:
        xs.add(r.x_min)
        xs.add(r.x_max)
    return sorted(xs)


def _subsample_ticks(positions: list[float], span: float, max_ticks: int = 22) -> list[float]:
    """Keep at most `max_ticks` evenly spaced positions."""
    if len(positions) <= max_ticks:
        return positions
    step = span / max_ticks
    kept, last = [], -np.inf
    for p in positions:
        if p - last >= step:
            kept.append(p)
            last = p
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Core draw
# ─────────────────────────────────────────────────────────────────────────────

def draw(
    geometry,
    *,
    figsize: tuple[float, float] = (15, 5),
    temp_cmap: str = "plasma",
    label_min_width_cm: float = 0.35,
    show_legend: bool = True,
    show_temp_colorbar: bool = True,
    show_temp_strip: bool = True,
    show_rod_index: bool = True,
    annotate_bc: bool = True,
    title: str | None = None,
    ax_mat: matplotlib.axes.Axes | None = None,
    ax_temp: matplotlib.axes.Axes | None = None,
) -> tuple[matplotlib.figure.Figure, dict]:
    """
    Draw the 1-D geometry.

    Parameters
    ----------
    geometry            : Geometry object with ._regions, .boundary_conditions
    figsize             : overall figure size in inches
    temp_cmap           : matplotlib colormap name for the temperature strip
    label_min_width_cm  : minimum region width [cm] at which text labels appear
    show_legend         : show material-class legend
    show_temp_colorbar  : show temperature colorbar
    show_temp_strip     : show the temperature heatmap row
    show_rod_index      : annotate rod / gap index above each fuel group
    annotate_bc         : draw boundary-condition markers
    title               : figure title (auto-generated if None)
    ax_mat, ax_temp     : pass existing axes to draw into (disables figure creation)

    Returns
    -------
    fig   : matplotlib Figure
    axes  : dict with keys 'mat', 'temp' (optional), 'cbar' (optional)
    """
    if not geometry._regions:
        print("[geometry_plot] No regions defined — nothing to draw.")
        return None, {}

    sr    = _sorted_regions(geometry)
    x0, x1 = sr[0].x_min, sr[-1].x_max
    span  = x1 - x0

    all_T   = _all_temps(geometry)
    T_min, T_max = min(all_T), max(all_T)
    if T_min == T_max:
        T_min, T_max = T_min - 50, T_max + 50
    t_norm = Normalize(vmin=T_min, vmax=T_max)
    t_cmap = plt.get_cmap(temp_cmap)

    # ── build figure / axes ──────────────────────────────────────────────────
    if ax_mat is None:
        n_rows         = 2 if show_temp_strip else 1
        height_ratios  = [1, 3] if show_temp_strip else [1]
        n_cols         = 2 if show_temp_colorbar else 1
        width_ratios   = [32, 1] if show_temp_colorbar else [1]

        fig = plt.figure(figsize=figsize, facecolor="white")
        gs  = gridspec.GridSpec(
            n_rows, n_cols,
            figure=fig,
            height_ratios=height_ratios,
            width_ratios=width_ratios,
            hspace=0.06,
            wspace=0.03,
        )

        ax_mat_new  = fig.add_subplot(gs[n_rows - 1, 0])
        ax_temp_new = fig.add_subplot(gs[0, 0], sharex=ax_mat_new) if show_temp_strip else None
        ax_cbar     = fig.add_subplot(gs[:, 1]) if show_temp_colorbar else None
    else:
        fig = ax_mat.figure
        ax_mat_new  = ax_mat
        ax_temp_new = ax_temp
        ax_cbar     = None

    axes_out = {"mat": ax_mat_new, "temp": ax_temp_new, "cbar": ax_cbar}

    # ── shared axis setup ────────────────────────────────────────────────────
    pad = 0.018 * span
    for ax in filter(None, [ax_mat_new, ax_temp_new]):
        ax.set_xlim(x0 - pad, x1 + pad)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        for spine in ("top", "left", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_facecolor("white")

    if ax_temp_new is not None:
        ax_temp_new.set_xticks([])
        ax_temp_new.spines["bottom"].set_visible(False)
        ax_temp_new.set_ylabel("T", fontsize=8, rotation=0,
                               labelpad=14, va="center", color="#555555")

    ax_mat_new.set_xlabel("x  [cm]", fontsize=10, labelpad=6)

    # ── track which fuel/water groups we've seen (for rod index labels) ──────
    # A "group" is a contiguous run of the same class
    seen_classes: dict[str, str] = {}   # class -> colour
    rod_groups: list[tuple[float, float, str, int]] = []  # (cx, width, cls, idx)
    _prev_cls, _group_start, _group_idx = None, x0, 0

    def _flush_group(cls, x_end):
        nonlocal _prev_cls, _group_start, _group_idx
        if cls is not None:
            rod_groups.append((_group_start, x_end, cls, _group_idx))
            _group_idx += 1
        _prev_cls  = cls
        _group_start = x_end

    for r in sr:
        cls = classify_region(r)
        if cls != _prev_cls:
            _flush_group(_prev_cls, r.x_min)
        _prev_cls = cls
    _flush_group(_prev_cls, x1)

    # ── draw each region ─────────────────────────────────────────────────────
    for r in sr:
        w   = r.x_max - r.x_min
        cx  = r.x_min + w / 2
        cls = classify_region(r)
        col = _CLASS_PALETTE.get(cls, _CLASS_PALETTE["other"])
        hatch = _CLASS_HATCH.get(cls, "")
        seen_classes[cls] = col

        # temperature strip
        if ax_temp_new is not None:
            t_col = t_cmap(t_norm(r.material.T))
            ax_temp_new.barh(
                0.5, w, left=r.x_min, height=0.80,
                color=t_col, edgecolor="none",
            )

        # material strip
        rect = mpatches.FancyBboxPatch(
            (r.x_min, 0.06), w, 0.88,
            boxstyle="square,pad=0",
            facecolor=col,
            edgecolor="black",
            linewidth=0.35,
            alpha=0.55,
            hatch=hatch,
        )
        ax_mat_new.add_patch(rect)

        # region separator lines
        for xb in (r.x_min, r.x_max):
            ax_mat_new.axvline(xb, color="black", linewidth=0.3,
                               linestyle="-", alpha=0.18, zorder=1)

        # text label — only when wide enough
        if w >= label_min_width_cm:
            ax_mat_new.text(
                cx, 0.50,
                f"{r.material.T:.0f} K",
                ha="center", va="center",
                fontsize=max(5.5, min(8.5, w * 8)),
                color="white",
                fontweight="bold",
                zorder=3,
            )

    # ── rod-index annotations above material strip ───────────────────────────
    if show_rod_index:
        fuel_counter, water_counter = 0, 0
        for gx_start, gx_end, gcls, gidx in rod_groups:
            gw = gx_end - gx_start
            gcx = gx_start + gw / 2
            if gcls == "fuel":
                fuel_counter += 1
                lbl = f"F{fuel_counter}"
            elif gcls == "water":
                water_counter += 1
                lbl = f"W{water_counter}"
            else:
                lbl = gcls[0].upper() + str(gidx)

            # only annotate if there's room
            if gw >= label_min_width_cm * 0.6:
                ref_ax = ax_temp_new if ax_temp_new is not None else ax_mat_new
                ref_ax.text(
                    gcx, 0.96,
                    lbl,
                    ha="center", va="top",
                    fontsize=max(5, min(7.5, gw * 6)),
                    color="#333333",
                    fontweight="bold",
                    clip_on=True,
                    zorder=4,
                )

    # ── boundary condition markers ───────────────────────────────────────────
    if annotate_bc and hasattr(geometry, "boundary_conditions"):
        bc = geometry.boundary_conditions
        for side, xpos, ha, xsign in (
            ("left",  x0, "right", -1),
            ("right", x1, "left",  +1),
        ):
            side_bc = bc.get(side, "vacuum")
            sym, col = _BC_META.get(side_bc, ("?", "black"))
            for ax in filter(None, [ax_mat_new, ax_temp_new]):
                ax.axvline(xpos, color=col, linewidth=2.2, alpha=0.85, zorder=5)
            ax_mat_new.text(
                xpos + xsign * 0.010 * span, 0.50,
                f"{sym} {side_bc}",
                ha=ha, va="center",
                fontsize=8, color=col, fontweight="bold",
                zorder=6,
            )

    # ── colorbar ─────────────────────────────────────────────────────────────
    if ax_cbar is not None:
        sm = ScalarMappable(cmap=t_cmap, norm=t_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=ax_cbar)
        cbar.set_label("T  [K]", fontsize=9, labelpad=6)
        cbar.ax.tick_params(labelsize=8)
        # add discrete tick at each unique temperature
        unique_T = sorted(set(all_T))
        if len(unique_T) <= 12:
            cbar.set_ticks(unique_T)
            cbar.ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda v, _: f"{v:.0f}")
            )

    # ── x-axis ticks ────────────────────────────────────────────────────────
    bxs = _boundary_xs(geometry)
    ticks = _subsample_ticks(bxs, span, max_ticks=20)
    ax_mat_new.set_xticks(ticks)
    ax_mat_new.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax_mat_new.tick_params(axis="x", labelsize=7.5, rotation=45, length=3)

    # ── legend ───────────────────────────────────────────────────────────────
    if show_legend and seen_classes:
        handles = [
            mpatches.Patch(
                facecolor=_CLASS_PALETTE.get(cls, _CLASS_PALETTE["other"]),
                edgecolor="black", linewidth=0.6, alpha=0.7,
                hatch=_CLASS_HATCH.get(cls, ""),
                label=cls.capitalize(),
            )
            for cls in sorted(seen_classes)
        ]
        ax_mat_new.legend(
            handles=handles,
            loc="upper right",
            fontsize=8,
            framealpha=0.85,
            edgecolor="#cccccc",
            title="Material class",
            title_fontsize=8,
        )

    # ── title ────────────────────────────────────────────────────────────────
    if title is None:
        n_reg = len(geometry._regions)
        T_range = f"{T_min:.0f} – {T_max:.0f} K"
        title = f"Geometry  ·  {n_reg} regions  ·  {x0:.2f} → {x1:.2f} cm  ·  T ∈ {T_range}"
    fig.suptitle(title, fontsize=11, y=1.01, color="#222222")

    try:
        fig.tight_layout()
    except Exception:
        pass
    return fig, axes_out


# ─────────────────────────────────────────────────────────────────────────────
# Temperature profile plot
# ─────────────────────────────────────────────────────────────────────────────

def draw_temperature_profile(
    geometry,
    *,
    figsize: tuple[float, float] = (13, 3.5),
    show_classes: bool = True,
    title: str | None = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Step-function plot of temperature vs x-position.
    Fuel and water regions are shaded in background.
    """
    if not geometry._regions:
        return None, None

    sr = _sorted_regions(geometry)
    x0, x1 = sr[0].x_min, sr[-1].x_max
    span = x1 - x0

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    # background shading per class
    if show_classes:
        for r in sr:
            cls = classify_region(r)
            col = _CLASS_PALETTE.get(cls, _CLASS_PALETTE["other"])
            ax.axvspan(r.x_min, r.x_max, alpha=0.10, color=col, linewidth=0)

    # step-function temperature line
    xs = []
    ys = []
    for r in sr:
        xs += [r.x_min, r.x_max]
        ys += [r.material.T, r.material.T]

    ax.plot(xs, ys, color="#1a1a1a", linewidth=1.5, drawstyle="steps-post", zorder=3)
    ax.fill_between(xs, ys, alpha=0.15, color="#555555", step="post", zorder=2)

    # mark unique temperature levels
    unique_T = sorted(set(ys))
    for T in unique_T:
        ax.axhline(T, color="#aaaaaa", linewidth=0.5, linestyle=":", zorder=1)
        ax.text(x1 + 0.005 * span, T, f"{T:.0f} K",
                va="center", ha="left", fontsize=7.5, color="#555555")

    ax.set_xlim(x0 - 0.01 * span, x1 + 0.06 * span)
    ax.set_xlabel("x  [cm]", fontsize=10)
    ax.set_ylabel("T  [K]", fontsize=10)
    ax.set_title(title or "Temperature profile", fontsize=11, pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Geometry summary table (text)
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(geometry) -> None:
    """
    Print a structured summary of the geometry:

      1. Global geometry info (span, region count, BCs)
      2. Per material-class block (fuel / water / …):
           - aggregate statistics (region count, total width, T range)
           - table of unique materials with nuclide compositions
    """
    sr = _sorted_regions(geometry)
    x0, x1 = sr[0].x_min, sr[-1].x_max
    W = 72   # total line width

    # ── helpers ──────────────────────────────────────────────────────────────
    def _nuclides(mat) -> list[tuple[str, float]]:
        """
        Return list of (nuclide_name, atom_density) from a material.

        Handles three storage formats:
          - list of (nuclide_obj, density)  where nuclide_obj has a .name attr
            (e.g. openmc.data.multipole.WindowedMultipole)
          - list of (name_str, density)
          - dict {name_str: density}
        """
        raw = getattr(mat, "nuclides", None)
        if raw is None:
            return []

        def _extract_name(n) -> str:
            # prefer .name attribute (covers WindowedMultipole and similar)
            if hasattr(n, "name"):
                return str(n.name)
            return str(n)

        if isinstance(raw, dict):
            return [(_extract_name(n), float(d)) for n, d in raw.items()]

        if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], (list, tuple)):
            return [(_extract_name(n), float(d)) for n, d in raw]

        return []

    def _total_atom_density(nuclides) -> float:
        return sum(d for _, d in nuclides)

    def _atom_fractions(nuclides) -> list[tuple[str, float, float]]:
        """Return [(name, density, at_frac), ...]  sorted by density desc."""
        total = _total_atom_density(nuclides) or 1.0
        return sorted(
            [(n, d, d / total) for n, d in nuclides],
            key=lambda x: x[1], reverse=True,
        )

    # ── group regions by class, then by unique material name ─────────────────
    classes: dict[str, list] = {}
    for r in sr:
        classes.setdefault(classify_region(r), []).append(r)

    # ── print ─────────────────────────────────────────────────────────────────
    print("=" * W)
    print("  GEOMETRY SUMMARY")
    print("=" * W)
    print(f"  Span        : {x0:.4f} → {x1:.4f} cm   (Δ = {x1 - x0:.4f} cm)")
    print(f"  Regions     : {len(sr)}")
    bc = getattr(geometry, "boundary_conditions", {})
    print(f"  BC left     : {bc.get('left',  '?')}")
    print(f"  BC right    : {bc.get('right', '?')}")

    for cls, regs in sorted(classes.items()):
        T_vals  = [r.material.T for r in regs]
        total_w = sum(r.x_max - r.x_min for r in regs)

        print()
        print("─" * W)
        print(f"  CLASS : {cls.upper()}")
        print(f"  {'Regions':<14}: {len(regs)}")
        print(f"  {'Total width':<14}: {total_w:.4f} cm")
        print(f"  {'T range':<14}: {min(T_vals):.0f} – {max(T_vals):.0f} K")
        print()

        # deduplicate by composition fingerprint  (nuclide names + densities,
        # rounded to 4 sig-figs so floating-point noise doesn't create false
        # duplicates).  For each unique composition we record the name of the
        # first material that carries it and all temperatures at which it appears.
        def _fingerprint(mat) -> tuple:
            nucs = _nuclides(mat)
            return tuple(sorted((n, round(d, 4 - int(np.floor(np.log10(abs(d)))) - 1) if d > 0 else 0)
                                for n, d in nucs))

        # ordered dict: fingerprint -> (representative_material, [T values], count)
        comp_groups: dict[tuple, list] = {}
        for r in regs:
            fp = _fingerprint(r.material)
            if fp not in comp_groups:
                comp_groups[fp] = [r.material, [], 0]
            comp_groups[fp][1].append(r.material.T)
            comp_groups[fp][2] += 1

        print(f"  {'Unique compositions':<20}: {len(comp_groups)}"
              f"  (out of {len(regs)} regions — sub-slabs with same composition grouped)")

        for fp, (mat, T_list, count) in comp_groups.items():
            nuclides = _nuclides(mat)
            T_uniq   = sorted(set(T_list))
            T_str    = ", ".join(f"{t:.0f}" for t in T_uniq)
            print()
            print(f"    ┌─ {mat.name}  ×{count} regions   T ∈ {{{T_str}}} K")

            if not nuclides:
                print(f"    │   (no nuclide data available)")
            else:
                fracs = _atom_fractions(nuclides)
                N_tot = _total_atom_density(nuclides)
                # header
                print(f"    │   {'Nuclide':<10}  {'N [at/cm³]':>14}  {'at. frac':>10}  {'wt. frac':>10}")
                print(f"    │   {'─'*10}  {'─'*14}  {'─'*10}  {'─'*10}")

                # approximate atomic masses for weight-fraction calculation
                _A = {
                    "H1": 1.008,   "H-1": 1.008,
                    "H2": 2.014,   "D":   2.014,
                    "O16": 15.999, "O-16": 15.999, "O": 15.999,
                    "U235": 235.044, "U-235": 235.044,
                    "U238": 238.051, "U-238": 238.051,
                    "ZR90": 89.905,  "ZR91": 90.906, "ZR92": 91.905,
                    "ZR94": 93.906,  "ZR96": 95.908,
                    "B10": 10.013,   "B11": 11.009,
                }
                def _mass(name: str) -> float:
                    key = name.upper().replace("-", "")
                    if key in _A:
                        return _A[key]
                    # fallback: parse trailing digits as mass number
                    import re
                    m = re.search(r"(\d+)$", name)
                    return float(m.group(1)) if m else 1.0

                # weight fractions
                rho_i   = [(n, d * _mass(n)) for n, d, _ in fracs]
                rho_tot = sum(r for _, r in rho_i) or 1.0
                wt_map  = {n: r / rho_tot for n, r in rho_i}

                for nuc, dens, at_fr in fracs:
                    print(f"    │   {nuc:<10}  {dens:>14.4e}  {at_fr:>10.4f}  {wt_map[nuc]:>10.4f}")

                print(f"    │   {'─'*10}  {'─'*14}  {'─'*10}  {'─'*10}")
                print(f"    │   {'TOTAL':<10}  {N_tot:>14.4e}  {'1.0000':>10}  {'1.0000':>10}")

            print(f"    └{'─' * (W - 6)}")

    print()
    print("=" * W)


# ─────────────────────────────────────────────────────────────────────────────
# High-level plotter object
# ─────────────────────────────────────────────────────────────────────────────

class GeometryPlotter:
    """
    Convenience wrapper around all plot functions.

    Usage
    -----
    >>> plotter = GeometryPlotter(geom)
    >>> plotter.draw()
    >>> plotter.draw_temperature_profile()
    >>> plotter.summary()
    >>> plotter.save("my_geometry.png", dpi=200)
    """

    def __init__(self, geometry):
        self.geometry = geometry
        self._last_fig: matplotlib.figure.Figure | None = None

    # ── main geometry view ───────────────────────────────────────────────────

    def draw(self, **kwargs) -> tuple:
        fig, axes = draw(self.geometry, **kwargs)
        self._last_fig = fig
        plt.show()
        return fig, axes

    # ── temperature profile ──────────────────────────────────────────────────

    def draw_temperature_profile(self, **kwargs) -> tuple:
        fig, ax = draw_temperature_profile(self.geometry, **kwargs)
        self._last_fig = fig
        plt.show()
        return fig, ax

    # ── combined dashboard ───────────────────────────────────────────────────

    def dashboard(
        self,
        figsize: tuple[float, float] = (15, 7),
        **draw_kwargs,
    ) -> tuple[matplotlib.figure.Figure, dict]:
        """
        Two-panel dashboard:  geometry view (top) + temperature profile (bottom).
        """
        fig = plt.figure(figsize=figsize, facecolor="white")
        gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.38,
                                height_ratios=[3, 2])

        ax_top = fig.add_subplot(gs[0])
        ax_bot = fig.add_subplot(gs[1])

        # geometry panel
        draw(
            self.geometry,
            ax_mat=ax_top,
            show_temp_strip=False,
            show_temp_colorbar=False,
            **{k: v for k, v in draw_kwargs.items()
               if k not in ("figsize",)},
        )

        # temperature profile panel
        sr = _sorted_regions(self.geometry)
        xs, ys = [], []
        for r in sr:
            xs += [r.x_min, r.x_max]
            ys += [r.material.T, r.material.T]

        for r in sr:
            cls = classify_region(r)
            col = _CLASS_PALETTE.get(cls, _CLASS_PALETTE["other"])
            ax_bot.axvspan(r.x_min, r.x_max, alpha=0.10, color=col, linewidth=0)

        ax_bot.plot(xs, ys, color="#1a1a1a", linewidth=1.5,
                    drawstyle="steps-post", zorder=3)
        ax_bot.fill_between(xs, ys, alpha=0.12, color="#555555",
                            step="post", zorder=2)

        x0, x1 = sr[0].x_min, sr[-1].x_max
        span = x1 - x0
        unique_T = sorted(set(ys))
        for T in unique_T:
            ax_bot.axhline(T, color="#cccccc", linewidth=0.5, linestyle=":", zorder=1)
        for T in unique_T:
            ax_bot.text(x1 + 0.005 * span, T, f"{T:.0f} K",
                        va="center", ha="left", fontsize=7, color="#555555")

        ax_bot.set_xlim(x0 - 0.01 * span, x1 + 0.07 * span)
        ax_bot.set_xlabel("x  [cm]", fontsize=10)
        ax_bot.set_ylabel("T  [K]", fontsize=10)
        ax_bot.set_title("Temperature profile", fontsize=10, pad=5)
        ax_bot.spines[["top", "right"]].set_visible(False)
        ax_bot.tick_params(labelsize=8)

        self._last_fig = fig
        plt.show()
        return fig, {"top": ax_top, "bottom": ax_bot}

    # ── save last figure ─────────────────────────────────────────────────────

    def save(self, path: str, dpi: int = 150, **kwargs) -> None:
        """Save the most recently produced figure."""
        if self._last_fig is None:
            warnings.warn("[GeometryPlotter] No figure to save — call draw() first.")
            return
        self._last_fig.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)
        print(f"[GeometryPlotter] Saved → {path}")

    # ── text summary ─────────────────────────────────────────────────────────

    def summary(self) -> None:
        print_summary(self.geometry)

    def __repr__(self) -> str:
        n = len(getattr(self.geometry, "_regions", []))
        return f"<GeometryPlotter  {n} regions>"


# ─────────────────────────────────────────────────────────────────────────────
# Monkey-patch helpers  (optional — call once at startup)
# ─────────────────────────────────────────────────────────────────────────────

def attach_to_geometry_class(GeometryClass) -> None:
    """
    Attach all plot methods directly to a Geometry class so they can be called
    as instance methods:

        geom.draw()
        geom.draw_temperature_profile()
        geom.dashboard()
        geom.plot_summary()

    Parameters
    ----------
    GeometryClass : the class object (not an instance) to patch
    """
    GeometryClass.draw = lambda self, **kw: draw(self, **kw)
    GeometryClass.draw_temperature_profile = (
        lambda self, **kw: draw_temperature_profile(self, **kw)
    )
    GeometryClass.dashboard = (
        lambda self, **kw: GeometryPlotter(self).dashboard(**kw)
    )
    GeometryClass.plot_summary = lambda self: print_summary(self)
    print(f"[geometry_plot] Attached draw / draw_temperature_profile / "
          f"dashboard / plot_summary to {GeometryClass.__name__}.")