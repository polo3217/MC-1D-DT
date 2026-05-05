import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize, to_rgba
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec


"""
exemple usage in a Jupyter notebook:
# 1 — standalone functions
gp.draw(geometry)
gp.draw_temperature_profile(geometry)
gp.print_summary(geometry)

# 2 — plotter object (tracks last figure for saving)
p = gp.GeometryPlotter(geometry)
p.draw()
p.draw_temperature_profile()
p.dashboard()          # geometry + T-profile stacked
p.summary()
p.save("out.png", dpi=200)
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_region(region) -> str:
    """Return a coarse material class label from the region/material name."""
    name = region.material.name.lower()
    if "fuel" in name or "u235" in name or "u238" in name or "uo2" in name:
        return "fuel"
    if "water" in name or "h2o" in name or "h1" in name:
        return "water"
    if "clad" in name or "zr" in name:
        return "clad"
    # fallback: use first word of material name
    return name.split("_")[0]


_CLASS_COLOURS = {
    "fuel":  "#e07b39",   # warm orange
    "water": "#4a90d9",   # cool blue
    "clad":  "#7f8c8d",   # grey
}
_CLASS_COLOURS_DEFAULT = "#aaaaaa"


# ---------------------------------------------------------------------------
# Main draw function
# ---------------------------------------------------------------------------

def draw(
    self,
    figsize: tuple = (14, 5),
    show_material_legend: bool = True,
    temp_cmap: str = "plasma",
    label_threshold_cm: float = 0.15,   # min region width [cm] to attempt any label
    annotate_boundaries: bool = True,
    title: str = "Geometry cross-section",
):
    """
    Draw the 1-D geometry in a way that remains legible for dense geometries
    (e.g. 139-region PWR benchmark).

    Layout
    ------
    Row 0 (thin)  : temperature colour-map strip + colorbar
    Row 1 (thick) : material-class colour bands + boundary condition markers
    Row 2 (thin)  : x-axis tick ruler

    Labels are only printed when the region is wide enough to hold them
    without overlap.  For narrow geometries the legend + colorbar carry all
    the information.
    """
    if not self._regions:
        print("[Geometry] No regions to draw.")
        return None, None

    sorted_regions = sorted(self._regions, key=lambda r: r.x_min)
    x0   = sorted_regions[0].x_min
    x1   = sorted_regions[-1].x_max
    span = x1 - x0

    # ── temperature range ────────────────────────────────────────────────────
    all_temps = [r.material.T for r in sorted_regions]
    T_min, T_max = min(all_temps), max(all_temps)
    # guard against uniform temperature
    if T_min == T_max:
        T_min, T_max = T_min - 1, T_max + 1
    t_norm = Normalize(vmin=T_min, vmax=T_max)
    t_cmap = plt.get_cmap(temp_cmap)

    # ── figure / axes ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(
        2, 2,
        height_ratios=[1, 2.5],
        width_ratios=[30, 1],
        hspace=0.08,
        wspace=0.03,
    )

    ax_temp   = fig.add_subplot(gs[0, 0])   # temperature strip
    ax_cbar   = fig.add_subplot(gs[:, 1])   # colorbar
    ax_mat    = fig.add_subplot(gs[1, 0])   # material-class strip

    # share x limits
    for ax in (ax_temp, ax_mat):
        ax.set_xlim(x0 - 0.02 * span, x1 + 0.02 * span)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        for spine in ("top", "left", "right"):
            ax.spines[spine].set_visible(False)

    ax_temp.set_xticks([])
    ax_temp.spines["bottom"].set_visible(False)
    ax_mat.set_xlabel("x  [cm]", fontsize=10)

    # ── titles / labels ──────────────────────────────────────────────────────
    ax_temp.set_ylabel("T  [K]", fontsize=8, labelpad=2, rotation=0, ha="right", va="center")
    ax_mat.set_ylabel("Material", fontsize=8, labelpad=2, rotation=0, ha="right", va="center")
    fig.suptitle(title, fontsize=12, y=1.01)

    # ── draw regions ─────────────────────────────────────────────────────────
    seen_classes = {}   # class_label -> colour  (for legend)

    for r in sorted_regions:
        w   = r.x_max - r.x_min
        cx  = r.x_min + w / 2
        cls = _classify_region(r)
        cls_colour = _CLASS_COLOURS.get(cls, _CLASS_COLOURS_DEFAULT)
        seen_classes[cls] = cls_colour

        # — temperature strip —
        t_colour = t_cmap(t_norm(r.material.T))
        ax_temp.barh(
            0.5, w, left=r.x_min, height=0.72,
            color=t_colour, edgecolor="none",
        )

        # — material strip —
        rect = mpatches.FancyBboxPatch(
            (r.x_min, 0.08), w, 0.84,
            boxstyle="square,pad=0",
            facecolor=cls_colour,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.6,
        )
        ax_mat.add_patch(rect)

        # — optional text label (only if region is wide enough) —
        if w >= label_threshold_cm:
            # material-strip label: show temperature
            ax_mat.text(
                cx, 0.52,
                f"{r.material.T:.0f} K",
                ha="center", va="center",
                fontsize=max(6, min(8, w * 6)),
                color="white", fontweight="bold",
            )

        # — boundary lines on material strip —
        for xb in (r.x_min, r.x_max):
            ax_mat.axvline(xb, color="black", linewidth=0.4,
                           linestyle="--", alpha=0.25)

    # ── boundary-condition markers ───────────────────────────────────────────
    if annotate_boundaries and hasattr(self, "boundary_conditions"):
        _bc_symbol = {
            "vacuum":     ("✕  vacuum",    "#c0392b"),
            "reflective": ("↔  reflective", "#2980b9"),
        }
        for side, xpos, ha, xoff_sign in (
            ("left",  x0,  "right", -1),
            ("right", x1, "left",  +1),
        ):
            bc = self.boundary_conditions.get(side, "vacuum")
            sym, col = _bc_symbol.get(bc, (bc, "black"))
            for ax in (ax_temp, ax_mat):
                ax.axvline(xpos, color=col, linewidth=1.8, linestyle="-", alpha=0.9)
            ax_mat.text(
                xpos + xoff_sign * 0.008 * span, 0.50,
                sym,
                ha=ha, va="center", fontsize=8,
                color=col, fontweight="bold",
            )

    # ── colorbar (temperature) ───────────────────────────────────────────────
    sm = ScalarMappable(cmap=t_cmap, norm=t_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_label("T  [K]", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # ── material-class legend ────────────────────────────────────────────────
    if show_material_legend and seen_classes:
        handles = [
            mpatches.Patch(
                facecolor=col, edgecolor="black",
                linewidth=0.6, alpha=0.7, label=cls.capitalize()
            )
            for cls, col in sorted(seen_classes.items())
        ]
        ax_mat.legend(
            handles=handles,
            loc="upper right",
            fontsize=8,
            framealpha=0.8,
            title="Material class",
            title_fontsize=8,
        )

    # ── x-axis: ticks at rod boundaries ─────────────────────────────────────
    # Show ticks only at fuel-plate edges to avoid clutter
    boundary_xs = sorted({r.x_min for r in sorted_regions} | {x1})
    # subsample: keep every Nth boundary so labels don't overlap
    min_tick_spacing = span / 25          # at most ~25 ticks
    kept, last = [], -np.inf
    for xb in boundary_xs:
        if xb - last >= min_tick_spacing:
            kept.append(xb)
            last = xb
    ax_mat.set_xticks(kept)
    ax_mat.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax_mat.tick_params(axis="x", labelsize=8, rotation=45)

    fig.tight_layout()
    plt.show()
    return fig, ax_mat