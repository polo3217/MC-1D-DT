import sys
import os
sys.path.append('/home/paule/open_mc_projects/windowed_multipole/02_working_notebook_vectfit')

from dataclasses import dataclass, field
import time
from datetime import datetime
from pathlib import Path

from typing import List, Optional
import numpy as np
import majorant_multipole as maj
import openmc


# ==========================================
# [NEW] --- Tally class ---
# ==========================================
class Tally:
    """
    Scalar Monte Carlo tally with inter-history variance estimation.

    Usage pattern (must follow this order per history):
        tally.score(x)          # call once or more during a history
        tally.end_history()     # commit history total; resets buffer

    After B histories the standard properties give:
        mean            = (1/B) Σ x_i
        variance        = sample variance of the mean  = s²/B
        relative_error  = sqrt(variance) / |mean|      (FOM in MC)

    Why inter-history variance?
    ----------------------------
    Scoring every track segment directly into _sum gives the wrong
    uncertainty because it treats each segment as an independent sample.
    The correct independent samples are the per-history totals x_i.
    Committing via end_history() ensures that structure.

    Attributes
    ----------
    _buffer  : running sum within the current history (reset each history)
    _sum     : Σ x_i  across all committed histories
    _sum_sq  : Σ x_i² across all committed histories
    _n       : number of committed histories
    """

    def __init__(self):
        self._buffer:  float = 0.0   # accumulator within current history
        self._sum:     float = 0.0   # Σ x_i
        self._sum_sq:  float = 0.0   # Σ x_i²
        self._n:       int   = 0     # number of histories committed

    # ── scoring ───────────────────────────────────────────────────────────────
    def score(self, x: float):
        """Add x to the current-history buffer (does NOT commit)."""
        self._buffer += x

    def end_history(self):
        """
        Commit the current-history buffer and reset it.
        Must be called exactly once per history, even if score() was never
        called (contributes 0 to sum and sum_sq, which is correct).
        """
        x             = self._buffer
        self._sum    += x
        self._sum_sq += x * x
        self._n      += 1
        self._buffer  = 0.0          # reset for next history

    def reset(self):
        """Full reset — use between batches or at geometry.reset()."""
        self._buffer  = 0.0
        self._sum     = 0.0
        self._sum_sq  = 0.0
        self._n       = 0

    # ── derived statistics ────────────────────────────────────────────────────
    @property
    def mean(self) -> float:
        """Per-history mean."""
        return self._sum / self._n if self._n > 0 else 0.0

    @property
    def variance(self) -> float:
        """
        Sample variance of the mean  s²(x̄) = [Σx² - n·x̄²] / [n(n-1)].
        Returns 0 when fewer than 2 histories have been committed.
        """
        if self._n < 2:
            return 0.0
        return (self._sum_sq - self._n * self.mean ** 2) / (self._n * (self._n - 1))

    @property
    def std(self) -> float:
        """Standard deviation of the mean (= sqrt of variance)."""
        return float(np.sqrt(max(self.variance, 0.0)))

    @property
    def relative_error(self) -> float:
        """
        σ(x̄) / |x̄| — the standard figure-of-merit in Monte Carlo.
        Returns inf when mean is zero.
        """
        m = self.mean
        return self.std / abs(m) if m != 0.0 else float("inf")

    # ── snapshot for batch collection ─────────────────────────────────────────
    def snapshot(self) -> dict:
        """Return {mean, std, relative_error} for external bookkeeping."""
        return {
            "mean"           : self.mean,
            "std"            : self.std,
            "relative_error" : self.relative_error,
        }


# ==========================================
# [NEW] --- TallyArray class ---
# ==========================================
class TallyArray:
    """
    Vectorised Monte Carlo tally for multi-dimensional scored quantities
    (e.g. space × energy arrays), using the same inter-history variance
    method as Tally but operating on NumPy arrays for performance.

    The history buffer and sum/sum_sq arrays all share the same shape.

    Usage pattern (identical to Tally):
        tally_arr.score(i, j, x)   # score value x into bin (i,j)
        tally_arr.end_history()    # commit entire buffer array; resets buffer

    Properties return NumPy arrays of the same shape as the tally bins.
    """

    def __init__(self, shape: tuple):
        """
        Parameters
        ----------
        shape : tuple  — e.g. (n_space, n_energy) or (n_energy,)
        """
        self._shape   = shape
        self._buffer  = np.zeros(shape)   # within-history accumulator
        self._sum     = np.zeros(shape)   # Σ x_i  (element-wise)
        self._sum_sq  = np.zeros(shape)   # Σ x_i² (element-wise)
        self._n       = 0                 # histories committed

    # ── scoring ───────────────────────────────────────────────────────────────
    def score(self, idx, x: float):
        """
        Add x to bin idx of the current-history buffer.
        idx can be a scalar index, a tuple of indices, or a slice.
        """
        self._buffer[idx] += x

    def end_history(self):
        """Commit buffer → sum/sum_sq, then zero the buffer."""
        self._sum    += self._buffer
        self._sum_sq += self._buffer ** 2
        self._n      += 1
        self._buffer[:] = 0.0           # reset; [:] preserves array identity

    def reset(self):
        """Full reset — use between batches or at geometry.reset()."""
        self._buffer[:] = 0.0
        self._sum[:]    = 0.0
        self._sum_sq[:] = 0.0
        self._n         = 0

    # ── derived statistics ────────────────────────────────────────────────────
    @property
    def mean(self) -> np.ndarray:
        """Per-history mean array."""
        return self._sum / self._n if self._n > 0 else np.zeros(self._shape)

    @property
    def variance(self) -> np.ndarray:
        """
        Element-wise sample variance of the mean.
        Returns zeros when fewer than 2 histories have been committed.
        """
        if self._n < 2:
            return np.zeros(self._shape)
        return (self._sum_sq - self._n * self.mean ** 2) / (self._n * (self._n - 1))

    @property
    def std(self) -> np.ndarray:
        """Element-wise standard deviation of the mean."""
        return np.sqrt(np.maximum(self.variance, 0.0))

    @property
    def relative_error(self) -> np.ndarray:
        """
        Element-wise σ(x̄) / |x̄|.
        Bins where mean == 0 get inf.
        """
        m = self.mean
        with np.errstate(divide="ignore", invalid="ignore"):
            re = np.where(m != 0.0, self.std / np.abs(m), np.inf)
        return re

    # ── snapshot for batch collection ─────────────────────────────────────────
    def snapshot(self) -> dict:
        """Return {mean, std, relative_error} as plain lists for serialisation."""
        return {
            "mean"           : self.mean.tolist(),
            "std"            : self.std.tolist(),
            "relative_error" : self.relative_error.tolist(),
        }


# ==========================================
# --- FluxTallyTLE class ---
# ==========================================
class FluxTallyTLE:
    """
    Track-length estimator (TLE) of scalar flux per energy group.

    The segment length is scored at the energy of the particle for that step.
    This matches OpenMC's internal TLE exactly.

    Every sampled distance is scored — real collisions, virtual collisions,
    and leakage steps alike — because the neutron physically travels through
    the material regardless of what type of event occurs at the end of the step.

    Result in group g:

        T_{g} = (1 / N) · Σ_k  l_k

    where
        N    : number of source histories
        l_k  : segment length [cm]

    Units : cm · source-neutron⁻¹

    Parameters
    ----------
    boundaries      : spatial bin edges along x [cm]
    energy_bins     : energy bin edges [eV]  (monotonically increasing)
    transverse_area : y-z cross-sectional area [cm²], default 1.0

    --- CHANGES FROM ORIGINAL ---
    [CHANGE] _flux (np.zeros array) replaced by a TallyArray of shape
             (n_energy,).  This gives correct inter-history variance via
             TallyArray.end_history().
    [CHANGE] score() now calls _flux_tally.score() instead of += on _flux.
    [CHANGE] end_history() now delegates to _flux_tally.end_history().
    [CHANGE] mean property now delegates to _flux_tally.mean (same values
             as before for the mean; additionally exposes .std and .relative_error).
    [NEW]    variance, std, relative_error properties forwarded from TallyArray.
    [NEW]    snapshot() method for batch bookkeeping.
    [CHANGE] _n_histories kept for backward compatibility with export code
             that reads ft._n_histories directly; it mirrors _flux_tally._n.
    """

    def __init__(self, boundaries: List[float], energy_bins: List[float],
                 transverse_area: float = 1.0):
        self.boundaries      = np.asarray(boundaries, dtype=float)
        self.energy_bins     = np.asarray(energy_bins, dtype=float)
        self.transverse_area = transverse_area

        n_energy = len(energy_bins) - 1

        self._volumes = (self.boundaries[-1] - self.boundaries[0]) * transverse_area
        print(self._volumes)

        # [CHANGE] Replace plain np.zeros with a TallyArray so that
        # variance can be computed correctly across histories.
        self._flux_tally = TallyArray(shape=(n_energy,))

        # [CHANGE] _n_histories is now a property (see below) for backward
        # compatibility; it is no longer stored as a plain int.

    # ------------------------------------------------------------------
    # [NEW] backward-compatible property: code that reads ft._n_histories
    # still works without modification.
    @property
    def _n_histories(self) -> int:
        return self._flux_tally._n

    # [NEW] backward-compatible property: export_simulation reads ft._flux
    # directly; expose the current mean array under that name.
    @property
    def _flux(self) -> np.ndarray:
        return self._flux_tally.mean

    # ------------------------------------------------------------------
    def _energy_bin(self, energy: float) -> int:
        ei = int(np.searchsorted(self.energy_bins, energy, side='right')) - 1
        if ei < 0 or ei >= self._flux_tally._shape[0]:
            return -1
        return ei

    # ------------------------------------------------------------------
    def score(self, energy: float, total_distance: float):
        """
        Score the neutron track segment.

        Parameters
        ----------
        energy         : neutron energy during the step [eV]
        total_distance : actual 3-D path length of the step [cm]

        [CHANGE] Now calls _flux_tally.score(ei, total_distance) instead
        of incrementing _flux[ei] directly.  Behaviour is identical for
        the mean; additionally accumulates sum_sq for variance.
        """
        ei = self._energy_bin(energy)
        if ei < 0:
            return
        # [CHANGE] delegate to TallyArray (was: self._flux[ei] += total_distance)
        self._flux_tally.score(ei, total_distance)

    # ------------------------------------------------------------------
    def end_history(self):
        # [CHANGE] delegate to TallyArray (was: self._n_histories += 1)
        self._flux_tally.end_history()

    # ------------------------------------------------------------------
    @property
    def mean(self) -> np.ndarray:
        # [CHANGE] delegate to TallyArray (was: self._flux / n)
        return self._flux_tally.mean

    # [NEW] variance, std, relative_error properties
    @property
    def variance(self) -> np.ndarray:
        return self._flux_tally.variance

    @property
    def std(self) -> np.ndarray:
        return self._flux_tally.std

    @property
    def relative_error(self) -> np.ndarray:
        return self._flux_tally.relative_error

    # ------------------------------------------------------------------
    # [NEW] snapshot() — returns mean ± std per energy bin as plain lists
    # for storage in geometry.batch_results.
    def snapshot(self) -> dict:
        """
        Return a dict suitable for batch bookkeeping.
        Keys: mean, std, relative_error  (each a list of length n_energy).
        """
        return {
            "energy_bins" : self.energy_bins.tolist(),
            "boundaries"  : self.boundaries.tolist(),   # ← add this
            "surface_xs"  : self.surface_xs.tolist(),
            "flux"           : self._flux_tally.snapshot(),
        }

    # ------------------------------------------------------------------
    def summary(self) -> str:
        lines = [
            "  FLUX TALLY (track-length estimator)",
            f"    Energy groups : {self._flux_tally._shape}",
            f"    Histories     : {self._n_histories}",
            f"    Bin volumes   : {np.round(self._volumes, 4).tolist()} cm³",
            "    Mean track length per source neutron [cm · src-n⁻¹]:",
            str(np.round(self.mean, 6)),
            # [NEW] also print uncertainty
            "    Std (of mean) [cm · src-n⁻¹]:",
            str(np.round(self.std, 6)),
            "    Relative error:",
            str(np.round(self.relative_error, 4)),
        ]
        return "\n".join(lines)


# ==========================================
# --- VerificationTally class ---
# ==========================================
class VerificationTally:
    """
    Verification tallies matching OpenMC's scoring definitions exactly.

    All results normalised per source neutron.

    Tallies
    -------
    1. Reaction rates  : absorption, scatter per cell × energy group
    2. Current         : forward and backward partial current at each surface
    3. Leakage         : left and right separately
    4. Spectrum        : fine-bin flux per cell (same TLE as FluxTallyTLE)

    Parameters
    ----------
    boundaries      : spatial bin edges along x [cm]
    energy_bins     : coarse energy bin edges [eV]
    fine_energy_bins: fine energy bin edges [eV]  for spectrum tally
    surface_xs      : list of x positions [cm] at which to tally current

    --- CHANGES FROM ORIGINAL ---
    [CHANGE] _absorption, _scatter replaced by TallyArray (shape n_space×n_energy).
    [CHANGE] _current_fwd, _current_bwd replaced by TallyArray (shape n_surf,).
    [CHANGE] _leak_left, _leak_right replaced by Tally objects.
    [CHANGE] score_collision, score_surface_crossing, score_leakage now call
             .score() on the appropriate TallyArray/Tally instead of += on arrays.
    [CHANGE] end_history() delegates to all TallyArray/Tally objects.
    [CHANGE] absorption, scatter, current_fwd, current_bwd, etc. properties now
             delegate to TallyArray.mean / Tally.mean (same numerical values;
             additionally expose _std and _re counterparts).
    [NEW]    absorption_std, scatter_std, current_fwd_std, etc. properties.
    [NEW]    snapshot() for batch bookkeeping.
    [CHANGE] n property unchanged but now reads from _leak_left._n.
    """

    def __init__(self, boundaries: List[float],
                 energy_bins: List[float],
                 surface_xs: List[float]):
        self.boundaries       = np.asarray(boundaries, dtype=float)
        self.energy_bins      = np.asarray(energy_bins, dtype=float)
        self.surface_xs       = np.asarray(surface_xs, dtype=float)

        n_space  = len(boundaries) - 1
        n_energy = len(energy_bins) - 1
        n_surf   = len(surface_xs)

        # ── 1. reaction rates ─────────────────────────────────────────────────
        # [CHANGE] was: np.zeros((n_space, n_energy)); now TallyArray
        self._absorption = TallyArray(shape=(n_space, n_energy))
        self._scatter    = TallyArray(shape=(n_space, n_energy))

        # ── 2. surface currents ───────────────────────────────────────────────
        # [CHANGE] was: np.zeros(n_surf); now TallyArray
        self._current_fwd = TallyArray(shape=(n_surf,))
        self._current_bwd = TallyArray(shape=(n_surf,))

        # ── 3. leakage ────────────────────────────────────────────────────────
        # [CHANGE] was: plain floats 0.0; now scalar Tally objects
        self._leak_left  = Tally()
        self._leak_right = Tally()
        self._total_leak = Tally()  

        # _n_histories no longer stored as a plain int; see property below.

    # ------------------------------------------------------------------
    # [NEW] backward-compatible _n_histories via the leak_left Tally counter.
    # Any Tally/_TallyArray._n would give the same value after end_history().
    @property
    def _n_histories(self) -> int:
        return self._leak_left._n

    # ------------------------------------------------------------------
    def _spatial_bin(self, x: float) -> int:
        si = int(np.searchsorted(self.boundaries, x, side='right')) - 1
        if si < 0 or si >= len(self.boundaries) - 1:
            return -1
        return si

    def _energy_bin(self, energy: float, bins: np.ndarray) -> int:
        ei = int(np.searchsorted(bins, energy, side='right')) - 1
        if ei < 0 or ei >= len(bins) - 1:
            return -1
        return ei

    # ------------------------------------------------------------------
    def score_collision(self, x: float, energy: float, event: str):
        si = self._spatial_bin(x)
        ei = self._energy_bin(energy, self.energy_bins)
        if si < 0 or ei < 0:
            return
        if event == 'absorption':
            # [CHANGE] was: self._absorption[si, ei] += 1.0
            self._absorption.score((si, ei), 1.0)
        elif event == 'scatter':
            # [CHANGE] was: self._scatter[si, ei] += 1.0
            self._scatter.score((si, ei), 1.0)

    # ------------------------------------------------------------------
    def score_surface_crossing(self, x_start: float, x_end: float):
        for si, sx in enumerate(self.surface_xs):
            if x_start < sx <= x_end:
                # [CHANGE] was: self._current_fwd[si] += 1.0
                self._current_fwd.score(si, 1.0)
            elif x_end < sx <= x_start:
                # [CHANGE] was: self._current_bwd[si] += 1.0
                self._current_bwd.score(si, 1.0)

    # ------------------------------------------------------------------
    def score_leakage(self, x_end: float):
        if x_end <= self.boundaries[0]:
            # [CHANGE] was: self._leak_left += 1.0
            self._leak_left.score(1.0)
        elif x_end >= self.boundaries[-1]:
            # [CHANGE] was: self._leak_right += 1.0
            self._leak_right.score(1.0)

    # ------------------------------------------------------------------
    def end_history(self):
        # [CHANGE] was: self._n_histories += 1
        # Now commits all TallyArrays and scalar Tallies simultaneously.
        self._absorption.end_history()
        self._scatter.end_history()
        self._current_fwd.end_history()
        self._current_bwd.end_history()
        self._leak_left.end_history()
        self._leak_right.end_history()
        self._total_leak.end_history()
    # ------------------------------------------------------------------
    @property
    def n(self) -> int:
        # [CHANGE] was: max(self._n_histories, 1)
        # Now reads from the Tally counter; falls back to 1 to avoid /0.
        return max(self._n_histories, 1)

    # ── mean properties (identical API to original) ───────────────────────────
    @property
    def absorption(self) -> np.ndarray:
        # [CHANGE] was: self._absorption / self.n
        return self._absorption.mean

    @property
    def scatter(self) -> np.ndarray:
        # [CHANGE] was: self._scatter / self.n
        return self._scatter.mean

    @property
    def total_rxn(self) -> np.ndarray:
        return self.absorption + self.scatter

    @property
    def current_fwd(self) -> np.ndarray:
        # [CHANGE] was: self._current_fwd / self.n
        return self._current_fwd.mean

    @property
    def current_bwd(self) -> np.ndarray:
        # [CHANGE] was: self._current_bwd / self.n
        return self._current_bwd.mean

    @property
    def current_net(self) -> np.ndarray:
        return self.current_fwd - self.current_bwd

    @property
    def leak_left(self) -> float:
        # [CHANGE] was: self._leak_left / self.n
        return self._leak_left.mean

    @property
    def leak_right(self) -> float:
        # [CHANGE] was: self._leak_right / self.n
        return self._leak_right.mean

    @property
    def leak_total(self) -> float:
        return self.leak_left + self.leak_right

    # ── [NEW] std properties ──────────────────────────────────────────────────
    @property
    def absorption_std(self) -> np.ndarray:
        return self._absorption.std

    @property
    def scatter_std(self) -> np.ndarray:
        return self._scatter.std

    @property
    def current_fwd_std(self) -> np.ndarray:
        return self._current_fwd.std

    @property
    def current_bwd_std(self) -> np.ndarray:
        return self._current_bwd.std

    @property
    def leak_left_std(self) -> float:
        return self._leak_left.std

    @property
    def leak_right_std(self) -> float:
        return self._leak_right.std

    # ── [NEW] relative error properties ──────────────────────────────────────
    @property
    def absorption_re(self) -> np.ndarray:
        return self._absorption.relative_error

    @property
    def scatter_re(self) -> np.ndarray:
        return self._scatter.relative_error

    @property
    def leak_left_re(self) -> float:
        return self._leak_left.relative_error

    @property
    def leak_right_re(self) -> float:
        return self._leak_right.relative_error

    # ------------------------------------------------------------------
    # [NEW] snapshot() — capture all mean ± std arrays for batch storage.
    def snapshot(self) -> dict:
        """
        Return a dict of all tally results for one batch.
        Suitable for storage in geometry.batch_results.
        """
        return {
            "boundaries"  : self.boundaries.tolist(),   # ← add this
            "surface_xs"  : self.surface_xs.tolist(),
            "energy_bins"    : self.energy_bins.tolist(),
            "surface_xs"     : self.surface_xs.tolist(),
            "absorption"     : self._absorption.snapshot(),
            "scatter"        : self._scatter.snapshot(),
            "current_fwd"    : self._current_fwd.snapshot(),
            "current_bwd"    : self._current_bwd.snapshot(),
            "leak_left"      : self._leak_left.snapshot(),
            "leak_right"     : self._leak_right.snapshot(),
        }

    # ------------------------------------------------------------------
    def summary(self) -> str:
        eb   = self.energy_bins
        sb   = self.boundaries
        sxs  = self.surface_xs
        n_g  = len(eb) - 1
        n_s  = len(sb) - 1

        e_labels = [f"{eb[i]:.0f}–{eb[i+1]:.0f} eV" for i in range(n_g)]
        s_labels = [f"{sb[i]:.1f}–{sb[i+1]:.1f} cm" for i in range(n_s)]

        def fmt_row(arr, ri):
            return "  ".join(f"{arr[ri, ei]:>12.4e}" for ei in range(n_g))

        # [NEW] also format std row
        def fmt_row_std(arr_std, ri):
            return "  ".join(f"{arr_std[ri, ei]:>12.4e}" for ei in range(n_g))

        lines = ["=" * 60, "  VERIFICATION TALLIES", "=" * 60]

        lines += ["", "  ABSORPTION RATE [reactions · src-n⁻¹]",
                  f"  {'Region':<20}" + "  ".join(f"{l:>12}" for l in e_labels)]
        for ri, sl in enumerate(s_labels):
            lines.append(f"  {sl:<20}  {fmt_row(self.absorption, ri)}")
        # [NEW] print std below mean block
        lines.append(f"  {'  ± std':<20}" +
                     "  ".join(f"  {'':>12}" for _ in e_labels))
        for ri, sl in enumerate(s_labels):
            lines.append(f"  {sl+' (±std)':<20}  {fmt_row_std(self.absorption_std, ri)}")

        lines += ["", "  SCATTER RATE [reactions · src-n⁻¹]",
                  f"  {'Region':<20}" + "  ".join(f"{l:>12}" for l in e_labels)]
        for ri, sl in enumerate(s_labels):
            lines.append(f"  {sl:<20}  {fmt_row(self.scatter, ri)}")
        for ri, sl in enumerate(s_labels):
            lines.append(f"  {sl+' (±std)':<20}  {fmt_row_std(self.scatter_std, ri)}")

        lines += ["", "  TOTAL REACTION RATE [reactions · src-n⁻¹]",
                  f"  {'Region':<20}" + "  ".join(f"{l:>12}" for l in e_labels)]
        for ri, sl in enumerate(s_labels):
            lines.append(f"  {sl:<20}  {fmt_row(self.total_rxn, ri)}")

        lines += ["", "  SURFACE CURRENTS [particles · src-n⁻¹]",
                  f"  {'Surface x (cm)':<20} {'Forward':>12} {'Backward':>12} {'Net':>12}"]
        for si, sx in enumerate(sxs):
            lines.append(f"  {sx:<20.2f} {self.current_fwd[si]:>12.4e} "
                         f"{self.current_bwd[si]:>12.4e} {self.current_net[si]:>12.4e}")
        # [NEW] print current std
        for si, sx in enumerate(sxs):
            lines.append(f"  {sx:<20.2f} ±{self.current_fwd_std[si]:>11.4e} "
                         f"±{self.current_bwd_std[si]:>11.4e}")

        lines += ["", "  LEAKAGE [particles · src-n⁻¹]",
                  f"  Left  (x={sb[0]:.1f}): {self.leak_left:.4e} ± {self.leak_left_std:.4e}",
                  f"  Right (x={sb[-1]:.1f}): {self.leak_right:.4e} ± {self.leak_right_std:.4e}",
                  f"  Total           : {self.leak_total:.4e}  ({self.leak_total*100:.2f}%)"]

        lines.append("=" * 60)

        # sanitty check 
        # leakage + absorption equal = 1
        lines += ["", "  SANITY CHECK: LEAKAGE + ABSORPTION ≈ 1.0",
                  f"  Leakage + Absorption = {self.leak_total + self.absorption.sum():.4e}"]
        return "\n".join(lines)