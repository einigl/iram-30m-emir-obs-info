import os
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from ...graphics import latex_line, Settings

from . import iram30m_emir

Settings.only_rotational = True

plt.rc("text", usetex=True)

def _load_noise_profile(
    obstime: float,
    ipwv: int
):
    """
    Obstime in seconds
    """
    f_array = np.load(
        os.path.join(os.path.dirname(__file__), "frequencies.npy")
    )
    n_array = np.load(
        os.path.join(os.path.dirname(__file__), f"rms_profile_ipwv_{ipwv}mm.npy")
    )

    n_array = n_array / np.sqrt(obstime / 60)

    return f_array, n_array

def plot_all_bands(
    freqs: np.ndarray,
    obstime: Optional[float],
    ipwv: int
):
    rms_min = 0 # mK
    rms_max = 800 # mK
    if obstime is not None:
        rms_max /= (obstime / 60)**2
    rms_delta = rms_max - rms_min

    for f in freqs:
        plt.axvline(f, 0.05, 0.95, linewidth=0.8)
        
    bands = iram30m_emir.IRAM30mEMIR.bands()

    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    lowest, highest = -float("inf"), float("inf")
    for name, (low, upp) in bands.items():
        lowest, highest = min(lowest, low), max(highest, upp)
        plt.plot([low, upp], 2*[rms_min+0.9*rms_delta], color="red")
        plt.text(
            (low+upp)/2, rms_min+0.80*rms_delta, name,
            horizontalalignment='center',
            verticalalignment='center',
            color="red", weight='bold', bbox=props
        )

    f_array, n_array = _load_noise_profile(obstime or 60, ipwv)
    mask = (lowest <= f_array) & (f_array <= highest)

    if obstime is not None:
        plt.plot(f_array[mask], 1e3*n_array[mask], color="tab:gray")
        plt.ylabel("Noise RMS (mK)", labelpad=10)

    plt.xlabel("$f$ (GHz)")
    plt.ylim([rms_min, rms_max])
    if obstime is None:
        plt.yticks([])
    plt.gca().xaxis.set_minor_locator(MultipleLocator(5))


def plot_specific_band(
    band: Literal["3mm", "2mm", "1mm", "0.9mm"],
    freqs: np.ndarray,
    lines: List[str],
    obstime: Optional[float],
    ipwv: int,
    short: bool
):

    min_space = 0.022 # [0, 1]
    global_offset_pos = -0.01

    percentile = 90

    low, upp = iram30m_emir.IRAM30mEMIR.bands()[band]

    f_array, n_array = _load_noise_profile(obstime or 60, ipwv)
    mask = (low <= f_array) & (f_array <= upp)

    rms_min = 1e3*np.min(n_array[mask]) # mK
    rms_max = min(1e3*np.percentile(n_array[mask], percentile), 900) # mK
    rms_min = max(rms_min - 0.1 * (rms_max - rms_min), 0)
    rms_delta = rms_max - rms_min

    plt.plot([low, upp], 2*[rms_min+0.9*rms_delta], color="red", label=f"{band} band")

    if obstime is not None:
        plt.plot(f_array[mask], 1e3*n_array[mask], color="tab:gray")
        plt.ylabel("Noise RMS (mK)", labelpad=10)

    _lines = []
    _freqs = []
    for line, f in zip(lines, freqs):
        if not low <= f <= upp:
            continue
        plt.axvline(f, 0.05, 0.95, linewidth=0.8)
        _lines.append(line)
        _freqs.append(f)

    xlim = plt.gca().get_xlim()
    factor = xlim[1] - xlim[0]
    min_df = min_space * factor

    _positions = []
    group_f, group_p = [], []

    i = 0
    while i < len(_freqs):
        f = _freqs[i]
        if len(group_f) == 0:
            group_f.append(f)
            group_p.append(f)
            i += 1
        elif f - group_f[-1] < min_df:
            group_f.append(f)
            group_p.append(group_p[-1] + min_df)
            i += 1
        else:
            group_offset = np.mean(group_p) - np.mean(group_f)
            _positions.extend([p - group_offset for p in group_p])
            group_f, group_p = [], []
            # No increment

    group_offset = np.mean(group_p) - np.mean(group_f)
    _positions.extend([p - group_offset for p in group_p])

    for line, f, pos in zip(_lines, _freqs, _positions):
        plt.annotate(
            "$" + latex_line(line, short=short) + "$",
            xy=(pos + factor * global_offset_pos, rms_max + 0.02*rms_delta),
            rotation=60, ha='left', va='bottom', annotation_clip=False,
            fontsize=7
        )

    plt.xlabel("$f$ (GHz)")
    plt.ylim([rms_min, rms_max])
    if obstime is None:
        plt.yticks([])

    plt.gca().xaxis.set_minor_locator(MultipleLocator(5))
    plt.legend(loc="lower left", handlelength=1, borderpad=0.5, fontsize=9)

