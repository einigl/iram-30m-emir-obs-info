import os
from typing import Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from matplotlib.ticker import MultipleLocator

from ...graphics import Settings, latex_line
from . import iram30m_emir

Settings.only_rotational = True

plt.rc("text", usetex=True)


def _load_noise_profile(obstime: float, ipwv: int):
    """
    Obstime in seconds
    """
    f_array = np.load(os.path.join(os.path.dirname(__file__), "frequencies.npy"))
    n_array = np.load(
        os.path.join(os.path.dirname(__file__), f"rms_profile_ipwv_{ipwv}mm.npy")
    )

    n_array = n_array / np.sqrt(obstime / 60)

    return f_array, n_array


def _avoid_overlapping(
    freqs: List[float], min_gap: Optional[float], lims: Tuple[float, float]
):
    x = [(f - lims[0]) / (lims[1] - lims[0]) for f in freqs]

    if min_gap is None:
        return x

    min_gap /= 100

    count = 0
    while True:
        offsets = len(x) * [0.0]
        for i in range(len(x) - 1):
            gap = x[i + 1] - x[i]
            diff = min_gap - gap
            if diff > 1e-5:  # To avoid staying in loop because of numerical errors
                offsets[i] -= diff / 2
                offsets[i + 1] += diff / 2
        x = [f + o for f, o in zip(x, offsets)]
        if max(offsets) == 0:
            break
        count += 1
        if count > 100:
            break

    return x


def plot_all_bands(
    lines: List[str],
    freqs: Dict[str, float],
    obstime: Optional[float],
    ipwv: Literal[0, 1, 2],
    rms_min: Optional[float] = None,
    rms_max: Optional[float] = None,
    fontsize=12,
    legend_fontsize=12,
):
    """
    ipwv [mm]
    rms_min [K]
    rms_max [K]
    """
    if rms_max is None:
        rms_max = 1.0 if obstime is None else 0.5 / (obstime / 60) ** 0.5
    if rms_min is None:
        rms_min = 0.0 if obstime is None else 0.05 / (obstime / 60) ** 0.5

    rms_min *= 1e3
    rms_max *= 1e3
    rms_delta = rms_max - rms_min

    for l in lines:
        plt.axvline(freqs[l], 0.05, 0.95, linewidth=0.8)

    bands = iram30m_emir.IRAM30mEMIR.bands()

    props = dict(boxstyle="round", facecolor="white", alpha=0.8)
    lowest, highest = -float("inf"), float("inf")
    for name, (low, upp) in bands.items():
        lowest, highest = min(lowest, low), max(highest, upp)
        plt.plot([low, upp], 2 * [rms_min + 0.9 * rms_delta], color="red")
        plt.text(
            (low + upp) / 2,
            rms_min + 0.80 * rms_delta,
            name,
            horizontalalignment="center",
            verticalalignment="center",
            color="red",
            weight="bold",
            bbox=props,
            fontsize=legend_fontsize,
        )

    f_array, n_array = _load_noise_profile(obstime or 60.0, ipwv)
    mask = (lowest <= f_array) & (f_array <= highest)

    if obstime is not None:
        plt.plot(f_array[mask], 1e3 * n_array[mask], color="tab:gray")
        plt.ylabel("Noise RMS (mK)", labelpad=10, fontsize=fontsize)

    plt.xlabel("$f$ (GHz)", fontsize=fontsize)
    plt.ylim([rms_min, rms_max])
    if obstime is None:
        plt.yticks([])
    plt.tick_params(labelsize=fontsize)
    plt.gca().xaxis.set_minor_locator(MultipleLocator(5))


def plot_specific_band(
    band: Literal["3mm", "2mm", "1mm", "0.9mm"],
    lines: List[str],
    freqs: Dict[str, float],
    obstime: Optional[float],
    ipwv: Literal[0, 1, 2],
    transitions: bool,
    rotation: int,
    rms_min: Optional[float] = None,
    rms_max: Optional[float] = None,
    min_gap: Optional[float] = None,
    global_offset: Optional[float] = None,
    fontsize: int = 12,
    lines_fontsize: int = 10,
    legend_fontsize: int = 12,
):
    """
    ipwv [mm]
    rms_min [K]
    rms_max [K]
    labels_gap [%]
    global_offset [%]
    """
    assert isinstance(band, str)
    band = band.lower().replace(" ", "")
    assert band in ["3mm", "2mm", "1mm", "0.9mm"]

    low, upp = iram30m_emir.IRAM30mEMIR.bands()[band]

    f_array, n_array = _load_noise_profile(obstime or 60, ipwv)
    mask = (low <= f_array) & (f_array <= upp)

    perc_d = {
        "3mm": 90,
        "2mm": 85,
        "1mm": 90,
        "0.9mm": 70,
    }

    if rms_max is None:
        rms_max = 1e3 * np.percentile(n_array[mask], perc_d[band])  # mK
    if rms_min is None:
        rms_min = 1e3 * np.min(n_array[mask])  # mK
        rms_min = max(rms_min - 0.1 * (rms_max - rms_min), 0)
    rms_delta = rms_max - rms_min

    plt.plot(
        [low, upp],
        2 * [rms_min + 0.9 * rms_delta],
        color="red",
        label=f"{band.replace('mm', '$~$mm')} band",
    )

    if obstime is not None:
        plt.plot(f_array[mask], 1e3 * n_array[mask], color="tab:gray")
        plt.ylabel("Noise RMS (mK)", labelpad=10, fontsize=fontsize)

    _freqs = []
    for line in lines:
        f = freqs[line]
        if not low <= f <= upp:
            continue
        plt.axvline(f, 0.05, 0.95, linewidth=0.8)
        _freqs.append(f)

    order = np.argsort(_freqs)
    lines = [lines[i] for i in order]
    _freqs = [_freqs[i] for i in order]

    xs = _avoid_overlapping(_freqs, min_gap=min_gap, lims=plt.gca().get_xlim())
    if global_offset is not None:
        xs = [x + global_offset / 100 for x in xs]

    for line, x in zip(lines, xs):
        t = plt.annotate(
            "$" + latex_line(line, transition=transitions) + "$",
            xy=(x, 1.025),
            xycoords="axes fraction",
            rotation=rotation,
            ha="center"
            if abs(rotation) == 90
            else "left"
            if rotation >= 0
            else "right",
            va="bottom",
            annotation_clip=False,
            fontsize=lines_fontsize,
        )

    plt.xlabel("$f$ (GHz)", fontsize=fontsize)
    plt.ylim([rms_min, rms_max])
    if obstime is None:
        plt.yticks([])
    else:
        plt.tick_params(labelsize=fontsize)

    plt.gca().xaxis.set_minor_locator(MultipleLocator(5))
    plt.legend(
        loc="lower left", handlelength=1, borderpad=0.5, fontsize=legend_fontsize
    )
