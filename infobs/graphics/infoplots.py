import itertools as itt
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from ._util import expformat, truncate_colormap

__all__ = ["InfoPlotter"]


class InfoPlotter:

    line_formatter: Callable
    param_formatter: Callable
    math_mode: bool

    def __init__(self, line_formatter: Callable, param_formatter: Callable):
        self.line_formatter = line_formatter
        self.param_formatter = param_formatter

        c = to_rgba("tab:blue")
        self.default_color = (c[0], c[1], c[2], 0.6)

        c = to_rgba("tab:orange")
        self.alt_color = (c[0], c[1], c[2], 0.6)

    # Probability plots

    def plot_prob_bar(
        self, lines: List[str], probs: List[float], transitions: bool = True
    ) -> Figure:
        ###
        dpi = 200
        width = 0.6
        xscale = 1.2
        yscale = 1.0
        capsize = 6
        ###

        # fig, ax = plt.subplots(1, 1, figsize = (xscale*6.4, yscale*4.8), dpi=dpi)
        ax = plt.gca()

        ax.bar(np.arange(len(probs)), probs, width=width, color="tab:blue")

        ax.set_xticks(np.arange(len(probs)))
        ax.set_xticklabels(
            [
                "$" + self.lines_comb_formatter(l, transition=transitions) + "$"
                for l in lines
            ],
            rotation=45,
            fontsize=12,
            ha="right",
        )

        ax.set_xlabel("Integrated molecular lines", labelpad=20)
        ax.set_ylabel("Mutual information (bits)", labelpad=20)

        return None  # fig TODO

    # MI plots (discrete)

    def plot_mi_bar(
        self,
        lines: List[str],
        mis: List[float],
        errs: Optional[List[Union[float, Tuple[float, float]]]] = None,
        colors: Optional[List[str]] = None,
        sort: bool = False,
        nfirst: Optional[int] = None,
        transitions: bool = True,
        rotation: int = 90,
        fontsize: int = 20,
        capsize: int = 8,
        barwidth: float = 0.6,
        bottom_val: Optional[float] = None,
    ) -> Axes:
        """
        TODO
        """
        assert len(mis) == len(lines)
        assert sort or (nfirst is None)

        ax = plt.gca()

        if sort:
            indices = np.array(mis).argsort()[::-1]
            if nfirst is not None:
                indices = indices[:nfirst]

            mis = [mis[i] for i in indices]
            lines = [lines[i] for i in indices]
            if errs is not None:
                errs = [errs[i] for i in indices]

        barlist = ax.bar(
            np.arange(len(mis)),
            mis,
            width=barwidth,
            color=self.default_color,
            edgecolor="black",
        )
        ax.errorbar(
            np.arange(len(mis)),
            mis,
            yerr=errs,
            fmt="none",
            capsize=capsize,
            color="tab:red",
        )

        if colors is not None:
            assert len(colors) == len(mis)
            if sort:
                colors = [colors[i] for i in indices]
            for i, c in enumerate(colors):
                barlist[i].set_facecolor(c)

        for b in barlist:
            b.set_linewidth(1.5)

        ha = "center" if rotation % 90 == 0 else "right"
        rotation_mode = "default" if rotation % 90 == 0 else "anchor"
        ax.set_xticks(np.arange(len(mis)))
        ax.set_xticklabels(
            [
                "$" + self.lines_comb_formatter(l, transition=transitions) + "$"
                for l in lines
            ],
            rotation=rotation,
            fontsize=fontsize,
            ha=ha,
            rotation_mode=rotation_mode,
        )
        plt.yticks(fontsize=fontsize)

        if errs is None:
            low = np.nanmin(mis)
            high = np.nanmax(mis)
        else:
            low = np.nanmin(np.array(mis) - np.array(errs))
            high = np.nanmax(np.array(mis) + np.array(errs))
        diff = high - low
        frac = 0.1

        plt.ylim(
            [
                low - frac * diff if bottom_val is None else bottom_val,
                high + frac * diff,
            ]
        )

        ax.set_ylabel("Mutual information (bits)", labelpad=24, fontsize=fontsize)

        return ax

    def plot_mi_matrix(
        self,
        lines: List[str],
        mis: List[List[float]],
        show_diag: bool = True,
        transitions: bool = True,
    ) -> Figure:
        ###
        cmap = "OrRd"
        ###

        # fig, ax = plt.subplots(1, 1, figsize = (xscale*6.4, yscale*4.8), dpi=dpi)
        ax = plt.gca()
        fig = ax.get_figure()

        mis = np.array(mis)
        mask = np.where(
            np.tril(np.ones_like(mis), k=-1 if show_diag else 0), float("nan"), 1.0
        )
        im = ax.imshow(mask * mis, origin="lower", cmap=cmap)

        cbar = fig.colorbar(im)
        cbar.set_label("Mutual information (bits)", labelpad=30, rotation=270)

        ax.set_xticks(np.arange(mis.shape[0]))
        ax.set_yticks(np.arange(mis.shape[0]))
        ax.set_xticklabels(
            ["$" + self.line_formatter(l, transition=transitions) + "$" for l in lines],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=10,
        )
        ax.set_yticklabels(
            ["$" + self.line_formatter(l, transition=transitions) + "$" for l in lines],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=10,
        )

        return fig

    # MI bar plots comparison

    def plot_mi_bar_comparison(
        self,
        lines: List[str],
        mis: Dict[str, List[float]],
        errs: Optional[Dict[str, List[float]]],
        labels: Dict[str, str],
        transitions: bool = True,
        rotation: int = 90,
        bottom_val: Optional[float] = None,
        show_legend: bool = False,
        fontsize: int = 20,
        capsize: int = 8,
        barwidth: float = 0.6,
    ) -> Axes:
        """
        TODO
        """
        ax = plt.gca()

        alt = ["el" in "_".join(l) for l in lines]
        idx_default = [i for i in range(len(lines)) if not alt[i]]
        idx_alt = [i for i in range(len(lines)) if alt[i]]

        keys = list(mis.keys())
        for i, key in enumerate(keys):
            if i == 0:
                barlist_0_default = ax.bar(
                    idx_default,
                    [mis[key][i] for i in idx_default],
                    width=barwidth,
                    color=self.default_color,
                    edgecolor="black",
                )
                barlist_0_alt = ax.bar(
                    idx_alt,
                    [mis[key][i] for i in idx_alt],
                    width=barwidth,
                    color=self.alt_color,
                    edgecolor="black",
                )
                ax.errorbar(
                    np.arange(len(mis[key])),
                    mis[key],
                    yerr=errs[key],
                    fmt="none",
                    capsize=capsize,
                    color="tab:red",
                    linewidth=1.5,
                )
            else:
                barlist_1 = ax.bar(
                    list(range(len(mis[key]))),
                    mis[key],
                    width=barwidth,
                    color="none",
                    label=labels[key],
                    edgecolor="black",
                    hatch="/",
                )
                ax.errorbar(
                    np.arange(len(mis[key])),
                    mis[key],
                    yerr=errs[key],
                    fmt="none",
                    capsize=capsize,
                    color="black",
                )

        for barlist in (barlist_0_default, barlist_0_alt, barlist_1):
            for b in barlist:
                b.set_linewidth(1.5)

        if bottom_val is not None:
            plt.ylim([bottom_val, None])

        ha = "center" if rotation % 90 == 0 else "right"
        rotation_mode = "default" if rotation % 90 == 0 else "anchor"
        ax.set_xticks(np.arange(len(lines)))
        ax.set_xticklabels(
            [
                "$" + self.lines_comb_formatter(l, transition=transitions) + "$"
                for l in lines
            ],
            rotation=rotation,
            fontsize=fontsize,
            ha=ha,
            rotation_mode=rotation_mode,
        )
        plt.yticks(fontsize=fontsize)

        ax.set_ylabel("Mutual information (bits)", labelpad=24, fontsize=fontsize)

        if errs is None:
            low = min([np.nanmin(mis[name]) for name in mis])
            high = max([np.nanmax(mis[name]) for name in mis])
        else:
            low = min(
                [np.nanmin(np.array(mis[name]) - np.array(errs[name])) for name in mis]
            )
            high = max(
                [np.nanmax(np.array(mis[name]) + np.array(errs[name])) for name in mis]
            )
        diff = high - low
        frac = 0.1

        plt.ylim(
            [
                low - frac * diff if bottom_val is None else bottom_val,
                high + frac * diff,
            ]
        )

        from matplotlib.collections import PatchCollection

        class MulticolorPatch(object):
            def __init__(self, colors):
                self.colors = colors

        class MulticolorPatchHandler(object):
            def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                width, height = handlebox.width, handlebox.height
                patches = []
                for i, c in enumerate(orig_handle.colors):
                    patches.append(
                        plt.Rectangle(
                            [
                                width / len(orig_handle.colors) * i
                                - handlebox.xdescent,
                                -handlebox.ydescent,
                            ],
                            width / len(orig_handle.colors),
                            height,
                            facecolor=c,
                            edgecolor="black",
                            linewidth=1.5,
                        )
                    )

                patch = PatchCollection(patches, match_original=True)

                handlebox.add_artist(patch)
                return patch

        h = []
        h.append(MulticolorPatch([self.default_color, self.alt_color]))
        h.append(barlist_1)

        if show_legend:
            ax.legend(
                h,
                list(labels.values()),
                fontsize=fontsize,
                loc="upper right",
                handler_map={MulticolorPatch: MulticolorPatchHandler()},
            )

        return ax

    # MI plots (continuous)

    def plot_mi_profile(self):
        raise NotImplementedError("TODO")

    def plot_mi_profile_comparison(self):
        raise NotImplementedError("TODO")

    def plot_mi_profiles_summary(self):
        raise NotImplementedError("TODO")

    def plot_mi_map(
        self,
        xticks: np.ndarray,
        yticks: np.ndarray,
        mat: np.ndarray,
        vmax: Optional[float] = None,
        cmap: str = "jet",
        paramx: Optional[str] = None,
        paramy: Optional[str] = None,
    ):
        ax = plt.gca()

        X, Y = np.meshgrid(xticks, yticks)

        im = ax.pcolor(X, Y, mat, cmap=cmap, vmin=0, vmax=vmax)

        cbar = plt.colorbar(im, ax=ax)  # fig.colorbar(...)
        cbar.set_label("Amount of information (bits)", labelpad=10)

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xlabel(f"${self.param_formatter(paramx)}$")
        ax.set_ylabel(f"${self.param_formatter(paramy)}$")

        return ax

    def plot_mi_map_comparison(
        self,
        xticks: np.ndarray,
        yticks: np.ndarray,
        mat1: np.ndarray,
        mat2: np.ndarray,
        diff: bool,
        title1: str,
        title2: str,
        titlediff: Optional[str] = None,
        vmax: Optional[float] = None,
        cmap: str = "jet",
        cmapdiff: str = "magma",
        params: Optional[List[str]] = None,
        lines: Optional[List[str]] = None,
        paramx: Optional[str] = None,
        paramy: Optional[str] = None,
    ):
        if diff:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(3 * 6.4, 4.8), dpi=125)
            matdiff = mat1 - mat2
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * 6.4, 4.8), dpi=125)

        X, Y = np.meshgrid(xticks, yticks)

        ax1.pcolor(X, Y, mat1, cmap=cmap, vmin=0, vmax=vmax)
        im = ax2.pcolor(X, Y, mat2, cmap=cmap, vmin=0, vmax=vmax)

        cbar = fig.colorbar(im, ax=[ax1, ax2])
        cbar.set_label("Amount of information (bits)", labelpad=10)

        if diff:
            im = ax3.pcolor(X, Y, matdiff, cmap=cmapdiff, vmin=0)

            cbar = fig.colorbar(im, ax=[ax3])
            cbar.set_label("Amount of information (bits)", labelpad=10)

        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        if diff:
            ax3.set_xscale("log")
            ax3.set_yscale("log")

        ax1.set_xlabel(f"${self.param_formatter(paramx)}$")
        ax1.set_ylabel(f"${self.param_formatter(paramy)}$")
        ax2.set_xlabel(f"${self.param_formatter(paramx)}$")
        ax2.set_yticks([])
        if diff:
            ax3.set_xlabel(f"${self.param_formatter(paramx)}$")
            ax3.set_yticks([])

        ax1.set_title(title1)
        ax2.set_title(title2)
        if diff:
            ax3.set_title(titlediff)

        if lines is not None:
            title = f"Informativity on ${self.params_comb_formatter(params)}$ of ${self.lines_comb_formatter(lines)}$"
        fig.suptitle(title, fontsize=18, x=0.5, y=1.0)

        if diff:
            return fig, (ax1, ax2, ax3)
        return fig, (ax1, ax2)

    def plot_mi_maps_summary(self):
        raise NotImplementedError("TODO")

    # Summaries

    def plot_summary_1d(
        self,
        parameters: Tuple[str, ...],
        regimes: Dict[str, Dict[str, Tuple]],
        best_lines: List[Tuple[str, ...]],
        confidences: List[float],
    ) -> Figure:
        """
        Plot the summary of the most informative lines. The constraint is on a single parameter.
        `parameter` is the set of physical parameter to estimate
        Format (example): ('g0',)
        `regimes` contains the bounds for all subregimes
        Format (example): {'av': {'1': [1, 2], '2': [2, None]}}
        `best_lines` contains a
        Format (example): [('13co10', 'c18o10'), ('n2hp10')]
        `confidence` contains the probabilities for the lines in `best_lines` to be the best.
        Format (example): [(line1, line2), (line3)]
        """
        ###
        xscale = 1.2
        yscale = 1.0
        dpi = 200
        ###

        fig, ax = plt.subplots(
            1, 1, figsize=(xscale * 6.4, 0.5 * yscale * 4.8), dpi=dpi
        )

        # Checking
        if isinstance(parameters, str):
            parameters = (parameters,)

        # Plot grid
        param_regime = list(regimes.keys())[0]
        x = []
        for val in regimes[param_regime].values():
            if val is None or val[0] is None:
                continue
            ax.axvline(len(x) + 1, color="black")

            if param_regime in ["g0"]:  # TODO
                x.append(f"${expformat(val[0])}$")
            else:
                x.append(f"${val[0]}$")
            if val[1] is None:
                x.append("$+\\infty$")

        # Static settings
        fontsizes = {1: 13, 2: 10, 3: 10, 4: 10}

        # Plot names and confidences
        cmap = plt.get_cmap("gist_rainbow")
        subcmap = truncate_colormap(cmap, 0.0, 0.35)
        for i, (l, c) in enumerate(zip(best_lines, confidences), 1):
            if l is not None:
                if isinstance(l, str):
                    l = (l,)
                    c = (c,)
                l = list(l)
                c = list(c)
                sign = [None] * len(l)

                ax.add_patch(Rectangle((i, 0), 1, 1, color=subcmap(c[0]), alpha=0.4))
                for k, _ in enumerate(c):
                    _c = 100 * c[k]
                    if _c > 99.9:
                        _c, _sign = 99.9, ">"
                    elif _c < 0.1:
                        _c, _sign = 0.1, "<"
                    else:
                        _sign = "="
                    c[k], sign[k] = _c, _sign
                ax.text(
                    i + 0.5,
                    0.5,
                    "\n\n".join(
                        [
                            f"${self.lines_comb_formatter(_l, transition=True)}$\n$p {_sign} {_c:.1f}\%$"
                            for _l, _c, _sign in zip(l, c, sign)
                        ]
                    ),
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=fontsizes[len(l)],
                )
            else:
                ax.add_patch(Rectangle((i, 0), 1, 1, color="gray", alpha=0.6))
                ax.add_patch(Rectangle((i, 0), 1, 1, fill=False, hatch="//"))

        # Settings
        ax.set_xticks(np.arange(1, len(x) + 1))
        ax.set_yticks([])
        ax.set_xticklabels(x)
        ax.set_xlabel("$" + self.param_formatter(param_regime) + "$", labelpad=10)
        ax.set_xlim([1, len(x)])
        ax.set_ylim([0, 1])

        return fig

    def plot_summary_2d(
        self,
        parameters: Tuple[str, ...],
        regimes: Dict[str, Dict[str, Tuple]],
        best_lines: List[List[Tuple[str, ...]]],
        confidences: List[List[float]],
    ):
        ###
        xscale = 1.2
        yscale = 1.0
        dpi = 200
        ###

        fig, ax = plt.subplots(1, 1, figsize=(xscale * 6.4, yscale * 4.8), dpi=dpi)

        # Checking
        if isinstance(parameters, str):
            parameters = (parameters,)

        # Plot grid
        param_regime_1, param_regime_2 = list(regimes.keys())[0:2]
        x, y = [], []
        for val in regimes[param_regime_1].values():
            if val is None or val[0] is None:
                continue
            ax.axvline(len(x) + 1, color="black")
            x.append(f"${val[0]}$")
            if val[1] is None:
                x.append("$+\\infty$")
        for val in regimes[param_regime_2].values():
            if val is None or val[0] is None:
                continue
            ax.axhline(len(y) + 1, color="black")
            y.append(f"${expformat(val[0])}$")
            if val[1] is None:
                y.append("$+\\infty$")

        # Static settings
        coords = {
            1: [(0.5, 0.5)],
            2: [(0.5, 0.7), (0.5, 0.3)],
            3: [(0.5, 0.7), (0.25, 0.3), (0.75, 0.3)],
            4: [(0.25, 0.7), (0.75, 0.7), (0.25, 0.3), (0.75, 0.3)],
        }
        fontsizes = {1: 13, 2: 10, 3: 8, 4: 8}

        # Plot names and confidences
        cmap = plt.get_cmap("gist_rainbow")
        subcmap = truncate_colormap(cmap, 0.0, 0.35)
        for i, j in itt.product(range(len(best_lines)), range(len(best_lines[0]))):
            l = best_lines[i][j]
            c = confidences[i][j]

            if l is not None:
                if isinstance(l, str):
                    l = (l,)
                    c = (c,)

                ax.add_patch(
                    Rectangle((i + 1, j + 1), 1, 1, color=subcmap(c[0]), alpha=0.4)
                )
                for k, _ in enumerate(l):
                    _c = 100 * c[k]
                    if _c > 99.9:
                        _c, _sign = 99.9, ">"
                    elif _c < 0.1:
                        _c, _sign = 0.1, "<"
                    else:
                        _sign = "="
                    _l = l[k]

                    i0, j0 = coords[len(l)][k]
                    ax.text(
                        i + 1 + i0,
                        j + 1 + j0,
                        f"${self.lines_comb_formatter(_l, transition=True)}$\n$p {_sign} {_c:.1f}\%$",
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=fontsizes[len(l)],
                    )
            else:
                ax.add_patch(Rectangle((i + 1, j + 1), 1, 1, color="gray", alpha=0.6))
                ax.add_patch(Rectangle((i + 1, j + 1), 1, 1, fill=False, hatch="//"))

        # Settings
        ax.set_xticks(np.arange(1, len(x) + 1))
        ax.set_yticks(np.arange(1, len(y) + 1))
        ax.set_xticklabels(x)
        ax.set_yticklabels(y)
        ax.set_xlabel("$" + self.param_formatter(param_regime_1) + "$", labelpad=10)
        ax.set_ylabel("$" + self.param_formatter(param_regime_2) + "$", labelpad=10)
        ax.set_xlim([1, len(x)])
        ax.set_ylim([1, len(y)])

        return fig

    # Helpers

    def lines_comb_formatter(
        self, lines: Union[List[str], str], transition: bool = True
    ) -> str:
        """
        Returns a printable latex version of the combination of lines `lines`.
        If the combination has only one element, it is treated as a single line.
        """
        assert isinstance(lines, (str, List, Tuple))

        if isinstance(lines, str):
            lines = [lines]

        # if len(lines) == 1:
        #     return self.line_formatter(lines[0], transition=transitions)
        # return r"\left(" + ','.join([self.line_formatter(line, transition=transitions) for line in lines]) + r"\right)"
        return ",".join(
            [self.line_formatter(line, transition=transition) for line in lines]
        )

    def params_comb_formatter(self, params: Union[List[str], str]) -> str:
        """
        Returns a printable latex version of the combination of physical parameters `params`.
        If the combination has only one element, it is treated as a single parameter.
        """
        assert isinstance(params, (str, List, Tuple))

        if isinstance(params, str):
            params = [params]

        if len(params) == 1:
            return self.param_formatter(params[0])
        return (
            r"\left("
            + ",".join([self.param_formatter(param) for param in params])
            + r"\right)"
        )

    def regime_formatter(
        self,
        param_name: str,
        reg: Optional[Tuple[Optional[float], Optional[float]]],
        lower_bound: Optional[float] = 0,
        upper_bound: Optional[float] = None,
    ) -> str:
        lb = "-\infty" if lower_bound is None else expformat(lower_bound)
        ub = "+\infty" if upper_bound is None else expformat(upper_bound)

        if reg is None or reg[0] is None and reg[1] is None:
            return f"${lb} < {self.param_formatter(param_name)} < {ub}$"
        if reg[0] is None:
            return f"${lb} < {self.param_formatter(param_name)} < {expformat(reg[1])}$"
        if reg[1] is None:
            return f"${expformat(reg[0])} < {self.param_formatter(param_name)} < {ub}$"
        return f"${expformat(reg[0])} < {self.param_formatter(param_name)} < {expformat(reg[1])}$"
