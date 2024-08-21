import os
from itertools import combinations
from typing import List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import (
    FixedLocator,
    FuncFormatter,
    MultipleLocator,
    NullFormatter,
)

from ..model import MeudonPDR
from .latex import LaTeX, latex_line, latex_param

__all__ = ["PDRPlotter"]


class PDRPlotter:
    """
    Class to handle the plotting of profiles and slices in the parameter space in a very user-friendly way.
    """

    parameters = ["Av", "G0", "Pth", "angle"]

    param_units_raw = {"Av": "mag", "G0": "", "Pth": "K.cm-3", "angle": "deg"}

    param_units_latex = {"Av": "mag", "G0": "", "Pth": "K.cm$^{-3}$", "angle": "deg"}

    param_scales = {"Av": "log", "G0": "log", "Pth": "log", "angle": "linear"}

    def __init__(self, kelvin: bool = True):
        """
        Plotter for 1D and 2D line profiles.
        """
        assert isinstance(kelvin, bool)
        self.kelvin = kelvin

        self.model = MeudonPDR(kelvin)

    def intensity_unit_latex(self) -> str:
        if self.kelvin:
            return "K km s$^{-1}$"
        return "erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$"

    def print_parameters_space(self) -> None:
        """TODO"""
        print("Parameters space")
        for param, bounds in self.model.bounds.items():
            print(f"{param}: {list(bounds)} {self.param_units_raw[param]}")

    def _parse_csv(self, csv_file: Union[str, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(csv_file, str):
            df = pd.read_csv(csv_file)
        elif isinstance(csv_file, pd.DataFrame):
            df = csv_file
        else:
            raise TypeError(
                f"csv_file must be a string path or a DataFrame, not {type(csv_file)}"
            )
        # Check the columns names
        inputs = [name.strip().lower() for name in self.parameters + ["lines"]]
        columns = [name.strip().lower() for name in df.columns]
        try:
            indices = [columns.index(name) for name in inputs]
        except ValueError:
            raise ValueError(
                f"Columns of loaded file does not match {inputs} even rearranging them."
            )
        columns = [df.columns[i] for i in indices]
        df = df[columns]
        df = df.rename(
            columns={
                name_low: name for name_low, name in zip(inputs[-1], self.parameters)
            }
        )
        return df

    def plot_profile(
        self,
        lines: Union[List[str], str],
        Av: Optional[float] = None,
        G0: Optional[float] = None,
        Pth: Optional[float] = None,
        angle: Optional[float] = None,
        n_samples: int = 100,
        logy: bool = True,
        legend: bool = True,
        latex: bool = True,
        fontsize: int = 12,
        legend_loc: str = "best",
    ) -> None:
        """
        Only one variable among Avmax, G0, Pth and angle has to be null.
        """

        # Argument checking (TODO: compléter)
        if isinstance(lines, str):
            lines = [lines]
        assert isinstance(lines, list)
        assert (Av is None) + (G0 is None) + (Pth is None) + (angle is None) == 1
        assert isinstance(n_samples, int)
        assert isinstance(legend, bool)
        assert isinstance(latex, bool)
        assert isinstance(fontsize, int)

        # Parameter to plot
        params = {"Av": Av, "G0": G0, "Pth": Pth, "angle": angle}
        for p in params:
            if params[p] is None:
                param_to_plot = p
                break

        # Parameters DataFrame
        d = dict.fromkeys(self.param_scales.keys(), None)
        for p, scale in self.param_scales.items():
            if p != param_to_plot:
                d[p] = params[p] * np.ones(n_samples)
                continue
            a, b = self.model.bounds[p]
            if scale == "log":
                d[p] = np.logspace(np.log10(a), np.log10(b), n_samples)
            else:
                d[p] = np.linspace(a, b, n_samples)
        df_params = pd.DataFrame.from_dict(d, orient="columns")

        # Evaluate model
        df_lines = self.model.predict(df_params, lines)

        # Plot profiles
        with LaTeX(activate=latex):

            for line in lines:
                lines = plt.plot(
                    df_params[param_to_plot],
                    df_lines[line],
                    label=f"${latex_line(line)}$",
                )
            if self.param_scales[param_to_plot] == "log":
                plt.xscale("log")
            if logy:
                plt.yscale("log")

            plt.ylabel(
                f"Integrated intensities ({self.intensity_unit_latex()})",
                labelpad=15,
                fontsize=fontsize,
            )

            plt.grid()
            plt.xlabel(
                f"${latex_param(param_to_plot)}$ ({self.param_units_latex[param_to_plot]})".replace(
                    "()", "(-)"
                ),
                labelpad=10,
                fontsize=fontsize,
            )
            if legend:
                plt.legend(
                    fontsize=fontsize,
                    handlelength=1.2,
                    loc=legend_loc,
                )

            plt.gca().tick_params(axis="both", labelsize=fontsize)

            str_title = ""
            for p in self.parameters:
                if p == param_to_plot:
                    continue
                str_title += (
                    "${}={:.1e}$ {}, "
                    if self.param_scales[p] == "log"
                    else "${}={:.2f}$ {}, "
                ).format(latex_param(p), params[p], self.param_units_latex[p])
            str_title = str_title.replace(" , ", ", ")
            if str_title.endswith(", "):  # Avoid using str.removesuffix for Python<3.9
                str_title = str_title[:-2]
            plt.title(str_title, pad=10, fontsize=fontsize)

        return plt.gca()

    def save_profiles_from_csv(
        self,
        csv_file: Union[str, pd.DataFrame],
        path_outputs: str,
        n_samples: int = 100,
        legend: bool = True,
        latex: bool = True,
        figsize: Tuple[float, float] = (6.4, 0.6 * 4.8),
        dpi: int = 150,
    ) -> None:
        """TODO"""
        # Parse CSV or DataFrame
        df = self._parse_csv(csv_file)

        # Remove potential leading and trailing whitespaces
        df = df.rename(columns=lambda x: x.strip())

        # Create directory
        if not os.path.isdir(path_outputs):
            os.mkdir(path_outputs)

        # Process each row
        for i in df.index:
            row = df.loc[i]
            # Ignore a row if it has no line to plot
            if row.isnull()["lines"]:
                continue
            lines = [subs for subs in row["lines"].strip().split(" ") if len(subs) > 0]

            row = row[self.parameters]
            # Ignore a row if more than one parameter is blank, an error is raised
            if (row.isnull()).sum() > 1:
                continue
            # If a row has exactly one blank value, we plot this profile
            if (row.isnull()).sum() == 1:
                names_blank = row.index[row.isnull()].to_list()
            # If a row has no blank value, we plot all the possible profiles
            else:
                names_blank = self.parameters
            # Save all programmed profiles
            for n_blank in names_blank:
                d = {
                    name: (row[name] if name != n_blank else None)
                    for name in self.parameters
                }

                fig = plt.figure(figsize=figsize, dpi=dpi)
                self.plot_profile(
                    lines,
                    **d,
                    n_samples=n_samples,
                    legend=legend,
                    latex=latex,
                )
                filename = os.path.join(path_outputs, f"{i}_{n_blank}")
                fig.savefig(filename, bbox_inches="tight")
                plt.close(fig)

    def _mimic_log_axes(
        self,
        ax: Axes,
        xscale: Literal["linear", "log"],
        yscale: Literal["linear", "log"],
    ):
        ## Locators for Y-axis
        # set tickmarks at multiples of 1.
        majorLocator = MultipleLocator(1.0)
        lim = (
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        )
        ra = np.array(
            [
                [
                    n + (1.0 + np.log10(i))
                    for n in range(int(lim[0]) - 1, int(lim[1]) + 1)
                ]
                for i in [2, 3, 4, 5, 6, 7, 8, 9][::-1]
            ]
        ).flatten()
        minorLocator = FixedLocator(ra)
        # majorFormatter= FuncFormatter(
        #     lambda x,p: "{:.1e}".format(10**x)
        # )
        majorFormatter = FuncFormatter(
            lambda x, p: r"$10^{" + "{x:d}".format(x=int(x)) + r"}$"
        )

        ax.minorticks_on()

        if xscale == "log":
            ax.xaxis.set_major_locator(majorLocator)
            ax.xaxis.set_major_formatter(majorFormatter)
            ax.xaxis.set_minor_locator(minorLocator)
            ax.xaxis.set_minor_formatter(NullFormatter())
        if yscale == "log":
            ax.yaxis.set_major_locator(majorLocator)
            ax.yaxis.set_major_formatter(majorFormatter)
            ax.yaxis.set_minor_locator(minorLocator)
            ax.yaxis.set_minor_formatter(NullFormatter())

    def plot_slice(
        self,
        line: str,
        Av: Optional[float] = None,
        G0: Optional[float] = None,
        Pth: Optional[float] = None,
        angle: Optional[float] = None,
        n_samples: int = 100,
        cmap: Optional[str] = None,
        transpose: bool = False,
        contour: bool = False,
        logz: bool = True,
        legend: bool = True,
        latex: bool = True,
        fontsize: int = 12,
        legend_loc: str = "best",
    ) -> Tuple[Axes, Colorbar]:
        """
        Only one variable among P, Avmax, radm and angle has to be null.
        """

        # Argument checking (TODO: compléter)
        assert isinstance(line, str)
        assert (Av is None) + (G0 is None) + (Pth is None) + (angle is None) == 2
        assert isinstance(n_samples, int)
        assert isinstance(legend, bool)
        assert isinstance(latex, bool)
        assert isinstance(fontsize, int)

        # Parameters to plot
        params = {"Av": Av, "G0": G0, "Pth": Pth, "angle": angle}
        params_to_plot = []
        for p in params:
            if params[p] is not None:
                continue
            params_to_plot.append(p)
            if len(params_to_plot) == 2:
                break

        if transpose:
            params_to_plot = params_to_plot[::-1]

        # Meshgrid
        l_mesh = [None, None]
        for i, p in enumerate(params_to_plot):
            a, b = self.model.bounds[p]
            if self.param_scales[p] == "log":
                l_mesh[i] = np.logspace(np.log10(a), np.log10(b), n_samples)
            else:
                l_mesh[i] = np.linspace(a, b, n_samples)

        X, Y = np.meshgrid(*l_mesh)

        d_mesh = {params_to_plot[0]: X.flatten(), params_to_plot[1]: Y.flatten()}

        # Parameters DataFrame
        d = dict.fromkeys(self.param_scales.keys(), None)
        for p in self.parameters:
            if p not in params_to_plot:
                d[p] = params[p] * np.ones(n_samples**2)
            else:
                d[p] = d_mesh[p]
        df_params = pd.DataFrame.from_dict(d, orient="columns")

        # Evaluate model
        df_line = self.model.predict(df_params, [line])
        Z = df_line[line].to_numpy().reshape(n_samples, n_samples)

        # Plot profiles
        with LaTeX(activate=latex):

            if logz:
                norm = LogNorm(vmin=Z.min(), vmax=Z.max())
            else:
                norm = Normalize(vmin=Z.min(), vmax=Z.max())

            if contour:
                _Z = np.log10(Z)
                max_levels = 20  # Can be modified
                resolution = 1  # decimal digit
                _min = np.min(np.around(_Z, resolution))
                _max = np.max(np.around(_Z, resolution))
                n_levels = int((_max - _min) * 10**resolution)
                levels = _min + 10 ** (-resolution) * np.arange(n_levels)
                levels = levels[:: int(np.ceil(n_levels / max_levels))]

                cs = plt.contour(
                    np.log10(X),
                    np.log10(Y),
                    _Z,
                    levels=levels,
                    cmap=cmap,
                    extent=[X.min(), X.max(), Y.min(), Y.max()],
                )
                plt.clabel(cs, cs.levels, fmt=lambda x: f"{x:.1f}", fontsize=8)

            else:
                im = plt.pcolor(
                    X,
                    Y,
                    Z,
                    norm=norm,
                    cmap=cmap,
                )

            plt.scatter([], [], label=f"${latex_line(line)}$")

            if contour:
                cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap))
            else:
                cbar = plt.colorbar(im)
            cbar.ax.set_ylabel(
                f"Integrated intensities ({self.intensity_unit_latex()})",
                rotation=-90,
                fontsize=fontsize,
                labelpad=20,
            )

            if not contour:
                if self.param_scales[params_to_plot[0]] == "log":
                    plt.xscale("log")
                if self.param_scales[params_to_plot[1]] == "log":
                    plt.yscale("log")
            else:
                self._mimic_log_axes(
                    plt.gca(),
                    self.param_scales[params_to_plot[0]],
                    self.param_scales[params_to_plot[1]],
                )
                pass

            plt.xlabel(
                f"${latex_param(params_to_plot[0])}$ ({self.param_units_latex[params_to_plot[0]]})".replace(
                    "()", "(-)"
                ),
                labelpad=10,
                fontsize=fontsize,
            )
            plt.ylabel(
                f"${latex_param(params_to_plot[1])}$ ({self.param_units_latex[params_to_plot[1]]})".replace(
                    "()", "(-)"
                ),
                labelpad=10,
                fontsize=fontsize,
            )

            if legend:
                leg = plt.legend(fontsize=fontsize, handletextpad=-2.0, loc=legend_loc)
                for item in leg.legendHandles:
                    item.set_visible(False)

            plt.gca().tick_params(axis="both", labelsize=fontsize)

            str_title = ""
            for p in self.parameters:
                if p in params_to_plot:
                    continue
                str_title += (
                    "${}={:.1e}$ {}, "
                    if self.param_scales[p] == "log"
                    else "${}={:.2f}$ {}, "
                ).format(latex_param(p), params[p], self.param_units_latex[p])
            str_title = str_title.replace(" , ", ", ")
            if str_title.endswith(", "):  # Avoid using str.removesuffix for Python<3.9
                str_title = str_title[:-2]
            plt.title(str_title, pad=10, fontsize=fontsize)

        return plt.gca(), cbar

    def save_slices_from_csv(
        self,
        csv_file: Union[str, pd.DataFrame],
        path_outputs: str,
        n_samples: int = 100,
        contour: bool = False,
        legend: bool = True,
        latex: bool = True,
        figsize: Tuple[float, float] = (6.4, 4.8),
        dpi: int = 150,
    ) -> None:
        """TODO"""
        # Parse CSV or DataFrame
        df = self._parse_csv(csv_file)

        # Create directory
        if not os.path.isdir(path_outputs):
            os.mkdir(path_outputs)

        # Process each row
        for i in df.index:
            row = df.loc[i]
            # Ignore a row if it has no line to plot
            if row.isnull()["lines"]:
                continue
            lines = [subs for subs in row["lines"].split(" ") if len(subs) > 0]
            assert len(lines[0])
            line = lines[0]

            # Ignore a row if more than two parameter are blank
            if (row[: len(self.parameters)].isnull()).sum() > 2:
                continue
            # If a row has exactly two blank value, we plot this profile
            if (row[: len(self.parameters)].isnull()).sum() == 2:
                names_blank = row.index[row.isnull()].to_list()
            # If a row has no blank value, we plot all the possible profiles
            else:
                names_blank = self.parameters
            # Save all programmed slices
            for n_blank_1, n_blank_2 in combinations(names_blank, 2):
                d = {
                    name: (row[name] if name not in (n_blank_1, n_blank_2) else None)
                    for name in self.parameters
                }

                fig = plt.figure(figsize=figsize, dpi=dpi)
                self.plot_slice(
                    line,
                    **d,
                    n_samples=n_samples,
                    contour=contour,
                    legend=legend,
                    latex=latex,
                )

                filename = os.path.join(path_outputs, f"{i}_{n_blank_1}_{n_blank_2}")
                fig.savefig(filename, bbox_inches="tight")
                plt.close(fig)
