import itertools as itt
from math import ceil
from typing import Dict, Iterable, List, Literal, Optional, Union

import infovar
import numpy as np
from matplotlib.axes import Axes

from .graphics import InfoPlotter, PDRPlotter, latex_line, latex_param
from .instruments import Instrument
from .model import MeudonPDR
from .sampling import Constant, Sampler
from .simulator import Simulator

__all__ = ["PDRGetter", "Infobs"]


class PDRGetter(infovar.StandardGetter):
    """
    TODO
    """

    def __init__(
        self,
        instrument: Instrument,
        n_samples: int,
        samplers: List[Sampler],
        obstime: Union[float, List[float]],
        kappa: Sampler = Constant(1.0),
        seed: Optional[int] = None,
    ):

        # Simulator
        self.simulator = Simulator(instrument)

        # Load and preprocess data
        df = self.simulator.simulate(
            n_samples, samplers, instrument.lines, obstime, kappa=kappa, seed=seed
        )

        # Attributes
        self.x_names = instrument.lines
        self.y_names = MeudonPDR.parameters
        self.x = df.loc[:, self.x_names].values
        self.y = df.loc[:, self.y_names].values


class Infobs:
    def __init__(
        self,
        instrument: Instrument,
        n_samples: int,
        samplers: List[Sampler],
        obstime: Union[float, List[float]],
        kappa: Sampler = Constant(1.0),
    ):
        self.discrete_handler = infovar.DiscreteHandler()
        self.continuous_handler = infovar.ContinuousHandler()

        getter = PDRGetter(
            instrument, n_samples, samplers, obstime=obstime, kappa=kappa, seed=None
        )

        self.discrete_handler.set_getter(getter.get)
        self.continuous_handler.set_getter(getter.get)

        self.plotter = InfoPlotter(latex_line, latex_param)

    def set_path(self, save_path: str) -> None:
        self.discrete_handler.set_path(save_path)
        self.continuous_handler.set_path(save_path)

    def set_regimes(self, regimes: Dict[str, Dict]) -> None:
        self.discrete_handler.set_restrictions(regimes)

    def reset(
        self, type: Literal["none", "discrete", "continuous", "both"] = "none"
    ) -> None:
        """Reset every data saved at the current path. Must be used carefully."""
        raise NotImplementedError("TODO")

    def compute_info(self) -> None:
        raise NotImplementedError("TODO")

    def get_discrete_max(
        self,
        lines_iterable: Iterable[List[str]],
        parameters: Union[str, List[str]],
        regime: str,
    ) -> float:
        raise NotImplementedError("TODO")

    def plot_info_bars(
        self,
        lines_iterable: Iterable[List[str]],
        parameters: Union[str, List[str]],
        restriction: str,
        errorbars: bool = True,
        sort: bool = True,
        nfirst: Optional[int] = None,
        transitions: bool = True,
        bottom_val: Optional[float] = None,
        progress_bar: bool = False,
    ):
        lines_iterable = list(lines_iterable)
        n = len(lines_iterable)

        inputs_dict = {
            "min_samples": 200,
            "statistics": ["mi"],
            "restrictions": [restriction],
        }

        if errorbars:
            inputs_dict.update(
                {"uncertainty": {"mi": {"name": "subsampling", "args": {"n": 5}}}}
            )

        self.discrete_handler.update(
            x_names=lines_iterable,
            y_names=parameters,
            inputs_dict=inputs_dict,
            iterable_x=True,
            save_every=ceil(n / 10),
            progress_bar=progress_bar,
            total_iter=n,
        )

        entries = self.discrete_handler.read(
            x_names=lines_iterable,
            y_names=parameters,
            restr=restriction,
            iterable_x=True,
        )

        mis = [el["mi"]["value"] for el in entries]
        if errorbars:
            stds = [el["mi"]["std"] for el in entries]
        else:
            stds = None

        ax = self.plotter.plot_mi_bar(
            lines_iterable,
            mis,
            errs=stds,
            sort=sort,
            nfirst=nfirst,
            transitions=transitions,
            bottom_val=bottom_val,
        )

        return ax

    def plot_info_matrix(
        self,
        lines_iterable: Iterable[List[str]],
        parameters: Union[str, List[str]],
        restriction: str,
        show_diag: bool = False,
        transitions: bool = True,
        progress_bar: bool = False,
    ):
        lines_iterable = list(lines_iterable)
        n = len(lines_iterable)

        inputs_dict = {
            "min_samples": 200,
            "statistics": ["mi"],
            "restrictions": [restriction],
        }

        self.discrete_handler.update(
            x_names=itt.combinations(lines_iterable, 2),
            y_names=parameters,
            inputs_dict=inputs_dict,
            iterable_x=True,
            save_every=ceil(n / 10),
            progress_bar=progress_bar,
            total_iter=n,
        )

        mis_mat = np.zeros((len(lines_iterable), len(lines_iterable))).tolist()
        for l1, l2 in itt.combinations_with_replacement(lines_iterable, 2):

            entry = self.discrete_handler.read(
                x_names=[l1, l2] if l1 != l2 else [l1],
                y_names=parameters,
                restr=restriction,
            )

            i, j = lines_iterable.index(l1), lines_iterable.index(l2)

            mis_mat[i][j] = entry["mi"]["value"]
            mis_mat[j][i] = entry["mi"]["value"]

        ax = self.plotter.plot_mi_matrix(
            lines_iterable,
            mis_mat,
            show_diag=show_diag,
            transitions=transitions,
        )

        return ax

    def plot_info_bars_comparison(
        self,
        ref: "Infobs",
        lines_iterable: Iterable[List[str]],
        parameters: Union[str, List[str]],
        restriction: str,
        errorbars: bool = True,
        sort: bool = True,  # TODO
        nfirst: Optional[int] = None,  # TODO
        labels: Optional[List[str]] = None,
        transitions: bool = True,
        bottom_val: Optional[float] = None,
        progress_bar: bool = False,
        fontsize: int = 24,
    ):
        lines_iterable = list(lines_iterable)
        n = len(lines_iterable)

        inputs_dict = {
            "min_samples": 200,
            "statistics": ["mi"],
            "restrictions": [restriction],
        }

        if errorbars:
            inputs_dict.update(
                {"uncertainty": {"mi": {"name": "subsampling", "args": {"n": 5}}}}
            )

        mis = {}
        if errorbars:
            stds = {}

        for i, infobs in enumerate([self, ref]):
            infobs.discrete_handler.update(
                x_names=lines_iterable,
                y_names=parameters,
                inputs_dict=inputs_dict,
                iterable_x=True,
                save_every=ceil(n / 10),
                progress_bar=progress_bar,
                total_iter=n,
            )

            entries = infobs.discrete_handler.read(
                x_names=lines_iterable,
                y_names=parameters,
                restr=restriction,
                iterable_x=True,
            )

            mis[i] = [el["mi"]["value"] for el in entries]
            if errorbars:
                stds[i] = [el["mi"]["std"] for el in entries]

        if labels is not None:
            labels = {0: labels[0], 1: labels[1]}

        # TODO: ajouter sort et nfirst à plotter.plot_mi_bar_comparison
        ax = self.plotter.plot_mi_bar_comparison(
            lines_iterable,
            mis,
            stds,
            labels=labels,
            transitions=transitions,
            bottom_val=bottom_val,
            show_legend=labels is not None,
            fontsize=fontsize,
        )

        return ax

    def compute_map(
        self,
        lines: Union[str, List[str]],
        parameters: Union[str, List[str]],
        xaxis_param: str,
        yaxis_param: str,
        points: int = 50,
        progress_bar: bool = True,  # TODO
    ) -> None:
        win_length = {"Av": 2, "G0": 10, "Pth": 10, "angle": 10}

        inputs = {
            "windows": [
                {
                    "features": [xaxis_param, yaxis_param],
                    "bounds": [
                        MeudonPDR.bounds[xaxis_param],
                        MeudonPDR.bounds[yaxis_param],
                    ],
                    "bounds_include_windows": True,
                    "scale": [
                        PDRPlotter.param_scales[xaxis_param],
                        PDRPlotter.param_scales[yaxis_param],
                    ],
                    "length": [win_length[xaxis_param], win_length[yaxis_param]],
                    "points": [points, points],
                }
            ],
            "min_samples": 200,
            "max_samples": 20_000,
            "statistics": ["mi"],
        }

        self.continuous_handler.update(lines, parameters, inputs)

    def get_continuous_max(
        self,
        lines_iterable: Iterable[List[str]],
        parameters: Union[str, List[str]],
        axis_params: Union[str, List[str]],
    ) -> float:

        l = []
        for lines in lines_iterable:
            entry = self.continuous_handler.read(lines, parameters, axis_params)
            l.append(np.nanmax(entry["stats"]["mi"]["data"]))

        return np.nanmax(l)

    def plot_info_map(
        self,
        lines: Union[str, List[str]],
        parameters: Union[str, List[str]],
        xaxis_param: str,
        yaxis_param: str,
        cmap: str = "jet",
        vmax: Optional[float] = None,
    ) -> Axes:
        entry = self.continuous_handler.read(
            lines, parameters, [xaxis_param, yaxis_param]
        )

        mat = entry["stats"]["mi"]["data"].T
        xticks, yticks = entry["stats"]["mi"]["coords"]
        paramx, paramy = entry["features"]

        ax = self.plotter.plot_mi_map(
            xticks,
            yticks,
            mat,
            paramx=paramx,
            paramy=paramy,
            vmax=vmax,
            cmap=cmap,
        )

        return ax
