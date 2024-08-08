import shutil

import matplotlib.pyplot as plt

from ._ism_lines_helpers import Settings, molecule, molecule_to_latex, line_to_latex, remove_hyperfine

Settings.math_mode = False


__all__ = [
    "Settings",
    "LaTeX",
    "latex_line",
    "latex_param"
]

class LaTeX:
    """
    Class to handle activation of the plotting with latex if available on the current installation.

    Example:
    ```
    with LaTeX():
        # Do some matplotlib stuff
    ```
    """

    activate: bool
    previous_mode: bool

    def __init__(self, activate: bool = True):
        if not isinstance(activate, bool):
            raise TypeError(f"activate must be a boolean value, not {type(activate)}")
        self.activate = activate
        self.previous_mode = plt.rcParams["text.usetex"]

    def __enter__(self):
        if self.activate:
            plt.rc("text", usetex=shutil.which("latex") is not None)
        else:
            plt.rc("text", usetex=False)
        return self

    def __exit__(self, _, __, ___):
        plt.rc("text", usetex=self.previous_mode)

def latex_line(line: str, short: bool=False) -> str:
    """ Returns a printable LaTeX version of the line `line_name` (without degenerate energy levels).
    If `short` is True, the transition is indicated, else it isn't."""

    if short:
        return molecule_to_latex(molecule(line))
    return line_to_latex(remove_hyperfine(line))

def latex_param(param: str) -> str:
    """ Returns a printable latex version of the physical parameter `param`. """

    param = param.strip().lower()

    if param == 'g0':
        s = 'G_0'
    elif param == 'av':
        s = 'A_V^{\\mathrm{tot}}'
    elif param == 'pth':
        s = 'P_{\\mathrm{th}}'
    elif param == 'angle':
        s = '\\alpha'
    elif param == 'kappa':
        s = '\\kappa'
    else:
        # By default, returns the input without raising an error
        return param
    
    if Settings.math_mode:
        return "$" + s + "$"
    return s
