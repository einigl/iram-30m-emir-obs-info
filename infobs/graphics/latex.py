from typing import Union

from ism_lines_helpers import Settings, molecule, molecule_to_latex, line_to_latex, remove_hyperfine

Settings.math_mode = False

def _latex_line_fir(line: str, short: bool=False) -> Union[str, None]:
    """
    TODO
    """
    if line == "cp_el2p_j3_2__el2p_j1_2":
        mol = "[CII]"
        lamda = 158
    
    elif line == "c_el3p_j1__el3p_j0":
        mol = "[CI]"
        lamda = 609

    elif line == "c_el3p_j2__el3p_j1":
        mol = "[CI]"
        lamda = 370
    
    else:
        return None

    if short:
        s = f"\\mathrm{{{mol}}}"
    else:
        s = f"\\mathrm{{{mol}}}\\,{lamda}\\,\\mathrm{{\\mu m}}"

    if Settings.math_mode:
        return "$" + s + "$"
    return s


def latex_line(line: str, short: bool=False) -> str:
    """ Returns a printable LaTeX version of the line `line_name` (without degenerate energy levels).
    If `short` is True, the transition is indicated, else it isn't."""

    res = _latex_line_fir(line, short)
    if res is not None:
        return res

    if short:
        return molecule_to_latex(molecule(line))
    return line_to_latex(remove_hyperfine(line))

def latex_param(param: str) -> str:
    """ Returns a printable latex version of the physical parameter `param`. """

    param = param.strip().lower()

    if param == 'radm' :
        s = 'radm'
    elif param == 'guv':
        s = 'G_0'
    elif param == 'avmax' :
        s = 'A_V^{\\mathrm{tot}}'
    elif param == 'p' :
        s = 'P_{\\mathrm{th}}'
    elif param == 'kappa' :
        s = '\\kappa'
    else:
        # By default, returns the input without raising an error
        return param
    
    if Settings.math_mode:
        return "$" + s + "$"
    return s
