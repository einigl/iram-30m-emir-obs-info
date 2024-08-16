from typing import Optional, Union

__all__ = [
    "radm_to_g0",
    "g0_to_radm",
    "erg_to_kelvin",
    "kelvin_to_erg",
    "integrate_noise"
]

def radm_to_g0(radm: float) -> float:
    """converts radm to G0

    Parameters
    ----------
    radm : float
        input parameter of the Meudon PDR code

    Returns
    -------
    float
        G0 factor
    """
    conv_fact = 1.2786 / 2  # G0 = 1.2786 * radm / 2
    return radm * conv_fact

def g0_to_radm(g0: float) -> float:
    """converts G0 to radm 

    Parameters
    ----------
    G0 : float
        G0 factor

    Returns
    -------
    float
        input parameter of the Meudon PDR code, radm
    """
    conv_fact = 1.2786 / 2  # G0 = 1.2786 * radm / 2
    return g0 / conv_fact

def kelvin_to_erg(I: float, nu: float) -> float:
    """Conversion of integrated intensity from radio astronomy [K.km.s^-1] to physics [erg cm-2 s-1 sr-1] conventions.

    [T]: K.km.s^-1
    [nu]: Hz
    """
    kb = 1.380_649e-23 # m^2.kg.s^-2.K^-1
    c = 299_792_458 # m.s^-1
    return (2 * 1e6 * (nu/c)**3 * kb) * I

def erg_to_kelvin(I: float, nu: float) -> float:
    """Conversion of integrated intensity from physics [erg cm-2 s-1 sr-1] to radio astronomy [K.km.s^-1] conventions.

    [I]: erg cm-2 s-1 sr-1
    [nu]: Hz
    """
    kb = 1.380_649e-23 # m^2.kg.s^-2.K^-1
    c = 299_792_458 # m.s^-1
    return I / (2 * 1e6 * (nu/c)**3 * kb)

def integrate_noise(rms: float, n: int, dv: Optional[float]=None) -> float:
    """Integrate noise intensity of RMS value `rms` over `n` velocity channels of width `dv`.
    The intensity is assumed to be independent of the velocity, which is generally appropriate in the case of small frequency excursions.

    If `dv` is omitted, we assume that `rms` is already integrated over a single velocity channel [K.km.s^-1].

    T: intensity [K] or integrated intensity [K.km.s^-1]
    n: number of velocity channel to integrate
    dv: velocity channel width [km.s] (optional)
    """
    return n**0.5 * rms * dv
