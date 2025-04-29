from __future__ import annotations
import numpy as np
from .constants import STEFAN_BOLTZMANN
from .validators import validate_inputs, to_array
from .thermo import vapor_pressure_slope, latent_heat, psychrometric_constant, actual_vapor_pressure, saturation_vapor_pressure
from typing import Final, Union, Tuple
import warnings

def canopy_gash(
    Pg: Union[float, np.ndarray], ER: float, S: float, p: float, pt: float, raise_warnings: bool = False
) -> Tuple[
    Union[float, np.ndarray],
    Union[float, np.ndarray],
    Union[float, np.ndarray],
    Union[float, np.ndarray],
]:
    """
    Calculate canopy water‐balance components with the Gash (1979) analytical interception model.

    The model partitions gross precipitation into throughfall, stemflow, and interception
    loss assuming (i) a constant mean evaporation rate from a wet canopy and
    (ii) instantaneous wetting of the canopy up to its storage capacity.

    Parameters
    ----------
    Pg : float or ndarray
        Gross precipitation for one or more rain events **[mm]**.
    ER : float
        Mean evaporation rate from a saturated canopy **[mm h⁻¹]**.
    S : float
        Canopy storage capacity (water retained when fully saturated) **[mm]**.
    p : float
        Free-throughfall coefficient
        Fraction of rainfall that passes the canopy without contact *(0 ≤ p ≤ 1)*.
    pt : float
        Stemflow coefficient
        Fraction of rainfall that drains down stems *(0 ≤ pt ≤ 1)*.
    raise_warnings : bool, optional
        Whether to raise warnings when an unexpected warning occurs.
        Defaults to False.

    Returns
    -------
    Pg : float or ndarray
        Echo of the input gross precipitation **[mm]**.
    TF : float or ndarray
        Throughfall reaching the ground between canopy gaps **[mm]**.
    SF : float or ndarray
        Stemflow delivered to the trunk bases **[mm]**.
    Ei : float or ndarray
        Interception loss (evaporation during and shortly after the storm) **[mm]**.

        All returned arrays have the same shape as *Pg*; scalars are returned when
        *Pg* is a scalar.

    Raises
    ------
    ValueError
        If any of the following conditions is violated:

        * ``p`` or ``pt`` lie outside the interval [0, 1].
        * ``p + pt > 1`` (fractions cannot exceed 100 % of rainfall).
        * ``S <= 0`` or ``ER <= 0``.

    Notes
    -----
    The rainfall depth that just saturates the canopy is

    :math:`PG_{sat} = -\\dfrac{S}{ER}\\,\\ln\\bigl[1-\\dfrac{ER}{1-p-pt}\\bigr]`.

    * When *Pg* < *PG\_sat*, the canopy does not saturate and interception is
      limited by available storage.
    * When *Pg* ≥ *PG\_sat*, the canopy saturates and further rainfall is partly
      evaporated at the rate *ER*.

    References
    ----------
    Gash, J. H. C. (1979). *An analytical model of rainfall interception by forests*.
    **Quarterly Journal of the Royal Meteorological Society, 105**, 43-55.
    """
    # Convert input to numpy array
    Pg = np.asarray(Pg)

    # Input validation
    if validate_inputs:
        validate_inputs(Pg=Pg, ER=ER, S=S)

        if not 0 <= p <= 1:
            raise ValueError(
                "Free throughfall coefficient (p) must be between 0 and 1"
            )
        if not 0 <= pt <= 1:
            raise ValueError("Stemflow coefficient (pt) must be between 0 and 1")
        if p + pt > 1:
            raise ValueError("Sum of p and pt must not exceed 1")
        if S <= 0:
            raise ValueError("Canopy storage capacity must be positive")
        if ER <= 0:
            raise ValueError("Evaporation rate must be positive")

    # Initialize output arrays
    rainfall_length = np.size(Pg)
    if rainfall_length < 2:
        # Single value case
        # Calculate saturation point (amount of rainfall needed to saturate canopy)
        PGsat = -(S / ER) * np.log((1 - (ER / (1 - p - pt))))

        # Calculate storages
        if Pg < PGsat and Pg > 0:
            # Case 1: Rainfall insufficient to saturate canopy
            canopy_storage = (1 - p - pt) * Pg
            trunk_storage = 0
            if Pg > canopy_storage / pt:
                trunk_storage = pt * Pg
        else:
            # Case 2: Rainfall sufficient to saturate canopy
            if Pg > 0:
                canopy_storage = (
                    ((1 - p - pt) * PGsat - S) + (ER * (Pg - PGsat)) + S
                )
                if Pg > (canopy_storage / pt):
                    trunk_storage = pt * Pg
                else:
                    trunk_storage = 0
            else:
                canopy_storage = 0
                trunk_storage = 0

        # Calculate components
        Ei = canopy_storage + trunk_storage
        TF = Pg - Ei
        SF = 0

    else:
        # Array case
        Ei = np.zeros(rainfall_length)
        TF = np.zeros(rainfall_length)
        SF = np.zeros(rainfall_length)
        PGsat = -(S / ER) * np.log((1 - (ER / (1 - p - pt))))

        # Calculate for each timestep
        for i in range(rainfall_length):
            if Pg[i] < PGsat and Pg[i] > 0:
                # Insufficient rainfall to saturate canopy
                canopy_storage = (1 - p - pt) * Pg[i]
                trunk_storage = 0
                if Pg[i] > canopy_storage / pt:
                    trunk_storage = pt * Pg[i]
            else:
                # Sufficient rainfall to saturate canopy
                if Pg[i] > 0:
                    canopy_storage = (
                        ((1 - p - pt) * PGsat - S) + (ER * (Pg[i] - PGsat)) + S
                    )
                    if Pg[i] > (canopy_storage / pt):
                        trunk_storage = pt * Pg[i]
                    else:
                        trunk_storage = 0
                else:
                    canopy_storage = 0
                    trunk_storage = 0

            Ei[i] = canopy_storage + trunk_storage
            TF[i] = Pg[i] - Ei[i]

    # Log warnings for potentially problematic values
    if raise_warnings:
        if np.any(Ei < 0):
            warnings.warn("Negative interception values detected")
        if np.any(TF < 0):
            warnings.warn("Negative throughfall values detected")
        if np.any(Ei > Pg):
            warnings.warn("Interception exceeds gross precipitation")

    return Pg, TF, SF, Ei


def _validate_gash_parameters(
    Pg: np.ndarray, ER: float, S: float, p: float, pt: float
) -> None:
    """Helper method to validate Gash model parameters"""
    try:
        if np.any(Pg < 0):
            raise ValueError("Gross precipitation cannot be negative")
        if ER <= 0:
            raise ValueError("Mean evaporation rate must be positive")
        if S <= 0:
            raise ValueError("Canopy storage capacity must be positive")
        if not 0 <= p <= 1:
            raise ValueError("Free throughfall coefficient must be between 0 and 1")
        if not 0 <= pt <= 1:
            raise ValueError("Stemflow coefficient must be between 0 and 1")
        if p + pt > 1:
            raise ValueError(
                "Sum of free throughfall and stemflow coefficients cannot exceed 1"
            )
    except Exception as e:
        warnings.warn(f"Parameter validation failed: {str(e)}")
        raise
