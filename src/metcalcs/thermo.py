from __future__ import annotations
import numpy as np
from .constants import Cpd, Lv0, VON_KARMAN, GRAVITY
from .validators import validate_inputs
from typing import Final, Union
import warnings


def actual_vapor_pressure(
    airtemp: Union[float, np.ndarray], rh: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate the actual vapor pressure from air temperature and relative humidity.

    Computes the actual vapor pressure (ea) by scaling the saturation vapor pressure (es)
    with the given relative humidity.

    Parameters
    ----------
    airtemp : float or numpy.ndarray
        Air temperature(s) in degrees Celsius (°C).
    rh : float or numpy.ndarray
        Relative humidity values in percent (%).

    Returns
    -------
    float or numpy.ndarray
        Actual vapor pressure(s) in Pascals (Pa).

    Notes
    -----
    - Saturation vapor pressure is calculated using `self.saturation_vapor_pressure`.
    - If input validation is enabled (`self.config.validate_inputs`), inputs are checked for validity.
    - Raises a `MeteoError` if relative humidity is outside the 0–100% range.
    - Formula: `ea = (rh / 100) × es`

    Examples
    --------
    >>> actual_vapor_pressure(25, 60)
    1900.09
    """
    airtemp = np.asarray(airtemp)
    rh = np.asarray(rh)

    if validate_inputs:
        validate_inputs(airtemp=airtemp, rh=rh)
        if np.any((rh < 0) | (rh > 100)):
            raise ValueError("Relative humidity must be between 0-100%")

    # Calculate saturation vapor pressure and actual vapor pressure
    es = saturation_vapor_pressure(airtemp)
    ea = rh / 100.0 * es

    return ea


def specific_heat(
    airtemp: Union[float, np.ndarray],
    rh: Union[float, np.ndarray],
    airpress: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate the specific heat capacity of moist air.

    This function computes the specific heat of air as a function of air temperature,
    relative humidity, and atmospheric pressure. The presence of water vapor modifies
    the specific heat relative to dry air.

    Parameters
    ----------
    airtemp : float or numpy.ndarray
        Air temperature(s) in degrees Celsius (°C).
    rh : float or numpy.ndarray
        Relative humidity values in percent (%).
    airpress : float or numpy.ndarray
        Atmospheric pressure values in Pascals (Pa).

    Returns
    -------
    float or numpy.ndarray
        Specific heat of moist air in joules per kilogram per kelvin (J kg⁻¹ K⁻¹).

    Notes
    -----
    - Internally calculates actual vapor pressure using `self.actual_vapor_pressure`.
    - If input validation is enabled (`self.config.validate_inputs`), the method checks inputs.
    - Based on a modified form of the specific heat formula accounting for humidity effects.
    - The dry air specific heat constant used is approximately 1004.52 J kg⁻¹ K⁻¹.
    - Calculations are vectorized for array input efficiency.
    """
    # Convert inputs to numpy arrays
    airtemp = np.asarray(airtemp)
    rh = np.asarray(rh)
    airpress = np.asarray(airpress)

    if validate_inputs:
        validate_inputs(airtemp=airtemp, rh=rh, airpress=airpress)

    # Calculate vapor pressures
    eact = actual_vapor_pressure(airtemp, rh)

    # Vectorized calculation
    cp = 0.24 * 4185.5 * (1 + 0.8 * (0.622 * eact / (airpress - eact)))
    return cp


def vapor_pressure_slope(airtemp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate the slope of the saturation vapor pressure curve with respect to air temperature.

    This function computes the rate of change of saturation vapor pressure with temperature,
    commonly used in evapotranspiration and meteorological calculations.

    Parameters
    ----------
    airtemp : float or numpy.ndarray
        Air temperature(s) in degrees Celsius (°C).

    Returns
    -------
    float or numpy.ndarray
        Slope of the saturation vapor pressure curve in Pascals per Kelvin (Pa K⁻¹).

    Notes
    -----
    - Internally calls `self.saturation_vapor_pressure` to compute saturation vapor pressure.
    - If input validation is enabled (`self.config.validate_inputs`), the method checks inputs.
    - Calculation is vectorized for efficient use with arrays of temperatures.
    - Based on the derivative of the Clausius-Clapeyron relation.
    """
    airtemp = np.asarray(airtemp)

    if validate_inputs:
        validate_inputs(airtemp=airtemp)

    # Calculate saturation vapor pressure
    es = saturation_vapor_pressure(airtemp)
    es_kpa = es / 1000.0

    # Vectorized calculation using numpy
    delta = es_kpa * 4098.0 / ((airtemp + 237.3) ** 2) * 1000
    return delta


def saturation_vapor_pressure(
    airtemp: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate saturation vapor pressure using the Buck (1981) equation.

    This method estimates the saturation vapor pressure over water and ice based
    on air temperature, blending the two formulations smoothly around 0°C.
    It provides results within approximately 0.05% accuracy compared to the
    full Goff-Gratch equation.

    Parameters
    ----------
    airtemp : float or numpy.ndarray
        Air temperature(s) in degrees Celsius (°C).

    Returns
    -------
    float or numpy.ndarray
        Saturation vapor pressure(s) in Pascals (Pa).

    Notes
    -----
    - Uses the Buck (1981) empirical equation for high accuracy and computational efficiency.
    - A smooth transition is applied between ice and water saturation vapor pressures around 0°C.
    - Reference saturation vapor pressure at 0°C is taken as 611.21 Pa.
    - Input validation is performed if `self.config.validate_inputs` is True.
    - A warning is issued if the temperature is outside the recommended range of -80°C to 50°C.

    References
    ----------
    Buck, A.L., 1981: New equations for computing vapor pressure and enhancement factor.
    Journal of Applied Meteorology, 20, 1527–1532.
    """

    airtemp = np.asarray(airtemp)

    # Input validation and warnings
    if validate_inputs:
        validate_inputs(airtemp=airtemp)

    # Check temperature range and issue warning
    TEMP_MIN = -80
    TEMP_MAX = 50
    if np.any(airtemp < TEMP_MIN) or np.any(airtemp > TEMP_MAX):
        warnings.warn(
            f"Temperature {airtemp}°C is outside recommended range "
            f"({TEMP_MIN} to {TEMP_MAX}°C). Results may be inaccurate.",
            UserWarning,
        )

    # Constants for Buck equation
    a_water = 17.502
    b_water = 240.97
    a_ice = 22.587
    b_ice = 273.86
    es0 = 611.21  # Reference vapor pressure at 0°C

    # For better numerical stability around 0°C, use a smooth transition
    # between ice and water formulations
    weight = 1 / (1 + np.exp(-2 * airtemp))  # Sigmoid function

    # Calculate both ice and water formulations
    es_water = es0 * np.exp(a_water * airtemp / (b_water + airtemp))
    es_ice = es0 * np.exp(a_ice * airtemp / (b_ice + airtemp))

    # Blend the two formulations smoothly
    es = es_ice * (1 - weight) + es_water * weight

    return es


def _saturation_vapor_pressure_ice(temp: np.ndarray) -> np.ndarray:
    """Calculate saturation vapor pressure over ice"""
    T = temp + 273.15
    return np.exp(
        (
            -9.09718 * (273.16 / T - 1.0)
            - 3.56654 * np.log10(273.16 / T)
            + 0.876793 * (1.0 - T / 273.16)
            + np.log10(6.1071)
        )
    )



def aerodynamic_resistance(
    z: float, z0: float, d: float, u: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Compute aerodynamic resistance to momentum transfer.

    Aerodynamic resistance is estimated based on the logarithmic wind profile,
    assuming neutral atmospheric stability.

    Parameters
    ----------
    z : float
        Height at which wind speed is measured, :math:`z` **[m]**.
    z0 : float
        Surface roughness length, :math:`z_0` **[m]**.
    d : float
        Displacement height (zero-plane displacement), :math:`d` **[m]**.
    u : float or ndarray
        Horizontal wind speed at height *z*, :math:`u` **[m s⁻¹]**.

    Returns
    -------
    ra : float or ndarray
        Aerodynamic resistance to momentum transfer **[s m⁻¹]**.
        Returns an array if *u* is an array; otherwise, returns a scalar.

    Raises
    ------
    ValueError
        If ``z <= d + z0`` (measurement height must exceed effective roughness height).

    Notes
    -----
    The aerodynamic resistance is calculated as

    .. math::

        r_a = \\frac{\\bigl(\\ln\\bigl(\\frac{z-d}{z_0}\\bigr)\\bigr)^2}{k^2 u},

    where

    * :math:`k \\approx 0.41` is the von Kármán constant (dimensionless).

    Assumptions:

    * The atmospheric flow is neutrally stratified (no buoyancy effects).
    * The surface is flat and horizontally homogeneous.

    """
    u = np.asarray(u)
    k = 0.41  # von Kármán constant

    if validate_inputs:
        if z <= (d + z0):
            raise ValueError("Measurement height must be greater than d + z0")
        validate_inputs(u=u)

    return (np.log((z - d) / z0)) ** 2 / (k ** 2 * u)

def h_tvardry(
    rho: Union[float, np.ndarray],
    cp: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    sigma_t: Union[float, np.ndarray],
    z: float,
    d: float = 0.0,
) -> Union[float, np.ndarray]:
    """
    Estimate the sensible heat flux using the **temperature-variance method for dry
    surfaces** (TVARDry).

    The algorithm relates high-frequency temperature fluctuations to turbulent heat
    transport following the formulation of De Bruin et al. (1992).

    Parameters
    ----------
    rho : float or ndarray
        Air density, :math:`\\rho` **[kg m⁻³]`.
    cp : float or ndarray
        Specific heat capacity of air at constant pressure, :math:`c_p`
        **[J kg⁻¹ K⁻¹]`.
    T : float or ndarray
        Mean air temperature **[°C]`.
    sigma_t : float or ndarray
        Standard deviation of detrended, high-frequency air-temperature
        fluctuations, :math:`\\sigma_T` **[°C]`.
    z : float
        Height of the temperature sensor above the ground surface, :math:`z` **[m]`.
    d : float, default 0.0
        Displacement height that accounts for vegetation or obstacles, :math:`d`
        **[m]**.

    Returns
    -------
    H : float or ndarray
        Sensible heat flux density **[W m⁻²]**.  The returned object has the same
        shape as *rho*, *cp*, *T*, and *sigma_t*.

    Raises
    ------
    ValueError
        If any of the following validation checks fails:

        * ``rho``, ``cp``, or ``sigma_t`` contain negative values.
        * ``z`` is not greater than ``d``.

    Notes
    -----
    The sensible heat flux is computed as

    .. math::

        H = \\rho c_p
            \\sqrt{\\left(\\frac{\\sigma_T}{C_1}\\right)^3
                   \\kappa g \\frac{z-d}{T + 273.15}\,C_2},

    where

    * :math:`\\kappa \\approx 0.41` (von Kármán constant),
    * :math:`g \\approx 9.81\\,\\text{m s⁻²}` (gravitational acceleration),
    * :math:`C_1 = 2.9` and :math:`C_2 = 28.4` are empirical constants
      calibrated by De Bruin et al. (1992).

    The formulation assumes horizontally homogeneous, dry terrain and
    near-neutral stability conditions.

    References
    ----------
    De Bruin, H. A. R., van den Hurk, B. J. J. M., & Kohsiek, W. (1992).
    **The scintillation method tested over a dry vineyard area**.
    *Boundary-Layer Meteorology, 62*(1), 89-106.
    """
    rho, cp, T, sigma_t = map(np.asarray, [rho, cp, T, sigma_t])

    if validate_inputs:
        validate_inputs(rho=rho, cp=cp, T=T, sigma_t=sigma_t)

    # Constants from De Bruin et al., 1992
    C1 = 2.9
    C2 = 28.4

    # Calculate sensible heat flux
    H = (
        rho
        * cp
        * np.sqrt(
            (sigma_t / C1) ** 3 * VON_KARMAN * GRAVITY * (z - d) / (T + 273.15) * C2
        )
    )

    return H

def psychrometric_constant(
    airtemp: Union[float, np.ndarray],
    rh: Union[float, np.ndarray],
    airpress: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate the psychrometric constant.

    The psychrometric constant relates the partial pressure of water vapor
    in air to the air temperature, accounting for the effects of humidity
    and atmospheric pressure.

    Parameters
    ----------
    airtemp : float or numpy.ndarray
        Air temperature(s) in degrees Celsius (°C).
    rh : float or numpy.ndarray
        Relative humidity values in percent (%).
    airpress : float or numpy.ndarray
        Air pressure values in Pascals (Pa).

    Returns
    -------
    float or numpy.ndarray
        Psychrometric constant(s) in Pascals per Kelvin (Pa K⁻¹).

    Notes
    -----
    - Specific heat is calculated using `self.specific_heat`.
    - Latent heat is calculated using `self.latent_heat`.
    - Formula: γ = (cp × airpress) / (0.622 × L)
      where cp is specific heat and L is latent heat of vaporization.
    - If input validation is enabled (`self.config.validate_inputs`),
      the method checks the validity of inputs.

    References
    ----------
    Bringfelt, B., 1986. The use of the Penman-Monteith equation for
    the calculation of the evaporation from a snow surface.
    """
    airtemp = np.asarray(airtemp)
    rh = np.asarray(rh)
    airpress = np.asarray(airpress)

    if validate_inputs:
        validate_inputs(airtemp=airtemp, rh=rh, airpress=airpress)

    # Calculate specific heat and latent heat
    cp = specific_heat(airtemp, rh, airpress)
    L = latent_heat(airtemp)

    # Calculate psychrometric constant
    gamma = cp * airpress / (0.622 * L)

    return gamma

def latent_heat(
    airtemp: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate the latent heat of vaporization of water as a function of air temperature.

    This function estimates the latent heat of vaporization (L) for water based on air temperature,
    valid within a typical meteorological range.

    Parameters
    ----------
    airtemp : float or numpy.ndarray
        Air temperature(s) in degrees Celsius (°C).

    Returns
    -------
    float or numpy.ndarray
        Latent heat of vaporization in joules per kilogram (J kg⁻¹).

    Notes
    -----
    - The calculation is based on a linear temperature dependence.
    - Valid for air temperatures in the range -40°C to +40°C.
    - If input validation is enabled (`self.config.validate_inputs`), the method checks input bounds.
    - Issues a warning if temperature is outside the recommended range.

    Formula
    -------
    L = 4185.5 × (751.78 - 0.5655 × (airtemp + 273.15))
    """
    airtemp = np.asarray(airtemp)

    if validate_inputs:
        validate_inputs(airtemp=airtemp)
        if np.any((airtemp < -40) | (airtemp > 40)):
            warnings.warn("Temperature outside recommended range (-40 to 40°C)")

    # Vectorized calculation
    L = 4185.5 * (751.78 - 0.5655 * (airtemp + 273.15))

    return L

def potential_temperature(
    airtemp: Union[float, np.ndarray],
    rh: Union[float, np.ndarray],
    airpress: Union[float, np.ndarray],
    ref_press: float = 100000.0,
) -> Union[float, np.ndarray]:
    """
    Calculate the potential temperature relative to a reference pressure level.

    Potential temperature is the temperature an air parcel would have if brought
    adiabatically to a reference pressure level. This is commonly used in
    meteorological and atmospheric stability analyses.

    Parameters
    ----------
    airtemp : float or numpy.ndarray
        Air temperature(s) in degrees Celsius (°C).
    rh : float or numpy.ndarray
        Relative humidity values in percent (%).
    airpress : float or numpy.ndarray
        Air pressure values in Pascals (Pa).
    ref_press : float, optional
        Reference pressure level in Pascals (Pa). Default is 100000 Pa (1000 hPa).

    Returns
    -------
    float or numpy.ndarray
        Potential temperature(s) in degrees Celsius (°C).

    Notes
    -----
    - Specific heat is computed from `self.specific_heat`, accounting for humidity.
    - Temperature is internally converted to Kelvin and then converted back to Celsius.
    - Uses Poisson's equation:
      θ = T × (P₀ / P)^(R / cp), where
      θ = potential temperature [K],
      T = temperature [K],
      P₀ = reference pressure [Pa],
      P = observed pressure [Pa],
      R = specific gas constant for dry air (287 J kg⁻¹ K⁻¹),
      cp = specific heat [J kg⁻¹ K⁻¹].
    - If input validation is enabled (`self.config.validate_inputs`), inputs are checked.
    """
    airtemp = np.asarray(airtemp)
    rh = np.asarray(rh)
    airpress = np.asarray(airpress)

    if validate_inputs:
        validate_inputs(airtemp=airtemp, rh=rh, airpress=airpress)

    # Calculate specific heat
    cp = specific_heat(airtemp, rh, airpress)

    # Convert temperature to Kelvin for calculation
    T = airtemp + 273.15

    # Calculate potential temperature using Poisson's equation
    theta = T * (ref_press / airpress) ** (287.0 / cp)

    # Convert back to Celsius
    return theta - 273.15

def air_density(
    airtemp: Union[float, np.ndarray],
    rh: Union[float, np.ndarray],
    airpress: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate the density of moist air using the ideal gas law with a humidity correction.

    This method estimates air density by accounting for the presence of water vapor,
    which reduces the effective molecular weight of air.

    Parameters
    ----------
    airtemp : float or numpy.ndarray
        Air temperature(s) in degrees Celsius (°C).
    rh : float or numpy.ndarray
        Relative humidity values in percent (%).
    airpress : float or numpy.ndarray
        Atmospheric pressure in Pascals (Pa).

    Returns
    -------
    float or numpy.ndarray
        Air density in kilograms per cubic meter (kg m⁻³).

    Notes
    -----
    - Actual vapor pressure is computed using `self.actual_vapor_pressure`.
    - Air temperature is converted to Kelvin internally.
    - Moisture correction uses the factor `0.378 × ea`, where `ea` is the actual vapor pressure.
    - Based on the ideal gas law:
      ρ = (P - 0.378 × ea) / (R_d × T)
      where R_d = 287.05 J kg⁻¹ K⁻¹ is the gas constant for dry air.
    - If input validation is enabled (`self.config.validate_inputs`), inputs are validated.
    """
    airtemp = np.asarray(airtemp)
    rh = np.asarray(rh)
    airpress = np.asarray(airpress)

    if validate_inputs:
        validate_inputs(airtemp=airtemp, rh=rh, airpress=airpress)

    # Calculate actual vapor pressure
    ea = actual_vapor_pressure(airtemp, rh)

    # Convert temperature to Kelvin
    T = airtemp + 273.15

    # Calculate density with moisture correction
    rho = (airpress - 0.378 * ea) / (287.05 * T)

    return rho
