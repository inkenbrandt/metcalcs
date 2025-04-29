from __future__ import annotations
import numpy as np
from .constants import STEFAN_BOLTZMANN
from .validators import validate_inputs, to_array
from .thermo import vapor_pressure_slope, latent_heat, psychrometric_constant, actual_vapor_pressure, saturation_vapor_pressure
from typing import Final, Union
import warnings


def penman_open_water(
    airtemp: Union[float, np.ndarray],
    rh: Union[float, np.ndarray],
    airpress: Union[float, np.ndarray],
    Rs: Union[float, np.ndarray],
    Rext: Union[float, np.ndarray],
    u: Union[float, np.ndarray],
    alpha: float = 0.08,
    Z: float = 0.0,
) -> Union[float, np.ndarray]:
    """
    Calculate open water evaporation using the original Penman (1948, 1956) equation.

    This method estimates daily evaporation from open water bodies by combining
    energy balance and aerodynamic terms, accounting for site elevation and albedo.

    Parameters
    ----------
    airtemp : float or ndarray
        Daily average air temperature **[°C]**.
    rh : float or ndarray
        Daily average relative humidity **[%]**.
    airpress : float or ndarray
        Daily average atmospheric pressure **[Pa]**.
    Rs : float or ndarray
        Incoming shortwave solar radiation **[J m⁻² day⁻¹]**.
    Rext : float or ndarray
        Daily extraterrestrial radiation **[J m⁻² day⁻¹]**.
    u : float or ndarray
        Daily average wind speed measured at 2 m height **[m s⁻¹]**.
    alpha : float, optional
        Surface albedo (reflectance) **[-]**, default is 0.08 for open water.
    Z : float, optional
        Site elevation above sea level **[m]**, default is 0.

    Returns
    -------
    E0 : float or ndarray
        Open water evaporation **[mm day⁻¹]**.
        Shape matches the input arrays.

    Raises
    ------
    ValueError
        If input arrays have invalid values (e.g., negative radiation, humidity out of bounds).

    Notes
    -----
    The Penman equation for evaporation is:

    .. math::

        E_0 = \\frac{\\Delta}{\\Delta + \\gamma} \\frac{R_n}{\\lambda}
            + \\frac{\\gamma}{\\Delta + \\gamma} \\frac{6430000\\, E_a}{\\lambda},

    where

    * :math:`\\Delta` is the slope of the saturation vapor pressure curve **[kPa °C⁻¹]**,
    * :math:`\\gamma` is the psychrometric constant **[kPa °C⁻¹]**,
    * :math:`\\lambda` is the latent heat of vaporization **[J kg⁻¹]**,
    * :math:`R_n` is the net radiation **[J m⁻² day⁻¹]**,
    * :math:`E_a` is the aerodynamic term representing drying power of air,
      calculated as:

    .. math::

        E_a = (1 + 0.536 u) (e_s - e_a),

    where *u* is the 2-m wind speed **[m s⁻¹]**, and *eₛ*, *eₐ* are saturation and
    actual vapor pressures **[kPa]**, respectively.

    Clear-sky radiation is estimated as:

    .. math::

        R_{so} = (0.75 + 2 \\times 10^{-5} Z) R_{ext}.

    Longwave radiation losses are approximated using cloudiness and vapor pressure corrections.

    Assumptions:

    * Daily-averaged meteorological conditions represent the entire day.
    * Water surface is homogeneous and free of significant heat storage effects.
    * No significant heat advection across the water body is considered.

    References
    ----------
    Penman, H.L. (1948).
    *Natural evaporation from open water, bare soil and grass*.
    Proceedings of the Royal Society of London. Series A, Mathematical and Physical Sciences, 193(1032), 120–145.
    https://doi.org/10.1098/rspa.1948.0037

    Penman, H.L. (1956).
    *Evaporation: an introductory survey*.
    Netherlands Journal of Agricultural Science, 4(1), 9–29.
    """
    airtemp, rh, airpress, Rs, Rext, u = map(
        np.asarray, [airtemp, rh, airpress, Rs, Rext, u]
    )

    if validate_inputs:
        validate_inputs(
            airtemp=airtemp, rh=rh, airpress=airpress, Rs=Rs, Rext=Rext, u=u
        )

    # Calculate parameters
    Delta = vapor_pressure_slope(airtemp)
    gamma = psychrometric_constant(airtemp, rh, airpress)
    Lambda = latent_heat(airtemp)
    es = saturation_vapor_pressure(airtemp)
    ea = actual_vapor_pressure(airtemp, rh)

    # Calculate radiation components
    Rns = (1.0 - alpha) * Rs
    Rs0 = (0.75 + 2e-5 * Z) * Rext
    f = 1.35 * Rs / Rs0 - 0.35
    epsilon = 0.34 - 0.14 * np.sqrt(ea / 1000)
    Rnl = f * epsilon * STEFAN_BOLTZMANN * (airtemp + 273.15) ** 4
    Rnet = Rns - Rnl

    # Calculate evaporation terms
    Ea = (1 + 0.536 * u) * (es / 1000 - ea / 1000)
    E0 = (
        Delta / (Delta + gamma) * Rnet / Lambda
        + gamma / (Delta + gamma) * 6430000 * Ea / Lambda
    )

    return E0

def pet_makkink(
    self,
    airtemp: Union[float, np.ndarray],
    rh: Union[float, np.ndarray],
    airpress: Union[float, np.ndarray],
    Rs: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate potential evaporation using the Makkink (1965) equation.

    This method estimates evaporation based primarily on incoming solar radiation
    and temperature-dependent vapor pressure dynamics.

    Parameters
    ----------
    airtemp : float or ndarray
        Daily average air temperature **[°C]**.
    rh : float or ndarray
        Daily average relative humidity **[%]**.
    airpress : float or ndarray
        Daily average atmospheric pressure **[Pa]**.
    Rs : float or ndarray
        Incoming shortwave solar radiation **[J m⁻² day⁻¹]**.

    Returns
    -------
    Em : float or ndarray
        Potential evaporation according to Makkink **[mm day⁻¹]**.
        Output has the same shape as input arrays.

    Raises
    ------
    ValueError
        If input arrays are invalid (e.g., negative radiation).

    Notes
    -----
    The Makkink equation for estimating potential evaporation is:

    .. math::

        E_m = 0.65 \\frac{\\Delta}{\\Delta + \\gamma} \\frac{R_s}{\\lambda},

    where

    * :math:`\\Delta` is the slope of the saturation vapor pressure curve **[kPa °C⁻¹]**,
    * :math:`\\gamma` is the psychrometric constant **[kPa °C⁻¹]**,
    * :math:`\\lambda` is the latent heat of vaporization **[J kg⁻¹]**,
    * :math:`R_s` is the incoming solar radiation **[J m⁻² day⁻¹]**.

    Assumptions:

    * The soil heat flux and net longwave radiation losses are neglected.
    * The proportionality coefficient 0.65 is empirical, based on Dutch climatic conditions.

    The method is simpler than Penman-Monteith but less accurate under very humid or arid conditions.

    References
    ----------
    Makkink, G.F. (1965).
    *Evaporation and crop yields*.
    Institute of Land and Water Management Research, Wageningen, Netherlands.
    """
    airtemp, rh, airpress, Rs = map(np.asarray, [airtemp, rh, airpress, Rs])

    if self.config.validate_inputs:
        validate_inputs(airtemp=airtemp, rh=rh, airpress=airpress, Rs=Rs)

    Delta = self.vapor_pressure_slope(airtemp)
    gamma = self.psychrometric_constant(airtemp, rh, airpress)
    Lambda = self.latent_heat(airtemp)

    return 0.65 * Delta / (Delta + gamma) * Rs / Lambda

def pet_preistleytaylor(
    self,
    airtemp: Union[float, np.ndarray],
    rh: Union[float, np.ndarray],
    airpress: Union[float, np.ndarray],
    Rn: Union[float, np.ndarray],
    G: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Estimate potential evaporation using the Priestley–Taylor (1972) equation.

    The Priestley–Taylor model relates net available energy to potential evaporation,
    assuming minimal aerodynamic resistance and large, saturated surfaces.

    Parameters
    ----------
    airtemp : float or ndarray
        Daily average air temperature **[°C]**.
    rh : float or ndarray
        Daily average relative humidity **[%]**.
    airpress : float or ndarray
        Daily average atmospheric pressure **[Pa]**.
    Rn : float or ndarray
        Daily average net radiation **[J m⁻² day⁻¹]**.
    G : float or ndarray
        Daily average soil heat flux **[J m⁻² day⁻¹]**.

    Returns
    -------
    Ept : float or ndarray
        Potential evaporation according to Priestley–Taylor **[mm day⁻¹]**.
        Output has the same shape as input arrays.

    Raises
    ------
    ValueError
        If inputs are physically invalid (e.g., negative radiation).

    Notes
    -----
    The Priestley–Taylor equation for evaporation is:

    .. math::

        E_{pt} = \\alpha \\frac{\\Delta}{\\Delta + \\gamma} \\frac{(R_n - G)}{\\lambda},

    where

    * :math:`\\alpha = 1.26` is the Priestley–Taylor empirical coefficient,
    * :math:`\\Delta` is the slope of the saturation vapor pressure curve **[kPa °C⁻¹]**,
    * :math:`\\gamma` is the psychrometric constant **[kPa °C⁻¹]**,
    * :math:`\\lambda` is the latent heat of vaporization **[J kg⁻¹]**,
    * :math:`R_n` is the net radiation **[J m⁻² day⁻¹]**,
    * :math:`G` is the soil heat flux **[J m⁻² day⁻¹]**.

    Assumptions:

    * The surface is wet enough that aerodynamic effects are minimal.
    * Conditions approximate large, wet, well-watered surfaces under near-equilibrium conditions.

    References
    ----------
    Priestley, C.H.B., & Taylor, R.J. (1972).
    *On the assessment of surface heat flux and evaporation using large-scale parameters*.
    Monthly Weather Review, 100(2), 81–92.
    https://doi.org/10.1175/1520-0493(1972)100<0081:OTAOSH>2.3.CO;2
    """
    airtemp, rh, airpress, Rn, G = map(np.asarray, [airtemp, rh, airpress, Rn, G])

    if self.config.validate_inputs:
        validate_inputs(airtemp=airtemp, rh=rh, airpress=airpress, Rn=Rn, G=G)

    Delta = self.vapor_pressure_slope(airtemp)
    gamma = self.psychrometric_constant(airtemp, rh, airpress)
    Lambda = self.latent_heat(airtemp)

    return 1.26 * Delta / (Delta + gamma) * (Rn - G) / Lambda

def refet_penmanmonteith(
    self,
    airtemp: Union[float, np.ndarray],
    rh: Union[float, np.ndarray],
    airpress: Union[float, np.ndarray],
    Rs: Union[float, np.ndarray],
    Rext: Union[float, np.ndarray],
    u: Union[float, np.ndarray],
    Z: float = 0.0,
) -> Union[float, np.ndarray]:
    """
    Calculate reference evapotranspiration (ET₀) using the FAO Penman-Monteith equation.

    This method estimates ET₀ for a well-watered, actively growing short grass reference crop
    based on meteorological data following FAO-56 guidelines.

    Parameters
    ----------
    airtemp : float or ndarray
        Daily average air temperature **[°C]**.
    rh : float or ndarray
        Daily average relative humidity **[%]**.
    airpress : float or ndarray
        Daily average atmospheric pressure **[Pa]**.
    Rs : float or ndarray
        Incoming shortwave solar radiation **[J m⁻² day⁻¹]**.
    Rext : float or ndarray
        Extraterrestrial radiation (top of atmosphere) **[J m⁻² day⁻¹]**.
    u : float or ndarray
        Wind speed measured at 2 m height **[m s⁻¹]**.
    Z : float, optional
        Elevation above sea level **[m]**. Default is 0.

    Returns
    -------
    ET0 : float or ndarray
        Reference evapotranspiration for short grass **[mm day⁻¹]**.
        Output has the same shape as input arrays.

    Raises
    ------
    ValueError
        If any inputs are physically invalid or inconsistent (e.g., negative radiation or pressure).

    Notes
    -----
    The method implements the FAO-56 formulation:

    .. math::

        ET_0 = \\frac{0.408 \\Delta (R_n - G) + \\gamma \\frac{900}{T + 273.16} u_2 (e_s - e_a)}
                     {\\Delta + \\gamma (1 + 0.34 u_2)},

    with radiation units converted to equivalent latent heat units using

    .. math::

        R_n = R_{ns} - R_{nl},

    where

    * :math:`R_{ns} = (1 - \\alpha) R_s` is net shortwave radiation,
    * :math:`R_{nl}` is net longwave radiation estimated using cloudiness factor :math:`f`,
    * :math:`\\alpha = 0.23` is the surface albedo for short grass,
    * :math:`f = 1.35 \\frac{R_s}{R_{so}} - 0.35`, where :math:`R_{so}` is clear-sky radiation,
    * :math:`\\Delta` is the slope of the saturation vapor pressure curve **[kPa °C⁻¹]**,
    * :math:`\\gamma` is the psychrometric constant **[kPa °C⁻¹]**.

    Assumptions:

    * Net soil heat flux *G* is assumed negligible at the daily timescale.
    * Radiation values must be provided as energy flux integrated over the day (not W/m²).
    * Elevation *Z* is used to adjust clear-sky radiation.

    References
    ----------
    Allen, R.G., Pereira, L.S., Raes, D., & Smith, M. (1998).
    *Crop Evapotranspiration – Guidelines for Computing Crop Water Requirements*.
    FAO Irrigation and Drainage Paper 56. FAO, Rome.
    https://www.fao.org/3/x0490e/x0490e00.htm
    """
    airtemp, rh, airpress, Rs, Rext, u = map(
        np.asarray, [airtemp, rh, airpress, Rs, Rext, u]
    )

    if self.config.validate_inputs:
        validate_inputs(
            airtemp=airtemp, rh=rh, airpress=airpress, Rs=Rs, Rext=Rext, u=u
        )

    # Constants for short grass
    albedo = 0.23

    # Calculate parameters
    Delta = self.vapor_pressure_slope(airtemp)
    gamma = self.psychrometric_constant(airtemp, rh, airpress)
    Lambda = self.latent_heat(airtemp)
    es = self.saturation_vapor_pressure(airtemp)
    ea = self.actual_vapor_pressure(airtemp, rh)

    # Calculate radiation terms
    Rns = (1.0 - albedo) * Rs
    Rs0 = (0.75 + 2e-5 * Z) * Rext
    f = 1.35 * Rs / Rs0 - 0.35
    epsilon = 0.34 - 0.14 * np.sqrt(ea / 1000)
    Rnl = f * epsilon * STEFAN_BOLTZMANN * (airtemp + 273.15) ** 4
    Rnet = Rns - Rnl

    # Calculate ET0
    ET0 = (
              Delta / 1000.0 * Rnet / Lambda
              + 900.0 / (airtemp + 273.16) * u * (es - ea) / 1000 * gamma / 1000
          ) / (Delta / 1000.0 + gamma / 1000 * (1.0 + 0.34 * u))

    return ET0

def et_penmanmonteith(
    self,
    airtemp: Union[float, np.ndarray],
    rh: Union[float, np.ndarray],
    airpress: Union[float, np.ndarray],
    Rn: Union[float, np.ndarray],
    G: Union[float, np.ndarray],
    ra: Union[float, np.ndarray],
    rs: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate actual evapotranspiration using the Penman-Monteith equation.

    This method estimates evapotranspiration by combining energy balance and
    aerodynamic terms according to Monteith's extension (1965) of Penman's original model.

    Parameters
    ----------
    airtemp : float or ndarray
        Daily average air temperature **[°C]**.
    rh : float or ndarray
        Daily average relative humidity **[%]**.
    airpress : float or ndarray
        Daily average air pressure **[Pa]**.
    Rn : float or ndarray
        Daily average net radiation **[J m⁻² day⁻¹]**.
    G : float or ndarray
        Daily average soil heat flux density **[J m⁻² day⁻¹]**.
    ra : float or ndarray
        Aerodynamic resistance to vapor transport **[s m⁻¹]**.
    rs : float or ndarray
        Surface (canopy or soil) resistance to vapor transport **[s m⁻¹]**.

    Returns
    -------
    Epm : float or ndarray
        Actual evapotranspiration **[mm day⁻¹]**.
        Shape matches input arrays.

    Raises
    ------
    ValueError
        If any input arrays fail validation (e.g., negative resistance, invalid humidity).

    Notes
    -----
    The Penman-Monteith equation is given by:

    .. math::

        E_{pm} = \\frac{\\Delta (R_n - G) + \\rho c_p (e_s - e_a)/r_a}
                      {\\lambda \\left( \\Delta + \\gamma \\left(1 + \\frac{r_s}{r_a}\\right) \\right)},

    where

    * :math:`\\Delta` is the slope of the saturation vapor pressure curve **[kPa °C⁻¹]**,
    * :math:`\\gamma` is the psychrometric constant **[kPa °C⁻¹]**,
    * :math:`\\lambda` is the latent heat of vaporization **[J kg⁻¹]**,
    * :math:`\\rho` is the air density **[kg m⁻³]**,
    * :math:`c_p` is the specific heat of air at constant pressure **[J kg⁻¹ K⁻¹]**,
    * :math:`e_s` is the saturation vapor pressure **[kPa]**,
    * :math:`e_a` is the actual vapor pressure **[kPa]**,
    * :math:`r_a` is the aerodynamic resistance **[s m⁻¹]**,
    * :math:`r_s` is the surface resistance **[s m⁻¹]**.

    Assumptions:

    * Daily-averaged meteorological variables are representative of daytime conditions.
    * Net radiation and soil heat flux should be integrated over the day.

    References
    ----------
    Monteith, J.L. (1965). **Evaporation and environment**.
    In: *The State and Movement of Water in Living Organisms*, 19th Symposia of the Society
    for Experimental Biology. Cambridge University Press, pp. 205–234.
    """
    airtemp, rh, airpress, Rn, G, ra, rs = map(
        np.asarray, [airtemp, rh, airpress, Rn, G, ra, rs]
    )

    if self.config.validate_inputs:
        validate_inputs(
            airtemp=airtemp, rh=rh, airpress=airpress, Rn=Rn, G=G, ra=ra, rs=rs
        )

    # Calculate parameters
    Delta = self.vapor_pressure_slope(airtemp) / 100.0  # [hPa/K]
    gamma = self.psychrometric_constant(airtemp, rh, airpress) / 100.0
    Lambda = self.latent_heat(airtemp)
    rho = self.air_density(airtemp, rh, airpress)
    cp = self.specific_heat(airtemp, rh, airpress)
    es = self.saturation_vapor_pressure(airtemp) / 100.0
    ea = self.actual_vapor_pressure(airtemp, rh) / 100.0

    # Calculate evaporation
    Epm = (
              (Delta * Rn + rho * cp * (es - ea) / ra) / (Delta + gamma * (1.0 + rs / ra))
          ) / Lambda

    return Epm
