from __future__ import annotations
import numpy as np
from .constants import SOLAR_CONSTANT, STEFAN_BOLTZMANN, DAYTIME_RANGE, NIGHTTIME_RANGE
from .validators import validate_inputs, to_array
from .thermo import actual_vapor_pressure
from typing import Final, Union, Dict
import warnings

def solar_parameters(doy: Union[float, np.ndarray], lat: float) -> (float,float):
    """
    Calculate solar parameters: maximum sunshine duration and extraterrestrial radiation.

    Computes the theoretical maximum daylight hours and daily extraterrestrial radiation
    at the top of the atmosphere for a given day of year and latitude, using standard
    astronomical formulas.

    Parameters
    ----------
    doy : float or numpy.ndarray
        Day of year (1–366).
    lat : float
        Latitude in degrees (positive for Northern Hemisphere, negative for Southern Hemisphere).

    Returns
    -------
    SolarResults
        A named tuple or dataclass containing:
        - sunshine_duration : float or numpy.ndarray
            Maximum possible sunshine duration in hours.
        - extraterrestrial_radiation : float or numpy.ndarray
            Daily extraterrestrial solar radiation at the top of the atmosphere in joules per day (J day⁻¹).

    Notes
    -----
    - Valid for latitudes between -67° and +67°; a warning is issued outside this range.
    - Raises `MeteoError` if `doy` is not between 1 and 366.
    - Solar constant is assumed to be 1367 W m⁻².
    - Extraterrestrial radiation is calculated using FAO-56 standard approach.
    - Sunshine duration and radiation are computed using declination, sunset hour angle,
        and Earth–Sun distance correction.

    References
    ----------
    - Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). Crop Evapotranspiration:
        Guidelines for Computing Crop Water Requirements. FAO Irrigation and Drainage Paper 56.
    """
    doy = np.asarray(doy)

    if validate_inputs:
        validate_inputs(doy=doy)
        if abs(lat) > 67:
            warnings.warn("Latitude outside valid range (-67° to +67°)")
        if np.any((doy < 1) | (doy > 366)):
            raise ValueError("Day of year must be between 1 and 366")

    # Convert latitude to radians
    lat_rad = np.radians(lat)

    # Solar constant [W m⁻²]
    S0 = 1367.0

    # Calculate solar declination [radians]
    decl = 0.409 * np.sin(2 * np.pi / 365 * doy - 1.39)

    # Calculate sunset hour angle [radians]
    ws = np.arccos(-np.tan(lat_rad) * np.tan(decl))

    # Calculate maximum sunshine duration [hours]
    sunshine_duration = 24 / np.pi * ws

    # Calculate relative distance to sun
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.25)

    # Calculate extraterrestrial radiation [J day⁻¹]
    extraterrestrial_radiation = (
        S0
        * 86400
        / np.pi
        * dr
        * (
            ws * np.sin(lat_rad) * np.sin(decl)
            + np.cos(lat_rad) * np.cos(decl) * np.sin(ws)
        )
    )

    return sunshine_duration, extraterrestrial_radiation

def calculate_radiation(
    airtemp: Union[float, np.ndarray],
    rh: Union[float, np.ndarray],
    rs: Union[float, np.ndarray],
    lat: float,
    doy: int,
    albedo: float = 0.23,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate radiation components at the surface and top of the atmosphere.

    Computes net shortwave radiation, net longwave radiation, net radiation, and
    extraterrestrial radiation using standard meteorological input data.

    Parameters
    ----------
    airtemp : float or numpy.ndarray
        Air temperature(s) in degrees Celsius (°C).
    rh : float or numpy.ndarray
        Relative humidity in percent (%).
    rs : float or numpy.ndarray
        Incoming shortwave radiation (solar radiation) in watts per square meter (W m⁻²).
    lat : float
        Latitude in degrees. Positive for Northern Hemisphere, negative for Southern.
    doy : int
        Day of year (1–366).
    albedo : float, optional
        Surface albedo (dimensionless reflectivity), default is 0.23 (typical for grass).

    Returns
    -------
    dict of str to float or numpy.ndarray
        Dictionary containing the following radiation components (in W m⁻²):
        - 'extraterrestrial': Incoming solar radiation at the top of the atmosphere
        - 'net_shortwave'  : Net shortwave radiation (absorbed solar)
        - 'net_longwave'   : Net outgoing longwave radiation
        - 'net_radiation'  : Net total radiation (shortwave - longwave)

    Notes
    -----
    - Extraterrestrial radiation is computed using the FAO-56 method.
    - Net shortwave is calculated as (1 - albedo) × rs.
    - Net longwave uses a simplified Brutsaert-style emissivity formulation.
    - Temperature is internally converted to Kelvin for Stefan-Boltzmann computation.
    - Radiation components are commonly used in evapotranspiration and energy balance models.
    - If input validation is enabled (`self.config.validate_inputs`), inputs are checked.

    References
    ----------
    - Allen et al., 1998. FAO Irrigation and Drainage Paper No. 56.
    - Brutsaert, W. (1975). On a Derivable Formula for Longwave Radiation from Clear Skies.
    """
    airtemp, rh, rs = to_array(airtemp, rh, rs)

    if validate_inputs:
        validate_inputs(airtemp=airtemp, rh=rh, rs=rs)

    # Calculate extraterrestrial radiation
    ra = extraterrestrial_radiation(lat, doy)

    # Net shortwave
    rns = (1 - albedo) * rs

    # Net longwave using improved formulation
    ea = actual_vapor_pressure(airtemp, rh)
    tk = airtemp + 273.15
    rnl = (
        STEFAN_BOLTZMANN
        * tk**4
        * (0.34 - 0.14 * np.sqrt(ea / 1000))
        * (1.35 * rs / ra - 0.35)
    )

    # Net radiation
    rn = rns - rnl

    return {
        "extraterrestrial": ra,
        "net_shortwave": rns,
        "net_longwave": rnl,
        "net_radiation": rn,
    }

def extraterrestrial_radiation(lat: float, doy: int) -> float:
    """
    Calculate daily extraterrestrial radiation at the top of the atmosphere.

    Extraterrestrial radiation is the solar radiation received on a horizontal surface
    outside the Earth's atmosphere, accounting for Earth's tilt and orbital position.

    Parameters
    ----------
    lat : float
        Latitude of the location **[degrees]**.
        Positive for the Northern Hemisphere, negative for the Southern Hemisphere.
    doy : int
        Day of year (1 = January 1st, 365 = December 31st, or 366 for leap years).

    Returns
    -------
    ra : float
        Daily extraterrestrial radiation **[MJ m⁻² day⁻¹]**.

    Notes
    -----
    The calculation follows FAO-56 guidelines and involves:

    * Solar declination:

        .. math::

            \\delta = 0.409 \\sin\\left( \\frac{2 \\pi \\text{DOY}}{365} - 1.39 \\right)

    * Inverse relative distance Earth–Sun:

        .. math::

            d_r = 1 + 0.033 \\cos\\left( \\frac{2 \\pi \\text{DOY}}{365} \\right)

    * Sunset hour angle:

        .. math::

            \\omega_s = \\arccos\\left( -\\tan(\\phi) \\tan(\\delta) \\right)

    * Daily extraterrestrial radiation:

        .. math::

            R_a = \\frac{24 \\times 60}{\\pi} G_{sc} d_r
            \\left( \\omega_s \\sin(\\phi) \\sin(\\delta)
                + \\cos(\\phi) \\cos(\\delta) \\sin(\\omega_s) \\right)

    where:

    * :math:`\\phi` is latitude **[radians]**,
    * :math:`G_{sc}` is the solar constant ≈ 0.082 MJ m⁻² min⁻¹.

    Assumptions:

    * A standard 365-day year is assumed (small errors may occur in leap years).
    * Atmospheric effects (e.g., scattering, absorption) are not considered.

    References
    ----------
    Allen, R.G., Pereira, L.S., Raes, D., & Smith, M. (1998).
    *Crop Evapotranspiration – Guidelines for Computing Crop Water Requirements*.
    FAO Irrigation and Drainage Paper 56. FAO, Rome.
    https://www.fao.org/3/x0490e/x0490e00.htm
    """
    lat_rad = np.radians(lat)

    # Solar declination
    decl = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)

    # Inverse relative distance Earth-Sun
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)

    # Sunset hour angle
    ws = np.arccos(-np.tan(lat_rad) * np.tan(decl))

    # Extraterrestrial radiation
    ra = (
        24
        * 60
        / np.pi
        * 0.082
        * dr
        * (
            ws * np.sin(lat_rad) * np.sin(decl)
            + np.cos(lat_rad) * np.cos(decl) * np.sin(ws)
        )
    )

    return ra

def solar_declination(day_of_year):
    """
    Calculate solar declination angle based on day of the year.

    The solar declination is the angle between the rays of the sun and the plane of the Earth's equator.
    It varies throughout the year due to the tilt of the Earth's rotational axis.

    Parameters
    ----------
    day_of_year : int or array-like
        Day of year (1 = Jan 1, 365 = Dec 31; 366 for leap years).

    Returns
    -------
    decl : float or ndarray
        Solar declination angle **[radians]**. Returns an array if input is array-like.

    Notes
    -----
    The approximation used is:

    .. math::

        \\delta = 23.44^\\circ \\cos\\left(\\frac{360}{365}(\\text{DOY} + 10)\\right),

    converted from degrees to radians.

    This is a simplified cosine-based model and is accurate to within ~1 degree.
    More accurate models use trigonometric expansions or solar ephemeris data.

    References
    ----------
    Spencer, J.W. (1971).
    *Fourier series representation of the position of the sun*.
    Search, 2(5), 172.
    """
    return 23.44 * np.cos(np.radians((360 / 365) * (day_of_year + 10))) * np.pi / 180

def solar_hour_angle(longitude, time_utc):
    """
    Calculate the solar hour angle at a given location and UTC time.

    The solar hour angle represents the angular displacement of the sun from the local solar noon.
    It is negative in the morning and positive in the afternoon.

    Parameters
    ----------
    longitude : float
        Longitude of the location in **[degrees]**.
        Positive for east of the prime meridian, negative for west.
    time_utc : datetime.datetime
        UTC time for which to compute the solar hour angle.

    Returns
    -------
    hour_angle : float
        Solar hour angle in **[radians]**.

    Notes
    -----
    The calculation assumes a simplified solar time correction using the standard meridian:

    * Local Standard Time Meridian (LSTM) is computed as:

      .. math::
         LSTM = 15^\circ \\times \\text{round}(\\text{longitude} / 15)

    * Solar time offset:

      .. math::
         \\text{offset} = 4 \\times (\\text{longitude} - LSTM) \\, [\\text{minutes}]

    * Solar time (in hours) is computed by adjusting UTC time with this offset:

      .. math::
         \\text{solar time} = \\text{UTC}_{\\text{decimal}} + \\frac{\\text{offset}}{60}

    * The hour angle is then:

      .. math::
         H = 15 \\times (\\text{solar time} - 12) \\, [\\text{degrees}]

    which is converted to radians.

    Assumptions:

    * This simplified form does not account for the Equation of Time (EoT).
    * Accurate within a few minutes depending on location and time of year.

    References
    ----------
    Duffie, J.A., & Beckman, W.A. (2013).
    *Solar Engineering of Thermal Processes* (4th ed.). Wiley.
    """
    # Convert time to fractional hours
    time_decimal = time_utc.hour + time_utc.minute / 60 + time_utc.second / 3600
    # Solar time correction (simplified, assumes standard meridian)
    lstm = 15 * round(longitude / 15)  # Standard meridian correction
    time_offset = 4 * (longitude - lstm)  # Minutes offset
    solar_time = time_decimal + time_offset / 60  # Adjusted time
    return np.radians(15 * (solar_time - 12))  # Convert to radians

def solar_zenith_angle(latitude, declination, hour_angle):
    """
    Calculate the solar zenith angle.

    The solar zenith angle is the angle between the local vertical and the line to the sun.
    It is 0° when the sun is directly overhead and increases as the sun moves toward the horizon.

    Parameters
    ----------
    latitude : float
        Latitude of the location in **[degrees]**.
        Positive for Northern Hemisphere, negative for Southern Hemisphere.
    declination : float
        Solar declination angle **[radians]**.
    hour_angle : float
        Solar hour angle **[radians]**.

    Returns
    -------
    zenith_angle : float
        Solar zenith angle **[radians]**.

    Notes
    -----
    The zenith angle is calculated using the spherical solar geometry equation:

    .. math::

        \\cos(\\theta_z) = \\sin(\\phi) \\sin(\\delta)
                         + \\cos(\\phi) \\cos(\\delta) \\cos(H)

    where:

    * :math:`\\theta_z` is the solar zenith angle,
    * :math:`\\phi` is the latitude (converted to radians),
    * :math:`\\delta` is the solar declination angle,
    * :math:`H` is the solar hour angle.

    Assumptions:

    * Earth's atmosphere and terrain are not considered.
    * Inputs should be in radians for angular parameters (except latitude).

    References
    ----------
    Duffie, J.A., & Beckman, W.A. (2013).
    *Solar Engineering of Thermal Processes* (4th ed.). Wiley.
    """
    latitude_rad = np.radians(latitude)
    return np.arccos(
        np.sin(latitude_rad) * np.sin(declination)
        + np.cos(latitude_rad) * np.cos(declination) * np.cos(hour_angle)
    )

def extraterrestrial_solar_radiation(day_of_year):
    """
    Calculate daily mean extraterrestrial solar radiation at the top of the atmosphere.

    This represents the solar irradiance received on a plane perpendicular to the Sun’s rays
    at the top of Earth's atmosphere, averaged over the day.

    Parameters
    ----------
    day_of_year : int or array-like
        Day of year (1 = Jan 1, 365 = Dec 31; 366 for leap years).

    Returns
    -------
    rad : float or ndarray
        Extraterrestrial solar radiation **[W m⁻²]**.

    Notes
    -----
    The calculation uses a simplified correction for Earth–Sun distance variation:

    .. math::

        G_{on} = G_{sc} \\left(1 + 0.033 \\cos\\left(\\frac{2\\pi d}{365}\\right)\\right),

    where:

    * :math:`G_{on}` is the extraterrestrial radiation **[W m⁻²]**,
    * :math:`G_{sc} = 1361` is the solar constant **[W m⁻²]**,
    * :math:`d` is the day of year.

    Assumptions:

    * Earth's orbit is elliptical, with maximum deviation ~3.3% from mean distance.
    * This function gives instantaneous average solar irradiance over the day.

    References
    ----------
    Duffie, J.A., & Beckman, W.A. (2013).
    *Solar Engineering of Thermal Processes* (4th ed.). Wiley.
    """
    solar_constant = 1361  # W/m²
    eccentricity_correction = 1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365)
    return solar_constant * eccentricity_correction

def atmospheric_transmissivity(zenith_angle):
    """
    Estimate atmospheric transmissivity based on solar zenith angle.

    This function provides a simplified empirical model for direct-beam atmospheric
    transmittance under clear-sky conditions, as a function of solar zenith angle.

    Parameters
    ----------
    zenith_angle : float
        Solar zenith angle **[radians]**.

    Returns
    -------
    tau : float
        Atmospheric transmissivity **[-]** (range: 0–1).

    Notes
    -----
    The model assumes exponential attenuation of solar radiation through the atmosphere:

    .. math::

        \\tau = 0.75 \\cdot \\exp\\left(-\\frac{0.15}{\\cos(\\theta_z)}\\right)

    where :math:`\\theta_z` is the solar zenith angle in radians.

    * If the sun is below the horizon (:math:`\\theta_z > 90^\\circ`), transmissivity is zero.
    * This approximation ignores humidity, aerosols, and elevation.

    References
    ----------
    Iqbal, M. (1983).
    *An Introduction to Solar Radiation*. Academic Press.

    Gueymard, C.A. (2000).
    *Parameterized transmittance model for direct beam and global solar irradiance*.
    Solar Energy, 71(5), 325–346.
    """
    if zenith_angle > np.pi / 2:  # Sun below the horizon
        return 0
    return max(
        0.75 * np.exp(-0.15 / np.cos(zenith_angle)), 0
    )  # Simple attenuation model

def incoming_solar_radiation(latitude, longitude, datetime_utc):
    """
    Estimate incoming shortwave solar radiation at the surface under clear-sky conditions.

    This function combines simplified solar geometry, extraterrestrial irradiance, and
    empirical atmospheric attenuation to estimate global horizontal irradiance (GHI).

    Parameters
    ----------
    latitude : float
        Site latitude in **[degrees]**.
    longitude : float
        Site longitude in **[degrees]** (used for hour angle calculation).
    datetime_utc : datetime.datetime
        UTC datetime of interest.

    Returns
    -------
    radiation : float
        Incoming shortwave solar radiation **[W m⁻²]** under clear-sky conditions.

    Notes
    -----
    This function calls:

    * `solar_declination()` – computes solar declination from day of year.
    * `solar_hour_angle()` – computes hour angle from time and longitude.
    * `solar_zenith_angle()` – computes solar zenith angle from position and time.
    * `extraterrestrial_solar_radiation()` – computes top-of-atmosphere irradiance.
    * `atmospheric_transmissivity()` – estimates transmissivity based on zenith angle.

    Final estimate:

    .. math::

        R_s = G_{on} \\cdot \\tau

    where:

    * :math:`G_{on}` is the extraterrestrial radiation **[W m⁻²]**,
    * :math:`\\tau` is atmospheric transmissivity **[-]**.

    Assumptions:

    * Atmosphere is clear and uniform.
    * Terrain and solar time corrections (e.g., Equation of Time) are ignored.

    References
    ----------
    Iqbal, M. (1983).
    *An Introduction to Solar Radiation*. Academic Press.

    Duffie, J.A., & Beckman, W.A. (2013).
    *Solar Engineering of Thermal Processes* (4th ed.). Wiley.
    """
    day_of_year = datetime_utc.timetuple().tm_yday
    declination = solar_declination(day_of_year)
    hour_angle = solar_hour_angle(longitude, datetime_utc)
    zenith_angle = solar_zenith_angle(latitude, declination, hour_angle)

    extra_terrestrial = extraterrestrial_solar_radiation(day_of_year)
    transmissivity = atmospheric_transmissivity(zenith_angle)

    return extra_terrestrial * transmissivity

def is_daytime(hour, sunrise, sunset):
    """
    Determine whether a given hour falls within the defined daytime period.

    This helper function checks if a time (given in hours) falls between a specified
    sunrise and sunset time. It is used in data quality checks and solar radiation filtering.

    Parameters
    ----------
    hour : int or float
        Hour of the day (0–23), typically derived from a datetime object.
    sunrise : int or float
        Local hour defining the beginning of the daytime period (e.g., 6 for 6:00 AM).
    sunset : int or float
        Local hour defining the end of the daytime period (e.g., 18 for 6:00 PM).

    Returns
    -------
    is_day : bool
        `True` if `hour` is between `sunrise` and `sunset`, inclusive; `False` otherwise.

    Notes
    -----
    This function assumes a fixed definition of sunrise and sunset times and does not
    calculate solar position. It is intended for approximate daytime filtering.

    For solar angle–based filtering, use solar zenith angle thresholds instead.

    Examples
    --------
    >>> is_daytime(10, 6, 18)
    True
    >>> is_daytime(20, 6, 18)
    False
    """
    return sunrise <= hour <= sunset

def estimate_clear_sky_radiation(lat, lon, timestamp):
    """
    Estimate clear-sky shortwave radiation at a given location and time.

    This function provides a simplified model of incoming shortwave radiation under clear-sky
    conditions using solar position geometry and exponential atmospheric attenuation.

    Parameters
    ----------
    lat : float
        Latitude in **[degrees]**.
    lon : float
        Longitude in **[degrees]** (not used in current model but reserved for future time corrections).
    timestamp : datetime.datetime
        UTC timestamp for which to compute solar radiation.

    Returns
    -------
    clear_sky_radiation : float
        Estimated clear-sky global horizontal irradiance (GHI) **[W m⁻²]**.

    Notes
    -----
    The model assumes:

    * Daily solar declination from:

      .. math::

         \\delta = 23.45^\\circ \\sin\\left(\\frac{360}{365}(\\text{DOY} - 81)\\right)

    * Hour angle:

      .. math::

         H = 15^\\circ \\times (\\text{hour} - 12)

    * Cosine of solar zenith angle:

      .. math::

         \\cos(\\theta) = \\sin(\\phi) \\sin(\\delta) + \\cos(\\phi) \\cos(\\delta) \\cos(H)

    * Atmospheric attenuation based on air mass:

      .. math::

         \\text{Transmittance} = \\exp(-0.1 \\times \\text{air mass})

    The final estimate is:

    .. math::

        R_{cs} = G_{sc} \\cdot \\cos(\\theta) \\cdot \\text{Transmittance}

    where :math:`G_{sc} \\approx 1361 \\text{ W m⁻²}` is the solar constant.

    Assumptions:

    * The atmosphere is cloud-free.
    * Atmospheric attenuation is approximated with a fixed optical depth (τ ≈ 0.1).
    * No correction for longitude-based solar time or Equation of Time.

    References
    ----------
    Iqbal, M. (1983).
    *An Introduction to Solar Radiation*. Academic Press.

    Duffie, J.A., & Beckman, W.A. (2013).
    *Solar Engineering of Thermal Processes* (4th ed.). Wiley.
    """
    # Approximate solar declination angle (valid for general validation purposes)
    day_of_year = timestamp.timetuple().tm_yday
    declination = 23.45 * np.sin(np.radians((360 / 365) * (day_of_year - 81)))

    # Approximate solar hour angle
    hour_angle = (timestamp.hour - 12) * 15  # Degrees (15° per hour from solar noon)

    # Approximate solar zenith angle
    latitude_rad = np.radians(lat)
    declination_rad = np.radians(declination)
    hour_angle_rad = np.radians(hour_angle)

    cos_theta = np.sin(latitude_rad) * np.sin(declination_rad) + np.cos(
        latitude_rad
    ) * np.cos(declination_rad) * np.cos(hour_angle_rad)

    if cos_theta <= 0:  # Sun is below the horizon
        return 0

    # Estimate atmospheric attenuation (simplified)
    air_mass = 1 / cos_theta if cos_theta > 0 else np.inf
    transmittance = np.exp(-0.1 * air_mass)  # Approximate atmospheric absorption

    # Compute clear-sky shortwave radiation
    clear_sky_radiation = SOLAR_CONSTANT * transmittance * cos_theta
    return clear_sky_radiation

def estimate_max_net_radiation(temp_c):
    """
    Estimate the theoretical maximum net radiation based on surface temperature.

    This method uses the Stefan–Boltzmann law to estimate the maximum outgoing longwave
    radiation from a surface at the given air temperature, assuming it behaves as a perfect blackbody.

    Parameters
    ----------
    temp_c : float or ndarray
        Air temperature **[°C]**.

    Returns
    -------
    max_radiation : float or ndarray
        Maximum theoretical net radiation **[W m⁻²]** based solely on longwave emission.

    Notes
    -----
    This estimate assumes:

    * The emitting surface is a perfect blackbody (emissivity = 1.0),
    * There is no incoming longwave radiation (i.e., fully radiative loss),
    * Temperature is converted to Kelvin as:

      .. math::

          T_K = T_C + 273.15

    The Stefan–Boltzmann law is then applied:

    .. math::

        R_{max} = \\sigma T_K^4

    where:

    * :math:`\\sigma \\approx 5.670 \\times 10^{-8}` W m⁻² K⁻⁴ is the Stefan–Boltzmann constant.

    This serves as a **conservative upper limit** for net radiation loss from a surface.

    References
    ----------
    Incropera, F.P., & DeWitt, D.P. (2006).
    *Fundamentals of Heat and Mass Transfer* (6th ed.). Wiley.

    Duffie, J.A., & Beckman, W.A. (2013).
    *Solar Engineering of Thermal Processes* (4th ed.). Wiley.
    """
    temp_k = temp_c + 273.15
    return STEFAN_BOLTZMANN * temp_k**4  # W/m²

def validate_net_radiation(
    radiation_values, timestamps, temp_values, lat, lon, sunrise=6, sunset=18
):
    """
    Validate net radiation values using physical bounds and solar geometry.

    This function flags net radiation measurements that fall outside expected physical
    ranges or exceed theoretical or empirical limits based on time of day, location, and
    temperature.

    Parameters
    ----------
    radiation_values : list of float
        Measured net radiation values **[W m⁻²]**.
    timestamps : list of datetime.datetime
        Corresponding timestamps for each radiation measurement.
    temp_values : list of float
        Air temperature values **[°C]** corresponding to each timestamp.
    lat : float
        Site latitude in **[degrees]**.
    lon : float
        Site longitude in **[degrees]**.
    sunrise : int, optional
        Approximate local hour for sunrise (default is 6).
    sunset : int, optional
        Approximate local hour for sunset (default is 18).

    Returns
    -------
    results : list of tuple
        List of validation results in the format:
        ``(timestamp, value, status, reason)``, where:

        * `status` is `"Valid"` or `"Invalid"`,
        * `reason` provides a comma-separated explanation if invalid.

    Notes
    -----
    A value may be flagged as invalid if it:

    * Falls outside empirically expected ranges for daytime or nighttime periods.
    * Exceeds estimated clear-sky solar radiation (based on location and time).
    * Surpasses the theoretical maximum net radiation given air temperature.

    The expected radiation bounds for daytime and nighttime periods
    are assumed to be stored in constants `DAYTIME_RANGE` and `NIGHTTIME_RANGE`.

    The following support functions are required:

    - `is_daytime(hour, sunrise, sunset)` — determines if the timestamp is during daylight hours.
    - `estimate_clear_sky_radiation(lat, lon, timestamp)` — returns an estimate of clear-sky radiation.
    - `estimate_max_net_radiation(temp_c)` — returns the maximum theoretical net radiation.

    Assumptions:

    * Sunrise and sunset are fixed by hour, not by solar geometry (approximation).
    * Atmospheric effects and cloud cover are not explicitly modeled.

    """
    results = []

    for i, value in enumerate(radiation_values):
        timestamp = timestamps[i]
        temp_c = temp_values[i]
        hour = timestamp.hour
        reason = ""

        # Check general expected range
        if is_daytime(hour, sunrise, sunset):
            valid_range = DAYTIME_RANGE
        else:
            valid_range = NIGHTTIME_RANGE

        if not (valid_range[0] <= value <= valid_range[1]):
            reason = "Outside expected range"

        # Check against estimated clear-sky shortwave radiation
        clear_sky_rad = estimate_clear_sky_radiation(lat, lon, timestamp)
        if value > clear_sky_rad + 200:  # Allow some margin for atmospheric effects
            reason += ", Exceeds clear-sky estimate"

        # Check against maximum theoretical net radiation
        max_theoretical_rad = estimate_max_net_radiation(temp_c)
        if value > max_theoretical_rad:
            reason += ", Exceeds theoretical max"

        # Assign validity status
        status = "Valid" if reason == "" else "Invalid"

        results.append((timestamp, value, status, reason.strip(", ")))

    return results
