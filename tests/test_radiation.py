import pytest
import numpy as np
import datetime

from src.metcalcs import radiation


def test_solar_parameters_typical():
    doy = 172
    lat = 40
    sunshine, rad = radiation.solar_parameters(doy, lat)
    assert 13 <= sunshine <= 15
    assert rad > 0


def test_calculate_radiation_typical():
    result = radiation.calculate_radiation(25, 50, 500, 40, 172)
    assert all(
        k in result
        for k in ["extraterrestrial", "net_shortwave", "net_longwave", "net_radiation"]
    )
    assert result["net_radiation"] < result["net_shortwave"]


def test_extraterrestrial_radiation_known():
    ra = radiation.extraterrestrial_radiation(40, 172)
    assert isinstance(ra, float)
    assert ra > 0


def test_solar_declination_bounds():
    decl = radiation.solar_declination(172)
    assert isinstance(decl, float)
    assert -0.5 < decl < 0.5


def test_solar_zenith_angle_at_noon():
    lat, doy = 40.0, 172
    decl = radiation.solar_declination(doy)
    ha = 0.0  # Solar noon
    zenith = radiation.solar_zenith_angle(lat, decl, ha)
    assert 0 <= zenith <= np.pi / 2


def test_atmospheric_transmissivity_low_angle():
    assert radiation.atmospheric_transmissivity(np.radians(60)) < 0.75
    assert radiation.atmospheric_transmissivity(np.radians(95)) == 0


def test_incoming_solar_radiation_daytime():
    rad = radiation.incoming_solar_radiation(
        40.0, -112.0, datetime.datetime(2023, 6, 21, 12, 0)
    )
    assert rad > 0


def test_is_daytime_true_false():
    assert radiation.is_daytime(12, 6, 18) is True
    assert radiation.is_daytime(3, 6, 18) is False


def test_estimate_clear_sky_radiation_day():
    result = radiation.estimate_clear_sky_radiation(
        40, -112, datetime.datetime(2023, 6, 21, 12, 0)
    )
    assert result > 0


def test_estimate_max_net_radiation():
    r = radiation.estimate_max_net_radiation(np.array([25.0]))
    assert r > 400.0


def test_validate_net_radiation_flags_invalid():
    timestamps = [datetime.datetime(2023, 6, 21, 12)]
    values = [1500]  # Unrealistically high
    temps = [25]
    result = radiation.validate_net_radiation(values, timestamps, temps, 40, -112)
    assert result[0][2] == "Invalid"
    assert "Exceeds" in result[0][3]


def test_validate_net_radiation_valid_day():
    timestamps = [datetime.datetime(2023, 6, 21, 12)]
    values = [300]
    temps = [25]
    result = radiation.validate_net_radiation(values, timestamps, temps, 40, -112)
    assert result[0][2] == "Valid"
