import numpy as np
import pytest
from src.metcalcs import evaporation


def test_penman_open_water_basic():
    result = evaporation.penman_open_water(
        airtemp=25.0,
        rh=60.0,
        airpress=101325.0,
        Rs=18e6,
        Rext=25e6,
        u=2.5,
    )
    assert result > 0 and np.isfinite(result)


def test_penman_open_water_array():
    airtemp = np.array([20.0, 25.0])
    rh = np.array([50.0, 60.0])
    airpress = np.array([101000.0, 101325.0])
    Rs = np.array([15e6, 18e6])
    Rext = np.array([22e6, 25e6])
    u = np.array([2.0, 2.5])

    result = evaporation.penman_open_water(airtemp, rh, airpress, Rs, Rext, u)
    assert result.shape == airtemp.shape
    assert np.all(result > 0)


class DummyConfig:
    validate_inputs = False


class DummyEvap:
    config = DummyConfig()

    vapor_pressure_slope = staticmethod(evaporation.vapor_pressure_slope)
    psychrometric_constant = staticmethod(evaporation.psychrometric_constant)
    latent_heat = staticmethod(evaporation.latent_heat)
    actual_vapor_pressure = staticmethod(evaporation.actual_vapor_pressure)
    saturation_vapor_pressure = staticmethod(evaporation.saturation_vapor_pressure)
    air_density = staticmethod(lambda T, rh, P: 1.2)
    specific_heat = staticmethod(lambda T, rh, P: 1005.0)


def test_pet_makkink():
    evap = DummyEvap()
    result = evaporation.pet_makkink(evap, 25.0, 50.0, 101325.0, 18e6)
    assert result > 0 and np.isfinite(result)


def test_pet_preistleytaylor():
    evap = DummyEvap()
    result = evaporation.pet_preistleytaylor(evap, 25.0, 50.0, 101325.0, 16e6, 2e6)
    assert result > 0 and np.isfinite(result)


def test_refet_penmanmonteith():
    evap = DummyEvap()
    result = evaporation.refet_penmanmonteith(
        evap, 25.0, 50.0, 101325.0, 18e6, 25e6, 2.0
    )
    assert result > 0 and np.isfinite(result)


def test_et_penmanmonteith():
    evap = DummyEvap()
    result = evaporation.et_penmanmonteith(
        evap,
        airtemp=25.0,
        rh=50.0,
        airpress=101325.0,
        Rn=18e6,
        G=0.0,
        ra=100.0,
        rs=50.0,
    )
    assert result > 0 and np.isfinite(result)
