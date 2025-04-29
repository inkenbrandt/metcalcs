import numpy as np
import pytest

from src.metcalcs import thermo


def test_actual_vapor_pressure_scalar():
    ea = thermo.actual_vapor_pressure(25.0, 50.0)
    assert np.isclose(ea, thermo.saturation_vapor_pressure(25.0) * 0.5, atol=1)


def test_actual_vapor_pressure_array():
    temps = np.array([20.0, 25.0])
    rhs = np.array([50.0, 60.0])
    ea = thermo.actual_vapor_pressure(temps, rhs)
    expected = thermo.saturation_vapor_pressure(temps) * (rhs / 100)
    np.testing.assert_allclose(ea, expected, rtol=0.01)


def test_specific_heat_scalar():
    cp = thermo.specific_heat(25.0, 50.0, 101325.0)
    assert cp > 1000


def test_vapor_pressure_slope():
    delta = thermo.vapor_pressure_slope(25.0)
    assert delta > 0


def test_saturation_vapor_pressure_scalar():
    es = thermo.saturation_vapor_pressure(25.0)
    assert es > 0


def test_saturation_vapor_pressure_warning():
    with pytest.warns(UserWarning):
        thermo.saturation_vapor_pressure(-100.0)


def test_aerodynamic_resistance_scalar():
    ra = thermo.aerodynamic_resistance(2.0, 0.1, 0.0, 3.0)
    assert ra > 0


def test_aerodynamic_resistance_invalid():
    with pytest.raises(ValueError):
        thermo.aerodynamic_resistance(0.1, 0.1, 0.0, 3.0)


def test_h_tvardry_scalar():
    H = thermo.h_tvardry(1.2, 1005, 25.0, 0.5, 2.0)
    assert H > 0


def test_psychrometric_constant_scalar():
    gamma = thermo.psychrometric_constant(25.0, 50.0, 101325.0)
    assert gamma > 0


def test_latent_heat_scalar():
    L = thermo.latent_heat(25.0)
    assert L > 2e6  # J/kg typically around 2.4 million J/kg


def test_latent_heat_warning():
    with pytest.warns(UserWarning):
        thermo.latent_heat(-50.0)


def test_potential_temperature_scalar():
    theta = thermo.potential_temperature(25.0, 50.0, 101325.0)
    assert theta > 0


def test_air_density_scalar():
    rho = thermo.air_density(25.0, 50.0, 101325.0)
    assert 0.5 < rho < 2.0  # kg/m^3 reasonable range for air density
