import numpy as np
import pytest
from src.metcalcs import canopy
# ----------------------
# canopy_gash() Tests
# ----------------------

def test_canopy_gash_single_value():
    Pg, TF, SF, Ei = canopy.canopy_gash(Pg=5, ER=0.5, S=1.0, p=0.1, pt=0.1)
    assert isinstance(Pg, (float, np.ndarray))
    assert isinstance(TF, (float, np.ndarray))
    assert isinstance(Ei, (float, np.ndarray))
    assert np.isclose(Pg, TF + Ei, atol=1e-6)

def test_canopy_gash_array_values():
    precipitation = np.array([1.0, 5.0, 10.0])
    Pg, TF, SF, Ei = canopy.canopy_gash(Pg=precipitation, ER=0.5, S=1.0, p=0.1, pt=0.1)
    assert isinstance(Pg, np.ndarray)
    assert Pg.shape == TF.shape == SF.shape == Ei.shape
    np.testing.assert_allclose(Pg, TF + Ei, atol=1e-6)

def test_canopy_gash_zero_precipitation():
    Pg, TF, SF, Ei = canopy.canopy_gash(Pg=0.0, ER=0.5, S=1.0, p=0.1, pt=0.1)
    assert TF == 0
    assert Ei == 0

def test_canopy_gash_invalid_p_raises():
    with pytest.raises(ValueError, match="Free throughfall coefficient"):
        canopy.canopy_gash(Pg=5, ER=0.5, S=1.0, p=1.5, pt=0.1)

def test_canopy_gash_invalid_pt_raises():
    with pytest.raises(ValueError, match="Stemflow coefficient"):
        canopy.canopy_gash(Pg=5, ER=0.5, S=1.0, p=0.1, pt=1.5)

def test_canopy_gash_sum_p_pt_exceeds_one():
    with pytest.raises(ValueError, match="Sum of p and pt must not exceed 1"):
        canopy.canopy_gash(Pg=5, ER=0.5, S=1.0, p=0.7, pt=0.4)

def test_canopy_gash_negative_storage():
    with pytest.raises(ValueError, match="Canopy storage capacity must be positive"):
        canopy.canopy_gash(Pg=5, ER=0.5, S=-1.0, p=0.1, pt=0.1)

def test_canopy_gash_negative_evaporation():
    with pytest.raises(ValueError, match="Evaporation rate must be positive"):
        canopy.canopy_gash(Pg=5, ER=-0.5, S=1.0, p=0.1, pt=0.1)

# ----------------------
# _validate_gash_parameters() Tests
# ----------------------

def test_validate_gash_parameters_valid():
    try:
        canopy._validate_gash_parameters(np.array([1.0, 2.0]), 0.5, 1.0, 0.1, 0.1)
    except Exception:
        pytest.fail("Unexpected Exception Raised")

def test_validate_gash_parameters_negative_precipitation():
    with pytest.raises(ValueError, match="Gross precipitation cannot be negative"):
        canopy._validate_gash_parameters(np.array([-1.0]), 0.5, 1.0, 0.1, 0.1)

def test_validate_gash_parameters_invalid_p():
    with pytest.raises(ValueError, match="Free throughfall coefficient must be between 0 and 1"):
        canopy._validate_gash_parameters(np.array([1.0]), 0.5, 1.0, -0.1, 0.1)

def test_validate_gash_parameters_invalid_pt():
    with pytest.raises(ValueError, match="Stemflow coefficient must be between 0 and 1"):
        canopy._validate_gash_parameters(np.array([1.0]), 0.5, 1.0, 0.1, -0.2)

def test_validate_gash_parameters_p_plus_pt_too_large():
    with pytest.raises(ValueError, match="Sum of free throughfall and stemflow coefficients cannot exceed 1"):
        canopy._validate_gash_parameters(np.array([1.0]), 0.5, 1.0, 0.8, 0.5)
