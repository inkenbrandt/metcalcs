import numpy as np
import pytest
from src.metcalcs import validators
# ------------------------
# Tests for validate_inputs
# ------------------------

def test_validate_inputs_all_valid():
    try:
        validators.validate_inputs(a=1.0, b=np.array([1, 2, 3]), c=[3.0, 4.5])
    except Exception:
        pytest.fail("validate_inputs raised unexpectedly for valid inputs.")


# ------------------------
# Tests for to_array
# ------------------------

def test_to_array_single_scalar():
    result = validators.to_array(5.5)
    assert result == 5.5

def test_to_array_single_list():
    result = validators.to_array([1, 2, 3])
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert (result == np.array([1, 2, 3])).all()

def test_to_array_multiple():
    a, b = validators.to_array([1, 2], (3, 4))
    assert isinstance(a, np.ndarray)
    assert isinstance(b, np.ndarray)
    assert (a == np.array([1, 2])).all()
    assert (b == np.array([3, 4])).all()
