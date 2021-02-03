import numpy as np
import pytest

from ratcal import calibrate


def test_input_dimensions():
    m = np.zeros((2, 3))
    calibrate(m)

    with pytest.raises(ValueError) as error:
        m = np.array([0, -1, 1])
        calibrate(m)
    assert 'got 1 dimension(s)' in str(error.value)

    with pytest.raises(ValueError) as error:
        m = np.zeros((1, 2, 3))
        calibrate(m)
    assert 'got 3 dimension(s)' in str(error.value)


def test_input_ratings():
    m = np.zeros((2, 3))
    calibrate(m)

    m = np.full((1, 5), -1.0)
    calibrate(m)

    m = np.random.rand(10, 1)
    calibrate(m)

    with pytest.raises(ValueError) as error:
        m = np.array([[2, 1, 0.5],
                      [5, -2, 1]])
        calibrate(m)
    assert 'should be -1 denoting null' in str(error.value)
