import numpy as np
import pytest

from ratcal import calibrate


def test_input_dimensions():

    with pytest.raises(ValueError) as error:
        m = np.array([0, -1, 1])
        calibrate(m)
    assert 'got 1 dimension(s)' in str(error.value)

    with pytest.raises(ValueError) as error:
        m = np.zeros((1, 2, 3))
        calibrate(m)
    assert 'got 3 dimension(s)' in str(error.value)


def test_input_ratings():

    with pytest.raises(ValueError) as error:
        m = np.array([[2, 1, 0.5],
                      [5, -2, 1]])
        calibrate(m)
    assert 'should be -1 denoting null' in str(error.value)


def test_paper_review_example():
    m = np.array([[.40, .44, -1., .67, -1., .81, .89, .95, .99],
                  [.15, .19, .32, .45, .49, .62, -1., -1., -1.],
                  [-1., .21, -1., .39, -1., .61, -1., .79, -1.],
                  [.20, -1., .20, -1., .60, -1., .70, -1., .80],
                  [-1., -1., .10, -1., .20, -1., .30, .40, .50]])
    z, p, q = calibrate(m)
    sz = np.interp(z, (z.min(), z.max()), (0.0, 1.0))
    assert np.array_equal(np.round(sz, decimals=2), [0.0, .05, .18, .40, .53, .68, .75, .92, 1.0])
