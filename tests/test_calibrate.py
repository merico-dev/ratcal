import numpy as np
import pytest

from ratcal import calibrate
from ratcal import average
from ratcal import find_min, find_max
from ratcal import check_rating_matrix


def test_input_dimensions():

    with pytest.raises(ValueError) as error:
        m = np.array([0, -1, 1])
        check_rating_matrix(m)
    assert 'got 1 dimension(s)' in str(error.value)

    with pytest.raises(ValueError) as error:
        m = np.zeros((1, 2, 3))
        check_rating_matrix(m)
    assert 'got 3 dimension(s)' in str(error.value)

    with pytest.raises(ValueError) as error:
        m = np.array([0, -1, 1])
        calibrate(m)
    assert 'got 1 dimension(s)' in str(error.value)

    with pytest.raises(ValueError) as error:
        m = np.array([0, -1, 1])
        average(m)
    assert 'got 1 dimension(s)' in str(error.value)

    with pytest.raises(ValueError) as error:
        m = np.array([0, -1, 1])
        find_min(m)
    assert 'got 1 dimension(s)' in str(error.value)

    with pytest.raises(ValueError) as error:
        m = np.array([0, -1, 1])
        find_max(m)
    assert 'got 1 dimension(s)' in str(error.value)


def test_input_ratings():

    with pytest.raises(ValueError) as error:
        m = np.array([[2, 1, 0.5],
                      [5, -2, 1]])
        calibrate(m)
    assert 'should be -1 denoting null' in str(error.value)

    with pytest.raises(ValueError) as error:
        m = np.array([[2, 1, 0.5],
                      [5, -2, 1]])
        average(m)
    assert 'should be -1 denoting null' in str(error.value)

    with pytest.raises(ValueError) as error:
        m = np.array([[2, 1, 0.5],
                      [5, -2, 1]])
        find_min(m)
    assert 'should be -1 denoting null' in str(error.value)

    with pytest.raises(ValueError) as error:
        m = np.array([[2, 1, 0.5],
                      [5, -2, 1]])
        find_max(m)
    assert 'should be -1 denoting null' in str(error.value)


def test_paper_review_example():
    M = np.array([[.40, .44, -1., .67, -1., .81, .89, .95, .99],
                  [.15, .19, .32, .45, .49, .62, -1., -1., -1.],
                  [-1., .21, -1., .39, -1., .61, -1., .79, -1.],
                  [.20, -1., .20, -1., .60, -1., .70, -1., .80],
                  [-1., -1., .10, -1., .20, -1., .30, .40, .50]])
    ratings, bias, leniency = calibrate(M, (0.0, 1.0), False)
    assert np.array_equal(np.round(ratings, decimals=2), [0.0, .05, .18, .40, .53, .68, .75, .92, 1.0])

    avg_rat = average(M)
    assert np.array_equal(np.round(avg_rat, decimals=2), [.25, .28, .21, .50, .43, .68, .63, .71, .76])

    assert find_min(M) == .10
    assert find_max(M) == .99


def test_performance_review():
    M = np.loadtxt('tests/rat_mat.data')
    assert M.shape[0] == M.shape[1]
    n = M.shape[0]
    ratings, bias, leniency = calibrate(M, (0, 100))
    assert len(ratings) == n and len(bias) == n and len(leniency) == n
    assert ratings.min() == 0 and ratings.max() == 100
