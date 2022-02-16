import numpy as np
import pytest

from ratcal import calibrate
from ratcal import average, rater_average, rater_median
from ratcal import rater_min, rater_max
from ratcal import find_min, find_max
from ratcal import check_rating_matrix


def test_input_dimensions():

    with pytest.raises(ValueError) as error:
        M = np.array([0, -1, 1])
        check_rating_matrix(M)
    assert 'got 1 dimension(s)' in str(error.value)

    with pytest.raises(ValueError) as error:
        M = np.zeros((1, 2, 3))
        check_rating_matrix(M)
    assert 'got 3 dimension(s)' in str(error.value)

    with pytest.raises(ValueError) as error:
        M = np.array([0, -1, 1])
        calibrate(M)
    assert 'got 1 dimension(s)' in str(error.value)

    with pytest.raises(ValueError) as error:
        M = np.array([0, -1, 1])
        average(M)
    assert 'got 1 dimension(s)' in str(error.value)

    with pytest.raises(ValueError) as error:
        M = np.array([0, -1, 1])
        find_min(M)
    assert 'got 1 dimension(s)' in str(error.value)

    with pytest.raises(ValueError) as error:
        M = np.array([0, -1, 1])
        find_max(M)
    assert 'got 1 dimension(s)' in str(error.value)


def test_input_ratings():

    with pytest.raises(ValueError) as error:
        M = np.array([[2, 1, 0.5],
                      [5, -2, 1]])
        calibrate(M)
    assert 'should be -1 denoting null' in str(error.value)

    with pytest.raises(ValueError) as error:
        M = np.array([[2, 1, 0.5],
                      [5, -2, 1]])
        average(M)
    assert 'should be -1 denoting null' in str(error.value)

    with pytest.raises(ValueError) as error:
        M = np.array([[2, 1, 0.5],
                      [5, -2, 1]])
        find_min(M)
    assert 'should be -1 denoting null' in str(error.value)

    with pytest.raises(ValueError) as error:
        M = np.array([[2, 1, 0.5],
                      [5, -2, 1]])
        find_max(M)
    assert 'should be -1 denoting null' in str(error.value)


def test_average():
    assert np.array_equal(average(np.array([[]])), [])

    M = np.array([[-1., -1., 2., 3.],
                  [-1., 1.1, 2., 5.],
                  [-1., -1., 2., 4.]])
    assert np.array_equal(average(M), [-1., 1.1, 2., 4.])
    assert np.array_equal(average(M, [0]), [-1.])
    assert np.array_equal(average(M, [-1]), [4.])


def test_rater_average():
    assert np.array_equal(rater_average(np.array([[]])), [-1.])

    M = np.array([[1., 2., 6.]])
    assert np.array_equal(rater_average(M, [0]), [3.])

    M = np.array([[1., 4., 3., 2.],
                  [-1., -1., 1., -1.],
                  [-1., -1., -1., -1.],
                  [700, 100, 100, -1.]])
    assert np.array_equal(rater_average(M), [2.5, 1., -1., 300])


def test_rater_median():
    assert rater_median(np.array([[]])) == [-1.]

    M = np.array([[1., 2., 6.]])
    assert np.array_equal(rater_median(M, [0]), [2.])

    M = np.array([[1., 4., 3., 2.],
                  [-1., -1., 1., -1.],
                  [-1., -1., -1., -1.],
                  [300, 100, 100, -1.]])
    assert np.array_equal(rater_median(M), [2.5, 1., -1., 100])


def test_rater_min():
    assert rater_min(np.array([[]])) == [-1.]

    M = np.array([[1., 2., 1.]])
    assert np.array_equal(rater_min(M, [0]), [1.])

    M = np.array([[1., 4., 3., 2.],
                  [-1., -1., 10., -1.],
                  [-1., -1., -1., -1.],
                  [300, 100, 100, -1.]])
    assert np.array_equal(rater_min(M), [1., 10., -1., 100])


def test_rater_max():
    assert rater_max(np.array([[]])) == [-1.]

    M = np.array([[1., 2., 1.]])
    assert np.array_equal(rater_max(M, [0]), [2.])

    M = np.array([[1., 4., 3., 2.],
                  [-1., -1., 10., -1.],
                  [-1., -1., -1., -1.],
                  [300, 300, 100, -1.]])
    assert np.array_equal(rater_max(M), [4., 10., -1., 300])


def test_paper_review_example():
    M = np.array([[.40, .44, -1., .67, -1., .81, .89, .95, .99],
                  [.15, .19, .32, .45, .49, .62, -1., -1., -1.],
                  [-1., .21, -1., .39, -1., .61, -1., .79, -1.],
                  [.20, -1., .20, -1., .60, -1., .70, -1., .80],
                  [-1., -1., .10, -1., .20, -1., .30, .40, .50]])
    ratings, distinct, lenient = calibrate(M, (0.0, 1.0), False)
    assert np.array_equal(np.round(ratings, decimals=2), [0.0, .05, .18, .40, .53, .68, .75, .92, 1.0])

    avg_rat = average(M)
    assert np.array_equal(np.round(avg_rat, decimals=2), [.25, .28, .21, .50, .43, .68, .63, .71, .76])

    assert find_min(M) == .10
    assert find_max(M) == .99


def test_performance_review_0():
    M = np.loadtxt('tests/rat_mat_0.data')
    assert M.shape[0] == M.shape[1]
    n = M.shape[0]

    ratings, distinct, lenient = calibrate(M, (0, 100))
    assert len(ratings) == n and len(distinct) == n and len(lenient) == n
    assert np.isclose(ratings.min(), 0) and np.isclose(ratings.max(), 100)


def test_performance_review_1():
    M = np.loadtxt('tests/rat_mat_1.data')
    assert M.shape[0] == M.shape[1]
    n = M.shape[0]

    avg_rat = average(M)
    ratings, distinct, lenient = calibrate(M, (avg_rat.min(), avg_rat.max()))
    assert len(ratings) == n and len(distinct) == n and len(lenient) == n
    assert np.isclose(ratings.min(), avg_rat.min()) and np.isclose(ratings.max(), avg_rat.max())
