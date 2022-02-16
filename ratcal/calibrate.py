from warnings import warn

from cvxopt import matrix
from cvxopt import spmatrix
from cvxopt import solvers
import numpy as np
import numpy.linalg as npl

from ratcal.utility import check_rating_matrix


def _prepare(M: np.array) -> (spmatrix, matrix, matrix):
    # m is the number of raters
    # n is the number of objects being rated
    m, n = M.shape

    H = spmatrix([], [], [], (n + 2 * m, m * n))

    for i in range(m):
        for j in range(n):
            if M[i, j] == -1.:
                continue
            r = j * m + i
            H[j, r] = 1
            H[n + i, r] = -M[i, j]
            H[n + m + i, r] = 1

    H = H[:-1, :]  # removes the last row for normalization
    Q = 2 * H * H.trans()

    h1 = matrix(0., (1, n))
    h2 = matrix(1., (1, m))
    h3 = matrix(0., (1, m - 1))
    A = matrix([[h1], [h2], [h3]])

    b = matrix(m, tc='d')

    return Q, A, b


def _check_ranks(P: spmatrix, A: matrix):
    rank_A = npl.matrix_rank(A)
    p = A.size[0]
    if rank_A < p:
        warn("cvxopt qp: rank(A) = %d lower than p = %d. Consider setting additive." % (rank_A, p))
    rank_PA = npl.matrix_rank(matrix([P, A]))
    n = P.size[1]
    if rank_PA < n:
        warn("cvxopt qp: rank([P, A]) = %d lower than n = %d. Consider setting additive." % (rank_PA, n))


def _qp(P: spmatrix, A: matrix, b: matrix) -> np.array:
    _check_ranks(P, A)

    q = matrix(np.zeros(P.size[1]))
    sol = solvers.qp(P, q, A=A, b=b)
    if sol['status'] != 'optimal':
        warn('calibrate() failed to find an optimal solution')
    return np.array(sol['x'])


def _calibrate(M: np.array) -> (np.array, np.array, np.array):
    P, A, b = _prepare(M)
    x = _qp(P, A, b)

    m, n = M.shape
    ratings = x[0:n].flatten()
    p = x[n:n + m].flatten()
    q = np.append(x[n + m:].flatten(), 0.)
    distinct = 1 / p
    lenient = q
    return ratings, distinct, lenient


def _scale(ratings: np.array, lenient: np.array, scale: np.array) -> (np.array, np.array):
    rat_min = ratings.min()
    rat_max = ratings.max()
    slope = (scale[1] - scale[0]) / (rat_max - rat_min)

    rat_check = np.interp(ratings, (rat_min, rat_max), scale)
    ratings = np.array([(r - rat_min) * slope + scale[0] for r in ratings])
    assert np.allclose(rat_check, ratings)

    lenient = np.array([d * slope for d in lenient])
    return ratings, lenient


def calibrate(M: np.array, scale: (float, float) = (0., 0.), additive: bool = True) -> (np.array, np.array, np.array):
    """
    Calibrate ratings and evaluate raters.

    :param M: The matrix of ratings. Ratings should be equal or greater than zero. -1 denotes null.
    :param scale: The range that the ratings are scaled to.
    :param additive: Whether to add three hypothetical objects, one that all raters give the highest rating,
           one that all raters give the lowest rating, and one that all raters give their average ratings.
    :return: The calibrated ratings, the distinction, and the leniency of each rater
    """

    check_rating_matrix(M)

    # m is the number of raters
    # n is the number of objects being rated
    m, n = M.shape

    if additive:
        max_rat = _find_max(M)
        min_rat = _find_min(M)
        best_column = np.full((m, 1), max_rat)
        worst_column = np.full((m, 1), min_rat)
        average_column = _rater_average(M)
        M = np.column_stack((M, best_column, worst_column, average_column))

        assert M.shape[1] == n + 3
        assert _object_average(M, [n, n + 1]) == [max_rat, min_rat]

        ratings, distinct, lenient = _calibrate(M)

        ratings = ratings[:-3]
    else:
        ratings, distinct, lenient = _calibrate(M)

    if scale != (0., 0.):
        ratings, lenient = _scale(ratings, lenient, scale)
    return ratings, distinct, lenient


def _object_average(M: np.array, indexes: list = None) -> list:
    # m is the number of raters
    # n is the number of objects being rated
    m, n = M.shape

    if indexes is None:
        indexes = range(n)

    rat = []
    for j in indexes:
        total = 0.0
        count = 0
        for i in range(m):
            if M[i, j] == -1.:
                continue
            total += M[i, j]
            count += 1
        rat.append(total / count if count else -1.)

    return rat


def average(M: np.array, indexes: list = None) -> np.array:
    """
    Calculate average ratings of objects.

    :param M: The matrix of ratings. Ratings should be equal or greater than zero. -1 denotes null.
    :param indexes: A list of columns to calculate their average ratings
    :return: The average ratings.
    """
    check_rating_matrix(M)

    return np.array(_object_average(M, indexes))


def _rater_average(M: np.array, indexes: list = None) -> list:
    # m is the number of raters
    # n is the number of objects being rated
    m, n = M.shape

    if indexes is None:
        indexes = range(m)

    rat = []
    for i in indexes:
        total = 0.0
        count = 0
        for j in range(n):
            if M[i, j] == -1.:
                continue
            total += M[i, j]
            count += 1
        rat.append(total / count if count else -1.)
    return rat


def rater_average(M: np.array, indexes: list = None) -> np.array:
    """
    Calculate the average ratings of raters.

    :param M: The matrix of ratings. Ratings should be equal or greater than zero. -1 denotes null.
    :param indexes: A list of rows to calculate their average ratings
    :return: The average ratings.
    """
    check_rating_matrix(M)

    return np.array(_rater_average(M, indexes))


def _rater_median(M: np.array, indexes: list = None) -> list:
    # m is the number of raters
    # n is the number of objects being rated
    m, n = M.shape

    if indexes is None:
        indexes = range(m)

    median_rat = []
    for i in indexes:
        rat = []
        for j in range(n):
            if M[i, j] == -1.:
                continue
            rat.append(M[i, j])
        median = np.median(rat) if rat else -1.
        median_rat.append(median)
    return median_rat


def rater_median(M: np.array, indexes: list = None) -> np.array:
    """
    Select the median ratings of raters.

    :param M: The matrix of ratings. Ratings should be equal or greater than zero. -1 denotes null.
    :param indexes: A list of rows to select their median ratings
    :return: The median ratings.
    """
    check_rating_matrix(M)

    return np.array(_rater_median(M, indexes))


def _rater_min(M: np.array, indexes: list = None) -> list:
    # m is the number of raters
    # n is the number of objects being rated
    m, n = M.shape

    if indexes is None:
        indexes = range(m)

    rat = []
    for i in indexes:
        min_rat = None
        for j in range(n):
            if M[i, j] == -1.:
                continue
            if min_rat is None or M[i, j] < min_rat:
                min_rat = M[i, j]
        rat.append(-1. if min_rat is None else min_rat)
    return rat


def rater_min(M: np.array, indexes: list = None) -> np.array:
    """
    Calculate the min ratings of raters.

    :param M: The matrix of ratings. Ratings should be equal or greater than zero. -1 denotes null.
    :param indexes: A list of rows to calculate their average ratings
    :return: The min ratings.
    """
    check_rating_matrix(M)

    return np.array(_rater_min(M, indexes))


def _rater_max(M: np.array, indexes: list = None) -> list:
    # m is the number of raters
    # n is the number of objects being rated
    m, n = M.shape

    if indexes is None:
        indexes = range(m)

    rat = []
    for i in indexes:
        max_rat = None
        for j in range(n):
            if M[i, j] == -1.:
                continue
            if max_rat is None or M[i, j] > max_rat:
                max_rat = M[i, j]
        rat.append(-1. if max_rat is None else max_rat)
    return rat


def rater_max(M: np.array, indexes: list = None) -> np.array:
    """
    Calculate the min ratings of raters.

    :param M: The matrix of ratings. Ratings should be equal or greater than zero. -1 denotes null.
    :param indexes: A list of rows to calculate their average ratings
    :return: The min ratings.
    """
    check_rating_matrix(M)

    return np.array(_rater_max(M, indexes))


def _find_min(M: np.array):
    m, n = M.shape
    rat = None
    for i in range(m):
        for j in range(n):
            if M[i, j] == -1.:
                continue
            if rat is None or M[i, j] < rat:
                rat = M[i, j]
    return rat


def find_min(M: np.array):
    """
    Find the min rating.

    :param M: The matrix of ratings. Ratings should be equal or greater than zero. -1 denotes null.
    :return: The min rating value. None if no ratings.
    """
    check_rating_matrix(M)
    return _find_min(M)


def _find_max(M: np.array):
    m, n = M.shape
    rat = None
    for i in range(m):
        for j in range(n):
            if M[i, j] == -1.:
                continue
            if rat is None or M[i, j] > rat:
                rat = M[i, j]
    return rat


def find_max(M: np.array):
    """
    Find the max rating.

    :param M: The matrix of ratings. Ratings should be equal or greater than zero. -1 denotes null.
    :return: The max rating value. None if no ratings.
    """
    check_rating_matrix(M)
    return _find_max(M)
