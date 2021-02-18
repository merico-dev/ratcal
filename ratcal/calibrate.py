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
            if M[i, j] >= 0:
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
        warn("cvxopt qp: rank(A) = %d lower than p = %d" % (rank_A, p))
        return False
    rank_PA = npl.matrix_rank(matrix([P, A]))
    n = P.size[1]
    if rank_PA < n:
        warn("cvxopt qp: rank([P, A]) = %d lower than n = %d" % (rank_PA, n))
        return False
    return True


def _qp(P, A, b):
    q = matrix(np.zeros(P.size[1]))
    sol = solvers.qp(P, q, A=A, b=b)
    if sol['status'] != 'optimal':
        warn('calibrate() failed to find an optimal solution')
    return sol['x']


def calibrate(M: np.array, scale: (float, float) = (0., 0.), coordinates: bool = None):
    """
    Calibrate ratings and evaluate raters.

    :param M: The matrix of ratings. Ratings should be equal or greater than zero. -1 denotes null.
    :param scale: The range that the ratings are scaled to.
    :param coordinates: Whether to add two hypothetical, common objects, one that all raters give highest ratings and
           one that all raters give lowest ratings, in order to coordinate raters. None refers to automatic selection.
    :return: The calibrated ratings, the bias, and the leniency of each rater
    """

    check_rating_matrix(M)

    # m is the number of raters
    # n is the number of objects being rated
    m, n = M.shape

    P, A, b = _prepare(M)

    if coordinates is True or (coordinates is None and not _check_ranks(P, A)):
        max_rat = find_max(M)
        min_rat = find_min(M)
        best_column = np.full((m, 1), max_rat)
        worst_column = np.full((m, 1), min_rat)
        ratings, bias, leniency = calibrate(np.column_stack((M, best_column, worst_column)), scale, False)
        return ratings[:-2], bias, leniency

    x = _qp(P, A, b)

    ratings = np.array(x[0:n]).flatten()
    p = np.array(x[n:n + m]).flatten()
    q = np.append(np.array(x[n + m:]).flatten(), 0.)
    bias = 1 / p
    leniency = q * bias

    if scale != (0., 0.):
        ratings = np.interp(ratings, (ratings.min(), ratings.max()), scale)
    return ratings, bias, leniency


def average(M: np.array):
    """
    Calculate average ratings.

    :param M: The matrix of ratings. Ratings should be equal or greater than zero. -1 denotes null.
    :return: The average ratings.
    """
    check_rating_matrix(M)

    # m is the number of raters
    # n is the number of objects being rated
    m, n = M.shape

    ratings = []
    for j in range(n):
        total = 0.0
        count = 0
        for i in range(m):
            if M[i, j] >= 0:
                total += M[i, j]
                count += 1
        ratings.append(total / count)

    return ratings


def find_min(M: np.array):
    """
    Find the min rating.

    :param M: The matrix of ratings. Ratings should be equal or greater than zero. -1 denotes null.
    :return: The min rating value. None if no ratings.
    """
    check_rating_matrix(M)

    m, n = M.shape
    rat = None
    for i in range(m):
        for j in range(n):
            if M[i, j] < 0:
                continue
            if rat is None or M[i, j] < rat:
                rat = M[i, j]
    return rat


def find_max(M: np.array):
    """
    Find the max rating.

    :param M: The matrix of ratings. Ratings should be equal or greater than zero. -1 denotes null.
    :return: The max rating value. None if no ratings.
    """
    check_rating_matrix(M)

    m, n = M.shape
    rat = None
    for i in range(m):
        for j in range(n):
            if M[i, j] < 0:
                continue
            if rat is None or M[i, j] > rat:
                rat = M[i, j]
    return rat
