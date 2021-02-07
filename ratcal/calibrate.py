import random
import numpy as np
from warnings import warn
from cvxopt import matrix
from cvxopt import spmatrix
from cvxopt import solvers


def _check_rating_matrix(M: np.array):
    if M.ndim != 2:
        raise ValueError("M must be two-dimension, got %d dimension(s)" % M.ndim)
    for i, v in np.ndenumerate(M):
        if v < 0 and v != -1:
            raise ValueError("M[%d, %d] = %f should be -1 denoting null" % (i[0], i[1], v))


def calibrate(M: np.array, noise=(0, 0), scale=(0, 0)):
    """
    Calibrate ratings and evaluate raters.

    :param M: The matrix of ratings. -1 denotes null. Ratings should be equal or greater than zero.
    :param noise: The range of random, tiny noise added to ratings to
           avoid linear dependence (when certain matrix ranks are low).
    :param scale: The range that the ratings are scaled to.
    :return: The calibrated ratings, the bias, and the leniency of each rater
    """

    _check_rating_matrix(M)

    # m is the number of raters
    # n is the number of objects being rated
    m, n = M.shape

    H = spmatrix([], [], [], (n + 2 * m, m * n))

    for i in range(m):
        for j in range(n):
            if M[i, j] >= 0:
                r = j * m + i
                H[j, r] = 1
                H[n + i, r] = -M[i, j] + random.uniform(noise[0], noise[1])
                H[n + m + i, r] = 1

    H = H[:-1, :]  # removes the last row for normalization
    Q = 2 * H * H.trans()
    q = matrix(np.zeros(n + 2 * m - 1))

    h1 = matrix(0., (1, n))
    h2 = matrix(1., (1, m))
    h3 = matrix(0., (1, m - 1))
    A = matrix([[h1], [h2], [h3]])

    b = matrix(m, tc='d')

    sol = solvers.qp(Q, q, A=A, b=b)
    if sol['status'] != 'optimal':
        warn('calibrate() failed to find an optimal solution')
    ratings = np.array(sol['x'][0:n]).flatten()
    bias = np.array(sol['x'][n:n + m]).flatten()
    leniency = np.append(np.array(sol['x'][n + m:]).flatten(), 0.)

    if scale != (0, 0):
        ratings = np.interp(ratings, (ratings.min(), ratings.max()), scale)
    return ratings, bias, leniency


def average(M: np.array):
    """
    Calculate average ratings.

    :param M: The matrix of ratings. -1 denotes null. Ratings should be equal or greater than zero.
    :return: The average ratings.
    """
    _check_rating_matrix(M)

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

    :param M: The matrix of ratings. -1 denotes null. Ratings should be equal or greater than zero.
    :return: The min rating value. None if no ratings.
    """
    _check_rating_matrix(M)

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

    :param M: The matrix of ratings. -1 denotes null. Ratings should be equal or greater than zero.
    :return: The max rating value. None if no ratings.
    """
    _check_rating_matrix(M)

    m, n = M.shape
    rat = None
    for i in range(m):
        for j in range(n):
            if M[i, j] < 0:
                continue
            if rat is None or M[i, j] > rat:
                rat = M[i, j]
    return rat
