import random
import numpy as np
from warnings import warn
from cvxopt import matrix
from cvxopt import spmatrix
from cvxopt import solvers


def check_rating_matrix(M: np.array):
    if M.ndim != 2:
        raise ValueError("M must be two-dimension, got %d dimension(s)" % M.ndim)
    for i, v in np.ndenumerate(M):
        if v < 0 and v != -1:
            raise ValueError("M[%d, %d] = %f should be -1 denoting null" % (i[0], i[1], v))


def calibrate(M: np.array, noise=(0, 0), scale=(0, 0)):
    """
    Calibrate ratings and evaluate raters.

    :param M: The matrix of ratings.
    :param noise: The range of random, tiny noise added to ratings to
           avoid linear dependence (when certain matrix ranks are low).
    :param scale: The range that the ratings are scaled to.
    :return:
    """

    check_rating_matrix(M)

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
