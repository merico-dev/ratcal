import numpy as np
from warnings import warn
from cvxopt import matrix
from cvxopt import spmatrix
from cvxopt import solvers


def calibrate(M: np.array, scale=(0, 0)):
    """
    Calibrate ratings and evaluate raters.

    :param M: The matrix of ratings.
    :param scale: The range that the ratings are scaled to.
    :return:
    """

    if M.ndim != 2:
        raise ValueError("M must be two-dimension, got %d dimension(s)" % M.ndim)
    # m is the number of raters
    # n is the number of objects being rated
    m, n = M.shape

    H = spmatrix([], [], [], (n + 2 * m, m * n))

    for i in range(m):
        for j in range(n):
            if M[i, j] < 0 and M[i, j] != -1:
                raise ValueError("M[%d, %d] = %f should be -1 denoting null" % (i, j, M[i, j]))

            if M[i, j] >= 0:
                rat = j * m + i
                H[j, rat] = 1
                H[n + i, rat] = -M[i, j]
                H[n + m + i, rat] = 1

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
    z = np.array(sol['x'][0:n]).flatten()
    p = np.array(sol['x'][n:n + m]).flatten()
    q = np.append(np.array(sol['x'][n + m:]).flatten(), 0.)

    if scale != (0, 0):
        z = np.interp(z, (z.min(), z.max()), scale)
    return z, p, q
