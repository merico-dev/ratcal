import numpy as np
from cvxopt import spmatrix


def calibrate(M: np.array):
    """
    Calibrate ratings and evaluate raters.

    :param M: The matrix of ratings.
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

    return H
