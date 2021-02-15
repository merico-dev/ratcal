import numpy as np


def is_rated(M: np.array, index: (int, int)):
    if M[index] < 0 and M[index] != -1:
        raise ValueError("M[%d, %d] = %f should be -1 denoting null" % (index[0], index[1], M[index]))
    return M[index] >= 0


def is_null(M: np.array, index: (int, int)):
    return not is_rated(M, index)


def check_rating_matrix(M: np.array):
    if M.ndim != 2:
        raise ValueError("M must be two-dimension, got %d dimension(s)" % M.ndim)
    for index in np.ndindex(M.shape):
        is_rated(M, index)
