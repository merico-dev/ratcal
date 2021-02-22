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


def sort_rat(name_list: list, ratings: list, reverse: bool = True):
    """
    Sort ratings associated with names. The sort is stable.

    :param name_list: The list of names to be associated with the ratings respectively
    :param ratings: The list of ratings to sort
    :param reverse: Descending by default. False means ascending.
    :return: The sorted list of names and their ratings and rankings
    """
    if len(name_list) != len(ratings):
        raise ValueError("# of names %d does not equal to # of ratings %d" % (len(name_list), len(ratings)))
    result = list()
    for i, name in enumerate(name_list):
        result.append((name, ratings[i]))

    def compare(name_rat):
        return name_rat[1]
    result.sort(key=compare, reverse=reverse)
    return [(r[0], r[1], i + 1) for i, r in enumerate(result)]
