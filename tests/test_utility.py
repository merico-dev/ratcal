from ratcal import sort_rat


def test_sort_rat():
    name_list = []
    ratings = []
    result = sort_rat(name_list, ratings)
    assert len(result) == 0

    name_list = ['a']
    ratings = [0]
    result = sort_rat(name_list, ratings)
    assert result == [('a', 0)]

    name_list = ['a', 'b', 'c']
    ratings = [1, 3, 2]
    result = sort_rat(name_list, ratings)
    assert result == [('b', 3), ('c', 2), ('a', 1)]

    name_list = ['d', 'a', 'b', 'c']
    ratings = [1, 3, 3, 2]
    result = sort_rat(name_list, ratings, False)
    assert result == [('d', 1), ('c', 2), ('a', 3), ('b', 3)]
