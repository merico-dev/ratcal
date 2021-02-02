import numpy as np
from ratcal import calibrate


def test_calibrate():
    m = np.array([[2, 0, 2],
                  [1, 0, 1]])
    assert calibrate(m) == (2, 3)
