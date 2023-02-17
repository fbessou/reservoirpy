import numpy as np
from numpy.testing import assert_array_equal

from ..perceptron import Perceptron


def test_perceptron():
    x = np.ones(2, 3)
    weight = np.ones(4, 3)
    biases = np.ones(1, 4)

    op = Perceptron(weight, biases)

    out = op(x)

    # ones-matrix multiplied by ones-vector should give the size of the vector
    # adding the bias this is size of the vector + 1
    assert_array_equal(out, 4 * np.ones((2, 4)))
