import numpy
from chainer import initializer
from chainer.initializers import Orthogonal
from chainer.initializers import LeCunNormal


class PossibleOrthogonal(initializer.Initializer):
    def __init__(self):
        super().__init__()

        self.orthogonal = Orthogonal()
        self.lecun = LeCunNormal()

    def __call__(self, array):
        flat_shape = (len(array), int(numpy.prod(array.shape[1:])))
        if flat_shape[0] > flat_shape[1]:
            self.lecun(array)
        else:
            self.orthogonal(array)
