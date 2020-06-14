from typing import Optional

import numpy as np
from chainer import initializer
from chainer.initializers import LeCunNormal, Orthogonal


class PossibleOrthogonal(initializer.Initializer):
    def __init__(self):
        super().__init__()

        self.orthogonal = Orthogonal()
        self.lecun = LeCunNormal()

    def __call__(self, array):
        flat_shape = (len(array), int(np.prod(array.shape[1:])))
        if flat_shape[0] > flat_shape[1]:
            self.lecun(array)
        else:
            self.orthogonal(array)


def get_weight_initializer(name: Optional[str]):
    if name is None:
        return None
    elif name == "GlorotNormal":
        from chainer.initializers.normal import GlorotNormal

        initializer = GlorotNormal()
    elif name == "HeNormal":
        from chainer.initializers.normal import HeNormal

        initializer = HeNormal()
    elif name == "LeCunNormal":
        from chainer.initializers.normal import LeCunNormal

        initializer = LeCunNormal()
    elif name == "Normal":
        from chainer.initializers.normal import Normal

        initializer = Normal()
    elif name == "Orthogonal":
        from chainer.initializers.orthogonal import Orthogonal

        initializer = Orthogonal()
    elif name == "GlorotUniform":
        from chainer.initializers.uniform import GlorotUniform

        initializer = GlorotUniform()
    elif name == "HeUniform":
        from chainer.initializers.uniform import HeUniform

        initializer = HeUniform()
    elif name == "LeCunUniform":
        from chainer.initializers.uniform import LeCunUniform

        initializer = LeCunUniform()
    elif name == "Uniform":
        from chainer.initializers.uniform import Uniform

        initializer = Uniform()
    elif name == "PossibleOrthogonal":
        initializer = PossibleOrthogonal()
    else:
        raise ValueError(name)
    return initializer
