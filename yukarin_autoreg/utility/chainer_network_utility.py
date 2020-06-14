import chainer
import numpy
from chainer import initializers, link, variable
from chainer.functions.array import permutate, transpose_sequence
from chainer.initializers import normal
from chainer.utils import argument


def argsort_list_descent(lst):
    return numpy.argsort([-len(x.data) for x in lst]).astype("i")


def permutate_list(lst, indices, inv):
    ret = [None] * len(lst)
    if inv:
        for i, ind in enumerate(indices):
            ret[ind] = lst[i]
    else:
        for i, ind in enumerate(indices):
            ret[i] = lst[ind]
    return ret


class ModifiedNStepRNNBase(link.ChainList):
    """for initialW"""

    def __init__(
        self,
        n_layers,
        in_size,
        out_size,
        dropout,
        initialW=None,
        initial_bias=None,
        **kwargs
    ):
        if kwargs:
            argument.check_unexpected_kwargs(
                kwargs,
                use_cudnn="use_cudnn argument is not supported anymore. "
                "Use chainer.using_config",
                use_bi_direction="use_bi_direction is not supported anymore",
                activation="activation is not supported anymore",
            )
            argument.assert_kwargs_empty(kwargs)

        if initialW is None:
            W_initializer = normal.LeCunNormal()
        else:
            W_initializer = initializers._get_initializer(initialW)

        if initial_bias is None:
            initial_bias = 0
        bias_initializer = initializers._get_initializer(initial_bias)

        weights = []
        if self.use_bi_direction:
            direction = 2
        else:
            direction = 1

        for i in range(n_layers):
            for di in range(direction):
                weight = link.Link()
                with weight.init_scope():
                    for j in range(self.n_weights):
                        if i == 0 and j < self.n_weights // 2:
                            w_in = in_size
                        elif i > 0 and j < self.n_weights // 2:
                            w_in = out_size * direction
                        else:
                            w_in = out_size
                        w = variable.Parameter(W_initializer, (out_size, w_in))
                        b = variable.Parameter(bias_initializer, (out_size,))
                        setattr(weight, "w%d" % j, w)
                        setattr(weight, "b%d" % j, b)
                weights.append(weight)

        super().__init__(*weights)

        self.ws = [
            [getattr(layer, "w%d" % i) for i in range(self.n_weights)] for layer in self
        ]
        self.bs = [
            [getattr(layer, "b%d" % i) for i in range(self.n_weights)] for layer in self
        ]

        self.n_layers = n_layers
        self.dropout = dropout
        self.out_size = out_size
        self.direction = direction

    def init_hx(self, xs):
        shape = (self.n_layers * self.direction, len(xs), self.out_size)
        with chainer.using_device(self.device):
            hx = variable.Variable(self.xp.zeros(shape, dtype=xs[0].dtype))
        return hx

    def rnn(self, *args):
        raise NotImplementedError

    @property
    def n_cells(self):
        return NotImplementedError

    def forward(self, hx, xs, **kwargs):
        (hy,), ys = self._call([hx], xs, **kwargs)
        return hy, ys

    def _call(self, hs, xs, **kwargs):
        if kwargs:
            argument.check_unexpected_kwargs(
                kwargs,
                train="train argument is not supported anymore. "
                "Use chainer.using_config",
            )
            argument.assert_kwargs_empty(kwargs)

        assert isinstance(xs, (list, tuple))
        indices = argsort_list_descent(xs)

        assert self.ws[0][0].shape[1] == xs[0].shape[1]  # input check

        xs = permutate_list(xs, indices, inv=False)
        hxs = []
        for hx in hs:
            if hx is None:
                hx = self.init_hx(xs)
            else:
                hx = permutate.permutate(hx, indices, axis=1, inv=False)
            hxs.append(hx)

        trans_x = transpose_sequence.transpose_sequence(xs)

        args = [self.n_layers, self.dropout] + hxs + [self.ws, self.bs, trans_x]
        result = self.rnn(*args)

        hys = [permutate.permutate(h, indices, axis=1, inv=True) for h in result[:-1]]
        trans_y = result[-1]
        ys = transpose_sequence.transpose_sequence(trans_y)
        ys = permutate_list(ys, indices, inv=True)

        return hys, ys


class ModifiedNStepGRUBase(ModifiedNStepRNNBase):
    n_weights = 6


class ModifiedNStepGRU(ModifiedNStepGRUBase):
    use_bi_direction = False

    def rnn(self, *args):
        from chainer.functions.rnn import n_step_gru as rnn

        return rnn.n_step_gru(*args)

    @property
    def n_cells(self):
        return 1


class ModifiedNStepBiGRU(ModifiedNStepGRUBase):
    use_bi_direction = True

    def rnn(self, *args):
        from chainer.functions.rnn import n_step_gru as rnn

        return rnn.n_step_bigru(*args)

    @property
    def n_cells(self):
        return 1
