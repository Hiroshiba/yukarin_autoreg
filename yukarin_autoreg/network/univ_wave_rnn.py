from typing import Union

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer.links import NStepGRU, NStepBiGRU, EmbedID

from yukarin_autoreg.network.wave_rnn import _call_1layer, _call_1step

ArrayLike = Union[np.ndarray, chainer.Variable]


class UnivWaveRNN(chainer.Chain):
    def __init__(
            self,
            dual_softmax: bool,
            bit_size: int,
            conditioning_size: int,
            embedding_size: int,
            hidden_size: int,
            linear_hidden_size: int,
            local_size: int,
            local_scale: int,
    ) -> None:
        super().__init__()
        assert not dual_softmax

        self.dual_softmax = dual_softmax
        self.bit_size = bit_size
        self.conditioning_size = conditioning_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.linear_hidden_size = linear_hidden_size
        self.local_size = local_size
        self.local_scale = local_scale
        with self.init_scope():
            self.local_gru = NStepBiGRU(
                n_layers=2,
                in_size=local_size,
                out_size=conditioning_size,
                dropout=0,
            ) if local_size > 0 else None
            self.x_embedder = EmbedID(self.bins, embedding_size)
            self.gru = NStepGRU(
                n_layers=1,
                in_size=embedding_size + (2 * conditioning_size if local_size > 0 else 0),
                out_size=hidden_size,
                dropout=0,
            )
            self.O1 = L.Linear(hidden_size, linear_hidden_size)
            self.O2 = L.Linear(linear_hidden_size, self.bins)

    @property
    def bins(self):
        return 2 ** self.bit_size

    def __call__(
            self,
            x_array: ArrayLike,
            l_array: ArrayLike,
            local_padding_size: int = 0,
            hidden: ArrayLike = None,
    ):
        """
        x: wave
        l: local
        :param x_array: int (batch_size, N+1)
        :param l_array: float (batch_size, lN, ?)
        :param local_padding_size:
        :param hidden: float (batch_size, hidden_size)
        :return:
            out_x_array: float (batch_size, bins, N)
            hidden: float (batch_size, hidden_size)
        """
        assert l_array.shape[2] == self.local_size, f'{l_array.shape[2]} {self.local_size}'

        l_array = self.forward_encode(l_array)  # (batch_size, N + pad, ?)
        if local_padding_size > 0:
            l_array = l_array[:, local_padding_size:-local_padding_size]  # (batch_size, N, ?)
        out_x_array, hidden = self.forward_rnn(
            x_array=x_array[:, :-1],
            l_array=l_array[:, 1:],
            hidden=hidden,
        )
        return out_x_array, hidden

    def forward_encode(self, l_array: ArrayLike):
        """
        :param l_array: float (batch_size, lN, ?)
        :return:
            l_array: float (batch_size, N, ?)
        """
        if self.local_size == 0:
            return l_array

        _, l_array = self.local_gru(hx=None, xs=F.separate(l_array, axis=0))
        l_array = F.stack(l_array, axis=0)

        l_array = F.expand_dims(l_array, axis=1)  # shape: (batch_size, 1, lN, ?)
        l_array = F.unpooling_2d(l_array, ksize=(self.local_scale, 1), stride=(self.local_scale, 1), cover_all=False)
        l_array = F.squeeze(l_array, axis=1)  # shape: (batch_size, N, ?)
        return l_array

    def forward_rnn(
            self,
            x_array: ArrayLike,
            l_array: ArrayLike,
            hidden: ArrayLike = None,
    ):
        """
        :param x_array: int (batch_size, N)
        :param l_array: (batch_size, N, ?)
        :param hidden: float (batch_size, hidden_size)
        :return:
            out_x_array: float (batch_size, bins, N)
            hidden: float (batch_size, hidden_size)
        """
        assert x_array.shape == l_array.shape[:2], f'{x_array.shape}, {l_array.shape[:2]}'

        batch_size = x_array.shape[0]
        length = x_array.shape[1]  # N

        x_array = x_array.reshape([batch_size * length, 1])  # (batchsize * N, 1)
        x_array = self.x_embedder(x_array).reshape([batch_size, length, -1])  # (batch_size, N, ?)

        xl_array = F.concat((x_array, l_array), axis=2)  # (batch_size, N, ?)

        out_hidden = _call_1layer(self.gru, hidden, xl_array)  # (batch_size, N, hidden_size)
        new_hidden = out_hidden[:, -1, :]

        out_hidden = F.reshape(out_hidden, shape=(batch_size * length, -1))  # (batch_size * N, ?)
        out_x_array = self.O2(F.relu(self.O1(out_hidden)))  # (batch_size * N, ?)
        out_x_array = F.reshape(out_x_array, shape=(batch_size, length, -1))  # (batch_size, N, ?)
        out_x_array = F.transpose(out_x_array, axes=(0, 2, 1))  # (batch_size, ?, N)

        return out_x_array, new_hidden

    def forward_one(
            self,
            prev_x: ArrayLike,
            prev_l: ArrayLike,
            hidden: ArrayLike = None,
    ):
        """
        :param prev_x: int (batch_size, )
        :param prev_l: (batch_size, ?)
        :param hidden: float (batch_size, single_hidden_size)
        :return:
            out_x: float (batch_size, bins)
            hidden: float (batch_size, hidden_size)
        """
        batch_size = prev_x.shape[0]

        prev_x = self.x_embedder(prev_x[:, np.newaxis]).reshape([batch_size, -1])  # (batch_size, ?)
        prev_xl = F.concat((prev_x, prev_l), axis=1)  # (batch_size, ?)

        new_hidden = _call_1step(
            self.gru,
            hidden,
            prev_xl,
            bug_fixed_gru_dimension=True,
        )  # (batch_size, single_hidden_size)

        out_x = self.O2(F.relu(self.O1(new_hidden)))  # (batch_size, ?)

        return out_x, new_hidden

    def sampling(self, softmax_dist: ArrayLike, maximum=True):
        xp = self.xp

        if maximum:
            sampled = xp.argmax(F.softmax(softmax_dist, axis=1).data, axis=1)
        else:
            prob_np = lambda x: x if isinstance(x, np.ndarray) else x.get()  # cupy don't have random.choice method

            prob_list = F.softmax(softmax_dist, axis=1)
            sampled = xp.array([
                np.random.choice(np.arange(self.bins), p=prob_np(prob))
                for prob in prob_list.data
            ])
        return sampled
