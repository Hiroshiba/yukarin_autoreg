from typing import List, Optional, Union

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

from yukarin_autoreg.network.residual_encoder import ResidualEncoder
from yukarin_autoreg.network.up_conv import UpConv
from yukarin_autoreg.utility.chainer_initializer_utility import PossibleOrthogonal
from yukarin_autoreg.utility.chainer_network_utility import ModifiedNStepGRU

ArrayLike = Union[np.ndarray, chainer.Variable]


def _call_1layer(net: ModifiedNStepGRU, hidden: ArrayLike, input: ArrayLike):
    if hidden is not None:
        hidden = hidden[np.newaxis]
    _, hidden = net(hx=hidden, xs=F.separate(input, axis=0))
    hidden = F.stack(hidden, axis=0)
    return hidden


def _call_1step(net: ModifiedNStepGRU, hidden: ArrayLike, input: ArrayLike):
    if hidden is None:
        hidden = net.init_hx(input)[0]

    x = input
    h = hidden
    w = net.ws[0]
    b = net.bs[0]

    xw = F.concat([w[0], w[1], w[2]], axis=0)
    hw = F.concat([w[3], w[4], w[5]], axis=0)
    xb = F.concat([b[0], b[1], b[2]], axis=0)
    hb = F.concat([b[3], b[4], b[5]], axis=0)

    gru_x = F.linear(x, xw, xb)
    gru_h = F.linear(h, hw, hb)

    W_r_x, W_z_x, W_x = F.split_axis(gru_x, 3, axis=1)
    U_r_h, U_z_h, U_x = F.split_axis(gru_h, 3, axis=1)

    r = F.sigmoid(W_r_x + U_r_h)
    z = F.sigmoid(W_z_x + U_z_h)
    h_bar = F.tanh(W_x + r * U_x)
    h = F.linear_interpolate(z, hidden, h_bar)
    return h


class WaveRNN(chainer.Chain):
    def __init__(
            self,
            upconv_scales: List[int],
            upconv_residual: bool,
            upconv_channel_ksize: int,
            residual_encoder_channel: Optional[int],
            residual_encoder_num_block: Optional[int],
            dual_softmax: bool,
            bit_size: int,
            hidden_size: int,
            local_size: int,
    ) -> None:
        super().__init__()
        initialW = PossibleOrthogonal()

        self.local_size = local_size
        self.dual_softmax = dual_softmax
        self.bit_size = bit_size
        self.hidden_size = hidden_size
        with self.init_scope():
            self.upconv = UpConv(
                scales=upconv_scales,
                residual=upconv_residual,
                c_ksize=upconv_channel_ksize,
                initialW=initialW,
            )
            self.residual_encoder = ResidualEncoder(
                n_channel=residual_encoder_channel,
                num_block=residual_encoder_num_block,
                initialW=initialW,
            ) if (residual_encoder_num_block is not None) and (residual_encoder_channel is not None) else None
            self.R_coarse = ModifiedNStepGRU(
                n_layers=1,
                in_size=(2 if self.dual_softmax else 1) + local_size,
                out_size=self.single_hidden_size,
                dropout=0.,
                initialW=initialW,
            )
            self.O1 = L.Linear(self.single_hidden_size, self.single_hidden_size, initialW=initialW)
            self.O2 = L.Linear(self.single_hidden_size, self.single_bins, initialW=initialW)

            if self.dual_softmax:
                self.R_fine = ModifiedNStepGRU(
                    n_layers=1,
                    in_size=3 + local_size,
                    out_size=self.single_hidden_size,
                    dropout=0.,
                    initialW=initialW,
                )
                self.O3 = L.Linear(self.single_hidden_size, self.single_hidden_size, initialW=initialW)
                self.O4 = L.Linear(self.single_hidden_size, self.single_bins, initialW=initialW)
            else:
                self.R_fine = None
                self.O3 = None
                self.O4 = None

    @property
    def single_bins(self):
        if self.dual_softmax:
            return 2 ** (self.bit_size // 2)
        else:
            return 2 ** self.bit_size

    @property
    def single_hidden_size(self):
        if self.dual_softmax:
            return self.hidden_size // 2
        else:
            return self.hidden_size

    def __call__(
            self,
            c_array: ArrayLike,
            f_array: Optional[ArrayLike],
            l_array: ArrayLike,
            hidden_coarse: ArrayLike = None,
            hidden_fine: ArrayLike = None,
    ):
        """
        c: coarse
        f: fine
        l: local
        :param c_array: float -1 ~ +1 (batch_size, N+1)
        :param f_array: float -1 ~ +1 (batch_size, N)
        :param l_array: float (batch_size, lN, ?)
        :param hidden_coarse: float (batch_size, single_hidden_size)
        :param hidden_fine: float (batch_size, single_hidden_size)
        :return:
            out_c_array: float (batch_size, single_bins, N)
            out_f_array: float (batch_size, single_bins, N)
            hidden: float (batch_size, hidden_size)
        """
        assert l_array.shape[2] == self.local_size, f'{l_array.shape[2]} {self.local_size}'

        l_array = self.forward_encode(l_array)  # (batch_size, N, ?)
        out_c_array, out_f_array, hidden_coarse, hidden_fine = self.forward_rnn(
            c_array=c_array[:, :-1],
            f_array=f_array,
            l_array=l_array[:, 1:],
            curr_c_array=c_array[:, 1:],
            hidden_coarse=hidden_coarse,
            hidden_fine=hidden_fine,
        )
        return out_c_array, out_f_array, hidden_coarse, hidden_fine

    def forward_encode(self, l_array: ArrayLike):
        """
        :param l_array: float (batch_size, lN, ?)
        :return:
            l_array: float (batch_size, N, ?)
        """
        if self.local_size == 0:
            return l_array

        if self.residual_encoder is not None:
            l_array = self.residual_encoder(l_array)

        return self.upconv(l_array)

    def forward_rnn(
            self,
            c_array: ArrayLike,
            f_array: Optional[ArrayLike],
            l_array: ArrayLike,
            curr_c_array: Optional[ArrayLike],
            hidden_coarse: ArrayLike = None,
            hidden_fine: ArrayLike = None,
    ):
        """
        :param c_array: float -1 ~ +1 (batch_size, N)
        :param f_array: float -1 ~ +1 (batch_size, N)
        :param l_array: (batch_size, N, ?)
        :param curr_c_array: float -1 ~ +1 (batch_size, N)
        :param hidden_coarse: float (batch_size, single_hidden_size)
        :param hidden_fine: float (batch_size, single_hidden_size)
        :return:
            out_c_array: float (batch_size, single_bins, N)
            out_f_array: float (batch_size, single_bins, N)
            hidden_coarse: float (batch_size, single_hidden_size)
            hidden_fine: float (batch_size, single_hidden_size)
        """
        if self.dual_softmax:
            assert c_array.shape == f_array.shape == l_array.shape[:2] == curr_c_array.shape, \
                f'{c_array.shape}, {f_array.shape}, {l_array.shape[:2]}, {curr_c_array.shape}'
        else:
            assert c_array.shape == l_array.shape[:2], \
                f'{c_array.shape}, {l_array.shape[:2]}'

        batch_size = c_array.shape[0]
        length = c_array.shape[1]  # N

        # coarse
        if self.dual_softmax:
            c_inputs = (c_array[:, :, np.newaxis], f_array[:, :, np.newaxis], l_array)
        else:
            c_inputs = (c_array[:, :, np.newaxis], l_array)
        xc_array = F.concat(c_inputs, axis=2)  # (batch_size, N, L + 1)

        hidden_coarse = _call_1layer(self.R_coarse, hidden_coarse, xc_array)  # (batch_size, N, single_hidden_size)
        new_hidden_coarse = hidden_coarse[:, -1, :]

        hidden_coarse = F.reshape(hidden_coarse, shape=(batch_size * length, -1))  # (batch_size * N, ?)
        out_c_array = self.O2(F.relu(self.O1(hidden_coarse)))  # (batch_size * N, ?)
        out_c_array = F.reshape(out_c_array, shape=(batch_size, length, -1))  # (batch_size, N, ?)
        out_c_array = F.transpose(out_c_array, axes=(0, 2, 1))  # (batch_size, ?, N)

        # fine
        if self.dual_softmax:
            f_inputs = (xc_array, curr_c_array[:, :, np.newaxis])
            xf_array = F.concat(f_inputs, axis=2)  # (batch_size, N, L + 1)

            hidden_fine = _call_1layer(self.R_fine, hidden_fine, xf_array)  # (batch_size, N, single_hidden_size)
            new_hidden_fine = hidden_fine[:, -1, :]

            hidden_fine = F.reshape(hidden_fine, shape=(batch_size * length, -1))  # (batch_size * N, ?)
            out_f_array = self.O4(F.relu(self.O3(hidden_fine)))  # (batch_size * N, ?)
            out_f_array = F.reshape(out_f_array, shape=(batch_size, length, -1))  # (batch_size, N, ?)
            out_f_array = F.transpose(out_f_array, axes=(0, 2, 1))  # (batch_size, ?, N)
        else:
            out_f_array = None
            new_hidden_fine = None

        return out_c_array, out_f_array, new_hidden_coarse, new_hidden_fine

    def forward_one(
            self,
            prev_c: ArrayLike,
            prev_f: Optional[ArrayLike],
            prev_l: ArrayLike,
            prev_corr_c: ArrayLike = None,
            hidden_coarse: ArrayLike = None,
            hidden_fine: ArrayLike = None,
    ):
        """
        :param prev_c: float -1 ~ +1 (batch_size, )
        :param prev_f: float -1 ~ +1 (batch_size, )
        :param prev_l: (batch_size, ?)
        :param prev_corr_c: float -1 ~ +1 (batch_size, )
        :param hidden_coarse: float (batch_size, single_hidden_size)
        :param hidden_fine: float (batch_size, single_hidden_size)
        :return:
            out_c: float (batch_size, single_bins)
            out_f: float (batch_size, single_bins)
            hidden_coarse: float (batch_size, single_hidden_size)
            hidden_fine: float (batch_size, single_hidden_size)
        """
        if self.dual_softmax:
            inputs = (prev_c[:, np.newaxis], prev_f[:, np.newaxis], prev_l)
        else:
            inputs = (prev_c[:, np.newaxis], prev_l)
        prev_x = F.concat(inputs, axis=1)  # (batch_size, L+1)

        new_hidden_coarse = _call_1step(
            self.R_coarse,
            hidden_coarse,
            prev_x,
        )  # (batch_size, single_hidden_size)

        out_c = self.O2(F.relu(self.O1(new_hidden_coarse)))  # (batch_size, ?)

        if self.dual_softmax:
            if prev_corr_c is None:
                curr_c = self.sampling(out_c).astype(prev_c.dtype) / 127.5 - 1  # (batch_size, )
            else:
                curr_c = prev_corr_c

            new_hidden_fine = _call_1step(
                self.R_fine,
                hidden_fine,
                F.concat((prev_x, curr_c[:, np.newaxis]), axis=1),
            )  # (batch_size, single_hidden_size)

            out_f = self.O4(F.relu(self.O3(new_hidden_fine)))  # (batch_size, ?)
        else:
            out_f = None
            new_hidden_fine = None
        return out_c, out_f, new_hidden_coarse, new_hidden_fine

    def sampling(self, softmax_dist: ArrayLike, maximum=True):
        xp = self.xp

        if maximum:
            sampled = xp.argmax(F.softmax(softmax_dist, axis=1).data, axis=1)
        else:
            prob_np = lambda x: x if isinstance(x, np.ndarray) else x.get()  # cupy don't have random.choice method

            prob_list = F.softmax(softmax_dist, axis=1)
            sampled = xp.array([
                np.random.choice(np.arange(self.single_bins), p=prob_np(prob))
                for prob in prob_list.data
            ])
        return sampled
