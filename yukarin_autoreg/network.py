import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

from yukarin_autoreg.config import ModelConfig
from yukarin_autoreg.utility.chainer_initializer_utility import PossibleOrthogonal
from yukarin_autoreg.utility.chainer_network_utility import ModifiedNStepGRU


class MaskGRU(ModifiedNStepGRU):
    def __init__(self, in_size: int, out_size: int, initialW=None, disable_mask=False) -> None:
        super().__init__(
            n_layers=1,
            in_size=in_size,
            out_size=out_size,
            dropout=0.,
            initialW=initialW,
        )
        self.in_size = in_size
        self.disable_mask = disable_mask
        self.mask_w = None

    def __call__(self, x_array, curr_c_array, hidden_coarse=None, hidden_fine=None):
        """
        :param x_array: float -1 ~ +1 (batch_size, N, L)
        :param curr_c_array: float -1 ~ +1 (batch_size, N)
        :param hidden_coarse: float (batch_size, half_hidden_size)
        :param hidden_fine: float (batch_size, half_hidden_size)
        :return:
            hidden_coarse: float (batch_size, N, half_hidden_size)
            hidden_fine: float (batch_size, N, half_hidden_size)
        """
        assert x_array.shape[2] == self.in_size - 1

        batch_size = x_array.shape[0]

        if hidden_coarse is None:
            hidden_coarse, hidden_fine = self.get_init_hidden(batch_size)

        input = F.concat((x_array, curr_c_array[:, :, np.newaxis]), axis=2)  # shape: (batch_size, N, L + 1)
        hidden = F.concat((hidden_coarse, hidden_fine), axis=1)
        hidden = F.expand_dims(hidden, axis=0)

        _, hidden = super().__call__(hx=hidden, xs=F.separate(input, axis=0))
        hidden = F.stack(hidden, axis=0)  # shape: (batch_size, N, hidden_size)
        hidden_coarse, hidden_fine = F.split_axis(hidden, 2, axis=2)  # shape: (batch_size, N, half_hidden_size)
        return hidden_coarse, hidden_fine

    def get_init_hidden(self, batch_size: int, dtype=np.float32):
        with chainer.cuda.get_device_from_id(self._device_id):
            hc = chainer.Variable(self.xp.zeros((batch_size, self.out_size // 2), dtype=dtype))
            hf = chainer.Variable(self.xp.zeros((batch_size, self.out_size // 2), dtype=dtype))
        return hc, hf

    def rnn(self, *args):
        ws = args[3][0]

        if self.mask_w is None:
            with chainer.cuda.get_device_from_id(self._device_id):
                mask = self.xp.ones_like(ws[0].data)
            mask[:ws[0].shape[0] // 2, -1] = 0
            self.mask_w = mask

        if not self.disable_mask:
            ws[0] *= self.mask_w
            ws[1] *= self.mask_w
            ws[2] *= self.mask_w

        from chainer.functions.connection import n_step_gru as rnn
        return rnn.n_step_gru(*args)

    def onestep_coarse(self, x_array, hidden_coarse=None, hidden_fine=None):
        """
        :param x_array: float -1 ~ +1 (batch_size, L)
        :param hidden_coarse: float (batch_size, half_hidden_size)
        :param hidden_fine: float (batch_size, half_hidden_size)
        :return:
            hidden_coarse: float (batch_size, half_hidden_size)
        """
        batch_size = x_array.shape[0]

        if hidden_coarse is None:
            hidden_coarse, hidden_fine = self.get_init_hidden(batch_size)

        x = x_array
        h = F.concat((hidden_coarse, hidden_fine), axis=1)
        w = [w[:self.out_size // 2] for w in self.ws[0]]
        b = [b[:self.out_size // 2] for b in self.bs[0]]

        w[0] = w[0][:, :-1]
        w[1] = w[1][:, :-1]
        w[2] = w[2][:, :-1]

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
        return F.linear_interpolate(z, hidden_coarse, h_bar)

    def onestep_fine(self, x_array, curr_c_array, hidden_coarse=None, hidden_fine=None):
        """
        :param x_array: float -1 ~ +1 (batch_size, L)
        :param curr_c_array: float -1 ~ +1 (batch_size, )
        :param hidden_coarse: float (batch_size, half_hidden_size)
        :param hidden_fine: float (batch_size, half_hidden_size)
        :return:
            hidden_coarse: float (batch_size, half_hidden_size)
        """
        batch_size = x_array.shape[0]

        if hidden_coarse is None:
            hidden_coarse, hidden_fine = self.get_init_hidden(batch_size)

        x = F.concat((x_array, curr_c_array[:, np.newaxis]), axis=1)  # shape: (batch_size, L + 1)
        h = F.concat((hidden_coarse, hidden_fine), axis=1)
        w = [w[self.out_size // 2:] for w in self.ws[0]]
        b = [b[self.out_size // 2:] for b in self.bs[0]]

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
        return F.linear_interpolate(z, hidden_fine, h_bar)


class WaveRNN(chainer.Chain):
    def __init__(self, config: ModelConfig, disable_mask=False) -> None:
        super().__init__()
        initialW = PossibleOrthogonal()

        self.half_bins = 2 ** (config.bit_size // 2)
        self.half_hidden_size = config.hidden_size // 2
        with self.init_scope():
            self.R = MaskGRU(
                in_size=3 + config.local_size,
                out_size=config.hidden_size,
                initialW=initialW,
                disable_mask=disable_mask,
            )
            self.O1 = L.Linear(self.half_hidden_size, self.half_hidden_size, initialW=initialW)
            self.O2 = L.Linear(self.half_hidden_size, self.half_bins, initialW=initialW)
            self.O3 = L.Linear(self.half_hidden_size, self.half_hidden_size, initialW=initialW)
            self.O4 = L.Linear(self.half_hidden_size, self.half_bins, initialW=initialW)

    def __call__(self, c_array, f_array, l_array, hidden_coarse=None, hidden_fine=None):
        """
        c: coarse
        f: fine
        l: local
        :param c_array: float -1 ~ +1 (batch_size, N+1)
        :param f_array: float -1 ~ +1 (batch_size, N)
        :param l_array: float (batch_size, N, ?)
        :param hidden: float (batch_size, hidden_size)
        :return:
            out_c_array: float (batch_size, half_bins, N)
            out_f_array: float (batch_size, half_bins, N)
            hidden: float (batch_size, hidden_size)
        """
        out_c_array, out_f_array, hidden_coarse, hidden_fine = self.forward(
            c_array=c_array[:, :-1],
            f_array=f_array,
            l_array=l_array,
            curr_c_array=c_array[:, 1:],
            hidden_coarse=hidden_coarse,
            hidden_fine=hidden_fine,
        )
        return out_c_array, out_f_array, hidden_coarse, hidden_fine

    def forward(self, c_array, f_array, l_array, curr_c_array, hidden_coarse=None, hidden_fine=None):
        """
        :param c_array: float -1 ~ +1 (batch_size, N)
        :param f_array: float -1 ~ +1 (batch_size, N)
        :param l_array: float -1 ~ +1 (batch_size, N, ?)
        :param curr_c_array: float -1 ~ +1 (batch_size, N)
        :param hidden_coarse: float (batch_size, half_hidden_size)
        :param hidden_fine: float (batch_size, half_hidden_size)
        :return:
            out_c_array: float (batch_size, half_bins, N)
            out_f_array: float (batch_size, half_bins, N)
            hidden_coarse: float (batch_size, half_hidden_size)
            hidden_fine: float (batch_size, half_hidden_size)
        """
        assert c_array.shape == f_array.shape == l_array.shape[:2] == curr_c_array.shape

        batch_size = c_array.shape[0]
        length = c_array.shape[1]  # N

        # shape: (batch_size, N, L + 1)
        x_array = F.concat((c_array[:, :, np.newaxis], f_array[:, :, np.newaxis], l_array), axis=2)

        hidden_coarse, hidden_fine = self.R(
            x_array=x_array,
            curr_c_array=curr_c_array,
            hidden_coarse=hidden_coarse,
            hidden_fine=hidden_fine,
        )  # shape: (batch_size, N, half_hidden_size)

        new_hidden_coarse = hidden_coarse[:, -1, :]
        new_hidden_fine = hidden_fine[:, -1, :]

        hidden_coarse = F.reshape(hidden_coarse, shape=(batch_size * length, -1))  # shape: (batch_size * N, ?)
        hidden_fine = F.reshape(hidden_fine, shape=(batch_size * length, -1))  # shape: (batch_size * N, ?)

        # Compute outputs
        out_c_array = self.O2(F.relu(self.O1(hidden_coarse)))  # shape: (batch_size * N, ?)
        out_f_array = self.O4(F.relu(self.O3(hidden_fine)))  # shape: (batch_size * N, ?)
        out_c_array = F.reshape(out_c_array, shape=(batch_size, length, -1))  # shape: (batch_size, N, ?)
        out_f_array = F.reshape(out_f_array, shape=(batch_size, length, -1))  # shape: (batch_size, N, ?)
        out_c_array = F.transpose(out_c_array, axes=(0, 2, 1))  # shape: (batch_size, ?, N)
        out_f_array = F.transpose(out_f_array, axes=(0, 2, 1))  # shape: (batch_size, ?, N)

        return out_c_array, out_f_array, new_hidden_coarse, new_hidden_fine

    def forward_one(self, prev_c, prev_f, prev_l, prev_corr_c=None, hidden_coarse=None, hidden_fine=None):
        """
        :param prev_c: float -1 ~ +1 (batch_size, )
        :param prev_f: float -1 ~ +1 (batch_size, )
        :param prev_l: float -1 ~ +1 (batch_size, ?)
        :param prev_corr_c: float -1 ~ +1 (batch_size, )
        :param hidden_coarse: float (batch_size, half_hidden_size)
        :param hidden_fine: float (batch_size, half_hidden_size)
        :return:
            out_c: float (batch_size, half_bins)
            out_f: float (batch_size, half_bins)
            hidden_coarse: float (batch_size, half_hidden_size)
            hidden_fine: float (batch_size, half_hidden_size)
        """
        prev_x = F.concat((prev_c[:, np.newaxis], prev_f[:, np.newaxis], prev_l), axis=1)

        new_hidden_coarse = self.R.onestep_coarse(
            x_array=prev_x,
            hidden_coarse=hidden_coarse,
            hidden_fine=hidden_fine,
        )  # shape: (batch_size, half_hidden_size)

        out_c = self.O2(F.relu(self.O1(new_hidden_coarse)))  # shape: (batch_size, ?)

        if prev_corr_c is None:
            curr_c = self.sampling(out_c).astype(prev_c.dtype) / 127.5 - 1  # shape: (batch_size, )
        else:
            curr_c = prev_corr_c

        new_hidden_fine = self.R.onestep_fine(
            x_array=prev_x,
            curr_c_array=curr_c,
            hidden_coarse=hidden_coarse,
            hidden_fine=hidden_fine,
        )  # shape: (batch_size, half_hidden_size)

        out_f = self.O4(F.relu(self.O3(new_hidden_fine)))  # shape: (batch_size, ?)
        return out_c, out_f, new_hidden_coarse, new_hidden_fine

    def sampling(self, softmax_dist, maximum=True):
        xp = self.xp

        if maximum:
            sampled = xp.argmax(F.softmax(softmax_dist, axis=1).data, axis=1)
        else:
            prob_np = lambda x: x if isinstance(x, np.ndarray) else x.get()  # cupy don't have random.choice method

            prob_list = F.softmax(softmax_dist, axis=1)
            sampled = xp.array([
                np.random.choice(np.arange(self.half_bins), p=prob_np(prob))
                for prob in prob_list.data
            ])
        return sampled


def create_predictor(config: ModelConfig):
    predictor = WaveRNN(config)
    return predictor
