import chainer
import chainer.functions as F
import numpy as np

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
        with chainer.using_device(self.device):
            hc = chainer.Variable(self.xp.zeros((batch_size, self.out_size // 2), dtype=dtype))
            hf = chainer.Variable(self.xp.zeros((batch_size, self.out_size // 2), dtype=dtype))
        return hc, hf

    def rnn(self, *args):
        ws = args[3][0]

        if self.mask_w is None:
            with chainer.using_device(self.device):
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
