import chainer
import chainer.functions as F
import chainer.links as L

from yukarin_autoreg.config import ModelConfig


class MaskGRU(L.NStepGRU):
    def __init__(self, out_size: int, disable_mask=False) -> None:
        super().__init__(n_layers=1, in_size=3, out_size=out_size, dropout=0.)
        self.disable_mask = disable_mask
        self.mask_w = None
        self.mask_b = None

    def __call__(self, c_array, f_array, curr_c_array, hidden_coarse=None, hidden_fine=None):
        """
        c: coarse
        f: fine
        :param c_array: float -1 ~ +1 (batch_size, N)
        :param f_array: float -1 ~ +1 (batch_size, N)
        :param curr_c_array: float -1 ~ +1 (batch_size, N)
        :param hidden_coarse: float (batch_size, half_hidden_size)
        :param hidden_fine: float (batch_size, half_hidden_size)
        :return:
            hidden_coarse: float (batch_size, N, half_hidden_size)
            hidden_fine: float (batch_size, N, half_hidden_size)
        """
        input = F.stack((c_array, f_array, curr_c_array), axis=2)  # shape: (batch_size, N, 3)
        if hidden_coarse is not None:
            hidden = F.concat((hidden_coarse, hidden_fine), axis=1)
            hidden = F.expand_dims(hidden, axis=0)
        else:
            hidden = None

        _, hidden = super().__call__(hx=hidden, xs=F.separate(input, axis=0))
        hidden = F.stack(hidden, axis=0)  # shape: (batch_size, N, hidden_size)
        hidden_coarse, hidden_fine = F.split_axis(hidden, 2, axis=2)  # shape: (batch_size, N, half_hidden_size)
        return hidden_coarse, hidden_fine

    def rnn(self, *args):
        ws = args[3][0]
        bs = args[4][0]

        if self.mask_w is None:
            mask = self.xp.ones_like(ws[0].data)
            mask[:ws[0].shape[0] // 2, 2] = 0
            self.mask_w = mask

        if self.mask_b is None:
            mask = self.xp.ones_like(bs[0].data)
            mask[:bs[0].shape[0] // 2] = 0
            self.mask_b = mask

        if not self.disable_mask:
            ws[0] *= self.mask_w
            ws[1] *= self.mask_w
            ws[2] *= self.mask_w
            bs[0] *= self.mask_b
            bs[1] *= self.mask_b
            bs[2] *= self.mask_b

        from chainer.functions.connection import n_step_gru as rnn
        return rnn.n_step_gru(*args)


class WaveRNN(chainer.Chain):
    def __init__(self, config: ModelConfig, disable_mask=False) -> None:
        super().__init__()

        self.half_bins = 2 ** (config.bit_size // 2)
        self.half_hidden_size = config.hidden_size // 2
        with self.init_scope():
            self.R = MaskGRU(out_size=config.hidden_size, disable_mask=disable_mask)
            self.O1 = L.Linear(self.half_hidden_size, self.half_hidden_size)
            self.O2 = L.Linear(self.half_hidden_size, self.half_bins)
            self.O3 = L.Linear(self.half_hidden_size, self.half_hidden_size)
            self.O4 = L.Linear(self.half_hidden_size, self.half_bins)

    def __call__(self, c_array, f_array, hidden_coarse=None, hidden_fine=None):
        """
        c: coarse
        f: fine
        :param c_array: float -1 ~ +1 (batch_size, N+1)
        :param f_array: float -1 ~ +1 (batch_size, N)
        :param hidden: float (batch_size, hidden_size)
        :return:
            out_c_array: float (batch_size, half_bins, N)
            out_f_array: float (batch_size, half_bins, N)
            hidden: float (batch_size, hidden_size)
        """
        out_c_array, out_f_array, hidden_coarse, hidden_fine = self.forward(
            c_array=c_array[:, :-1],
            f_array=f_array[:, :],
            curr_c_array=c_array[:, 1:],
            hidden_coarse=hidden_coarse,
            hidden_fine=hidden_fine,
        )
        return out_c_array, out_f_array, hidden_coarse, hidden_fine

    def forward(self, c_array, f_array, curr_c_array, hidden_coarse=None, hidden_fine=None):
        """
        :param c_array: float -1 ~ +1 (batch_size, N)
        :param f_array: float -1 ~ +1 (batch_size, N)
        :param curr_c_array: float -1 ~ +1 (batch_size, N)
        :param hidden_coarse: float (batch_size, half_hidden_size)
        :param hidden_fine: float (batch_size, half_hidden_size)
        :return:
            out_c_array: float (batch_size, half_bins, N)
            out_f_array: float (batch_size, half_bins, N)
            hidden_coarse: float (batch_size, half_hidden_size)
            hidden_fine: float (batch_size, half_hidden_size)
        """
        assert c_array.shape == f_array.shape == curr_c_array.shape

        batch_size = c_array.shape[0]
        length = c_array.shape[1]  # N

        hidden_coarse, hidden_fine = self.R(
            c_array=c_array,
            f_array=f_array,
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

    def forward_one(self, prev_c, prev_f, hidden_coarse=None, hidden_fine=None):
        """
        :param prev_c: float -1 ~ +1 (batch_size, )
        :param prev_f: float -1 ~ +1 (batch_size, )
        :param hidden_coarse: float (batch_size, half_hidden_size)
        :param hidden_fine: float (batch_size, half_hidden_size)
        :return:
            out_c: float (batch_size, half_bins)
            out_f: float (batch_size, half_bins)
            hidden_coarse: float (batch_size, half_hidden_size)
            hidden_fine: float (batch_size, half_hidden_size)
        """
        prev_c = F.expand_dims(prev_c, axis=1)  # shape: (batch_size, 1)
        prev_f = F.expand_dims(prev_f, axis=1)  # shape: (batch_size, 1)
        curr_c_dummy = self.xp.zeros_like(prev_c.data)  # shape: (batch_size, 1)

        new_hidden_coarse, _ = self.R(
            c_array=prev_c,
            f_array=prev_f,
            curr_c_array=curr_c_dummy,
            hidden_coarse=hidden_coarse,
            hidden_fine=hidden_fine,
        )  # shape: (batch_size, 1, half_hidden_size)
        new_hidden_coarse = new_hidden_coarse[:, 0, :]  # shape: (batch_size, half_hidden_size)

        out_c = self.O2(F.relu(self.O1(new_hidden_coarse)))  # shape: (batch_size, ?)
        curr_c = self.sampling(out_c)  # shape: (batch_size, )

        curr_c = F.expand_dims(curr_c, axis=1)  # shape: (batch_size, 1)

        _, new_hidden_fine = self.R(
            c_array=prev_c,
            f_array=prev_f,
            curr_c_array=curr_c,
            hidden_coarse=hidden_coarse,
            hidden_fine=hidden_fine,
        )  # shape: (batch_size, 1, half_hidden_size)
        new_hidden_fine = new_hidden_fine[:, 0, :]  # shape: (batch_size, half_hidden_size)

        out_f = self.O4(F.relu(self.O3(new_hidden_fine)))  # shape: (batch_size, ?)
        return out_c, out_f, new_hidden_coarse, new_hidden_fine

    def sampling(self, softmax_dist, maximum=True):
        xp = self.xp

        if maximum:
            indexes = xp.argmax(F.softmax(softmax_dist, axis=1).data, axis=1)
            sampled = xp.linspace(-1, 1, self.half_bins, dtype=softmax_dist.dtype)[indexes]
        else:
            prob_list = F.softmax(softmax_dist, axis=1)
            sampled = xp.array([
                xp.random.choice(xp.linspace(-1, 1, self.half_bins, dtype=prob.dtype), p=prob)
                for prob in prob_list.data
            ])
        return sampled


def create_predictor(config: ModelConfig):
    predictor = WaveRNN(config)
    return predictor
