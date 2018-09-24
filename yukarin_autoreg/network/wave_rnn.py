import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

from yukarin_autoreg.utility.chainer_initializer_utility import PossibleOrthogonal
from yukarin_autoreg.utility.chainer_network_utility import ModifiedNStepGRU


def _call_1step(net: ModifiedNStepGRU, hidden, input):
    if hidden is not None:
        hidden = hidden[np.newaxis]
    _, hidden = net(hx=hidden, xs=F.separate(input, axis=0))
    hidden = F.stack(hidden, axis=0)
    return hidden


class WaveRNN(chainer.Chain):
    def __init__(
            self,
            bit_size: int,
            hidden_size: int,
            local_size: int,
    ) -> None:
        super().__init__()
        initialW = PossibleOrthogonal()

        self.half_bins = 2 ** (bit_size // 2)
        self.half_hidden_size = hidden_size // 2
        with self.init_scope():
            self.R_coarse = ModifiedNStepGRU(
                n_layers=1,
                in_size=2 + local_size,
                out_size=self.half_hidden_size,
                dropout=0.,
                initialW=initialW,
            )
            self.R_fine = ModifiedNStepGRU(
                n_layers=1,
                in_size=3 + local_size,
                out_size=self.half_hidden_size,
                dropout=0.,
                initialW=initialW,
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
        xc_array = F.concat((c_array[:, :, np.newaxis], f_array[:, :, np.newaxis], l_array), axis=2)
        xf_array = F.concat((xc_array, curr_c_array[:, :, np.newaxis]), axis=2)

        # shape: (batch_size, N, half_hidden_size)
        hidden_coarse = _call_1step(self.R_coarse, hidden_coarse, xc_array)
        hidden_fine = _call_1step(self.R_fine, hidden_fine, xf_array)

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
        prev_x = F.concat((prev_c[:, np.newaxis], prev_f[:, np.newaxis], prev_l), axis=1)  # shape: (batch_size, L+1)
        prev_x = F.expand_dims(prev_x, axis=1)  # shape: (batch_size, 1, L+1)

        new_hidden_coarse = _call_1step(
            self.R_coarse,
            hidden_coarse,
            prev_x,
        )[:, -1]  # shape: (batch_size, half_hidden_size)

        out_c = self.O2(F.relu(self.O1(new_hidden_coarse)))  # shape: (batch_size, ?)

        if prev_corr_c is None:
            curr_c = self.sampling(out_c).astype(prev_c.dtype) / 127.5 - 1  # shape: (batch_size, )
        else:
            curr_c = prev_corr_c

        new_hidden_fine = _call_1step(
            self.R_fine,
            hidden_fine,
            F.concat((prev_x, curr_c[:, np.newaxis, np.newaxis]), axis=2),
        )[:, -1]  # shape: (batch_size, half_hidden_size)

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
