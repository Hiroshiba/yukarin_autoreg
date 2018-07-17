import chainer
import chainer.functions as F
import chainer.links as L

from yukarin_autoreg.config import ModelConfig


class WaveRNN(chainer.Chain):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.split_size = config.hidden_size // 2
        self.half_bins = 2 ** (config.bit_size // 2)
        with self.init_scope():
            self.R = L.Linear(self.hidden_size, 3 * self.hidden_size)
            self.O1 = L.Linear(self.split_size, self.split_size)
            self.O2 = L.Linear(self.split_size, self.half_bins)
            self.O3 = L.Linear(self.split_size, self.split_size)
            self.O4 = L.Linear(self.split_size, self.half_bins)
            self.I_coarse = L.Linear(2, 3 * self.split_size)
            self.I_fine = L.Linear(3, 3 * self.split_size)

    def __call__(self, c_array, f_array, hidden=None):
        """
        :param c_array: float -1 ~ +1 (batch_size, N+1)
        :param f_array: float -1 ~ +1 (batch_size, N)
        :param hidden: float (batch_size, hidden_size)
        :return:
            out_c_array: float (batch_size, half_bins, N)
            out_f_array: float (batch_size, half_bins, N)
            hidden: float (batch_size, hidden_size)
        """
        out_c_array, out_f_array, hidden = self.forward(
            c_array=c_array[:, :-1],
            f_array=f_array[:, :],
            curr_c_array=c_array[:, 1:],
            prev_hidden=hidden,
        )
        return out_c_array, out_f_array, hidden

    def forward(self, c_array, f_array, curr_c_array, prev_hidden=None):
        """
        c: coarse
        f: fine
        :param c_array: float -1 ~ +1 (batch_size, N)
        :param f_array: float -1 ~ +1 (batch_size, N)
        :param curr_c_array: float -1 ~ +1 (batch_size, N)
        :param prev_hidden: float (batch_size, hidden_size)
        :return:
            out_c_array: float (batch_size, half_bins, N)
            out_f_array: float (batch_size, half_bins, N)
            hidden: float (batch_size, hidden_size)
        """
        assert c_array.shape == f_array.shape == curr_c_array.shape

        batch_size = c_array.shape[0]
        length = c_array.shape[1]  # N
        if prev_hidden is None:
            prev_hidden = self.get_init_hidden(batch_size=batch_size)

        # batch
        c_array = F.reshape(c_array, shape=(batch_size * length,))  # shape: (batch_size * N)
        f_array = F.reshape(f_array, shape=(batch_size * length,))  # shape: (batch_size * N)
        curr_c_array = F.reshape(curr_c_array, shape=(batch_size * length,))  # shape: (batch_size * N)

        # Project the input and split
        coarse_input_proj = self.I_coarse(F.stack((c_array, f_array), axis=1))  # shape: (batch_size * N, ?)
        fine_input_proj = self.I_fine(F.stack((c_array, f_array, curr_c_array), axis=1))  # shape: (batch_size * N, ?)

        coarse_input_proj = F.reshape(coarse_input_proj, shape=(batch_size, length, -1))  # shape: (batch_size, N, ?)
        fine_input_proj = F.reshape(fine_input_proj, shape=(batch_size, length, -1))  # shape: (batch_size, N, ?)

        hidden = prev_hidden
        hidden_list = []
        for cip, fip in zip(F.separate(coarse_input_proj, axis=1), F.separate(fine_input_proj, axis=1)):
            I_coarse_u, I_coarse_r, I_coarse_e = F.split_axis(cip, 3, axis=1)
            I_fine_u, I_fine_r, I_fine_e = F.split_axis(fip, 3, axis=1)
            Iu = F.concat((I_coarse_u, I_fine_u), axis=1)
            Ir = F.concat((I_coarse_r, I_fine_r), axis=1)
            Ie = F.concat((I_coarse_e, I_fine_e), axis=1)

            # Main matmul - the projection is split 6 ways
            # for the coarse/fine gates
            R_hidden = self.R(hidden)
            R_coarse_u, R_coarse_r, R_coarse_e, \
            R_fine_u, R_fine_r, R_fine_e = F.split_axis(R_hidden, 6, axis=1)
            Ru = F.concat((R_coarse_u, R_fine_u), axis=1)
            Rr = F.concat((R_coarse_r, R_fine_r), axis=1)
            Re = F.concat((R_coarse_e, R_fine_e), axis=1)

            # Compute the first round of gates: coarse
            u = F.sigmoid(Ru + Iu)
            r = F.sigmoid(Rr + Ir)
            e = F.tanh(r * Re + Ie)
            hidden = u * hidden + (1 - u) * e

            hidden_list.append(hidden)

        hidden = F.stack(hidden_list, axis=1)  # shape: (batch_size, N, ?)
        hidden = F.reshape(hidden, shape=(batch_size * length, -1))  # shape: (batch_size * N, ?)
        hidden_coarse, hidden_fine = F.split_axis(hidden, 2, axis=1)  # shape: (batch_size * N, ?)

        # Compute outputs
        out_c_array = self.O2(F.relu(self.O1(hidden_coarse)))  # shape: (batch_size * N, ?)
        out_f_array = self.O4(F.relu(self.O3(hidden_fine)))  # shape: (batch_size * N, ?)
        out_c_array = F.reshape(out_c_array, shape=(batch_size, length, -1))  # shape: (batch_size, N, ?)
        out_f_array = F.reshape(out_f_array, shape=(batch_size, length, -1))  # shape: (batch_size, N, ?)
        out_c_array = F.transpose(out_c_array, axes=(0, 2, 1))  # shape: (batch_size, ?, N)
        out_f_array = F.transpose(out_f_array, axes=(0, 2, 1))  # shape: (batch_size, ?, N)

        return out_c_array, out_f_array, hidden

    def forward_one(self, prev_c, prev_f, prev_hidden=None, curr_c=None):
        """
        c: coarse
        f: fine
        :param prev_c: float -1 ~ +1 (batch_size, )
        :param prev_f: float -1 ~ +1 (batch_size, )
        :param prev_hidden: float (batch_size, hidden_size)
        :param curr_c: float -1 ~ +1 (batch_size, )
        :return:
            out_c: float (batch_size, half_bins)
            out_f: float (batch_size, half_bins)
            hidden: float (batch_size, hidden_size)
        """
        batch_size = len(prev_c)
        if prev_hidden is None:
            prev_hidden = self.get_init_hidden(batch_size=batch_size)

        # Main matmul - the projection is split 6 ways
        # for the coarse/fine gates
        R_hidden = self.R(prev_hidden)
        R_coarse_u, R_coarse_r, R_coarse_e, \
        R_fine_u, R_fine_r, R_fine_e = F.split_axis(R_hidden, 6, axis=1)

        # Project the input and split for coarse gates
        coarse_input_proj = self.I_coarse(F.stack((prev_c, prev_f), axis=1))
        I_coarse_u, I_coarse_r, I_coarse_e = F.split_axis(coarse_input_proj, 3, axis=1)

        # The hidden state needs to be split up too
        hidden_coarse, hidden_fine = F.split_axis(prev_hidden, 2, axis=1)

        # Compute the first round of gates: coarse
        u_coarse = F.sigmoid(R_coarse_u + I_coarse_u)
        r_coarse = F.sigmoid(R_coarse_r + I_coarse_r)
        e_coarse = F.tanh(r_coarse * R_coarse_e + I_coarse_e)
        hidden_coarse = u_coarse * hidden_coarse + (1 - u_coarse) * e_coarse

        # Compute outputs for coarse
        out_c = self.O2(F.relu(self.O1(hidden_coarse)))

        # If generating pick a coarse sample
        if curr_c is None:
            curr_c = self.sampling(out_c)

        # concatenate the prev c/f samples with the coarse
        # predicted (while generating) or ground truth(while training)
        fine_input_proj = self.I_fine(F.stack((prev_c, prev_f, curr_c), axis=1))
        I_fine_u, I_fine_r, I_fine_e = F.split_axis(fine_input_proj, 3, axis=1)

        # Compute the second round of gates: fine
        u_fine = F.sigmoid(R_fine_u + I_fine_u)
        r_fine = F.sigmoid(R_fine_r + I_fine_r)
        e_fine = F.tanh(r_fine * R_fine_e + I_fine_e)
        hidden_fine = u_fine * hidden_fine + (1. - u_fine) * e_fine

        # Compute outputs for fine
        out_f = self.O4(F.relu(self.O3(hidden_fine)))

        # put the hidden state back together
        hidden = F.concat((hidden_coarse, hidden_fine), axis=1)

        return out_c, out_f, hidden

    def get_init_hidden(self, batch_size=1):
        return self.xp.zeros((batch_size, self.hidden_size), dtype=self.xp.float32)

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
