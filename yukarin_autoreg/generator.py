from pathlib import Path

import chainer
import numpy as np
from chainer import cuda

from yukarin_autoreg.config import Config
from yukarin_autoreg.dataset import encode_16bit
from yukarin_autoreg.dataset import decode_16bit
from yukarin_autoreg.dataset import normalize
from yukarin_autoreg.model import create_predictor
from yukarin_autoreg.wave import Wave


class Generator(object):
    def __init__(
            self,
            config: Config,
            model_path: Path,
            gpu: int = None,
    ) -> None:
        self.config = config
        self.model_path = model_path
        self.gpu = gpu

        self.model = model = create_predictor(config.model)
        chainer.serializers.load_npz(str(model_path), model)
        if self.gpu is not None:
            model.to_gpu(self.gpu)
            cuda.get_device_from_id(self.gpu).use()

        chainer.global_config.train = False
        chainer.global_config.enable_backprop = False

    def forward(self, w: np.ndarray):
        coarse, fine = encode_16bit(self.model.xp.asarray(w))
        coarse = normalize(coarse).astype(np.float32)[np.newaxis]
        fine = normalize(fine).astype(np.float32)[:-1][np.newaxis]
        c, f, hc, hf = self.model(coarse, fine)
        c = self.model.sampling(c[:, :, -1], maximum=True)
        f = self.model.sampling(f[:, :, -1], maximum=True)
        return c, f, hc, hf

    def generate(
            self,
            time_length: int,
            sampling_maximum: bool,
            coarse=None,
            fine=None,
            hidden_coarse=None,
            hidden_fine=None,
    ):
        w_list = []

        if coarse is None:
            c = self.model.xp.random.rand(1).astype(np.float32)
            f = self.model.xp.random.rand(1).astype(np.float32)
        else:
            c, f = coarse, fine

        hc, hf = hidden_coarse, hidden_fine
        for _ in range(self.config.dataset.sampling_rate * time_length):
            c, f, hc, hf = self.model.forward_one(prev_c=c, prev_f=f, hidden_coarse=hc, hidden_fine=hf)

            c = self.model.sampling(c, maximum=sampling_maximum)
            f = self.model.sampling(f, maximum=sampling_maximum)

            w = decode_16bit(
                coarse=chainer.cuda.to_cpu((c[0] + 1) * 255 // 2),
                fine=chainer.cuda.to_cpu((f[0] + 1) * 255 // 2),
            )
            w_list.append(w)

        return Wave(wave=np.array(w_list).astype(np.float32), sampling_rate=self.config.dataset.sampling_rate)
