from pathlib import Path

import chainer
from chainer import cuda
import numpy as np

from yukarin_autoreg.config import Config
from yukarin_autoreg.dataset import decode_16bit
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

    def generate(self, time_length: int, sampling_maximum: bool):
        w_list = []

        c = f = self.model.xp.zeros((1,), dtype=np.float32)
        hidden = None
        for _ in range(self.config.dataset.sampling_rate * time_length):
            with chainer.using_config('train', False):
                c, f, hidden = self.model.forward_one(prev_c=c, prev_f=f, prev_hidden=hidden)

            c = self.model.sampling(c, maximum=sampling_maximum)
            f = self.model.sampling(f, maximum=sampling_maximum)

            w = decode_16bit(
                coarse=chainer.cuda.to_cpu((c[0] + 1) * 255 // 2),
                fine=chainer.cuda.to_cpu((f[0] + 1) * 255 // 2),
            )
            w_list.append(w)

        return Wave(wave=np.array(w_list).astype(np.float32), sampling_rate=self.config.dataset.sampling_rate)
