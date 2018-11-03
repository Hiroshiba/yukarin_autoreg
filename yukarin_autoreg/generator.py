from pathlib import Path
from typing import List, Union

import chainer
import numpy as np
from chainer import cuda

from yukarin_autoreg.config import Config
from yukarin_autoreg.dataset import decode_16bit, encode_16bit, normalize
from yukarin_autoreg.model import create_predictor
from yukarin_autoreg.utility.chainer_link_utility import mean_params
from yukarin_autoreg.wave import Wave


class Generator(object):
    def __init__(
            self,
            config: Config,
            model_path: Union[Path, List[Path]],
            gpu: int = None,
    ) -> None:
        self.config = config
        self.model_path = model_path
        self.gpu = gpu

        if isinstance(model_path, Path):
            self.model = model = create_predictor(config.model)
            chainer.serializers.load_npz(str(model_path), model)
        else:
            # mean weights
            models = []
            for p in model_path:
                model = create_predictor(config.model)
                chainer.serializers.load_npz(str(p), model)
                models.append(model)
            self.model = model = create_predictor(config.model)
            mean_params(models, model)

        if self.gpu is not None:
            model.to_gpu(self.gpu)
            cuda.get_device_from_id(self.gpu).use()

        chainer.global_config.train = False
        chainer.global_config.enable_backprop = False

    def forward(self, w: np.ndarray, l: np.ndarray):
        coarse, fine = encode_16bit(self.model.xp.asarray(w))

        coarse = self.model.xp.expand_dims(normalize(coarse).astype(np.float32), axis=0)
        fine = self.model.xp.expand_dims(normalize(fine).astype(np.float32)[:-1], axis=0)

        local = self.model.xp.expand_dims(self.model.xp.asarray(l), axis=0)

        c, f, hc, hf = self.model(coarse, fine, local)
        c = normalize(self.model.sampling(c[:, :, -1], maximum=True).astype(np.float32))
        f = normalize(self.model.sampling(f[:, :, -1], maximum=True).astype(np.float32))
        return c, f, hc, hf

    def generate(
            self,
            time_length: float,
            sampling_maximum: bool,
            coarse=None,
            fine=None,
            local_array: np.ndarray = None,
            hidden_coarse=None,
            hidden_fine=None,
    ):
        length = int(self.config.dataset.sampling_rate * time_length)

        if local_array is None:
            local_array = self.model.xp.expand_dims(self.model.xp.empty((length, 0), dtype=np.float32), axis=0)
        else:
            local_array = self.model.xp.expand_dims(self.model.xp.asarray(local_array), axis=0)
            local_array = self.model.forward_encode(local_array)

        w_list = []

        if coarse is None:
            c = self.model.xp.random.rand(1).astype(np.float32)
            f = self.model.xp.random.rand(1).astype(np.float32)
        else:
            c, f = coarse, fine

        hc, hf = hidden_coarse, hidden_fine
        for i in range(length):
            c, f, hc, hf = self.model.forward_one(
                prev_c=c,
                prev_f=f,
                prev_l=local_array[:, i],
                hidden_coarse=hc,
                hidden_fine=hf,
            )

            c = self.model.sampling(c, maximum=sampling_maximum)
            f = self.model.sampling(f, maximum=sampling_maximum)

            w = decode_16bit(
                coarse=chainer.cuda.to_cpu(c[0]),
                fine=chainer.cuda.to_cpu(f[0]),
            )
            w_list.append(w)

            c = normalize(c.astype(np.float32))
            f = normalize(f.astype(np.float32))

        return Wave(wave=np.array(w_list), sampling_rate=self.config.dataset.sampling_rate)
