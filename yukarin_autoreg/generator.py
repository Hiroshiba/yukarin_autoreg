from enum import Enum
from pathlib import Path
from typing import List, Union

import chainer
import numpy as np
from chainer import cuda

from yukarin_autoreg.config import Config
from yukarin_autoreg.dataset import decode_16bit, encode_16bit, normalize, decode_single
from yukarin_autoreg.model import create_predictor
from yukarin_autoreg.utility.chainer_link_utility import mean_params
from yukarin_autoreg.wave import Wave


class SamplingPolicy(str, Enum):
    random = 'random'
    maximum = 'maximum'
    mix = 'mix'


class Generator(object):
    def __init__(
            self,
            config: Config,
            model_path: Union[Path, List[Path]],
            gpu: int = None,
    ) -> None:
        self.model_path = model_path
        self.gpu = gpu

        self.sampling_rate = config.dataset.sampling_rate

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

    def forward(self, w: np.ndarray, l: np.ndarray):
        if self.model.dual_softmax:
            coarse, fine = encode_16bit(self.model.xp.asarray(w))
            coarse = self.model.xp.expand_dims(normalize(coarse).astype(np.float32), axis=0)
            fine = self.model.xp.expand_dims(normalize(fine).astype(np.float32)[:-1], axis=0)
        else:
            coarse = self.model.xp.expand_dims(self.model.xp.asarray(w), axis=0)
            fine = None

        local = self.model.xp.expand_dims(self.model.xp.asarray(l), axis=0)

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            c, f, hc, hf = self.model(coarse, fine, local)

        c = normalize(self.model.sampling(c[:, :, -1], maximum=True).astype(np.float32))
        if self.model.dual_softmax:
            f = normalize(self.model.sampling(f[:, :, -1], maximum=True).astype(np.float32))
        else:
            f = None
        return c, f, hc, hf

    def generate(
            self,
            time_length: float,
            sampling_policy: SamplingPolicy,
            coarse=None,
            fine=None,
            local_array: np.ndarray = None,
            hidden_coarse=None,
            hidden_fine=None,
    ):
        length = int(self.sampling_rate * time_length)

        if local_array is None:
            local_array = self.model.xp.expand_dims(self.model.xp.empty((length, 0), dtype=np.float32), axis=0)
        else:
            local_array = self.model.xp.expand_dims(self.model.xp.asarray(local_array), axis=0)
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                local_array = self.model.forward_encode(local_array)

        w_list = []

        if coarse is None:
            c = self.model.xp.random.rand(1).astype(np.float32)
            if self.model.dual_softmax:
                f = self.model.xp.random.rand(1).astype(np.float32)
            else:
                f = None
        else:
            c, f = coarse, fine

        hc, hf = hidden_coarse, hidden_fine
        for i in range(length):
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                c, f, hc, hf = self.model.forward_one(
                    prev_c=c,
                    prev_f=f,
                    prev_l=local_array[:, i],
                    hidden_coarse=hc,
                    hidden_fine=hf,
                )

            if sampling_policy == SamplingPolicy.random:
                is_random = True
            elif sampling_policy == SamplingPolicy.maximum:
                is_random = False
            elif sampling_policy == SamplingPolicy.mix:
                if len(w_list) < 2:
                    is_random = True
                elif w_list[-2] == w_list[-1]:
                    is_random = True
                else:
                    is_random = False
            else:
                raise ValueError(sampling_policy)

            c = self.model.sampling(c, maximum=not is_random)
            if self.model.dual_softmax:
                f = self.model.sampling(f, maximum=not is_random)
                w = decode_16bit(
                    coarse=chainer.cuda.to_cpu(c[0]),
                    fine=chainer.cuda.to_cpu(f[0]),
                )

                c = normalize(c.astype(np.float32))
                f = normalize(f.astype(np.float32))
            else:
                c = decode_single(c.astype(np.float32), bit=self.model.bit_size)
                w = chainer.cuda.to_cpu(c[0])

            w_list.append(w)

        return Wave(wave=np.array(w_list), sampling_rate=self.sampling_rate)
