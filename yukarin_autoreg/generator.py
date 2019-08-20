from enum import Enum
from pathlib import Path
from typing import List, Union

import chainer
import numpy as np
from chainer import cuda

from yukarin_autoreg.config import Config
from yukarin_autoreg.data import encode_16bit, decode_16bit, decode_single, encode_mulaw, decode_mulaw, encode_single
from yukarin_autoreg.model import create_predictor
from yukarin_autoreg.network import WaveRNN
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
        self.mulaw = config.dataset.mulaw

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

    @property
    def dual_softmax(self):
        return self.model.dual_softmax

    @property
    def single_bit(self):
        return self.model.bit_size // (2 if self.dual_softmax else 1)

    def forward(self, w: np.ndarray, l: np.ndarray):
        if self.mulaw:
            w = encode_mulaw(w, mu=2 ** self.model.bit_size)
            w = self.model.xp.expand_dims(self.model.xp.asarray(w), axis=0)

        if self.dual_softmax:
            encoded_coarse, encoded_fine = encode_16bit(w)
            coarse = decode_single(encoded_coarse)
            fine = decode_single(encoded_fine)[:-1]
        else:
            encoded_coarse = encode_single(w, bit=self.single_bit)
            coarse = w
            fine = None

        local = self.model.xp.expand_dims(self.model.xp.asarray(l), axis=0)

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            if isinstance(self.model, WaveRNN):
                c, f, hc, hf = self.model(coarse, fine, local)
            else:
                c, hc = self.model(encoded_coarse, local)
                f, hf = None, None

        c = self.model.sampling(c[:, :, -1], maximum=True)
        if self.dual_softmax:
            f = self.model.sampling(f[:, :, -1], maximum=True)
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
            if self.dual_softmax:
                c, f = encode_16bit(self.model.xp.zeros([1], dtype=np.float32))
            else:
                c = encode_single(self.model.xp.zeros([1], dtype=np.float32), bit=self.single_bit)
                f = None
        else:
            c, f = coarse, fine

        hc, hf = hidden_coarse, hidden_fine
        for i in range(length):
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                if isinstance(self.model, WaveRNN):
                    c = decode_single(c.astype(np.float32), bit=self.single_bit)
                    if self.dual_softmax:
                        f = decode_single(f.astype(np.float32), bit=self.single_bit)

                    c, f, hc, hf = self.model.forward_one(
                        prev_c=c,
                        prev_f=f,
                        prev_l=local_array[:, i],
                        hidden_coarse=hc,
                        hidden_fine=hf,
                    )
                else:
                    c, hc = self.model.forward_one(
                        prev_x=c,
                        prev_l=local_array[:, i],
                        hidden=hc,
                    )
                    f, hf = None, None

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
            if self.dual_softmax:
                f = self.model.sampling(f, maximum=not is_random)
                w = decode_16bit(
                    coarse=chainer.cuda.to_cpu(c[0]),
                    fine=chainer.cuda.to_cpu(f[0]),
                )
            else:
                w = decode_single(chainer.cuda.to_cpu(c[0]), bit=self.single_bit)

            w_list.append(w)

        wave = np.array(w_list)
        if self.mulaw:
            wave = decode_mulaw(wave, mu=2 ** self.single_bit)

        return Wave(wave=wave, sampling_rate=self.sampling_rate)
