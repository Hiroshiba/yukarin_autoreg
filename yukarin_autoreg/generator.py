from enum import Enum
from pathlib import Path
from typing import List, Union

import chainer
import numpy as np
from chainer import cuda

from yukarin_autoreg.config import Config
from yukarin_autoreg.data import decode_single, encode_mulaw, decode_mulaw, encode_single
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

        assert not self.dual_softmax

        if self.gpu is not None:
            model.to_gpu(self.gpu)
            cuda.get_device_from_id(self.gpu).use()

    @property
    def dual_softmax(self):
        return self.model.dual_softmax

    @property
    def single_bit(self):
        return self.model.bit_size // (2 if self.dual_softmax else 1)

    @property
    def input_categorical(self):
        return self.model.input_categorical

    @property
    def output_categorical(self):
        return not self.model.gaussian

    @property
    def xp(self):
        return self.model.xp

    def forward(self, w: np.ndarray, l: np.ndarray):
        assert not self.model.with_speaker

        if self.mulaw:
            w = encode_mulaw(w, mu=2 ** self.model.bit_size)
            w = self.xp.expand_dims(self.xp.asarray(w), axis=0)

        x_array = w
        if self.input_categorical:
            x_array = encode_single(x_array, bit=self.single_bit)

        local = self.xp.expand_dims(self.xp.asarray(l), axis=0)

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            c, hc = self.model(x_array, local)

        c = self.model.sampling(c[:, :, -1], maximum=True)
        return c, hc

    def generate(
            self,
            time_length: float,
            sampling_policy: SamplingPolicy,
            coarse=None,
            fine=None,
            local_array: np.ndarray = None,
            speaker_num: int = None,
            hidden_coarse=None,
            hidden_fine=None,
    ):
        length = int(self.sampling_rate * time_length)

        if local_array is None:
            local_array = self.xp.expand_dims(self.xp.empty((length, 0), dtype=np.float32), axis=0)
        else:
            local_array = self.xp.expand_dims(self.xp.asarray(local_array), axis=0)
            if speaker_num is not None:
                speaker_num = self.xp.asarray(speaker_num).reshape(shape=(-1,))
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                local_array = self.model.forward_encode(l_array=local_array, s_one=speaker_num)

        w_list = []

        if coarse is None:
            c = self.xp.zeros([1], dtype=np.float32)
            if self.output_categorical:
                c = encode_single(c, bit=self.single_bit)
        else:
            c = coarse

        hc = hidden_coarse
        for i in range(length):
            if self.output_categorical and not self.input_categorical:
                c = decode_single(c, bit=self.single_bit)

            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                c, hc = self.model.forward_one(
                    prev_x=c,
                    prev_l=local_array[:, i],
                    hidden=hc,
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
            if not self.output_categorical:
                c[c < -1] = -1
                c[c > 1] = 1

            w = chainer.cuda.to_cpu(c[0])
            if self.output_categorical:
                w = decode_single(w, bit=self.single_bit)
            w_list.append(w)

        wave = np.array(w_list)
        if self.mulaw:
            wave = decode_mulaw(wave, mu=2 ** self.single_bit)

        return Wave(wave=wave, sampling_rate=self.sampling_rate)
