from enum import Enum
from pathlib import Path
from typing import List, Optional

import chainer
import numpy as np
from acoustic_feature_extractor.data.wave import Wave
from chainer import cuda

from yukarin_autoreg.config import Config, ModelConfig
from yukarin_autoreg.data import decode_single, decode_mulaw, encode_single
from yukarin_autoreg.model import create_predictor
from yukarin_autoreg.network.fast_forward import get_fast_forward_params, fast_generate
from yukarin_autoreg.network.wave_rnn import WaveRNN


class SamplingPolicy(str, Enum):
    random = 'random'
    # maximum = 'maximum'
    # mix = 'mix'


class Generator(object):
    def __init__(
            self,
            config: Config,
            model: WaveRNN,
    ) -> None:
        self.config = config
        self.model = model

        self.sampling_rate = config.dataset.sampling_rate
        self.mulaw = config.dataset.mulaw

        assert not self.dual_softmax

    @staticmethod
    def load_model(
            model_config: ModelConfig,
            model_path: Path,
            gpu: int = None,
    ):
        predictor = create_predictor(model_config)
        chainer.serializers.load_npz(str(model_path), predictor)

        if gpu is not None:
            predictor.to_gpu(gpu)
            cuda.get_device_from_id(gpu).use()

        return predictor

    @property
    def dual_softmax(self):
        return self.model.dual_softmax

    @property
    def single_bit(self):
        return self.model.bit_size // (2 if self.dual_softmax else 1)

    @property
    def xp(self):
        return self.model.xp

    def generate(
            self,
            time_length: Optional[float],
            sampling_policy: SamplingPolicy,
            num_generate: int,
            coarse=None,
            local_array1: np.ndarray = None,
            local_array2: np.ndarray = None,
            speaker_nums1: List[int] = None,
            speaker_nums2: List[int] = None,
            morph_rates: List[float] = None,
            hidden_coarse=None,
    ):
        assert coarse is None or len(coarse) == num_generate
        assert local_array1 is None or len(local_array1) == num_generate
        assert local_array2 is None or len(local_array2) == num_generate
        assert speaker_nums1 is None or len(speaker_nums1) == num_generate
        assert speaker_nums2 is None or len(speaker_nums2) == num_generate
        assert hidden_coarse is None or len(hidden_coarse) == num_generate
        assert sampling_policy == SamplingPolicy.random

        length = int(self.sampling_rate * time_length)

        if local_array1 is None:
            local_array1 = self.xp.empty((num_generate, length, 0), dtype=np.float32)
            local_array2 = self.xp.empty((num_generate, length, 0), dtype=np.float32)
        else:
            local_array1 = self.xp.asarray(local_array1)
            local_array2 = self.xp.asarray(local_array2)

        if speaker_nums1 is not None:
            speaker_nums1 = self.xp.asarray(speaker_nums1).reshape((-1,))
            speaker_nums2 = self.xp.asarray(speaker_nums2).reshape((-1,))
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                s_one1 = self.model.forward_speaker(speaker_nums1).data
                s_one2 = self.model.forward_speaker(speaker_nums2).data

        morph_rates = self.xp.array(morph_rates, dtype=np.float32)

        if self.model.with_local:
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                morph_rates = morph_rates.reshape((-1, 1))
                s_one = s_one1 * morph_rates + s_one2 * (1 - morph_rates)

                morph_rates = morph_rates.reshape((-1, 1, 1))
                local_array = local_array1 * morph_rates + local_array2 * (1 - morph_rates)
                local_array = self.model.forward_encode(l_array=local_array, s_one=s_one).data

        if coarse is None:
            c = self.xp.zeros([num_generate], dtype=np.float32)
            c = encode_single(c, bit=self.single_bit)
        else:
            c = coarse

        fast_forward_params = get_fast_forward_params(self.model)
        # fast_forward_params['xp'] = self.model.xp

        if hidden_coarse is None:
            hidden_coarse = self.model.gru.init_hx(local_array)[0].data

        output = fast_generate(
            length=length,
            x=c,
            l_array=local_array,
            h=hidden_coarse,
            **fast_forward_params,
        )

        wave = self.xp.stack(output).T
        wave = decode_single(wave, bit=self.single_bit)
        wave = chainer.cuda.to_cpu(wave)
        if self.mulaw:
            wave = decode_mulaw(wave, mu=2 ** self.single_bit)

        return [
            Wave(wave=w_one, sampling_rate=self.sampling_rate)
            for w_one in wave
        ]
