from enum import Enum
from pathlib import Path
from typing import List, Optional

import chainer
import numpy as np
from acoustic_feature_extractor.data.wave import Wave
from chainer import cuda
from tqdm import tqdm

from yukarin_autoreg.config import Config, ModelConfig
from yukarin_autoreg.data import decode_single, decode_mulaw, encode_single
from yukarin_autoreg.model import create_predictor
from yukarin_autoreg.network.fast_forward import get_fast_forward_params, fast_forward_one
from yukarin_autoreg.network.wave_rnn import WaveRNN


class SamplingPolicy(str, Enum):
    random = 'random'
    maximum = 'maximum'
    mix = 'mix'


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
            local_array: np.ndarray = None,
            speaker_nums: List[int] = None,
            hidden_coarse=None,
    ):
        assert coarse is None or len(coarse) == num_generate
        assert local_array is None or len(local_array) == num_generate
        assert speaker_nums is None or len(speaker_nums) == num_generate
        assert hidden_coarse is None or len(hidden_coarse) == num_generate

        length = int(self.sampling_rate * time_length)

        if local_array is None:
            local_array = self.xp.empty((num_generate, length, 0), dtype=np.float32)
        else:
            local_array = self.xp.asarray(local_array)

        if speaker_nums is not None:
            speaker_nums = self.xp.asarray(speaker_nums).reshape((-1,))
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                s_one = self.model.forward_speaker(speaker_nums).data
        else:
            s_one = None

        if self.model.with_local:
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                local_array = self.model.forward_encode(l_array=local_array, s_one=s_one).data

        w_list = []

        if coarse is None:
            c = self.xp.zeros([num_generate], dtype=np.float32)
            c = encode_single(c, bit=self.single_bit)
        else:
            c = coarse

        fast_forward_params = get_fast_forward_params(self.model)
        fast_forward_params['xp'] = self.model.xp

        if hidden_coarse is None:
            hidden_coarse = self.model.gru.init_hx(local_array)[0].data

        hc = hidden_coarse
        for i in tqdm(range(length), desc='generate'):
            c, hc = fast_forward_one(
                prev_x=c,
                prev_l=local_array[:, i],
                hidden=hc,
                **fast_forward_params,
            )

            if sampling_policy == SamplingPolicy.random:
                is_random = True
            elif sampling_policy == SamplingPolicy.maximum:
                is_random = False
            elif sampling_policy == SamplingPolicy.mix:
                if len(w_list) < 2:
                    is_random = True
                elif np.all(w_list[-2] == w_list[-1]):
                    is_random = True
                else:
                    is_random = False
            else:
                raise ValueError(sampling_policy)

            c = self.model.sampling(c, maximum=not is_random)

            w = decode_single(c, bit=self.single_bit)
            w_list.append(w)

        wave = self.xp.stack(w_list).T
        wave = chainer.cuda.to_cpu(wave)
        if self.mulaw:
            wave = decode_mulaw(wave, mu=2 ** self.single_bit)

        return [
            Wave(wave=w_one, sampling_rate=self.sampling_rate)
            for w_one in wave
        ]
