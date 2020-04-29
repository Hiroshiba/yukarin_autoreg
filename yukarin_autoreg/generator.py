from enum import Enum
from pathlib import Path
from typing import List, Optional

import chainer
import cupy as cp
import numpy as np
from acoustic_feature_extractor.data.wave import Wave
from chainer import cuda

import yukarin_autoreg_cpp
from yukarin_autoreg.config import Config, ModelConfig
from yukarin_autoreg.data import decode_single, decode_mulaw, encode_single
from yukarin_autoreg.model import create_predictor
from yukarin_autoreg.network.fast_forward import get_fast_forward_params
from yukarin_autoreg.network.wave_rnn import WaveRNN


class SamplingPolicy(str, Enum):
    random = 'random'
    # maximum = 'maximum'
    # mix = 'mix'


def to_numpy(a):
    if isinstance(a, chainer.Variable):
        a = a.data
    if isinstance(a, cp.ndarray):
        a = cp.asnumpy(a)
    return np.ascontiguousarray(a)


class Generator(object):
    def __init__(
            self,
            config: Config,
            model: WaveRNN,
            max_batch_size: int = 10,
    ) -> None:
        self.config = config
        self.model = model
        self.max_batch_size = max_batch_size

        self.sampling_rate = config.dataset.sampling_rate
        self.mulaw = config.dataset.mulaw

        assert not self.dual_softmax

        # setup cpp inference
        params = get_fast_forward_params(self.model)
        local_size = config.model.conditioning_size * 2 if config.model.conditioning_size is not None else 0
        yukarin_autoreg_cpp.initialize(
            graph_length=1000,
            max_batch_size=max_batch_size,
            local_size=local_size,
            hidden_size=config.model.hidden_size,
            embedding_size=config.model.embedding_size,
            linear_hidden_size=config.model.linear_hidden_size,
            output_size=2 ** config.model.bit_size,
            x_embedder_W=to_numpy(params['x_embedder_W']),
            gru_xw=to_numpy(params['gru_xw']),
            gru_xb=to_numpy(params['gru_xb']),
            gru_hw=to_numpy(params['gru_hw']),
            gru_hb=to_numpy(params['gru_hb']),
            O1_W=to_numpy(params['O1_W']),
            O1_b=to_numpy(params['O1_b']),
            O2_W=to_numpy(params['O2_W']),
            O2_b=to_numpy(params['O2_b']),
        )

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
        assert num_generate <= self.max_batch_size
        assert coarse is None or len(coarse) == num_generate
        assert local_array is None or len(local_array) == num_generate
        assert speaker_nums is None or len(speaker_nums) == num_generate
        assert hidden_coarse is None or len(hidden_coarse) == num_generate
        assert sampling_policy == SamplingPolicy.random

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

        if coarse is None:
            c = self.xp.zeros([num_generate], dtype=np.float32)
            c = encode_single(c, bit=self.single_bit)
        else:
            c = coarse

        if hidden_coarse is None:
            hidden_coarse = self.model.gru.init_hx(local_array)[0].data

        wave = np.zeros((length, num_generate), dtype=np.int32)
        yukarin_autoreg_cpp.inference(
            batch_size=num_generate,
            length=length,
            output=wave,
            x=to_numpy(c),
            l_array=to_numpy(self.xp.transpose(local_array, (1, 0, 2))),
            hidden=to_numpy(hidden_coarse),
        )

        wave = wave.T
        wave = decode_single(wave, bit=self.single_bit)
        if self.mulaw:
            wave = decode_mulaw(wave, mu=2 ** self.single_bit)

        return [
            Wave(wave=w_one, sampling_rate=self.sampling_rate)
            for w_one in wave
        ]
