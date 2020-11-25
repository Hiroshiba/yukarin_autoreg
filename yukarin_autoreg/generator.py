from enum import Enum
from pathlib import Path
from typing import Sequence

import chainer
import cupy as cp
import numpy as np
from acoustic_feature_extractor.data.wave import Wave
from chainer import cuda
from tqdm import tqdm

from yukarin_autoreg.config import Config, ModelConfig
from yukarin_autoreg.data import decode_mulaw, decode_single, encode_single
from yukarin_autoreg.model import create_predictor
from yukarin_autoreg.network.fast_forward import fast_generate, get_fast_forward_params
from yukarin_autoreg.network.wave_rnn import WaveRNN


class SamplingPolicy(str, Enum):
    random = "random"
    maximum = "maximum"
    # mix = "mix"


class MorphingPolicy(str, Enum):
    linear = "linear"
    sphere = "sphere"


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
        use_cpp_inference: bool = True,
    ) -> None:
        self.config = config
        self.model = model
        self.max_batch_size = max_batch_size
        self.use_cpp_inference = use_cpp_inference

        self.sampling_rate = config.dataset.sampling_rate
        self.mulaw = config.dataset.mulaw

        assert not self.dual_softmax

        # setup cpp inference
        if use_cpp_inference:
            import yukarin_autoreg_cpp

            params = get_fast_forward_params(self.model)
            local_size = (
                config.model.conditioning_size * 2
                if config.model.conditioning_size is not None
                else 0
            )
            yukarin_autoreg_cpp.initialize(
                graph_length=1000,
                max_batch_size=max_batch_size,
                local_size=local_size,
                hidden_size=config.model.hidden_size,
                embedding_size=config.model.embedding_size,
                linear_hidden_size=config.model.linear_hidden_size,
                output_size=2 ** config.model.bit_size,
                x_embedder_W=to_numpy(params["x_embedder_W"]),
                gru_xw=to_numpy(params["gru_xw"]),
                gru_xb=to_numpy(params["gru_xb"]),
                gru_hw=to_numpy(params["gru_hw"]),
                gru_hb=to_numpy(params["gru_hb"]),
                O1_W=to_numpy(params["O1_W"]),
                O1_b=to_numpy(params["O1_b"]),
                O2_W=to_numpy(params["O2_W"]),
                O2_b=to_numpy(params["O2_b"]),
            )

    @staticmethod
    def load_model(
        model_config: ModelConfig, model_path: Path, gpu: int = None,
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
        time_length: float,
        sampling_policy: SamplingPolicy,
        num_generate: int,
        local_array: np.ndarray = None,
        speaker_nums: Sequence[int] = None,
    ):
        assert num_generate <= self.max_batch_size
        assert local_array is None or len(local_array) == num_generate
        assert speaker_nums is None or len(speaker_nums) == num_generate

        length = int(self.sampling_rate * time_length)

        if local_array is None:
            local_array = self.xp.empty((num_generate, length, 0), dtype=np.float32)
        else:
            local_array = self.xp.asarray(local_array)

        if speaker_nums is not None:
            speaker_nums = self.xp.asarray(speaker_nums).reshape((-1,))
            with chainer.using_config("train", False), chainer.using_config(
                "enable_backprop", False
            ):
                s_one = self.model.forward_speaker(speaker_nums).data
        else:
            s_one = None

        return self.main_forward(
            length=length,
            sampling_policy=sampling_policy,
            num_generate=num_generate,
            local_array=local_array,
            s_one=s_one,
        )

    def morphing(
        self,
        time_length: float,
        sampling_policy: SamplingPolicy,
        morphing_policy: MorphingPolicy,
        local_array1: np.ndarray,
        local_array2: np.ndarray,
        speaker_nums1: Sequence[int],
        speaker_nums2: Sequence[int],
        start_rates: Sequence[float],
        stop_rates: Sequence[float],
    ):
        num_generate = len(local_array1)
        assert num_generate <= self.max_batch_size
        assert len(local_array2) == num_generate
        assert len(speaker_nums1) == num_generate
        assert len(speaker_nums2) == num_generate
        assert len(start_rates) == num_generate
        assert len(stop_rates) == num_generate

        length = int(self.sampling_rate * time_length)
        local_length = local_array1.shape[1]

        local_array1 = self.xp.asarray(local_array1)
        local_array2 = self.xp.asarray(local_array2)

        speaker_nums1 = self.xp.asarray(speaker_nums1).reshape((-1,))
        speaker_nums2 = self.xp.asarray(speaker_nums2).reshape((-1,))
        with chainer.using_config("train", False), chainer.using_config(
            "enable_backprop", False
        ):
            s_one1 = self.xp.repeat(
                self.model.forward_speaker(speaker_nums1).data[:, None],
                local_length,
                axis=1,
            )
            s_one2 = self.xp.repeat(
                self.model.forward_speaker(speaker_nums2).data[:, None],
                local_length,
                axis=1,
            )

        # morphing
        start_rates = np.asarray(start_rates)
        stop_rates = np.asarray(stop_rates)
        morph_rates = self.xp.asarray(
            np.linspace(
                start_rates, stop_rates, num=local_length, axis=1, dtype=np.float32,
            )
        ).reshape((num_generate, local_length, 1))

        local_array = local_array1 * morph_rates + local_array2 * (1 - morph_rates)

        if morphing_policy == MorphingPolicy.linear:
            s_one = s_one1 * morph_rates + s_one2 * (1 - morph_rates)
        elif morphing_policy == MorphingPolicy.sphere:
            omega = self.xp.arccos(
                self.xp.sum(
                    (s_one1 * s_one2)
                    / (
                        self.xp.linalg.norm(s_one1, axis=2, keepdims=True)
                        * self.xp.linalg.norm(s_one2, axis=2, keepdims=True)
                    ),
                    axis=2,
                    keepdims=True,
                )
            )
            sin_omega = self.xp.sin(omega)
            s_one = (
                self.xp.sin(morph_rates * omega) / sin_omega * s_one1
                + self.xp.sin((1.0 - morph_rates) * omega) / sin_omega * s_one2
            ).astype(np.float32)
        else:
            raise ValueError(morphing_policy)

        return self.main_forward(
            length=length,
            sampling_policy=sampling_policy,
            num_generate=num_generate,
            local_array=local_array,
            s_one=s_one,
        )

    def main_forward(
        self,
        length: int,
        sampling_policy: SamplingPolicy,
        num_generate: int,
        local_array: np.ndarray = None,
        s_one: np.ndarray = None,
    ):
        if self.model.with_local:
            with chainer.using_config("train", False), chainer.using_config(
                "enable_backprop", False
            ):
                local_array = self.model.forward_encode(
                    l_array=local_array, s_one=s_one
                ).data

        c = self.xp.zeros([num_generate], dtype=np.float32)
        c = encode_single(c, bit=self.single_bit)

        hidden_coarse = self.model.gru.init_hx(local_array)[0].data

        if self.use_cpp_inference and sampling_policy == SamplingPolicy.random:
            import yukarin_autoreg_cpp
            
            wave = np.zeros((length, num_generate), dtype=np.int32)
            yukarin_autoreg_cpp.inference(
                batch_size=num_generate,
                length=length,
                output=wave,
                x=to_numpy(c),
                l_array=to_numpy(self.xp.transpose(local_array, (1, 0, 2))),
                hidden=to_numpy(hidden_coarse),
            )
        else:
            if sampling_policy == SamplingPolicy.random:
                fast_forward_params = get_fast_forward_params(self.model)
                w_list = fast_generate(
                    length=length,
                    x=c,
                    l_array=local_array,
                    h=hidden_coarse,
                    **fast_forward_params,
                )
            else:
                w_list = []
                hc = hidden_coarse
                for i in tqdm(range(length), desc="generate"):
                    with chainer.using_config("train", False), chainer.using_config(
                        "enable_backprop", False
                    ):
                        c, hc = self.model.forward_one(
                            prev_x=c, prev_l=local_array[:, i], hidden=hc,
                        )

                    if sampling_policy == SamplingPolicy.random:
                        is_random = True
                    elif sampling_policy == SamplingPolicy.maximum:
                        is_random = False
                    else:
                        raise ValueError(sampling_policy)

                    c = self.model.sampling(c, maximum=not is_random)
                    w_list.append(c)

            wave = self.xp.stack(w_list)
            wave = cuda.to_cpu(wave)

        wave = wave.T
        wave = decode_single(wave, bit=self.single_bit)
        if self.mulaw:
            wave = decode_mulaw(wave, mu=2 ** self.single_bit)

        return [Wave(wave=w_one, sampling_rate=self.sampling_rate) for w_one in wave]
