import unittest
from collections import namedtuple
from pathlib import Path

from yukarin_autoreg.generator import Generator, SamplingPolicy

gpu = 3


class TestGenerator(unittest.TestCase):
    def test_generator(self):
        for to_double, bit, mulaw, use_univ in (
                (True, 16, False, False),
                (False, 8, False, False),
                (False, 8, True, False),
                (False, 9, True, True),
        ):
            with self.subTest(to_double=to_double, bit=bit, mulaw=mulaw, use_univ=use_univ):
                config = namedtuple('Config', ['dataset', 'model'])(
                    dataset=namedtuple('DatasetConfig', [
                        'sampling_rate',
                        'mulaw',
                    ])(
                        sampling_rate=8000,
                        mulaw=mulaw,
                    ),
                    model=namedtuple('ModelConfig', [
                        'upconv_scales',
                        'upconv_residual',
                        'upconv_channel_ksize',
                        'residual_encoder_channel',
                        'residual_encoder_num_block',
                        'dual_softmax',
                        'bit_size',
                        'hidden_size',
                        'local_size',
                        'use_univ_wavernn',
                        'conditioning_size',
                        'embedding_size',
                        'linear_hidden_size',
                        'local_scale',
                        'local_layer_num',
                        'bug_fixed_gru_dimension',
                    ])(
                        upconv_scales=[],
                        upconv_residual=False,
                        upconv_channel_ksize=0,
                        residual_encoder_channel=None,
                        residual_encoder_num_block=None,
                        dual_softmax=to_double,
                        bit_size=bit,
                        hidden_size=896,
                        local_size=0,
                        use_univ_wavernn=use_univ,
                        conditioning_size=128,
                        embedding_size=256,
                        linear_hidden_size=512,
                        local_scale=1,
                        local_layer_num=2,
                        bug_fixed_gru_dimension=True,
                    ),
                )

                generator = Generator(
                    config,
                    model_path=Path(
                        f'tests/data/TestTrainingWaveRNN'
                        f'-to_double={to_double}'
                        f'-bit={bit}'
                        f'-mulaw={mulaw}'
                        f'-use_univ={use_univ}'
                        f'-iteration=1000.npz'
                    ),
                    gpu=gpu,
                )

                for sampling_policy in SamplingPolicy.__members__.values():
                    with self.subTest(sampling_policy=sampling_policy):
                        wave = generator.generate(
                            time_length=0.3,
                            sampling_policy=sampling_policy,
                        )
                        wave.save(Path(
                            f'test_generator_audio'
                            f'-sampling_policy={sampling_policy}'
                            f'-to_double={to_double}'
                            f'-bit={bit}'
                            f'-mulaw={mulaw}'
                            f'-use_univ={use_univ}'
                            '.wav'
                        ))
