import unittest
from collections import namedtuple
from pathlib import Path

from yukarin_autoreg.generator import Generator, SamplingPolicy


class TestGenerator(unittest.TestCase):
    def test_generator(self):
        for to_double, bit in zip([True, False], [16, 8]):
            with self.subTest(to_double=to_double, bit=bit):
                config = namedtuple('Config', ['dataset', 'model'])(
                    dataset=namedtuple('DatasetConfig', [
                        'sampling_rate',
                    ])(
                        sampling_rate=8000,
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
                        bug_fixed_gru_dimension=False,
                    ),
                )

                generator = Generator(
                    config,
                    model_path=Path(
                        f'tests/data/TestTrainingWaveRNN'
                        f'-to_double={to_double}'
                        f'-bit={bit}'
                        f'-iteration=1000.npz'
                    ),
                    gpu=3,
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
                            f'-bit={bit}.wav'
                        ))
