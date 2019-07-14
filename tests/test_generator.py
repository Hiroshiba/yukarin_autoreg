import unittest
from collections import namedtuple
from pathlib import Path

from yukarin_autoreg.generator import Generator, SamplingPolicy


class TestGenerator(unittest.TestCase):
    def test_generator(self):
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
            ])(
                upconv_scales=[],
                upconv_residual=False,
                upconv_channel_ksize=0,
                residual_encoder_channel=None,
                residual_encoder_num_block=None,
                dual_softmax=True,
                bit_size=16,
                hidden_size=896,
                local_size=0,
            ),
        )

        generator = Generator(
            config,
            model_path=Path('tests/data/TestTrainingWaveRNN.npz'),
            gpu=3,
        )

        for sampling_policy in SamplingPolicy.__members__.values():
            with self.subTest(sampling_policy=sampling_policy):
                wave = generator.generate(
                    time_length=0.3,
                    sampling_policy=sampling_policy,
                )
                wave.save(Path(f'test_generator_audio_{sampling_policy}.wav'))
