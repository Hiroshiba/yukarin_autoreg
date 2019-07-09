import unittest
from collections import namedtuple
from pathlib import Path

from yukarin_autoreg.generator import Generator, SamplingPolicy


class TestGenerator(unittest.TestCase):
    def setUp(self):
        pass

    def test_generator(self):
        config = namedtuple('Config', ['dataset', 'model'])(
            dataset=namedtuple('DatasetConfig', [
                'sampling_rate',
            ])(
                sampling_rate=8000,
            ),
            model=namedtuple('ModelConfig', [
                'hidden_size',
                'bit_size',
                'local_size',
                'upconv_scales',
                'upconv_residual',
                'upconv_channel_ksize',
                'residual_encoder_channel',
                'residual_encoder_num_block',
            ])(
                hidden_size=896,
                bit_size=16,
                local_size=0,
                upconv_scales=[],
                upconv_residual=False,
                upconv_channel_ksize=0,
                residual_encoder_channel=None,
                residual_encoder_num_block=None,
            ),
        )

        generator = Generator(
            config,
            Path('tests/data/TestTrainingWaveRNN.npz'),
            gpu=3,
        )
        wave = generator.generate(
            time_length=0.3,
            sampling_policy=SamplingPolicy.random,
        )
        wave.save(Path('test_generator_audio_random.wav'))

        wave = generator.generate(
            time_length=0.3,
            sampling_policy=SamplingPolicy.mix,
        )
        wave.save(Path('test_generator_audio_mix.wav'))

        wave = generator.generate(
            time_length=0.3,
            sampling_policy=SamplingPolicy.maximum,
        )
        wave.save(Path('test_generator_audio_maximum.wav'))

