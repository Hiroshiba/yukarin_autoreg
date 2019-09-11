import unittest
from collections import namedtuple
from pathlib import Path

from yukarin_autoreg.generator import Generator, SamplingPolicy

gpu = 3


class TestGenerator(unittest.TestCase):
    def test_generator(self):
        for to_double, bit, mulaw in (
                (False, 9, True),
        ):
            with self.subTest(to_double=to_double, bit=bit, mulaw=mulaw):
                config = namedtuple('Config', ['dataset', 'model'])(
                    dataset=namedtuple('DatasetConfig', [
                        'sampling_rate',
                        'mulaw',
                    ])(
                        sampling_rate=8000,
                        mulaw=mulaw,
                    ),
                    model=namedtuple('ModelConfig', [
                        'dual_softmax',
                        'bit_size',
                        'hidden_size',
                        'local_size',
                        'conditioning_size',
                        'embedding_size',
                        'linear_hidden_size',
                        'local_scale',
                        'local_layer_num',
                    ])(
                        dual_softmax=to_double,
                        bit_size=bit,
                        hidden_size=896,
                        local_size=0,
                        conditioning_size=128,
                        embedding_size=256,
                        linear_hidden_size=512,
                        local_scale=1,
                        local_layer_num=2,
                    ),
                )

                generator = Generator(
                    config,
                    model_path=Path(
                        f'tests/data/TestTrainingWaveRNN'
                        f'-to_double={to_double}'
                        f'-bit={bit}'
                        f'-mulaw={mulaw}'
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
                            '.wav'
                        ))
