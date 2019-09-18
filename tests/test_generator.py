import unittest
from collections import namedtuple
from pathlib import Path

from yukarin_autoreg.config import ModelConfig
from yukarin_autoreg.generator import Generator, SamplingPolicy

gpu = 0


class TestGenerator(unittest.TestCase):
    def test_generator(self):
        to_double = False
        bit = 10
        mulaw = True
        iteration = 3000
        for input_categorical, gaussian, speaker_size in (
                (True, False, 0),
                (False, True, 0),
                (True, False, 4),
        ):
            with self.subTest(input_categorical=input_categorical, gaussian=gaussian):
                config = namedtuple('Config', ['dataset', 'model'])(
                    dataset=namedtuple('DatasetConfig', [
                        'sampling_rate',
                        'mulaw',
                    ])(
                        sampling_rate=8000,
                        mulaw=mulaw,
                    ),
                    model=ModelConfig(
                        dual_softmax=to_double,
                        bit_size=bit,
                        gaussian=gaussian,
                        input_categorical=input_categorical,
                        hidden_size=896,
                        local_size=0,
                        conditioning_size=128,
                        embedding_size=256,
                        linear_hidden_size=512,
                        local_scale=1,
                        local_layer_num=2,
                        speaker_size=speaker_size,
                        speaker_embedding_size=speaker_size // 4,
                        weight_initializer=None,
                    ),
                )

                generator = Generator(
                    config,
                    model_path=Path(
                        f'tests/data/test_training_wavernn'
                        f'-to_double={to_double}'
                        f'-bit={bit}'
                        f'-mulaw={mulaw}'
                        f'-input_categorical={input_categorical}'
                        f'-gaussian={gaussian}'
                        f'-speaker_size={speaker_size}'
                        f'-iteration={iteration}.npz'
                    ),
                    gpu=gpu,
                )

                for sampling_policy in SamplingPolicy.__members__.values():
                    with self.subTest(sampling_policy=sampling_policy):
                        wave = generator.generate(
                            time_length=0.3,
                            sampling_policy=sampling_policy,
                            speaker_num=0 if speaker_size > 0 else None,
                        )
                        wave.save(Path(
                            f'test_generator_audio'
                            f'-sampling_policy={sampling_policy}'
                            f'-to_double={to_double}'
                            f'-bit={bit}'
                            f'-mulaw={mulaw}'
                            f'-input_categorical={input_categorical}'
                            f'-gaussian={gaussian}'
                            f'-speaker_size={speaker_size}'
                            f'-iteration={iteration}'
                            '.wav'
                        ))
