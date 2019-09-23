import unittest
from collections import namedtuple
from pathlib import Path

from parameterized import parameterized

from yukarin_autoreg.config import ModelConfig
from yukarin_autoreg.generator import Generator, SamplingPolicy

gpu = 0

to_double = False
bit = 10
mulaw = True
iteration = 3000


class TestGenerator(unittest.TestCase):
    @parameterized.expand([
        (True, False, 0, 1),
        (False, True, 0, 1),
        (True, False, 4, 1),
        (True, False, 4, 4),
    ])
    def test_generator(self, input_categorical, gaussian, speaker_size, num_generate):
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

        model = Generator.load_model(
            model_config=config.model,
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
        generator = Generator(
            config=config,
            model=model,
        )

        for sampling_policy in SamplingPolicy.__members__.values():
            with self.subTest(sampling_policy=sampling_policy):
                waves = generator.generate(
                    time_length=0.3,
                    sampling_policy=sampling_policy,
                    num_generate=num_generate,
                    speaker_nums=list(range(num_generate)) if speaker_size > 0 else None,
                )
                for num, wave in enumerate(waves):
                    wave.save(Path(
                        f'test_generator_audio'
                        f'-sampling_policy={sampling_policy}'
                        f'-to_double={to_double}'
                        f'-bit={bit}'
                        f'-mulaw={mulaw}'
                        f'-input_categorical={input_categorical}'
                        f'-gaussian={gaussian}'
                        f'-speaker_size={speaker_size}'
                        f'-num={num}'
                        f'-iteration={iteration}'
                        '.wav'
                    ))
