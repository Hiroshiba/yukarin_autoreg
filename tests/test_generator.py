import unittest
from pathlib import Path

from parameterized import parameterized

from tests.utility import get_test_model_path, get_test_config
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
        config = get_test_config(
            to_double=to_double,
            bit=bit,
            mulaw=mulaw,
            input_categorical=input_categorical,
            gaussian=gaussian,
            speaker_size=speaker_size,
        )

        model = Generator.load_model(
            model_config=config.model,
            model_path=get_test_model_path(
                to_double=to_double,
                bit=bit,
                mulaw=mulaw,
                input_categorical=input_categorical,
                gaussian=gaussian,
                speaker_size=speaker_size,
                iteration=iteration,
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
