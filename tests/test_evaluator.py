import unittest

import numpy as np
from parameterized import parameterized

from tests.utility import get_test_model_path, get_test_config
from yukarin_autoreg.evaluator import GenerateEvaluator
from yukarin_autoreg.generator import Generator, SamplingPolicy

gpu = 0

to_double = False
bit = 10
mulaw = True
iteration = 3000


class TestEvaluator(unittest.TestCase):
    @parameterized.expand([
        (4, 4),
    ])
    def test_generate_evaluator(self, speaker_size, num_generate):
        config = get_test_config(
            to_double=to_double,
            bit=bit,
            mulaw=mulaw,
            speaker_size=speaker_size,
        )

        model = Generator.load_model(
            model_config=config.model,
            model_path=get_test_model_path(
                to_double=to_double,
                bit=bit,
                mulaw=mulaw,
                speaker_size=speaker_size,
                iteration=iteration,
            ),
            gpu=gpu,
        )
        generator = Generator(
            config=config,
            model=model,
        )

        speaker_nums = list(range(num_generate)) if speaker_size > 0 else None
        waves = generator.generate(
            time_length=1,
            sampling_policy=SamplingPolicy.random,
            num_generate=num_generate,
            speaker_nums=speaker_nums,
        )

        wave = np.array([w.wave for w in waves])
        assert np.var(wave) > 0.2

        evaluator = GenerateEvaluator(
            generator=generator,
            time_length=1,
            local_padding_time_length=0,
            sampling_policy=SamplingPolicy.random,
        )
        scores = evaluator(
            wave=wave,
            local=None,
            speaker_num=speaker_nums,
        )
        assert scores['mcd'][0].mean() < 1
