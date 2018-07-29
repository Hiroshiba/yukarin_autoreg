import argparse
import re
from functools import partial
from pathlib import Path
import glob

import numpy as np
from yukarin_autoreg.config import create_from_json as create_config
from yukarin_autoreg.generator import Generator
from yukarin_autoreg.utility.json_utility import save_arguments
from yukarin_autoreg.wave import Wave

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', '-md', type=Path)
parser.add_argument('--model_iteration', '-mi', type=int)
parser.add_argument('--model_config', '-mc', type=Path)
parser.add_argument('--time_length', '-tl', type=float, default=1)
parser.add_argument('--num_test', '-nt', type=int, default=5)
parser.add_argument('--sampling_maximum', '-sm', action='store_true')
parser.add_argument('--output_dir', '-o', type=Path, default='./output/')
parser.add_argument('--gpu', type=int)
arguments = parser.parse_args()

model_dir: Path = arguments.model_dir
model_iteration: int = arguments.model_iteration
model_config: Path = arguments.model_config
time_length: int = arguments.time_length
num_test: int = arguments.num_test
sampling_maximum: bool = arguments.sampling_maximum
output_dir: Path = arguments.output_dir
gpu: int = arguments.gpu

output_dir.mkdir(exist_ok=True)

output = output_dir / model_dir.name
output.mkdir(exist_ok=True)


def _extract_number(f):
    s = re.findall("\d+", str(f))
    return int(s[-1]) if s else -1


def _get_predictor_model_path(model_dir: Path, iteration: int = None, prefix: str = 'main_'):
    if iteration is None:
        paths = model_dir.glob(prefix + '*.npz')
        model_path = list(sorted(paths, key=_extract_number))[-1]
    else:
        fn = prefix + '{}.npz'.format(iteration)
        model_path = model_dir / fn
    return model_path


def process_wo_context(filename: str, generator: Generator):
    wave = generator.generate(
        time_length=time_length,
        sampling_maximum=sampling_maximum,
    )
    wave.save(output / filename)


def process_resume(path: Path, generator: Generator, sampling_rate: int, sampling_length: int):
    w = Wave.load(path, sampling_rate=sampling_rate)
    c, f, hc, hf = generator.forward(w.wave[:sampling_length])
    wave = generator.generate(
        time_length=time_length,
        sampling_maximum=sampling_maximum,
        coarse=c,
        fine=f,
        hidden_coarse=hc,
        hidden_fine=hf,
    )
    wave.save(output / path.name)


def main():
    save_arguments(arguments, output / 'arguments.json')

    config = create_config(model_config)
    model = _get_predictor_model_path(model_dir, model_iteration)
    generator = Generator(
        config,
        model,
        gpu=gpu,
    )
    print(f'Loaded generator "{model}"')

    # random
    process_wo_context('wo_context.wav', generator=generator)

    # resume
    if config.dataset.input_glob is not None:
        input_paths = sorted([Path(p) for p in glob.glob(str(config.dataset.input_glob))])
        np.random.RandomState(config.dataset.seed).shuffle(input_paths)
        test_paths = input_paths[:num_test]

        process_partial = partial(
            process_resume,
            generator=generator,
            sampling_rate=config.dataset.sampling_rate,
            sampling_length=config.dataset.sampling_length,
        )
        list(map(process_partial, test_paths))


if __name__ == '__main__':
    main()
