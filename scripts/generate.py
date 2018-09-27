import argparse
import glob
import re
from functools import partial
from itertools import starmap
from pathlib import Path
from typing import List

import numpy as np

from yukarin_autoreg.config import create_from_json as create_config
from yukarin_autoreg.generator import Generator
from yukarin_autoreg.sampling_data import SamplingData
from yukarin_autoreg.utility.json_utility import save_arguments
from yukarin_autoreg.wave import Wave

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', '-md', type=Path)
parser.add_argument('--model_iterations', '-mi', nargs='*', type=int)
parser.add_argument('--model_config', '-mc', type=Path)
parser.add_argument('--time_length', '-tl', type=float, default=1)
parser.add_argument('--num_test', '-nt', type=int, default=5)
parser.add_argument('--sampling_maximum', '-sm', action='store_true')
parser.add_argument('--num_mean_model', '-nmm', type=int, default=1)
parser.add_argument('--output_dir', '-o', type=Path, default='./output/')
parser.add_argument('--gpu', type=int)
arguments = parser.parse_args()

model_dir: Path = arguments.model_dir
model_iterations: List[int] = arguments.model_iterations
model_config: Path = arguments.model_config
time_length: int = arguments.time_length
num_test: int = arguments.num_test
sampling_maximum: bool = arguments.sampling_maximum
num_mean_model: int = arguments.num_mean_model
output_dir: Path = arguments.output_dir
gpu: int = arguments.gpu

output_dir.mkdir(exist_ok=True)

output = output_dir / model_dir.name
output.mkdir(exist_ok=True)


def _extract_number(f):
    s = re.findall("\d+", str(f))
    return int(s[-1]) if s else -1


def _get_predictor_model_paths(
        model_dir: Path,
        iterations: List[int] = None,
        num_mean_model: int = None,
        prefix: str = 'main_',
):
    if iterations is None:
        assert isinstance(num_mean_model, int)
        paths = model_dir.glob(prefix + '*.npz')
        model_paths = list(sorted(paths, key=_extract_number))[-num_mean_model:]
    else:
        model_paths = [model_dir / (prefix + '{}.npz'.format(i)) for i in iterations]
    return model_paths


def process_wo_context(local_path: Path, generator: Generator, postfix='_woc'):
    try:
        l = SamplingData.load(local_path).array
        wave = generator.generate(
            time_length=time_length,
            sampling_maximum=sampling_maximum,
            local_array=l,
        )
        wave.save(output / (local_path.stem + postfix + '.wav'))
    except:
        import traceback
        traceback.print_exc()


def process_resume(wave_path: Path, local_path: Path, generator: Generator, sampling_rate: int, sampling_length: int):
    try:
        w = Wave.load(wave_path, sampling_rate=sampling_rate).wave
        l_data = SamplingData.load(local_path)
        l = l_data.array

        l_scale = int(sampling_rate // l_data.rate)
        l_sl = sampling_length // l_scale
        sampling_length = l_sl * l_scale

        c, f, hc, hf = generator.forward(w[:sampling_length], l[:l_sl])

        wave = generator.generate(
            time_length=time_length,
            sampling_maximum=sampling_maximum,
            coarse=c,
            fine=f,
            local_array=l[l_sl:],
            hidden_coarse=hc,
            hidden_fine=hf,
        )
        wave.save(output / (wave_path.stem + '.wav'))
    except:
        import traceback
        traceback.print_exc()


def main():
    save_arguments(arguments, output / 'arguments.json')

    config = create_config(model_config)
    models = _get_predictor_model_paths(
        model_dir=model_dir,
        iterations=model_iterations,
        num_mean_model=num_mean_model,
    )
    generator = Generator(
        config,
        models,
        gpu=gpu,
    )
    print(f'Loaded generator "{models}"')

    if config.dataset.input_wave_glob is not None:
        wave_paths = sorted([Path(p) for p in glob.glob(str(config.dataset.input_wave_glob))])
        local_paths = sorted([Path(p) for p in glob.glob(str(config.dataset.input_local_glob))])
        assert len(wave_paths) == len(local_paths)

        np.random.RandomState(config.dataset.seed).shuffle(wave_paths)
        np.random.RandomState(config.dataset.seed).shuffle(local_paths)
        wave_paths = wave_paths[:num_test]
        local_paths = local_paths[:num_test]

        # random
        process_partial = partial(
            process_wo_context,
            generator=generator,
        )
        list(map(process_partial, local_paths))

        # resume
        process_partial = partial(
            process_resume,
            generator=generator,
            sampling_rate=config.dataset.sampling_rate,
            sampling_length=config.dataset.sampling_rate,
        )
        list(starmap(process_partial, zip(wave_paths, local_paths)))


if __name__ == '__main__':
    main()
