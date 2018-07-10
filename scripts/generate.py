import argparse
import multiprocessing
import re
from functools import partial
from pathlib import Path

from yukarin_autoreg.config import create_from_json as create_config
from yukarin_autoreg.generator import Generator
from yukarin_autoreg.utility.json_utility import save_arguments

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', '-md', type=Path)
parser.add_argument('--model_iteration', '-mi', type=int)
parser.add_argument('--model_config', '-mc', type=Path)
parser.add_argument('--output_dir', '-o', type=Path, default='./output/')
parser.add_argument('--gpu', type=int)
arguments = parser.parse_args()

model_dir: Path = arguments.model_dir
model_iteration: int = arguments.model_iteration
model_config: Path = arguments.model_config
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


def process(path: Path, generator: Generator):
    try:
        wave = generator.generate()
        wave.save(path)
    except:
        import traceback
        traceback.print_exc()


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

    paths_test = [output / f'{name}.wav' for name in range(5)]

    process_partial = partial(process, generator=generator)
    if gpu is None:
        pool = multiprocessing.Pool()
        pool.map(process_partial, paths_test)
    else:
        list(map(process_partial, paths_test))


if __name__ == '__main__':
    main()
