import argparse
import glob
import re
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from acoustic_feature_extractor.data.sampling_data import SamplingData

from yukarin_autoreg.config import create_from_json as create_config
from yukarin_autoreg.dataset import WavesDataset, SpeakerWavesDataset
from yukarin_autoreg.generator import Generator, SamplingPolicy
from yukarin_autoreg.utility.json_utility import save_arguments

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', '-md', type=Path)
parser.add_argument('--model_iteration', '-mi', type=int)
parser.add_argument('--model_config', '-mc', type=Path)
parser.add_argument('--time_length', '-tl', type=float, default=1)
parser.add_argument('--num_test', '-nt', type=int, default=5)
parser.add_argument('--sampling_policy', '-sp', type=SamplingPolicy, default=SamplingPolicy.random)
parser.add_argument('--val_local_glob1', '-vlg1')
parser.add_argument('--val_local_glob2', '-vlg2')
parser.add_argument('--val_speaker_num1', '-vsn1', type=int)
parser.add_argument('--val_speaker_num2', '-vsn2', type=int)
parser.add_argument('--output_dir', '-o', type=Path, default='./output/')
parser.add_argument('--gpu', type=int)
arguments = parser.parse_args()

model_dir: Path = arguments.model_dir
model_iteration: int = arguments.model_iteration
model_config: Path = arguments.model_config
time_length: int = arguments.time_length
num_test: int = arguments.num_test
sampling_policy: SamplingPolicy = arguments.sampling_policy
val_local_glob1: str = arguments.val_local_glob1
val_local_glob2: str = arguments.val_local_glob2
val_speaker_num1: Optional[int] = arguments.val_speaker_num1
val_speaker_num2: Optional[int] = arguments.val_speaker_num2
output_dir: Path = arguments.output_dir
gpu: int = arguments.gpu


def _extract_number(f):
    s = re.findall(r'\d+', str(f))
    return int(s[-1]) if s else -1


def _get_predictor_model_path(
        model_dir: Path,
        iteration: int = None,
        prefix: str = 'main_',
):
    if iteration is None:
        paths = model_dir.glob(prefix + '*.npz')
        model_path = list(sorted(paths, key=_extract_number))[-1]
    else:
        model_path = model_dir / (prefix + '{}.npz'.format(iteration))
        assert model_path.exists()
    return model_path


def process_wo_context(
        local_paths: Tuple[Path, Path],
        speaker_num1: int,
        speaker_num2: int,
        generator: Generator,
        postfix='_woc',
):
    try:
        morph_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        local_path1, local_path2 = local_paths
        l_data1 = SamplingData.load(local_path1)
        l1 = np.expand_dims(l_data1.array[:int((time_length + 5) * l_data1.rate)], axis=0)
        l_data2 = SamplingData.load(local_path2)
        l2 = np.expand_dims(l_data2.array[:int((time_length + 5) * l_data2.rate)], axis=0)

        waves = generator.generate(
            time_length=time_length,
            sampling_policy=sampling_policy,
            num_generate=len(morph_rates),
            local_array1=np.concatenate([l1] * len(morph_rates)),
            local_array2=np.concatenate([l2] * len(morph_rates)),
            speaker_nums1=[speaker_num1] * len(morph_rates) if speaker_num1 is not None else None,
            speaker_nums2=[speaker_num2] * len(morph_rates) if speaker_num2 is not None else None,
            morph_rates=morph_rates,
        )
        for wave, morph_rate in zip(waves, morph_rates):
            wave.save(output_dir / (local_path1.stem + '-' + local_path2.stem + '-' + str(morph_rate) + postfix + '.wav'))
    except:
        import traceback
        traceback.print_exc()


def main():
    model_path = _get_predictor_model_path(
        model_dir=model_dir,
        iteration=model_iteration,
    )

    output_dir.mkdir(exist_ok=True, parents=True)
    save_arguments(arguments, output_dir / 'arguments.json')

    config = create_config(model_config)
    model = Generator.load_model(
        model_config=config.model,
        model_path=model_path,
        gpu=gpu,
    )
    print(f'Loaded generator "{model_path}"')

    generator = Generator(
        config=config,
        model=model,
    )

    # validation
    if val_local_glob1 is not None:
        local_paths1 = sorted([Path(p) for p in glob.glob(val_local_glob1)])
        local_paths2 = sorted([Path(p) for p in glob.glob(val_local_glob2)])
        process_partial = partial(
            process_wo_context,
            generator=generator,
            speaker_num1=val_speaker_num1,
            speaker_num2=val_speaker_num2,
        )
        list(map(process_partial, zip(local_paths1, local_paths2)))


if __name__ == '__main__':
    main()
