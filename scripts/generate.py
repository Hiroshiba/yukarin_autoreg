import argparse
import glob
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from acoustic_feature_extractor.data.sampling_data import SamplingData
from more_itertools import chunked

from yukarin_autoreg.config import create_from_json as create_config
from yukarin_autoreg.dataset import SpeakerWavesDataset, WavesDataset
from yukarin_autoreg.generator import Generator, SamplingPolicy
from yukarin_autoreg.utility.json_utility import save_arguments

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", "-md", type=Path)
parser.add_argument("--model_iteration", "-mi", type=int)
parser.add_argument("--model_config", "-mc", type=Path)
parser.add_argument("--time_length", "-tl", type=float, default=1)
parser.add_argument("--input_batchsize", "-ib", type=int)
parser.add_argument("--num_test", "-nt", type=int, default=5)
parser.add_argument(
    "--sampling_policy", "-sp", type=SamplingPolicy, default=SamplingPolicy.random
)
parser.add_argument("--val_local_glob", "-vlg")
parser.add_argument("--val_speaker_num", "-vsn", type=int)
parser.add_argument("--output_dir", "-o", type=Path, default="./output/")
parser.add_argument("--gpu", type=int)
arguments = parser.parse_args()

model_dir: Path = arguments.model_dir
model_iteration: int = arguments.model_iteration
model_config: Path = arguments.model_config
time_length: int = arguments.time_length
input_batchsize: Optional[int] = arguments.input_batchsize
num_test: int = arguments.num_test
sampling_policy: SamplingPolicy = arguments.sampling_policy
val_local_glob: str = arguments.val_local_glob
val_speaker_num: Optional[int] = arguments.val_speaker_num
output_dir: Path = arguments.output_dir
gpu: int = arguments.gpu


def _extract_number(f):
    s = re.findall(r"\d+", str(f))
    return int(s[-1]) if s else -1


def _get_predictor_model_path(
    model_dir: Path, iteration: int = None, prefix: str = "main_",
):
    if iteration is None:
        paths = model_dir.glob(prefix + "*.npz")
        model_path = list(sorted(paths, key=_extract_number))[-1]
    else:
        model_path = model_dir / (prefix + "{}.npz".format(iteration))
        assert model_path.exists()
    return model_path


def process_wo_context(
    local_paths: Sequence[Path],
    speaker_nums: Optional[Sequence[int]],
    generator: Generator,
    postfix="_woc",
):
    try:
        local_datas = [SamplingData.load(local_path) for local_path in local_paths]
        size = int((time_length + 5) * local_datas[0].rate)
        local_arrays = [
            local_data.array[:size]
            if len(local_data.array) >= size
            else np.pad(
                local_data.array,
                ((0, size - len(local_data.array)), (0, 0)),
                mode="edge",
            )
            for local_data in local_datas
        ]

        waves = generator.generate(
            time_length=time_length,
            sampling_policy=sampling_policy,
            num_generate=len(local_arrays),
            local_array=np.stack(local_arrays),
            speaker_nums=speaker_nums,
        )
        for wave, local_path in zip(waves, local_paths):
            wave.save(output_dir / (local_path.stem + postfix + ".wav"))
    except:
        import traceback

        traceback.print_exc()


def main():
    model_path = _get_predictor_model_path(
        model_dir=model_dir, iteration=model_iteration,
    )

    output_dir.mkdir(exist_ok=True, parents=True)
    save_arguments(arguments, output_dir / "arguments.json")

    config = create_config(model_config)
    model = Generator.load_model(
        model_config=config.model, model_path=model_path, gpu=gpu,
    )
    print(f'Loaded generator "{model_path}"')

    batchsize = (
        input_batchsize if input_batchsize is not None else config.train.batchsize
    )
    generator = Generator(config=config, model=model,)

    from yukarin_autoreg.dataset import create

    dataset = create(config.dataset)["test"]

    if isinstance(dataset, WavesDataset):
        inputs = dataset.inputs
        local_paths = [input.path_local for input in inputs[:num_test]]
        speaker_nums = [None] * num_test
    elif isinstance(dataset, SpeakerWavesDataset):
        inputs = dataset.wave_dataset.inputs
        local_paths = [input.path_local for input in inputs[:num_test]]
        speaker_nums = dataset.speaker_nums[:num_test]
    else:
        raise ValueError(dataset)

    # random
    for local_path, speaker_num in zip(
        chunked(local_paths, batchsize), chunked(speaker_nums, batchsize)
    ):
        process_wo_context(
            generator=generator,
            local_paths=local_path,
            speaker_nums=speaker_num if speaker_num[0] is not None else None,
        )

    # validation
    if val_local_glob is not None:
        local_paths = sorted([Path(p) for p in glob.glob(val_local_glob)])
        speaker_nums = [val_speaker_num] * len(local_paths)
        for local_path, speaker_num in zip(
            chunked(local_paths, batchsize), chunked(speaker_nums, batchsize)
        ):
            process_wo_context(
                generator=generator,
                local_paths=local_path,
                speaker_nums=speaker_num if speaker_num[0] is not None else None,
            )


if __name__ == "__main__":
    main()
