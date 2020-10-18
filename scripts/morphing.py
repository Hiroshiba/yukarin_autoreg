import argparse
import glob
import traceback
from itertools import product
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from acoustic_feature_extractor.data.sampling_data import SamplingData
from more_itertools import chunked
from yukarin_autoreg.config import create_from_json as create_config
from yukarin_autoreg.generator import Generator, MorphingPolicy, SamplingPolicy
from yukarin_autoreg.utility.json_utility import save_arguments


def process_local_data(local_paths: Sequence[Path], time_length: float):
    local_datas = [SamplingData.load(local_path) for local_path in local_paths]
    size = int((time_length + 1) * local_datas[0].rate)
    local_arrays = [
        local_data.array[:size]
        if len(local_data.array) >= size
        else np.pad(
            local_data.array, ((0, size - len(local_data.array)), (0, 0)), mode="edge",
        )
        for local_data in local_datas
    ]
    return local_arrays


def process(
    local_paths1: Sequence[Path],
    local_paths2: Sequence[Path],
    speaker_nums1: Sequence[int],
    speaker_nums2: Sequence[int],
    start_rates: Sequence[float],
    stop_rates: Sequence[float],
    generator: Generator,
    time_length: float,
    sampling_policy: SamplingPolicy,
    morphing_policy: MorphingPolicy,
    output_dir: Path,
):
    try:
        local_arrays1 = process_local_data(local_paths1, time_length=time_length)
        local_arrays2 = process_local_data(local_paths2, time_length=time_length)

        waves = generator.morphing(
            time_length=time_length,
            sampling_policy=sampling_policy,
            morphing_policy=morphing_policy,
            local_array1=np.stack(local_arrays1),
            local_array2=np.stack(local_arrays2),
            speaker_nums1=speaker_nums1,
            speaker_nums2=speaker_nums2,
            start_rates=start_rates,
            stop_rates=stop_rates,
        )
        for wave, local_path1, local_path2, start_rate, stop_rate in zip(
            waves, local_paths1, local_paths2, start_rates, stop_rates
        ):
            wave.save(
                output_dir.joinpath(
                    local_path1.stem
                    + "-"
                    + local_path2.stem
                    + "-"
                    + str(start_rate)
                    + "-"
                    + str(stop_rate)
                    + ".wav"
                )
            )
    except:
        traceback.print_exc()


def main(
    model_path: Path,
    model_config: Path,
    time_length: int,
    input_batchsize: Optional[int],
    sampling_policy: SamplingPolicy,
    morphing_policy: MorphingPolicy,
    local_glob1: str,
    local_glob2: str,
    speaker_num1: int,
    speaker_num2: int,
    start_rates: Sequence[float],
    stop_rates: Sequence[float],
    output_dir: Path,
    gpu: int,
):
    output_dir.mkdir(exist_ok=True, parents=True)
    save_arguments(locals(), output_dir / "arguments.json")

    config = create_config(model_config)
    model = Generator.load_model(
        model_config=config.model, model_path=model_path, gpu=gpu,
    )
    print(f'Loaded generator "{model_path}"')

    batchsize = (
        input_batchsize if input_batchsize is not None else config.train.batchsize
    )
    generator = Generator(config=config, model=model)

    local_paths1 = sorted([Path(p) for p in glob.glob(local_glob1)])
    local_paths2 = sorted([Path(p) for p in glob.glob(local_glob2)])
    assert len(local_paths1) == len(local_paths2)

    for (_local_paths1, _local_paths2), (start_rate, stop_rate) in product(
        zip(chunked(local_paths1, batchsize), chunked(local_paths2, batchsize)),
        zip(start_rates, stop_rates),
    ):
        num = len(local_paths1)
        process(
            local_paths1=_local_paths1,
            local_paths2=_local_paths2,
            speaker_nums1=[speaker_num1] * num,
            speaker_nums2=[speaker_num2] * num,
            start_rates=[start_rate] * num,
            stop_rates=[stop_rate] * num,
            generator=generator,
            time_length=time_length,
            sampling_policy=sampling_policy,
            morphing_policy=morphing_policy,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path)
    parser.add_argument("--model_config", type=Path)
    parser.add_argument("--time_length", type=float, default=1)
    parser.add_argument("--input_batchsize", type=int)
    parser.add_argument(
        "--sampling_policy", type=SamplingPolicy, default=SamplingPolicy.random
    )
    parser.add_argument(
        "--morphing_policy", type=MorphingPolicy, default=MorphingPolicy.sphere
    )
    parser.add_argument("--local_glob1")
    parser.add_argument("--local_glob2")
    parser.add_argument("--speaker_num1", type=int)
    parser.add_argument("--speaker_num2", type=int)
    parser.add_argument("--start_rates", type=float, nargs="+")
    parser.add_argument("--stop_rates", type=float, nargs="+")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--gpu", type=int)
    main(**vars(parser.parse_args()))
