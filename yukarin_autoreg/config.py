import json
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Union

from yukarin_autoreg.utility.json_utility import JSONEncoder


class DatasetConfig(NamedTuple):
    sampling_rate: int
    sampling_length: int
    input_wave_glob: str
    input_silence_glob: str
    input_local_glob: str
    bit_size: int
    seed: int
    num_test: int
    sign_wave_dataset: bool


class ModelConfig(NamedTuple):
    hidden_size: int
    bit_size: int
    local_size: int


class LossConfig(NamedTuple):
    pass


class TrainConfig(NamedTuple):
    batchsize: int
    gpu: int
    log_iteration: int
    snapshot_iteration: int
    stop_iteration: int
    optimizer: Dict[str, Any]


class ProjectConfig(NamedTuple):
    name: str
    tags: List[str]


class Config(NamedTuple):
    dataset: DatasetConfig
    model: ModelConfig
    loss: LossConfig
    train: TrainConfig
    project: ProjectConfig

    def save_as_json(self, path):
        d = _namedtuple_to_dict(self)
        json.dump(d, open(path, 'w'), indent=2, sort_keys=True, cls=JSONEncoder)


def _namedtuple_to_dict(o: NamedTuple):
    return {
        k: v if not hasattr(v, '_asdict') else _namedtuple_to_dict(v)
        for k, v in o._asdict().items()
    }


def create_from_json(s: Union[str, Path]):
    d = json.load(open(s))
    backward_compatible(d)

    return Config(
        dataset=DatasetConfig(
            sampling_rate=d['dataset']['sampling_rate'],
            sampling_length=d['dataset']['sampling_length'],
            input_wave_glob=d['dataset']['input_wave_glob'],
            input_silence_glob=d['dataset']['input_silence_glob'],
            input_local_glob=d['dataset']['input_local_glob'],
            bit_size=d['dataset']['bit_size'],
            seed=d['dataset']['seed'],
            num_test=d['dataset']['num_test'],
            sign_wave_dataset=d['dataset']['sign_wave_dataset'],
        ),
        model=ModelConfig(
            hidden_size=d['model']['hidden_size'],
            bit_size=d['model']['bit_size'],
            local_size=d['model']['local_size'],
        ),
        loss=LossConfig(
        ),
        train=TrainConfig(
            batchsize=d['train']['batchsize'],
            gpu=d['train']['gpu'],
            log_iteration=d['train']['log_iteration'],
            snapshot_iteration=d['train']['snapshot_iteration'],
            stop_iteration=d['train']['stop_iteration'],
            optimizer=d['train']['optimizer'],
        ),
        project=ProjectConfig(
            name=d['project']['name'],
            tags=d['project']['tags'],
        )
    )


def backward_compatible(d: Dict):
    if 'sign_wave_dataset' not in d['dataset']:
        d['dataset']['sign_wave_dataset'] = False

    if 'silence_top_db' not in d['dataset']:
        d['dataset']['silence_top_db'] = None

    if 'clipping_range' not in d['dataset']:
        d['dataset']['clipping_range'] = None

    if 'input_wave_glob' not in d['dataset']:
        d['dataset']['input_wave_glob'] = d['dataset']['input_glob']

    if 'input_silence_glob' not in d['dataset']:
        d['dataset']['input_silence_glob'] = None

    if 'input_local_glob' not in d['dataset']:
        d['dataset']['input_local_glob'] = None

    if 'local_size' not in d['model']:
        d['model']['local_size'] = 0
