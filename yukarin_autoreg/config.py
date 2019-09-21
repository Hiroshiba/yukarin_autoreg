import json
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Union

from yukarin_autoreg.utility.json_utility import JSONEncoder


class DatasetConfig(NamedTuple):
    sampling_rate: int
    sampling_length: int
    input_wave_glob: str
    input_silence_glob: str
    input_local_glob: str
    bit_size: Optional[int]
    gaussian_noise_sigma: float
    only_coarse: bool
    mulaw: bool
    local_padding_size: int
    speaker_dict_path: Optional[str]
    seed: int
    num_train: Optional[int]
    num_test: int
    fix_contain_not_silence: bool = True


class ModelConfig(NamedTuple):
    dual_softmax: bool
    bit_size: Optional[int]
    gaussian: bool
    input_categorical: bool
    hidden_size: int
    local_size: int
    conditioning_size: int
    embedding_size: int
    linear_hidden_size: int
    local_scale: int
    local_layer_num: int
    speaker_size: int
    speaker_embedding_size: int
    weight_initializer: Optional[str]


class LossConfig(NamedTuple):
    disable_fine: bool
    eliminate_silence: bool
    mean_silence: bool


class TrainConfig(NamedTuple):
    batchsize: int
    gpu: List[int]
    log_iteration: int
    snapshot_iteration: int
    stop_iteration: int
    optimizer: Dict[str, Any]
    optimizer_gradient_clipping: float
    linear_shift: Dict[str, Any]
    step_shift: Dict[str, Any]
    trained_model: Optional[str]


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
            gaussian_noise_sigma=d['dataset']['gaussian_noise_sigma'],
            only_coarse=d['dataset']['only_coarse'],
            mulaw=d['dataset']['mulaw'],
            seed=d['dataset']['seed'],
            num_train=d['dataset']['num_train'],
            num_test=d['dataset']['num_test'],
            local_padding_size=d['dataset']['local_padding_size'],
            speaker_dict_path=d['dataset']['speaker_dict_path'],
        ),
        model=ModelConfig(
            hidden_size=d['model']['hidden_size'],
            bit_size=d['model']['bit_size'],
            gaussian=d['model']['gaussian'],
            input_categorical=d['model']['input_categorical'],
            dual_softmax=d['model']['dual_softmax'],
            local_size=d['model']['local_size'],
            conditioning_size=d['model']['conditioning_size'],
            embedding_size=d['model']['embedding_size'],
            linear_hidden_size=d['model']['linear_hidden_size'],
            local_scale=d['model']['local_scale'],
            local_layer_num=d['model']['local_layer_num'],
            speaker_size=d['model']['speaker_size'],
            speaker_embedding_size=d['model']['speaker_embedding_size'],
            weight_initializer=d['model']['weight_initializer'],
        ),
        loss=LossConfig(
            disable_fine=d['loss']['disable_fine'],
            eliminate_silence=d['loss']['eliminate_silence'],
            mean_silence=d['loss']['mean_silence'],
        ),
        train=TrainConfig(
            batchsize=d['train']['batchsize'],
            gpu=d['train']['gpu'],
            log_iteration=d['train']['log_iteration'],
            snapshot_iteration=d['train']['snapshot_iteration'],
            stop_iteration=d['train']['stop_iteration'],
            optimizer=d['train']['optimizer'],
            optimizer_gradient_clipping=d['train']['optimizer_gradient_clipping'],
            linear_shift=d['train']['linear_shift'],
            step_shift=d['train']['step_shift'],
            trained_model=d['train']['trained_model'],
        ),
        project=ProjectConfig(
            name=d['project']['name'],
            tags=d['project']['tags'],
        )
    )


def backward_compatible(d: Dict):
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

    if 'optimizer_gradient_clipping' not in d['train']:
        d['train']['optimizer_gradient_clipping'] = None

    if 'linear_shift' not in d['train']:
        d['train']['linear_shift'] = None

    if 'gaussian_noise_sigma' not in d['dataset']:
        d['dataset']['gaussian_noise_sigma'] = 0.0

    if 'trained_model' not in d['train']:
        d['train']['trained_model'] = None

    if 'only_coarse' not in d['dataset']:
        d['dataset']['only_coarse'] = False

    if 'dual_softmax' not in d['model']:
        d['model']['dual_softmax'] = True

    if 'disable_fine' not in d['loss']:
        d['loss']['disable_fine'] = False

    if 'mulaw' not in d['dataset']:
        d['dataset']['mulaw'] = False

    if 'step_shift' not in d['train']:
        d['train']['step_shift'] = None

    if 'eliminate_silence' not in d['loss']:
        d['loss']['eliminate_silence'] = True

    if 'conditioning_size' not in d['model']:
        d['model']['conditioning_size'] = None

    if 'embedding_size' not in d['model']:
        d['model']['embedding_size'] = None

    if 'linear_hidden_size' not in d['model']:
        d['model']['linear_hidden_size'] = None

    if 'local_scale' not in d['model']:
        d['model']['local_scale'] = None

    if 'local_padding_size' not in d['dataset']:
        d['dataset']['local_padding_size'] = 0

    if 'local_layer_num' not in d['model']:
        d['model']['local_layer_num'] = 2

    if 'num_train' not in d['dataset']:
        d['dataset']['num_train'] = None

    if 'gaussian' not in d['model']:
        d['model']['gaussian'] = False

    if 'input_categorical' not in d['model']:
        d['model']['input_categorical'] = True

    if 'weight_initializer' not in d['model']:
        d['model']['weight_initializer'] = None

    if 'fix_contain_not_silence' not in d['dataset']:
        d['dataset']['fix_contain_not_silence'] = False

    if 'mean_silence' not in d['loss']:
        d['loss']['mean_silence'] = True

    if 'speaker_size' not in d['model']:
        d['model']['speaker_size'] = 0

    if 'speaker_embedding_size' not in d['model']:
        d['model']['speaker_embedding_size'] = 0

    if 'speaker_dict_path' not in d['dataset']:
        d['dataset']['speaker_dict_path'] = None


def assert_config(config: Config):
    assert config.dataset.bit_size == config.model.bit_size

    if not config.dataset.only_coarse:
        assert config.model.dual_softmax
        assert not config.loss.disable_fine
    else:
        assert not config.model.dual_softmax
        assert config.loss.disable_fine
