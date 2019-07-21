import json
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Union
from warnings import warn

from yukarin_autoreg.utility.json_utility import JSONEncoder


class DatasetConfig(NamedTuple):
    sampling_rate: int
    sampling_length: int
    input_wave_glob: str
    input_silence_glob: str
    input_local_glob: str
    bit_size: int
    gaussian_noise_sigma: float
    only_coarse: bool
    seed: int
    num_test: int


class ModelConfig(NamedTuple):
    upconv_scales: List[int]
    upconv_residual: bool
    upconv_channel_ksize: int
    residual_encoder_channel: int
    residual_encoder_num_block: int
    dual_softmax: bool
    bit_size: int
    hidden_size: int
    local_size: int
    bug_fixed_gru_dimension: bool = True


class LossConfig(NamedTuple):
    disable_fine: bool


class TrainConfig(NamedTuple):
    batchsize: int
    gpu: List[int]
    log_iteration: int
    snapshot_iteration: int
    stop_iteration: int
    optimizer: Dict[str, Any]
    optimizer_gradient_clipping: float
    linear_shift: Dict[str, Any]
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
            seed=d['dataset']['seed'],
            num_test=d['dataset']['num_test'],
        ),
        model=ModelConfig(
            hidden_size=d['model']['hidden_size'],
            bit_size=d['model']['bit_size'],
            dual_softmax=d['model']['dual_softmax'],
            local_size=d['model']['local_size'],
            upconv_scales=d['model']['upconv_scales'],
            upconv_residual=d['model']['upconv_residual'],
            upconv_channel_ksize=d['model']['upconv_channel_ksize'],
            residual_encoder_channel=d['model']['residual_encoder_channel'],
            residual_encoder_num_block=d['model']['residual_encoder_num_block'],
            bug_fixed_gru_dimension=d['model']['bug_fixed_gru_dimension'],
        ),
        loss=LossConfig(
            disable_fine=d['loss']['disable_fine'],
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

    if 'using_modified_model' not in d['model']:
        d['model']['using_modified_model'] = False

    if 'upconv_scales' not in d['model']:
        d['model']['upconv_scales'] = []

    if 'upconv_residual' not in d['model']:
        d['model']['upconv_residual'] = False

    if 'optimizer_gradient_clipping' not in d['train']:
        d['train']['optimizer_gradient_clipping'] = None

    if 'residual_encoder_channel' not in d['model']:
        d['model']['residual_encoder_channel'] = None

    if 'residual_encoder_num_block' not in d['model']:
        d['model']['residual_encoder_num_block'] = None

    if 'upconv_channel_ksize' not in d['model']:
        d['model']['upconv_channel_ksize'] = 3

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

    if 'bug_fixed_gru_dimension' not in d['model']:
        warn('this config is not bug fixed "gru dimension" https://github.com/Hiroshiba/yukarin_autoreg/pull/2')
        d['model']['bug_fixed_gru_dimension'] = False


def assert_config(config: Config):
    assert config.dataset.bit_size == config.model.bit_size

    if not config.dataset.only_coarse:
        assert config.model.dual_softmax
        assert not config.loss.disable_fine
    else:
        assert not config.model.dual_softmax
        assert config.loss.disable_fine
