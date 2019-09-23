import argparse
import importlib
import sys
from pathlib import Path

import optuna
from optuna.samplers import RandomSampler

from yukarin_autoreg.config import create_from_json, Config
from yukarin_autoreg.trainer import create_trainer


def create_config(config_path: Path, trial: optuna.Trial) -> Config:
    sys.path.append(str(config_path.parent))
    config = importlib.import_module(config_path.stem).create_config(trial)
    return config


def objective(trial):
    x = trial.suggest_categorical('x', (0, 1, 2, 3, 4))
    return x


def train_optuna(
        # config_json_path: Path,
        # output: Path,
):
    study = optuna.create_study(
        study_name='hoge',
        storage='sqlite:///example.db',
        sampler=RandomSampler(),
    )

    # config = create_from_json(config_json_path)
    # trainer = create_trainer(config=config, output=output)
    # trainer.run()

    study.optimize(objective, n_trials=10)

    import code
    code.interact(local=locals())


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('config_json_path', type=Path)
    # parser.add_argument('output', type=Path)
    # arguments = parser.parse_args()

    train_optuna(
        # config_json_path=arguments.config_json_path,
        # output=arguments.output,
    )
