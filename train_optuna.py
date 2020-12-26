import argparse
import importlib
import sys
import traceback
from functools import partial
from pathlib import Path
from typing import Any, Dict

import chainer
import optuna
from optuna.integration import ChainerPruningExtension
from optuna.pruners import PercentilePruner
from optuna.storages import RDBStorage
from optuna.structs import TrialPruned
from yukarin_autoreg.config import Config
from yukarin_autoreg.trainer import create_trainer


def param_dict_to_name(param_dict: Dict[str, Any]):
    return ",".join(
        f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}"
        for k, v in sorted(param_dict.items())
    )


def create_config(config_path: Path, trial: optuna.Trial) -> Config:
    sys.path.append(str(config_path.parent))
    config = importlib.import_module(config_path.stem).create_config(trial)
    return config


def objective(
    trial: optuna.Trial, config_path: Path, root_output: Path,
):
    config = create_config(config_path=config_path, trial=trial)
    postfix = param_dict_to_name(trial.params)
    output = root_output / (f"{trial.number}-" + postfix)

    try:
        trainer = create_trainer(config=config, output=output)
        trainer.extend(
            ChainerPruningExtension(
                trial=trial,
                observation_key=config.train.optuna["key"],
                pruner_trigger=(config.train.optuna["iteration"], "iteration"),
            ),
        )
        trainer.run()
    except chainer.cuda.cupy.cuda.memory.OutOfMemoryError as e:
        traceback.print_exc()
        raise TrialPruned()

    log_last = trainer.get_extension("LogReport").log[-1]
    return log_last[config.train.optuna["key"]]


def train_optuna(
    config_path: Path, root_output: Path, name: str, storage: str, num_trials: int,
):
    study = optuna.create_study(
        storage=RDBStorage(storage),
        pruner=PercentilePruner(25),
        study_name=name,
        load_if_exists=True,
    )
    objective_wrapper = partial(
        objective, config_path=config_path, root_output=root_output,
    )
    study.optimize(func=objective_wrapper, n_trials=num_trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=Path)
    parser.add_argument("root_output", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--storage")
    parser.add_argument("--num_trials", type=int)
    arguments = parser.parse_args()

    train_optuna(**vars(arguments))
