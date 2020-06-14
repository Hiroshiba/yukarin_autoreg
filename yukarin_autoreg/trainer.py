import argparse
from copy import copy
from pathlib import Path
from typing import Any, Dict

import chainer
from chainer import cuda, optimizer_hooks, optimizers, training
from chainer.iterators import MultiprocessIterator
from chainer.training import ParallelUpdater, extensions
from chainer.training.updaters import StandardUpdater
from tensorboardX import SummaryWriter

from utility.extension_utility import TensorBoardReport
from yukarin_autoreg.config import Config, assert_config
from yukarin_autoreg.dataset import create as create_dataset
from yukarin_autoreg.evaluator import GenerateEvaluator
from yukarin_autoreg.generator import Generator
from yukarin_autoreg.model import Model, create_predictor
from yukarin_autoreg.utility.chainer_converter_utility import concat_optional


def create_trainer(
    config: Config, output: Path,
):
    assert_config(config)
    if output.exists():
        raise Exception(f"output directory {output} already exists.")

    # model
    predictor = create_predictor(config.model)
    if config.train.trained_model is not None:
        chainer.serializers.load_npz(
            config.train.trained_model["predictor_path"], predictor
        )
    model = Model(
        loss_config=config.loss,
        predictor=predictor,
        local_padding_size=config.dataset.local_padding_size,
    )

    model.to_gpu(config.train.gpu[0])
    cuda.get_device_from_id(config.train.gpu[0]).use()

    # dataset
    dataset = create_dataset(config.dataset)
    batchsize_devided = config.train.batchsize // len(config.train.gpu)
    train_iter = MultiprocessIterator(dataset["train"], config.train.batchsize)
    test_iter = MultiprocessIterator(
        dataset["test"], batchsize_devided, repeat=False, shuffle=True
    )
    train_test_iter = MultiprocessIterator(
        dataset["train_test"], batchsize_devided, repeat=False, shuffle=True
    )

    if dataset["test_eval"] is not None:
        test_eval_iter = MultiprocessIterator(
            dataset["test_eval"], batchsize_devided, repeat=False, shuffle=True
        )
    else:
        test_eval_iter = None

    # optimizer
    def create_optimizer(model):
        cp: Dict[str, Any] = copy(config.train.optimizer)
        n = cp.pop("name").lower()

        if n == "adam":
            optimizer = optimizers.Adam(**cp)
        elif n == "sgd":
            optimizer = optimizers.SGD(**cp)
        else:
            raise ValueError(n)

        optimizer.setup(model)

        if config.train.optimizer_gradient_clipping is not None:
            optimizer.add_hook(
                optimizer_hooks.GradientClipping(
                    config.train.optimizer_gradient_clipping
                )
            )

        return optimizer

    optimizer = create_optimizer(model)
    if config.train.trained_model is not None:
        chainer.serializers.load_npz(
            config.train.trained_model["optimizer_path"], optimizer
        )

    # updater
    if len(config.train.gpu) <= 1:
        updater = StandardUpdater(
            iterator=train_iter,
            optimizer=optimizer,
            converter=concat_optional,
            device=config.train.gpu[0],
        )
    else:
        updater = ParallelUpdater(
            iterator=train_iter,
            optimizer=optimizer,
            converter=concat_optional,
            devices={
                "main" if i == 0 else f"gpu{gpu}": gpu
                for i, gpu in enumerate(config.train.gpu)
            },
        )
    if config.train.trained_model is not None:
        updater.iteration = optimizer.t

    # trainer
    output.mkdir()
    config.save_as_json((output / "config.json").absolute())

    trigger_log = (config.train.log_iteration, "iteration")
    trigger_snapshot = (config.train.snapshot_iteration, "iteration")
    trigger_stop = (
        (config.train.stop_iteration, "iteration")
        if config.train.stop_iteration is not None
        else None
    )

    trainer = training.Trainer(updater, stop_trigger=trigger_stop, out=output)
    tb_writer = SummaryWriter(Path(output))

    shift_ext = None
    if config.train.linear_shift is not None:
        shift_ext = extensions.LinearShift(**config.train.linear_shift)
    if config.train.step_shift is not None:
        shift_ext = extensions.StepShift(**config.train.step_shift)
    if shift_ext is not None:
        if config.train.trained_model is not None:
            shift_ext._t = optimizer.t
        trainer.extend(shift_ext)

    ext = extensions.Evaluator(
        test_iter, model, concat_optional, device=config.train.gpu[0]
    )
    trainer.extend(ext, name="test", trigger=trigger_log)
    ext = extensions.Evaluator(
        train_test_iter, model, concat_optional, device=config.train.gpu[0]
    )
    trainer.extend(ext, name="train", trigger=trigger_log)

    if test_eval_iter is not None:
        generator = Generator(
            config=config, model=predictor, max_batch_size=config.train.batchsize
        )
        generate_evaluator = GenerateEvaluator(
            generator=generator,
            time_length=config.dataset.time_length_evaluate,
            local_padding_time_length=config.dataset.local_padding_time_length_evaluate,
        )
        ext = extensions.Evaluator(
            test_eval_iter,
            generate_evaluator,
            concat_optional,
            device=config.train.gpu[0],
        )
        trainer.extend(ext, name="eval", trigger=trigger_snapshot)

    ext = extensions.snapshot_object(
        predictor, filename="main_{.updater.iteration}.npz"
    )
    trainer.extend(ext, trigger=trigger_snapshot)
    ext = extensions.snapshot_object(
        optimizer, filename="optimizer_{.updater.iteration}.npz"
    )
    trainer.extend(ext, trigger=trigger_snapshot)

    trainer.extend(extensions.FailOnNonNumber(), trigger=trigger_log)
    trainer.extend(extensions.observe_lr(), trigger=trigger_log)
    trainer.extend(extensions.LogReport(trigger=trigger_log))
    trainer.extend(
        extensions.PrintReport(["iteration", "main/loss", "test/main/loss"]),
        trigger=trigger_log,
    )
    trainer.extend(TensorBoardReport(writer=tb_writer), trigger=trigger_log)

    trainer.extend(extensions.dump_graph(root_name="main/loss"))

    if trigger_stop is not None:
        trainer.extend(extensions.ProgressBar(trigger_stop))

    return trainer
