import argparse
import multiprocessing
from copy import copy
from functools import partial
from pathlib import Path
from typing import Any, Dict

from chainer import cuda, optimizer_hooks, optimizers, training
from chainer.dataset import convert
from chainer.iterators import MultiprocessIterator
from chainer.training import extensions
from chainer.training.updaters import MultiprocessParallelUpdater, StandardUpdater
from tb_chainer import SummaryWriter

from utility.chainer_extension_utility import TensorBoardReport
from yukarin_autoreg.config import create_from_json
from yukarin_autoreg.dataset import create as create_dataset
from yukarin_autoreg.model import Model, create_predictor

parser = argparse.ArgumentParser()
parser.add_argument('config_json_path', type=Path)
parser.add_argument('output', type=Path)
arguments = parser.parse_args()

config = create_from_json(arguments.config_json_path)
arguments.output.mkdir(exist_ok=True)
config.save_as_json((arguments.output / 'config.json').absolute())

# model
predictor = create_predictor(config.model)
model = Model(loss_config=config.loss, predictor=predictor)

if len(config.train.gpu) == 1:
    model.to_gpu(config.train.gpu[0])
    cuda.get_device_from_id(config.train.gpu[0]).use()

# dataset
dataset = create_dataset(config.dataset)
batchsize_devided = config.train.batchsize // len(config.train.gpu)
train_iters = [
    MultiprocessIterator(
        dataset['train'],
        batchsize_devided,
        n_processes=multiprocessing.cpu_count() // len(config.train.gpu),
    )
    for _ in config.train.gpu
]
test_iter = MultiprocessIterator(dataset['test'], batchsize_devided, repeat=False, shuffle=False)
train_eval_iter = MultiprocessIterator(dataset['train_eval'], batchsize_devided, repeat=False, shuffle=False)


# optimizer
def create_optimizer(model):
    cp: Dict[str, Any] = copy(config.train.optimizer)
    n = cp.pop('name').lower()

    if n == 'adam':
        optimizer = optimizers.Adam(**cp)
    elif n == 'sgd':
        optimizer = optimizers.SGD(**cp)
    else:
        raise ValueError(n)

    optimizer.setup(model)

    if config.train.optimizer_gradient_clipping is not None:
        optimizer.add_hook(optimizer_hooks.GradientClipping(config.train.optimizer_gradient_clipping))

    return optimizer


optimizer = create_optimizer(model)

# updater
converter = partial(convert.concat_examples)
if len(config.train.gpu) <= 1:
    updater = StandardUpdater(
        iterator=train_iters[0],
        optimizer=optimizer,
        converter=converter,
        device=config.train.gpu[0],
    )
else:
    updater = MultiprocessParallelUpdater(
        iterators=train_iters,
        optimizer=optimizer,
        converter=converter,
        devices=config.train.gpu,
    )

# trainer
trigger_log = (config.train.log_iteration, 'iteration')
trigger_snapshot = (config.train.snapshot_iteration, 'iteration')
trigger_stop = (config.train.stop_iteration, 'iteration') if config.train.stop_iteration is not None else None

trainer = training.Trainer(updater, stop_trigger=trigger_stop, out=arguments.output)
tb_writer = SummaryWriter(Path(arguments.output))

ext = extensions.Evaluator(test_iter, model, converter, device=config.train.gpu[0])
trainer.extend(ext, name='test', trigger=trigger_log)
ext = extensions.Evaluator(train_eval_iter, model, converter, device=config.train.gpu[0])
trainer.extend(ext, name='train', trigger=trigger_log)

ext = extensions.snapshot_object(predictor, filename='main_{.updater.iteration}.npz')
trainer.extend(ext, trigger=trigger_snapshot)

trainer.extend(extensions.FailOnNonNumber(), trigger=trigger_log)
trainer.extend(extensions.observe_lr(), trigger=trigger_log)
trainer.extend(extensions.LogReport(trigger=trigger_log))
trainer.extend(TensorBoardReport(writer=tb_writer), trigger=trigger_log)

trainer.extend(extensions.dump_graph(root_name="main/loss"))

if trigger_stop is not None:
    trainer.extend(extensions.ProgressBar(trigger_stop))

trainer.run()
