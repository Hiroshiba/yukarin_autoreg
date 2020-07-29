from pathlib import Path

import chainer
import cupy
from tensorboardX import SummaryWriter


@cupy.fuse
def interp(p, x, y):
    return p * x + (1 - p) * y


class TensorBoardReport(chainer.training.Extension):
    def __init__(self, writer=None):
        self.writer = writer

    def __call__(self, trainer: chainer.training.Trainer):
        if self.writer is None:
            self.writer = SummaryWriter(Path(trainer.out))

        observations = trainer.observation
        n_iter = trainer.updater.iteration
        for n, v in observations.items():
            if isinstance(v, chainer.Variable):
                v = v.data
            if isinstance(v, chainer.cuda.cupy.ndarray):
                v = chainer.cuda.to_cpu(v)

            self.writer.add_scalar(n, v, n_iter)

    def finalize(self):
        super().finalize()
        self.writer.flush()


class ExponentialMovingAverage(chainer.training.Extension):

    priority = chainer.training.PRIORITY_WRITER + 50

    def __init__(self, target: chainer.Chain, ema_target: chainer.Chain, decay: float):
        self.target = target
        self.ema_target = ema_target
        self.decay = decay

    def __call__(self, trainer):
        for t, ema_t in zip(self.target.params(), self.ema_target.params()):
            ema_t.array = interp(self.decay, ema_t.array, t.array)
