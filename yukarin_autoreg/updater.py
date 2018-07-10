import chainer
import chainer.functions as F
import numpy

from yukarin_autoreg.config import LossConfig
from yukarin_autoreg.model import WaveRNN


class Updater(chainer.training.StandardUpdater):
    def __init__(
            self,
            loss_config: LossConfig,
            predictor: WaveRNN,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loss_config = loss_config
        self.predictor = predictor

    def forward(
            self,
            input_coarse: numpy.ndarray,
            input_fine: numpy.ndarray,
            target_coarse: numpy.ndarray,
            target_fine: numpy.ndarray,
    ):
        out_c_array, out_f_array, hidden = self.predictor(c_array=input_coarse, t_array=input_fine)

        nll_coarse = F.softmax_cross_entropy(out_c_array, target_coarse)
        nll_fine = F.softmax_cross_entropy(out_f_array, target_fine)
        loss = nll_coarse + nll_fine

        chainer.report(dict(
            nll_coarse=nll_coarse,
            nll_fine=nll_fine,
            loss=loss,
        ), self.predictor)
        return loss

    def update_core(self):
        optimizer = self.get_optimizer('main')
        batch = self.get_iterator('main').next()
        batch = self.converter(batch, self.device)
        optimizer.update(self.forward, **batch)
