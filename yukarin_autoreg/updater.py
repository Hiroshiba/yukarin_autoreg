import chainer
import chainer.functions as F
import numpy as np

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
            input_coarse: np.ndarray,
            input_fine: np.ndarray,
            target_coarse: np.ndarray,
            target_fine: np.ndarray,
            local: np.ndarray,
            silence: np.ndarray,
    ):
        out_c_array, out_f_array, _, _ = self.predictor(
            c_array=input_coarse,
            f_array=input_fine,
            l_array=local,
        )

        nll_coarse = F.softmax_cross_entropy(out_c_array, target_coarse, reduce='no')
        nll_fine = F.softmax_cross_entropy(out_f_array, target_fine, reduce='no')

        nll_coarse = F.mean(nll_coarse[~silence])
        nll_fine = F.mean(nll_fine[~silence])

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
