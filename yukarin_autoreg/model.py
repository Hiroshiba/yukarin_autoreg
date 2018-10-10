import chainer
import chainer.functions as F
import numpy as np
from chainer import Chain

from yukarin_autoreg.config import ModelConfig
from yukarin_autoreg.network import WaveRNN


def create_predictor(config: ModelConfig):
    predictor = WaveRNN(
        bit_size=config.bit_size,
        hidden_size=config.hidden_size,
        local_size=config.local_size,
        upconv_scales=config.upconv_scales,
        upconv_residual=config.upconv_residual,
    )
    return predictor


class Model(Chain):
    def __init__(self, predictor: WaveRNN) -> None:
        super().__init__()
        with self.init_scope():
            self.predictor = predictor

    def __call__(
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
        ), self)
        return loss
