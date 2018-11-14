import chainer
import chainer.functions as F
import numpy as np
from chainer import Chain

from yukarin_autoreg.config import LossConfig, ModelConfig
from yukarin_autoreg.network import WaveRNN


def create_predictor(config: ModelConfig):
    predictor = WaveRNN(
        bit_size=config.bit_size,
        hidden_size=config.hidden_size,
        local_size=config.local_size,
        upconv_scales=config.upconv_scales,
        upconv_residual=config.upconv_residual,
        upconv_channel_ksize=config.upconv_channel_ksize,
        residual_encoder_channel=config.residual_encoder_channel,
        residual_encoder_num_block=config.residual_encoder_num_block,
    )
    return predictor


class Model(Chain):
    def __init__(self, loss_config: LossConfig, predictor: WaveRNN) -> None:
        super().__init__()
        self.loss_config = loss_config
        with self.init_scope():
            self.predictor = predictor

        self.loss_clopping_count = 0

    @property
    def loss_clopping_flag(self):
        return self.loss_clopping_count > 100  # magic number

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

        nll_coarse = F.softmax_cross_entropy(out_c_array, target_coarse, reduce='no')[~silence]
        nll_fine = F.softmax_cross_entropy(out_f_array, target_fine, reduce='no')[~silence]

        loss = nll_coarse + nll_fine
        if self.loss_config.clipping is not None:
            if self.loss_clopping_flag:
                loss = F.clip(loss, 0.0, float(self.loss_config.clipping))
        loss = F.mean(loss)

        nll_coarse = F.mean(nll_coarse)
        nll_fine = F.mean(nll_fine)

        if self.loss_config.clipping is not None:
            if not self.loss_clopping_flag:
                if loss.data < self.loss_config.clipping:
                    self.loss_clopping_count += 1

        chainer.report(dict(
            nll_coarse=nll_coarse,
            nll_fine=nll_fine,
            loss=loss,
        ), self)
        return loss
