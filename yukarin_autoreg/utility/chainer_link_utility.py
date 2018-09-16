from typing import Dict, List

import chainer
import chainer.functions as F


def mean_params(srcs: List[chainer.Link], dst: chainer.Link):
    params: List[Dict[str, chainer.Parameter]] = []
    for src in srcs:
        ps = {kp[0]: kp[1] for kp in src.namedparams()}
        params.append(ps)

    meaned = {
        k: F.mean(F.stack([param[k] for param in params]), axis=0)
        for k in params[0].keys()
    }

    for k, p in dst.namedparams():
        p.copydata(meaned[k])
