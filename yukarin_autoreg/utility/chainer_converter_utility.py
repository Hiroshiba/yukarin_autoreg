from typing import Any, Dict, List

import chainer


def concat_optional(batch: List[Dict[str, Any]], *args, **kwargs):
    none_keys = [key for key, value in batch[0].items() if value is None]

    for elem in batch:
        for key in none_keys:
            elem.pop(key)

    batch = chainer.dataset.convert.concat_examples(batch, *args, **kwargs)
    for key in none_keys:
        batch[key] = None
    return batch
