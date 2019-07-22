import numpy as np
from chainer.serializers import NpzDeserializer


class FixGruDimensionNpzDeserializer(NpzDeserializer):
    def __getitem__(self, key):
        key = key.strip('/')
        return FixGruDimensionNpzDeserializer(
            self.npz,
            self.path + key + '/', strict=self.strict,
            ignore_names=self.ignore_names)

    def __call__(self, key, value):
        if key in ('w0', 'w1', 'w2') and self.path[-2:] == '0/':
            print(f'{self.__class__}: auto fix {self.path + key}')
            data = self.npz[self.path + key]
            assert isinstance(value, np.ndarray)
            np.copyto(value, data[:, :value.shape[1]])
            return value
        return super().__call__(key=key, value=value)


def load_fix_gru_dimension(
        file,
        obj,
        path='',
        strict=True,
        ignore_names=None,
):
    with np.load(file, allow_pickle=True) as f_load:
        d = FixGruDimensionNpzDeserializer(
            f_load,
            path=path,
            strict=strict,
            ignore_names=ignore_names,
        )
        d.load(obj)
