import unittest

import chainer

from utility.test_train_utility import LocalRandomDataset, RandomDataset, setup_support, train_support
from yukarin_autoreg.dataset import SignWaveDataset
from yukarin_autoreg.network import ModifiedWaveRNN

sampling_rate = 8000
sampling_length = 880

gpu = 1
bit_size = 16
batch_size = 16
hidden_size = 896
iteration = 300

if gpu is not None:
    chainer.cuda.get_device_from_id(gpu).use()


def _create_network(
        local_size: int,
):
    return ModifiedWaveRNN(
        hidden_size=hidden_size,
        bit_size=bit_size,
        local_size=local_size,
    )


class TestTrainingModifiedWaveRNN(unittest.TestCase):
    def setUp(self):
        wave_rnn = _create_network(local_size=0)
        dataset = SignWaveDataset(sampling_rate=sampling_rate, sampling_length=sampling_length)

        self.updater, self.reporter = setup_support(batch_size, gpu, wave_rnn, dataset)

    def test_train(self):
        def _first_hook(o):
            self.assertTrue(o['main/nll_coarse'].data > 4)
            self.assertTrue(o['main/nll_fine'].data > 4)

        def _last_hook(o):
            self.assertTrue(o['main/nll_coarse'].data < 1)
            self.assertTrue(o['main/nll_fine'].data < 5)

        train_support(iteration, self.reporter, self.updater, _first_hook, _last_hook)


class TestCannotTrainingModifiedWaveRNN(unittest.TestCase):
    def setUp(self):
        wave_rnn = _create_network(local_size=0)
        dataset = RandomDataset(sampling_length=sampling_length)

        self.updater, self.reporter = setup_support(batch_size, gpu, wave_rnn, dataset)

    def test_train(self):
        def _first_hook(o):
            self.assertTrue(o['main/nll_coarse'].data > 4)
            self.assertTrue(o['main/nll_fine'].data > 4)

        def _last_hook(o):
            self.assertTrue(o['main/nll_coarse'].data > 4)
            self.assertTrue(o['main/nll_fine'].data > 4)

        train_support(iteration, self.reporter, self.updater, _first_hook, _last_hook)


class TestLocalTrainingWaveRNN(unittest.TestCase):
    def setUp(self):
        wave_rnn = _create_network(local_size=2)
        dataset = LocalRandomDataset(sampling_length=sampling_length)

        self.updater, self.reporter = setup_support(batch_size, gpu, wave_rnn, dataset)

    def test_train(self):
        def _first_hook(o):
            self.assertTrue(o['main/nll_coarse'].data > 4)
            self.assertTrue(o['main/nll_fine'].data > 4)

        def _last_hook(o):
            self.assertTrue(o['main/nll_coarse'].data < 5)
            self.assertTrue(o['main/nll_fine'].data < 5)

        train_support(iteration, self.reporter, self.updater, _first_hook, _last_hook)


if __name__ == '__main__':
    unittest.main()
