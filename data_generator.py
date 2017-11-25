# Modified from https://github.com/igul222/improved_wgan_training/blob/master/gan_toy.py

import numpy as np
import sklearn.datasets
import random

# This module referes 'Python Generator', not 'Generative Model'.


class GeneratorGaussians8(object):
    def __init__(self,
                 batch_size: int=256,
                 scale: float=2.,
                 center_coor_min: float=-1.,
                 center_coor_max: float=1.,
                 stdev: float=1.414):
        self.batch_size = batch_size
        self.stdev = stdev
        scale = scale
        diag_len = np.sqrt(center_coor_min**2 + center_coor_max**2)
        centers = [
            (center_coor_max, 0.),
            (center_coor_min, 0.),
            (0., center_coor_max),
            (0., center_coor_min),
            (center_coor_max / diag_len, center_coor_max / diag_len),
            (center_coor_max / diag_len, center_coor_min / diag_len),
            (center_coor_min / diag_len, center_coor_max / diag_len),
            (center_coor_min / diag_len, center_coor_min / diag_len)
        ]
        self.centers = [(scale * x, scale * y) for x, y in centers]

    def __iter__(self):
        while True:
            dataset = []
            for i in range(self.batch_size):
                point = np.random.randn(2) * .02
                center = random.choice(self.centers)
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            dataset /= self.stdev
            yield dataset


class GeneratorGaussians25(object):
    def __init__(self,
                 batch_size: int=256,
                 n_init_loop: int=4000,
                 x_iter_range_min: int=-2,
                 x_iter_range_max: int=2,
                 y_iter_range_min: int=-2,
                 y_iter_range_max: int=2,
                 noise_const: float = 0.05,
                 stdev: float=2.828):
        self.batch_size = batch_size
        self.dataset = []
        for i in range(n_init_loop):
            for x in range(x_iter_range_min, x_iter_range_max+1):
                for y in range(y_iter_range_min, y_iter_range_max+1):
                    point = np.random.randn(2) * noise_const
                    point[0] += 2 * x
                    point[1] += 2 * y
                    self.dataset.append(point)
        self.dataset = np.array(self.dataset, dtype='float32')
        np.random.shuffle(self.dataset)
        self.dataset /= stdev

    def __iter__(self):
        while True:
            for i in range(int(len(self.dataset) / self.batch_size)):
                yield self.dataset[i * self.batch_size:(i + 1)*self.batch_size]


class GeneratorSwissRoll(object):
    def __init__(self,
                 batch_size: int=256,
                 noise_stdev: float=0.25,
                 stdev: float=7.5):
        self.batch_size = batch_size
        self.noise_stdev = noise_stdev
        self.stdev = stdev

    def __iter__(self):
        while True:
            data = sklearn.datasets.make_swiss_roll(
                n_samples=self.batch_size,
                noise=self.noise_stdev
            )[0]
            data = data.astype('float32')[:, [0, 2]]
            data /= self.stdev  # stdev plus a little
            yield data
