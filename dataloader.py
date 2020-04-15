# Copyright 2019-2020 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import dareblopy as db
import random

import numpy as np
import torch
import torch.tensor
import torch.utils
import torch.utils.data
import time
import math

cpu = torch.device('cpu')


class TFRecordsDataset:
    def __init__(self, cfg, logger, rank=0, world_size=1, buffer_size_mb=200, channels=3, seed=None, train=True, needs_labels=False):
        self.cfg = cfg
        self.logger = logger
        self.rank = rank
        self.last_data = ""
        if train:
            self.part_count = cfg.DATASET.PART_COUNT
            self.part_size = cfg.DATASET.SIZE // self.part_count
        else:
            self.part_count = cfg.DATASET.PART_COUNT_TEST
            self.part_size = cfg.DATASET.SIZE_TEST // self.part_count
        self.workers = []
        self.workers_active = 0
        self.iterator = None
        self.filenames = {}
        self.batch_size = 512
        self.features = {}
        self.channels = channels
        self.seed = seed
        self.train = train
        self.needs_labels = needs_labels

        assert self.part_count % world_size == 0

        self.part_count_local = self.part_count // world_size

        if train:
            path = cfg.DATASET.PATH
        else:
            path = cfg.DATASET.PATH_TEST

        for r in range(2, cfg.DATASET.MAX_RESOLUTION_LEVEL + 1):
            files = []
            for i in range(self.part_count_local * rank, self.part_count_local * (rank + 1)):
                file = path % (r, i)
                files.append(file)
            self.filenames[r] = files

        self.buffer_size_b = 1024 ** 2 * buffer_size_mb

        self.current_filenames = []

    def reset(self, lod, batch_size):
        assert lod in self.filenames.keys()
        self.current_filenames = self.filenames[lod]
        self.batch_size = batch_size

        img_size = 2 ** lod

        if self.needs_labels:
            self.features = {
                # 'shape': db.FixedLenFeature([3], db.int64),
                'data': db.FixedLenFeature([self.channels, img_size, img_size], db.uint8),
                'label': db.FixedLenFeature([], db.int64)
            }
        else:
            self.features = {
                # 'shape': db.FixedLenFeature([3], db.int64),
                'data': db.FixedLenFeature([self.channels, img_size, img_size], db.uint8)
            }

        buffer_size = self.buffer_size_b // (self.channels * img_size * img_size)

        if self.seed is None:
            seed = np.uint64(time.time() * 1000)
        else:
            seed = self.seed
            self.logger.info('!' * 80)
            self.logger.info('! Seed is used for to shuffle data in TFRecordsDataset!')
            self.logger.info('!' * 80)

        self.iterator = db.ParsedTFRecordsDatasetIterator(self.current_filenames, self.features, self.batch_size, buffer_size, seed=seed)

    def __iter__(self):
        return self.iterator

    def __len__(self):
        return self.part_count_local * self.part_size


def make_dataloader(cfg, logger, dataset, GPU_batch_size, local_rank, numpy=False):
    class BatchCollator(object):
        def __init__(self, device=torch.device("cpu")):
            self.device = device
            self.flip = cfg.DATASET.FLIP_IMAGES
            self.numpy = numpy

        def __call__(self, batch):
            with torch.no_grad():
                x, = batch
                if self.flip:
                    flips = [(slice(None, None, None), slice(None, None, None), slice(None, None, random.choice([-1, None]))) for _ in range(x.shape[0])]
                    x = np.array([img[flip] for img, flip in zip(x, flips)])
                if self.numpy:
                    return x
                x = torch.tensor(x, requires_grad=True, device=torch.device(self.device), dtype=torch.float32)
                return x

    batches = db.data_loader(iter(dataset), BatchCollator(local_rank), len(dataset) // GPU_batch_size)

    return batches


def make_dataloader_y(cfg, logger, dataset, GPU_batch_size, local_rank):
    class BatchCollator(object):
        def __init__(self, device=torch.device("cpu")):
            self.device = device
            self.flip = cfg.DATASET.FLIP_IMAGES

        def __call__(self, batch):
            with torch.no_grad():
                x, y = batch
                if self.flip:
                    flips = [(slice(None, None, None), slice(None, None, None), slice(None, None, random.choice([-1, None]))) for _ in range(x.shape[0])]
                    x = np.array([img[flip] for img, flip in zip(x, flips)])
                x = torch.tensor(x, requires_grad=True, device=torch.device(self.device), dtype=torch.float32)
                return x, y

    batches = db.data_loader(iter(dataset), BatchCollator(local_rank), len(dataset) // GPU_batch_size)

    return batches


class TFRecordsDatasetImageNet:
    def __init__(self, cfg, logger, rank=0, world_size=1, buffer_size_mb=200, channels=3, seed=None, train=True, needs_labels=False):
        self.cfg = cfg
        self.logger = logger
        self.rank = rank
        self.last_data = ""
        self.part_count = cfg.DATASET.PART_COUNT
        if train:
            self.part_size = cfg.DATASET.SIZE // cfg.DATASET.PART_COUNT
        else:
            self.part_size = cfg.DATASET.SIZE_TEST // cfg.DATASET.PART_COUNT
        self.workers = []
        self.workers_active = 0
        self.iterator = None
        self.filenames = {}
        self.batch_size = 512
        self.features = {}
        self.channels = channels
        self.seed = seed
        self.train = train
        self.needs_labels = needs_labels

        assert self.part_count % world_size == 0

        self.part_count_local = cfg.DATASET.PART_COUNT // world_size

        if train:
            path = cfg.DATASET.PATH
        else:
            path = cfg.DATASET.PATH_TEST

        for r in range(2, cfg.DATASET.MAX_RESOLUTION_LEVEL + 1):
            files = []
            for i in range(self.part_count_local * rank, self.part_count_local * (rank + 1)):
                file = path % (r, i)
                files.append(file)
            self.filenames[r] = files

        self.buffer_size_b = 1024 ** 2 * buffer_size_mb

        self.current_filenames = []

    def reset(self, lod, batch_size):
        assert lod in self.filenames.keys()
        self.current_filenames = self.filenames[lod]
        self.batch_size = batch_size

        if self.train:
            img_size = 2 ** lod + 2 ** (lod - 3)
        else:
            img_size = 2 ** lod

        if self.needs_labels:
            self.features = {
                'data': db.FixedLenFeature([3, img_size, img_size], db.uint8),
                'label': db.FixedLenFeature([], db.int64)
            }
        else:
            self.features = {
                'data': db.FixedLenFeature([3, img_size, img_size], db.uint8)
            }

        buffer_size = self.buffer_size_b // (self.channels * img_size * img_size)

        if self.seed is None:
            seed = np.uint64(time.time() * 1000)
        else:
            seed = self.seed
            self.logger.info('!' * 80)
            self.logger.info('! Seed is used for to shuffle data in TFRecordsDataset!')
            self.logger.info('!' * 80)

        self.iterator = db.ParsedTFRecordsDatasetIterator(self.current_filenames, self.features, self.batch_size, buffer_size, seed=seed)

    def __iter__(self):
        return self.iterator

    def __len__(self):
        return self.part_count_local * self.part_size


def make_imagenet_dataloader(cfg, logger, dataset, GPU_batch_size, target_size, local_rank, do_random_crops=True):
    class BatchCollator(object):
        def __init__(self, device=torch.device("cpu")):
            self.device = device
            self.flip = cfg.DATASET.FLIP_IMAGES
            self.size = target_size
            p = math.log2(target_size)
            self.source_size = 2 ** p + 2 ** (p - 3)
            self.do_random_crops = do_random_crops

        def __call__(self, batch):
            with torch.no_grad():
                x, = batch

                if self.do_random_crops:
                    images = []
                    for im in x:
                        deltax = self.source_size - target_size
                        deltay = self.source_size - target_size
                        offx = np.random.randint(deltax + 1)
                        offy = np.random.randint(deltay + 1)
                        im = im[:, offy:offy + self.size, offx:offx + self.size]
                        images.append(im)
                    x = np.stack(images)

                if self.flip:
                    flips = [(slice(None, None, None), slice(None, None, None), slice(None, None, random.choice([-1, None]))) for _ in range(x.shape[0])]
                    x = np.array([img[flip] for img, flip in zip(x, flips)])
                x = torch.tensor(x, requires_grad=True, device=torch.device(self.device), dtype=torch.float32)

                return x

    batches = db.data_loader(iter(dataset), BatchCollator(local_rank), len(dataset) // GPU_batch_size)

    return batches


def make_imagenet_dataloader_y(cfg, logger, dataset, GPU_batch_size, target_size, local_rank, do_random_crops=True):
    class BatchCollator(object):
        def __init__(self, device=torch.device("cpu")):
            self.device = device
            self.flip = cfg.DATASET.FLIP_IMAGES
            self.size = target_size
            p = math.log2(target_size)
            self.source_size = 2 ** p + 2 ** (p - 3)
            self.do_random_crops = do_random_crops

        def __call__(self, batch):
            with torch.no_grad():
                x, y = batch

                if self.do_random_crops:
                    images = []
                    for im in x:
                        deltax = self.source_size - target_size
                        deltay = self.source_size - target_size
                        offx = np.random.randint(deltax + 1)
                        offy = np.random.randint(deltay + 1)
                        im = im[:, offy:offy+self.size, offx:offx+self.size]
                        images.append(im)
                    x = np.stack(images)

                if self.flip:
                    flips = [(slice(None, None, None), slice(None, None, None), slice(None, None, random.choice([-1, None]))) for _ in range(x.shape[0])]
                    x = np.array([img[flip] for img, flip in zip(x, flips)])
                x = torch.tensor(x, requires_grad=True, device=torch.device(self.device), dtype=torch.float32)
                return x, y

    batches = db.data_loader(iter(dataset), BatchCollator(local_rank), len(dataset) // GPU_batch_size)

    return batches
