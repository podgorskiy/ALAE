# Copyright 2019 Stanislav Pidhorskyi
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

import pickle
import dareblopy as db
from threading import Thread, Lock, Event
import random
import threading

import numpy as np
import torch
import torch.tensor
import torch.utils
import torch.utils.data
import time

from dlutils.batch_provider import batch_provider
from dlutils.shuffle import shuffle_ndarray

cpu = torch.device('cpu')


class TFRecordsDataset:
    def __init__(self, cfg, logger, rank=0, world_size=1, buffer_size_mb=200):
        self.cfg = cfg
        self.logger = logger
        self.rank = rank
        self.last_data = ""
        self.part_count = cfg.DATASET.PART_COUNT
        self.part_size = cfg.DATASET.SIZE // cfg.DATASET.PART_COUNT
        self.workers = []
        self.workers_active = 0
        self.iterator = None
        self.filenames = {}
        self.batch_size = 512
        self.features = {}

        assert self.part_count % world_size == 0

        self.part_count_local = cfg.DATASET.PART_COUNT // world_size

        for r in range(2, cfg.DATASET.MAX_RESOLUTION_LEVEL + 1):
            files = []
            for i in range(self.part_count_local * rank, self.part_count_local * (rank + 1)):
                file = cfg.DATASET.PATH % (r, i)
                files.append(file)
            self.filenames[r] = files

        self.buffer_size_b = 1024 ** 2 * buffer_size_mb

        self.current_filenames = []

    def reset(self, lod, batch_size):
        assert lod in self.filenames.keys()
        self.current_filenames = self.filenames[lod]
        self.batch_size = batch_size

        img_size = 2 ** lod

        self.features = {
            # 'shape': db.FixedLenFeature([3], db.int64),
            'data': db.FixedLenFeature([3, img_size, img_size], db.uint8)
        }
        buffer_size = self.buffer_size_b // (3 * img_size * img_size)

        self.iterator = db.ParsedTFRecordsDatasetIterator(self.current_filenames, self.features, self.batch_size, buffer_size, seed=np.uint64(time.time() * 1000))

    def __iter__(self):
        return self.iterator

    def __len__(self):
        return self.part_count_local * self.part_size


def make_dataloader(cfg, logger, dataset, GPU_batch_size, local_rank):
    class BatchCollator(object):
        def __init__(self, device=torch.device("cpu")):
            self.device = device

        def __call__(self, batch):
            with torch.no_grad():
                x, = batch
                flips = [(slice(None, None, None), slice(None, None, None), slice(None, None, random.choice([-1, None]))) for _ in range(x.shape[0])]
                x = np.array([img[flip] for img, flip in zip(x, flips)])
                x = torch.tensor(x, requires_grad=True, device=torch.device(self.device), dtype=torch.float32)
                return x

    batches = db.data_loader(iter(dataset), BatchCollator(local_rank), len(dataset) // GPU_batch_size)

    return batches
