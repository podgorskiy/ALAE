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

import torch
import math
import time
from collections import defaultdict


class LODDriver:
    def __init__(self, cfg, logger, world_size, dataset_size):
        if world_size == 8:
            self.lod_2_batch = cfg.TRAIN.LOD_2_BATCH_8GPU
        if world_size == 4:
            self.lod_2_batch = cfg.TRAIN.LOD_2_BATCH_4GPU
        if world_size == 2:
            self.lod_2_batch = cfg.TRAIN.LOD_2_BATCH_2GPU
        if world_size == 1:
            self.lod_2_batch = cfg.TRAIN.LOD_2_BATCH_1GPU

        self.world_size = world_size
        self.minibatch_base = 16
        self.cfg = cfg
        self.dataset_size = dataset_size
        self.current_epoch = 0
        self.lod = -1
        self.in_transition = False
        self.logger = logger
        self.iteration = 0
        self.epoch_end_time = 0
        self.epoch_start_time = 0
        self.per_epoch_ptime = 0
        self.reports = cfg.TRAIN.REPORT_FREQ
        self.snapshots = cfg.TRAIN.SNAPSHOT_FREQ
        self.tick_start_nimg_report = 0
        self.tick_start_nimg_snapshot = 0

    def get_lod_power2(self):
        return self.lod + 2

    def get_batch_size(self):
        return self.lod_2_batch[min(self.lod, len(self.lod_2_batch) - 1)]

    def get_dataset_size(self):
        return self.dataset_size

    def get_per_GPU_batch_size(self):
        return self.get_batch_size() // self.world_size

    def get_blend_factor(self):
        if self.cfg.TRAIN.EPOCHS_PER_LOD == 0:
            return 1
        blend_factor = float((self.current_epoch % self.cfg.TRAIN.EPOCHS_PER_LOD) * self.dataset_size + self.iteration)
        blend_factor /= float(self.cfg.TRAIN.EPOCHS_PER_LOD // 2 * self.dataset_size)
        blend_factor = math.sin(blend_factor * math.pi - 0.5 * math.pi) * 0.5 + 0.5

        if not self.in_transition:
            blend_factor = 1

        return blend_factor

    def is_time_to_report(self):
        if self.iteration >= self.tick_start_nimg_report + self.reports[min(self.lod, len(self.reports) - 1)] * 1000:
            self.tick_start_nimg_report = self.iteration
            return True
        return False

    def is_time_to_save(self):
        if self.iteration >= self.tick_start_nimg_snapshot + self.snapshots[min(self.lod, len(self.snapshots) - 1)] * 1000:
            self.tick_start_nimg_snapshot = self.iteration
            return True
        return False

    def step(self):
        self.iteration += self.get_batch_size()
        self.epoch_end_time = time.time()
        self.per_epoch_ptime = self.epoch_end_time - self.epoch_start_time

    def set_epoch(self, epoch, optimizers):
        self.current_epoch = epoch
        self.iteration = 0
        self.tick_start_nimg_report = 0
        self.tick_start_nimg_snapshot = 0
        self.epoch_start_time = time.time()

        if self.cfg.TRAIN.EPOCHS_PER_LOD == 0:
            self.lod = self.cfg.MODEL.LAYER_COUNT - 1
            return

        new_lod = min(self.cfg.MODEL.LAYER_COUNT - 1, epoch // self.cfg.TRAIN.EPOCHS_PER_LOD)
        if new_lod != self.lod:
            self.lod = new_lod
            self.logger.info("#" * 80)
            self.logger.info("# Switching LOD to %d" % self.lod)
            self.logger.info("# Starting transition")
            self.logger.info("#" * 80)
            self.in_transition = True
            for opt in optimizers:
                opt.state = defaultdict(dict)

        is_in_first_half_of_cycle = (epoch % self.cfg.TRAIN.EPOCHS_PER_LOD) < (self.cfg.TRAIN.EPOCHS_PER_LOD // 2)
        is_growing = epoch // self.cfg.TRAIN.EPOCHS_PER_LOD == self.lod > 0
        new_in_transition = is_in_first_half_of_cycle and is_growing

        if new_in_transition != self.in_transition:
            self.in_transition = new_in_transition
            self.logger.info("#" * 80)
            self.logger.info("# Transition ended")
            self.logger.info("#" * 80)
