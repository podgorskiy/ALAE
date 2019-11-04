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

from __future__ import print_function
import torch.utils.data
from scipy import misc
from torch import optim
from torchvision.utils import save_image
from net import *
import numpy as np
import pickle
import time
import random
import os
from model import Model
from net import *
from checkpointer import Checkpointer
from scheduler import ComboMultiStepLR
from model_z_gan import Model
from launcher import run
from defaults import get_cfg_defaults
import lod_driver


from checkpointer import Checkpointer
from scheduler import ComboMultiStepLR

from dlutils import batch_provider
from dlutils.pytorch.cuda_helper import *
from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
import argparse
import logging
import sys
import bimpy
from dataloader import *
import lreq
from PIL import Image


im_size = 128


def save_sample(model, sample, i):
    os.makedirs('results', exist_ok=True)

    with torch.no_grad():
        model.eval()
        x_rec = model.generate(model.generator.layer_count - 1, 1, z=sample)

        def save_pic(x_rec):
            resultsample = x_rec * 0.5 + 0.5
            resultsample = resultsample.cpu()
            save_image(resultsample,
                       'sample_%i_lr.png' % i, nrow=16)

        save_pic(x_rec)


def sample(cfg, logger):
    torch.cuda.set_device(0)

    dataset = TFRecordsDataset(cfg, logger, rank=0, world_size=1, buffer_size_mb=1, channels=cfg.MODEL.CHANNELS, seed=1)

    batch_size = 256

    dataset.reset(7, batch_size)

    ctx = bimpy.Context()

    ctx.init(1800, 1600, "Styles")

    batches = iter(dataset)
    batch = next(batches)[0]

    selected = [bimpy.Bool(False) for _ in range(batch_size)]
    images_src = [im.transpose(1, 2, 0) for im in batch]
    images = [bimpy.Image(im) for im in images_src]

    b = 0

    while not ctx.should_close():
        with ctx:
            for i in range(batch_size):
                bimpy.push_id_int(i)
                bimpy.image(images[i])
                bimpy.same_line()
                if bimpy.selectable("", selected[i], 0, bimpy.Vec2(16, 128)):
                    im = Image.fromarray(images_src[i])
                    im.save('selected/%d_%d.png' % (b, i))

                if not i % 8 == 7:
                    bimpy.same_line()
                bimpy.pop_id()

            if bimpy.button('NEXT'):
                batch = next(batches)[0]
                selected = [bimpy.Bool(False) for _ in range(batch_size)]
                images_src = [im.transpose(1, 2, 0) for im in batch]
                images = [bimpy.Image(im) for im in images_src]
                b += 1
    exit()


if __name__ == "__main__":
    gpu_count = 1
    run(sample, get_cfg_defaults(), description='StyleGAN', default_config='configs/experiment_z.yaml',
        world_size=gpu_count, write_log=False)
