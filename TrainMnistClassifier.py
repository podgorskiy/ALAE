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
from torchvision.utils import save_image
from net import *
import numpy as np
import pickle
import time
import random
import os
import utils
from tracker import LossTracker
from checkpointer import Checkpointer
from scheduler import ComboMultiStepLR
from custom_adam import LREQAdam
from torch.optim.adam import Adam
from dataloader import *
from tqdm import tqdm
from dlutils import batch_provider
from dlutils.pytorch.cuda_helper import *
from dlutils.pytorch import count_parameters
import dlutils.pytorch.count_parameters as count_param_override
from tracker import LossTracker
import math
from model_z_gan import Model
from launcher import run
from defaults import get_cfg_defaults
import lod_driver
from PIL import Image
from net import *


def eval(cfg, logger, encoder, do_svm=False):
    local_rank = 0
    world_size = 1
    dataset_train = TFRecordsDataset(cfg, logger, rank=local_rank, world_size=world_size, buffer_size_mb=1024, channels=cfg.MODEL.CHANNELS, train=True, needs_labels=True)
    dataset_test = TFRecordsDataset(cfg, logger, rank=local_rank, world_size=world_size, buffer_size_mb=1024, channels=cfg.MODEL.CHANNELS, train=False, needs_labels=True)

    encoder.eval()

    batch_size = cfg.TRAIN.LOD_2_BATCH_1GPU[len(cfg.TRAIN.LOD_2_BATCH_1GPU) - 1]

    dataset_train.reset(cfg.DATASET.MAX_RESOLUTION_LEVEL, batch_size)
    dataset_test.reset(cfg.DATASET.MAX_RESOLUTION_LEVEL, batch_size)

    batches_train = make_dataloader_y(cfg, logger, dataset_train, batch_size, 0)
    batches_test = make_dataloader_y(cfg, logger, dataset_test, batch_size, 0)

    # @utils.cache
    def compute_train():
        train_X = []
        train_X2 = []
        train_Y = []

        for x_orig, y in tqdm(batches_train):
            with torch.no_grad():
                x = (x_orig / 255)

                Z, E = encoder(x, cfg.DATASET.MAX_RESOLUTION_LEVEL, 1, report_feature=True)
                train_X += torch.split(Z, 1)
                train_X2 += torch.split(E, 1)
                train_Y += list(y)

        train_X2 = torch.cat(train_X2)
        train_X = torch.cat(train_X)
        return train_X, train_X2, train_Y

    # @utils.cache
    def compute_test():
        test_X = []
        test_X2 = []
        test_Y = []

        for x_orig, y in tqdm(batches_test):
            with torch.no_grad():
                x = (x_orig / 255)

                Z, E = encoder(x, cfg.DATASET.MAX_RESOLUTION_LEVEL, 1, report_feature=True)
                test_X += torch.split(Z, 1)
                test_X2 += torch.split(E, 1)
                test_Y += list(y)
        test_X = torch.cat(test_X)
        test_X2 = torch.cat(test_X2)
        return test_X, test_X2, test_Y

    train_X, train_X2, train_Y = compute_train()
    test_X, test_X2, test_Y = compute_test()

    train_Y = np.asarray(train_Y)
    test_Y = np.asarray(test_Y)



# Number of channels in the training images. For color images this is 3
nc = 1
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # state size. (ndf) x 32 x 32
            nn.Conv2d(nc, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 10, 4, 1, 0, bias=False),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        return self.main(input).view(input.shape[0], -1)


def test(model, test_loader, dataset_size):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x, target in test_loader:
            with torch.no_grad():
                x = (x / 127.5 - 1.)
            x.requires_grad = True
            output = model(x)
            target = torch.tensor(target)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= dataset_size

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, dataset_size,
        100. * correct / dataset_size))


def train(cfg, logger, local_rank, world_size, distributed):
    torch.cuda.set_device(local_rank)
    classififer = Discriminator()

    d = torch.load("classififer.pt")
    classififer.load_state_dict(d)

    count_param_override.print = lambda a: logger.info(a)

    logger.info("Trainable parameters classififer:")
    count_parameters(classififer)

    arguments = dict()
    arguments["iteration"] = 0

    optimizer = Adam([
        {'params': classififer.parameters()}
    ], lr=cfg.TRAIN.BASE_LEARNING_RATE, weight_decay=0)

    tracker = LossTracker(cfg.OUTPUT_DIR)

    dataset_train = TFRecordsDataset(cfg, logger, rank=local_rank, world_size=1, buffer_size_mb=1024, channels=cfg.MODEL.CHANNELS, train=True, needs_labels=True)
    dataset_test = TFRecordsDataset(cfg, logger, rank=local_rank, world_size=1, buffer_size_mb=1024, channels=cfg.MODEL.CHANNELS, train=False, needs_labels=True)

    for epoch in range(9, 20):
        classififer.train()

        if epoch == 10:
            for g in optimizer.param_groups:
                g['lr'] /= 10.0

        dataset_train.reset(5, 128)
        batches = make_dataloader_y(cfg, logger, dataset_train, 128, local_rank)

        epoch_start_time = time.time()

        i = 0
        with torch.autograd.profiler.profile(use_cuda=True, enabled=False) as prof:
            for x, y in tqdm(batches):
                i += 1
                with torch.no_grad():
                    x = (x / 127.5 - 1.)

                x.requires_grad = True

                optimizer.zero_grad()
                output = classififer(x)
                target = torch.tensor(y)
                loss = F.nll_loss(output, target)
                tracker.update(dict(loss_d=loss))
                loss.backward()
                optimizer.step()

                epoch_end_time = time.time()
                per_epoch_ptime = epoch_end_time - epoch_start_time

        torch.save(classififer.state_dict(), "classififer.pt")

        dataset_test.reset(5, 128)
        batches = make_dataloader_y(cfg, logger, dataset_test, 128, local_rank)
        test(classififer, batches, len(dataset_test))


if __name__ == "__main__":
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpu_count = 1
    run(train, get_cfg_defaults(), description='StyleGAN', default_config='configs/experiment_mnist.yaml',
        world_size=gpu_count)
