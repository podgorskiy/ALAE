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
from dataloader import *
from tqdm import tqdm
from dlutils import batch_provider
from dlutils.pytorch.cuda_helper import *
from dlutils.pytorch import count_parameters
import dlutils.pytorch.count_parameters as count_param_override
from tracker import LossTracker
import math
from model import Model
from launcher import run
from defaults import get_cfg_defaults
import lod_driver
from PIL import Image
from net import *
from dlutils import batch_provider


class EmbMapping(nn.Module):
    def __init__(self, mapping_layers=5, latent_size=256, dlatent_size=256, mapping_fmaps=256):
        super(EmbMapping, self).__init__()
        inputs = latent_size
        self.mapping_layers = mapping_layers
        for i in range(mapping_layers):
            outputs = dlatent_size if i == mapping_layers - 1 else mapping_fmaps
            block = MappingBlock(inputs, outputs, lrmul=0.1)
            inputs = outputs
            setattr(self, "block_%d" % (i + 1), block)

    def forward(self, x):
        for i in range(self.mapping_layers):
            x = getattr(self, "block_%d" % (i + 1))(x)
        x = F.normalize(x, p=2, dim=1)
        return x


def train(cfg, logger, local_rank, world_size, distributed):
    torch.cuda.set_device(local_rank)
    model = EmbMapping(mapping_layers=8, latent_size=512, dlatent_size=512, mapping_fmaps=1024)

    model.cuda(local_rank)
    model.train()

    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            bucket_cap_mb=25,
            find_unused_parameters=True)
        model.device_ids = None

    optimizer = LREQAdam([
        {'params': model.parameters()},
    ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)

    model_dict = {
        'model': model,
    }
    tracker = LossTracker(cfg.OUTPUT_DIR)

    info = {'epoch': 0}

    checkpointer = Checkpointer(cfg,
                                model_dict,
                                {
                                    'optimizer': optimizer,
                                    'tracker': tracker,
                                },
                                logger=logger,
                                save=local_rank == 0)

    extra_checkpoint_data = checkpointer.load()

    dlats = np.load("dlat.npy")
    embeddings = np.load("embeddings.npy")

    data = []

    for i in range(dlats.shape[0]):
        data.append((dlats[i], embeddings[i]))

    for epoch in range(info['epoch'], cfg.TRAIN.TRAIN_EPOCHS):
        info['epoch'] = epoch
        model.train()

        model.train()

        epoch_start_time = time.time()

        i = 0
        batches = batch_provider(data, 128)

        for d in batches:
            x = [x[0] for x in d]
            y = [x[1] for x in d]
            x = torch.tensor(x)  # [:, 0]
            y = torch.tensor(y)

            i += 1
            optimizer.zero_grad()
            pred = model(x)
            loss = torch.mean((pred - y.detach())**2)
            tracker.update(dict(loss=loss))
            loss.backward()
            optimizer.step()

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time

        print(str(tracker))
        tracker.register_means(epoch)
        tracker.plot()

    logger.info("Training finish!... save training results")
    if local_rank == 0:
        checkpointer.save("emb_model_final").wait()


if __name__ == "__main__":
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpu_count = torch.cuda.device_count()
    run(train, get_cfg_defaults(), description='StyleGAN', default_config='configs/experiment_embedding.yaml',
        world_size=gpu_count)
