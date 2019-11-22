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
import lreq
from skimage.transform import resize
import utils

from PIL import Image
import sys
import tqdm
import torchvision.models as models
rmodel = models.vgg16(pretrained=True)
layer = rmodel.features[9]
rmodel.eval()


lreq.use_implicit_lreq.set(True)

im_size = 1024


def place(canvas, image, x, y):
    im_size = image.shape[2]
    if len(image.shape) == 4:
        image = image[0]
    canvas[:, y: y + im_size, x: x + im_size] = image * 0.5 + 0.5


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
    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
        truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER)
    model.cuda(0)
    model.eval()
    model.requires_grad_(True)

    decoder = model.decoder
    encoder = model.encoder
    mapping_tl = model.mapping_tl
    mapping_fl = model.mapping_fl
    dlatent_avg = model.dlatent_avg

    logger.info("Trainable parameters generator:")
    count_parameters(decoder)

    logger.info("Trainable parameters discriminator:")
    count_parameters(encoder)

    arguments = dict()
    arguments["iteration"] = 0

    model_dict = {
        'discriminator_s': encoder,
        'generator_s': decoder,
        'mapping_tl_s': mapping_tl,
        'mapping_fl_s': mapping_fl,
        'dlatent_avg': dlatent_avg
    }

    checkpointer = Checkpointer(cfg,
                                model_dict,
                                {},
                                logger=logger,
                                save=False)

    extra_checkpoint_data = checkpointer.load()

    model.eval()

    layer_count = cfg.MODEL.LAYER_COUNT

    def encode(x):
        Z, _ = model.encode(x, layer_count - 1, 1)
        Z = Z.repeat(1, model.mapping_fl.num_layers, 1)
        return Z

    def decode(x):
        layer_idx = torch.arange(2 * cfg.MODEL.LAYER_COUNT)[np.newaxis, :, np.newaxis]
        ones = torch.ones(layer_idx.shape, dtype=torch.float32)
        coefs = torch.where(layer_idx < model.truncation_cutoff, ones, ones)
        # x = torch.lerp(model.dlatent_avg.buff.data, x, coefs)
        return model.decoder(x, layer_count - 1, 1, noise=True)

    def optimize(im, w):
        model.decoder.requires_grad_(True)
        im = torch.tensor(im.detach(), requires_grad=True)
        lr = 0.01
        noise = 0.5
        # z = torch.tensor(np.load('putyn.npy'), requires_grad=True)
        z = torch.tensor(w, requires_grad=True)
        z = torch.nn.parameter.Parameter(z)
        opt = torch.optim.Adam([z], lr)
        my_embedding = []

        def copy_data(m, i, o):
            my_embedding.clear()
            my_embedding.append(o)

        h = layer.register_forward_hook(copy_data)
        imd = F.avg_pool2d(im, 32, 32).detach()
        rmodel(imd)
        target = my_embedding[0].detach()

        for i in tqdm.tqdm(range(1000)):
            # res = decode(z)
            res = model.decoder(z, layer_count - 1, 1, noise=True)
            resd = F.avg_pool2d(res, 32, 32)
            loss = torch.pow(resd - imd, 2.0).view(im.shape[0], -1)

            loss = loss.mean()

            #rmodel(resd)

            loss = loss#  + torch.pow(my_embedding[0] - target, 2.0).mean()

            print(loss.item(), lr)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (i + 1) % 100 == 0:
                lr /= 1.1

            # with torch.no_grad():
            #     # grad = z.grad / (z.grad.std() + 1e-16)
            #     # z -= lr * grad
            #     zdev = torch.pow(z, 2.0)
            #     amax, imax = torch.max(zdev, 1, keepdim=True)
            #     mask = (zdev != amax) | (amax < 2)
            #     z *= mask.type(FloatTensor)
            #     z += (~mask).type(FloatTensor) * torch.randn([1, model.mapping_fl.num_layers, cfg.MODEL.LATENT_SPACE_SIZE])
            #     z.grad.zero_()
            if i % 100 == 0:
                save_image(res * 0.5 + 0.5, 'reconstruction_iter.png')
        np.save('putyn', z.cpu().detach().numpy())
        return z

    rnd = np.random.RandomState(4)
    latents = rnd.randn(1, cfg.MODEL.LATENT_SPACE_SIZE)

    path = 'realign1024x1024_'
    # path = 'realign128x128'

    paths = list(os.listdir(path))

    paths = sorted(paths)
    random.seed(21)
    random.shuffle(paths)

    def make(paths):
        src = []
        for filename in paths:
            img = np.asarray(Image.open(path + '/' + filename))
            if img.shape[2] == 4:
                img = img[:, :, :3]
            im = img.transpose((2, 0, 1))
            x = torch.tensor(np.asarray(im, dtype=np.float32), requires_grad=True).cuda() / 127.5 - 1.
            if x.shape[0] == 4:
                x = x[:3]
            src.append(x)

        reconstructions = []
        for s in src:
            with torch.no_grad():
                latents = encode(s[None, ...])
            w = optimize(s[None, ...], latents)
            with torch.no_grad():
                f = decode(w)
            reconstructions.append(f)
        return src, reconstructions

    src0, rec0 = make(paths[:1])

    save_image(rec0[0] * 0.5 + 0.5, 'reconstruction_iter.png')


if __name__ == "__main__":
    gpu_count = 1
    run(sample, get_cfg_defaults(), description='StyleGAN', default_config='configs/experiment_ffhq_z.yaml',
        world_size=gpu_count, write_log=False)
