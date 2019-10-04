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
from torchvision.utils import save_image, make_grid
from net import *
import numpy as np
import pickle
import time
import random
import os
from dlutils import batch_provider
from dlutils.pytorch.cuda_helper import *
from dlutils.pytorch import count_parameters
import bimpy
import math

im_size = 128


def process_batch(batch):
    data = [x.transpose((2, 0, 1)) for x in batch]

    x = torch.from_numpy(np.asarray(data, dtype=np.float32)).cuda() / 127.5 - 1.
    x = x.view(-1, 3, im_size, im_size)
    return x


def place(canvas, image, x, y):
    image = image.cpu().detach().numpy()
    im_size = image.shape[1]
    canvas[:, y * im_size: (y + 1) * im_size, x * im_size: (x + 1) * im_size] = image * 0.5 + 0.5


def main(model_filename_enc, model_filename_dec):
    layer_count = 6
    latent_size = 256
    encoder = Encoder(layer_count=layer_count, startf=64, maxf=256, latent_size=latent_size, channels=3)
    decoder = Decoder(layer_count=layer_count, startf=64, maxf=256, latent_size=latent_size, channels=3)
    encoder.cuda()
    decoder.cuda()
    try:
        encoder.load_state_dict(torch.load(model_filename_enc))
        decoder.load_state_dict(torch.load(model_filename_dec))
    except RuntimeError:
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)
        encoder.load_state_dict(torch.load(model_filename_enc))
        decoder.load_state_dict(torch.load(model_filename_dec))
        encoder = encoder.module
        decoder = decoder.module
    encoder.eval()
    decoder.eval()

    ids = list(range(1))

    print("Trainable parameters:")
    count_parameters(encoder)
    count_parameters(decoder)

    with open('data_selected_old.pkl', 'rb') as pkl:
        data_train = pickle.load(pkl)

    im_size = 128
    im_count = 1

    xs_ = process_batch([data_train[x] for x in ids])

    xs_ = make_grid(xs_, nrow=2)
    xs__ = xs_

    def rollx(x, n):
        return torch.cat((x[:,:, -n:], x[:,:, :-n]), dim=2)

    rx = bimpy.Int()
    rx.value = 0
    ry = bimpy.Int()
    ry.value = 0

    def update_image():
        def rolly(x, n):
            return torch.cat((x[:, -n:], x[:, :-n]), dim=1)

        xs_ = rolly(xs__, ry.value)
        xs_ = rollx(xs_, rx.value)

        xs = xs_.view(1, xs_.shape[0], xs_.shape[1], xs_.shape[2])

        styless = encoder.encode(xs, layer_count - 1)
        w = encoder.style_encode(styless, layer_count - 1)

        mu, logvar = torch.split(w, [decoder.latent_size, decoder.latent_size], dim=1)

        styless = decoder.style_decode(mu, layer_count - 1)
        recs = decoder.decode(styless, layer_count - 1, True)

        im_width = math.ceil(pow(im_count, 0.5) + 0.5)
        im_height = math.ceil(im_count / im_width)
        canvas2 = np.zeros([3, im_size * (im_height), im_size * (im_width)])

        for i in range(im_height):
            for j in range(im_width):
                if i * im_width + j < len(xs):
                    place(canvas2, recs[i * im_width + j], j, i)

        canvas2 = np.clip(canvas2.transpose([1, 2, 0]) * 255, 0, 255).astype(np.uint8)
        return canvas2

    ctx = bimpy.Context()

    ctx.init(1800, 1600, "Styles")

    im = bimpy.Image(update_image())
    im_source = bimpy.Image(np.clip((xs_ * 0.5 + 0.5).cpu().detach().numpy().transpose([1, 2, 0]) * 255, 0, 255).astype(np.uint8))

    str = bimpy.String()

    while(not ctx.should_close()):
        with ctx:
            if bimpy.slider_int('rollx', rx, -30, 30):
                im = bimpy.Image(update_image())
            if bimpy.slider_int('rolly', ry, -30, 30):
                im = bimpy.Image(update_image())



            bimpy.image(im_source)
            bimpy.image(im)


            bimpy.set_window_font_scale(3.0)
            bimpy.input_text("", str, 255)
            bimpy.set_window_font_scale(2.0)


if __name__ == '__main__':
    main("encoder.pkl", "decoder.pkl")
