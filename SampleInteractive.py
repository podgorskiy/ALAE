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
from dlutils import batch_provider
from dlutils.pytorch.cuda_helper import *
from dlutils.pytorch import count_parameters
import bimpy

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

    ids = [18, 2, 10, 1, 13, 4]
    idsd = [3, 4, 14, 21, 20, 19]

    print("Trainable parameters:")
    count_parameters(encoder)
    count_parameters(decoder)

    with open('data_selected.pkl', 'rb') as pkl:
        data_train = pickle.load(pkl)

    im_size = 128
    im_count = 5

    # ids = [i for i in range(len(data_train))]
    im_count = max(len(ids), len(idsd))

    xs = process_batch([data_train[x] for x in ids])
    xd = process_batch([data_train[x] for x in idsd])
    # x = process_batch(data_train[im_count * 2:im_count * 3])

    styless = encoder.encode(xs, layer_count - 1)
    w = encoder.style_encode(styless, layer_count - 1)

    mu, logvar = torch.split(w, [decoder.latent_size, decoder.latent_size], dim=1)

    styless = decoder.style_decode(mu, layer_count - 1)

    recs = decoder.decode(styless, layer_count - 1, True)
    stylesd = encoder.encode(xd, layer_count - 1)
    w = encoder.style_encode(stylesd, layer_count - 1)

    mu, logvar = torch.split(w, [decoder.latent_size, decoder.latent_size], dim=1)

    stylesd = decoder.style_decode(mu, layer_count - 1)

    recd = decoder.decode(stylesd, layer_count - 1, True)

    layer_start = bimpy.Int()
    layer_end = bimpy.Int()
    layer_start.value = 0
    layer_end.value = 4

    def update_image():
        canvas = np.zeros([3, im_size * (im_count + 2), im_size * (im_count + 2)])

        cut_layer_b = layer_start.value
        cut_layer_e = layer_end.value

        for i in range(im_count):
            place(canvas, xs[i], 0, 2 + i)
            place(canvas, recs[i], 1, 2 + i)

            place(canvas, xd[i], 2 + i, 0)
            place(canvas, recd[i], 2 + i, 1)

        for i in range(im_count):
            for j in range(im_count):
                style_a = [x[i].unsqueeze(0) for x in stylesd[:cut_layer_b]]
                style_b = [x[j].unsqueeze(0) for x in styless[cut_layer_b:cut_layer_e]]
                style_c = [x[i].unsqueeze(0) for x in stylesd[cut_layer_e:]]
                style = style_a + style_b + style_c

                rec = decoder.decode(style, layer_count - 1, True)
                place(canvas, rec[0], 2 + i, 2 + j)
        return np.clip(canvas.transpose([1, 2, 0]) * 255, 0, 255).astype(np.uint8)

    ctx = bimpy.Context()

    ctx.init(1800, 1600, "Styles")

    im = bimpy.Image(update_image())

    str = bimpy.String()

    while(not ctx.should_close()):
        with ctx:
            if bimpy.slider_int('Crop, start layer', layer_start, 0, len(styless)):
                im = bimpy.Image(update_image())
            if bimpy.slider_int('Crop, end layer', layer_end, 0, len(styless)):
                im = bimpy.Image(update_image())

            bimpy.image(im)

            bimpy.set_window_font_scale(3.0)
            bimpy.input_text("", str, 255)
            bimpy.set_window_font_scale(2.0)


if __name__ == '__main__':
    main("encoder.pkl", "decoder.pkl")

