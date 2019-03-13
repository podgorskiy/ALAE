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

im_size = 128


def process_batch(batch):
    data = [misc.imresize(x, [im_size, im_size]).transpose((2, 0, 1)) for x in batch]

    x = torch.from_numpy(np.asarray(data, dtype=np.float32)).cuda() / 127.5 - 1.
    x = x.view(-1, 3, im_size, im_size)
    return x


def place(canvas, image, x, y):
    image = image.cpu().detach().numpy()
    im_size = image.shape[1]
    canvas[:, y * im_size : (y + 1) * im_size, x * im_size : (x + 1) * im_size] = image * 0.5 + 0.5



def main(model_filename):
    z_size = 512
    vae = VAE(zsize=z_size, maxf=128, layer_count=5)
    vae.cuda()
    try:
        vae.load_state_dict(torch.load(model_filename))
    except RuntimeError:
        vae = nn.DataParallel(vae)
        vae.load_state_dict(torch.load(model_filename))
        vae = vae.module
    vae.eval()

    print("Trainable parameters:")
    count_parameters(vae, verbose=True)
    
    with open('data_selected.pkl', 'rb') as pkl:
        data_train = pickle.load(pkl)

        im_size = 128
        im_count = 8

        x = process_batch(data_train[im_count * 2:im_count * 3])

        styles = vae.encode(x)
        rec = vae.decode(styles)

        canvas = np.zeros([3, im_size * (im_count + 2), im_size * (im_count + 2)])

        cut_layer_b = 0
        cut_layer_e = 8
        
        for i in range(im_count):
            place(canvas, x[i], 0, 2 + i)
            place(canvas, rec[i], 1, 2 + i)

            place(canvas, x[i], 2 + i, 0)
            place(canvas, rec[i], 2 + i, 1)

        for i in range(im_count):
            for j in range(im_count):
                style_a = [(x[0][i].unsqueeze(0), x[1][i].unsqueeze(0)) for x in styles[:cut_layer_b]]
                style_b = [(x[0][j].unsqueeze(0), x[1][j].unsqueeze(0)) for x in styles[cut_layer_b:cut_layer_e]]
                style_c = [(x[0][i].unsqueeze(0), x[1][i].unsqueeze(0)) for x in styles[cut_layer_e:]]
                style = style_a + style_b + style_c
                
                rec = vae.decode(style)
                place(canvas, rec[0], 2 + i, 2 + j)

        save_image(torch.Tensor(canvas), 'reconstruction.png')

        del data_train


if __name__ == '__main__':
    main("VAEmodel_128.pkl")
