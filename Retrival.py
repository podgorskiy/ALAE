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
    try:
        image = image.cpu().detach().numpy()
    except:
        pass
    im_size = image.shape[1]
    canvas[:, y * im_size : (y + 1) * im_size, x * im_size : (x + 1) * im_size] = image * 0.5 + 0.5

    
def calc_distance(style_a, style_b):
    distance = 0
    for a, b in zip(style_a[:5], style_b[:5]):
        mu_a = a[0]
        std_a = a[1]
        mu_b = b[0]
        std_b = b[1]
        distance += (np.log(std_b / std_a) + (std_a * std_a + (mu_a - mu_b)**2) / 2 / std_b / std_b - 1./2).mean()
        distance += (np.log(std_a / std_b) + (std_b * std_b + (mu_b - mu_a)**2) / 2 / std_a / std_a - 1./2).mean()
        
    return distance

def main(model_filename):
    z_size = 512
    layer_count = 6
    latent_size = 128
    batch = 128
    vae = Autoencoder(layer_count=layer_count, startf=64, maxf=128, latent_size=latent_size, channels=3)
    vae.cuda()
    try:
        vae.load_state_dict(torch.load(model_filename))
    except RuntimeError:
        vae = nn.DataParallel(vae)
        vae.load_state_dict(torch.load(model_filename))
        vae = vae.module
    vae.eval()

    print("Trainable parameters:")
    count_parameters(vae)
    
    with open('data_selected.pkl', 'rb') as pkl:
        data_query = pickle.load(pkl)

    with open('../VAE/data_fold_4_lod_5.pkl', 'rb') as pkl:
        data_db = pickle.load(pkl)

    im_size = 128
    im_count = 8

    styles_db = []
    
    with torch.no_grad():        
        x = process_batch(data_query[im_count * 2:im_count * 3])

        styles_q = vae.encode(x, layer_count - 1)
        
        #batches = batch_provider(data_db, batch, process_batch, report_progress=True)
        #for x in batches:
        #    styles = vae.encode(x, layer_count - 1)
        #    #    styles = list(sum(styles, ()))
        #    #    styles = torch.cat(styles, dim=1)
        #    #    styles = styles.view(styles.shape[0], styles.shape[1])
        #    for i in range(x.shape[0]):
        #        style = [(x[0][i].unsqueeze(0).detach().cpu().numpy(), x[1][i].unsqueeze(0).detach().cpu().numpy()) for x in styles]
        #        styles_db.append(style)

        #with open('styles_db.pkl', 'wb') as pkl:
        #    pickle.dump(styles_db, pkl)

        with open('styles_db.pkl', 'rb') as pkl:
            styles_db = pickle.load(pkl)

        #styles_db = torch.cat(styles_db, dim=0)
        
        canvas = np.zeros([3, im_size * (im_count + 2), im_size * (im_count + 2)])
            
        for i in range(im_count):
            style = [(x[0][i].unsqueeze(0).detach().cpu().numpy(), x[1][i].unsqueeze(0).detach().cpu().numpy()) for x in styles_q]
            
            
            distances = []
            for j in range(len(styles_db)):
                dist = calc_distance(styles_db[j], style)
                distances.append(dist)
                
            idx = np.argsort(distances)
            print(len(idx))
            
            place(canvas, data_query[im_count * 2 + i].transpose((2, 0, 1))/ 127.5 - 1., 2 + i, 0)
            for j in range(im_count):
                place(canvas, data_db[idx[j]].transpose((2, 0, 1))/ 127.5 - 1., 2 + i, 2 + j)
            
        save_image(torch.Tensor(canvas), 'retrival.png')


if __name__ == '__main__':
    main("autoencoder.pkl")
