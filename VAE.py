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


def save_model(x, name):
    if isinstance(x, nn.DataParallel):
        torch.save(x.module.state_dict(), name)
    else:
        torch.save(x.state_dict(), name)


def loss_function(recon_x, x):#, mu, logvar):
    BCE = torch.mean((recon_x - x)**2)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))
    return BCE#, KLD * 0.1


def process_batch(batch):
    data = [x.transpose((2, 0, 1)) for x in batch]

    x = torch.from_numpy(np.asarray(data, dtype=np.float32)).cuda() / 127.5 - 1.
    return x

lod_2_batch = [512, 256, 128, 64, 32]


def main(parallel=False):
    z_size = 512
    layer_count = 5
    epochs_per_lod = 8
    vae = VAE(zsize=z_size, layer_count=layer_count, maxf=128)
    vae.cuda()
    vae.train()
    vae.weight_init(mean=0, std=0.02)

    print("Trainable parameters:")
    count_parameters(vae)

    if parallel:
        vae = nn.DataParallel(vae)
        vae.layer_to_resolution = vae.module.layer_to_resolution

    lr = 0.0005

    vae_optimizer = optim.Adam(vae.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0)
 
    train_epoch = 45

    sample1 = torch.randn(128, z_size).view(-1, z_size, 1, 1)

    lod = 0
    in_transition = False

    for epoch in range(train_epoch):
        vae.train()

        new_lod = min(layer_count - 1, epoch // epochs_per_lod)
        if new_lod != lod:
            lod = new_lod
            print("#" * 80, "\n# Switching LOD to %d" % lod, "\n" + "#" * 80)
            print("Start transition")
            in_transition = True

        new_in_transition = (epoch % epochs_per_lod) < (epochs_per_lod // 2) and lod > 0 and epoch // epochs_per_lod == lod
        if new_in_transition != in_transition:
            in_transition = new_in_transition
            print("#" * 80, "\n# Transition ended", "\n" + "#" * 80)

        if lod == layer_count - 1:
            with open('../VAE/data_fold_%d.pkl' % (epoch % 5), 'rb') as pkl:
                data_train = pickle.load(pkl)
        else:
            with open('../VAE/data_fold_%d_lod_%d.pkl' % (epoch % 5, lod), 'rb') as pkl:
                data_train = pickle.load(pkl)

        print("Train set size:", len(data_train))

        random.shuffle(data_train)

        batches = batch_provider(data_train, lod_2_batch[lod], process_batch, report_progress=True)

        rec_loss = 0
        kl_loss = 0

        epoch_start_time = time.time()

        if (epoch + 1) % 20 == 0:
            vae_optimizer.param_groups[0]['lr'] /= 4
            print("learning rate change!")

        i = 0
        for x_orig in batches:
            vae.train()
            vae.zero_grad()

            blend_factor = float((epoch % epochs_per_lod) * len(data_train) + i * lod_2_batch[lod]) / float(epochs_per_lod // 2 * len(data_train))

            if not in_transition:
                blend_factor = 1


            #rec, mu, logvar = vae(x)

            needed_resolution = vae.layer_to_resolution[lod]
            x = resize2d(x_orig, needed_resolution)

            x_prev = None

            if in_transition:
                needed_resolution_prev = vae.layer_to_resolution[lod - 1]
                x_prev = resize2d(x_orig, needed_resolution_prev)
                x_prev_2x = F.interpolate(x_prev, size=needed_resolution)
                x = x * blend_factor + x_prev_2x * (1.0 - blend_factor)

            rec = vae(x, x_prev, lod, blend_factor)

            loss_re = loss_function(rec, x)#, mu, logvar)
            (loss_re).backward()
            vae_optimizer.step()
            rec_loss += loss_re.item()
            #kl_loss += loss_kl.item()

            #############################################

            os.makedirs('results_rec', exist_ok=True)
            os.makedirs('results_gen', exist_ok=True)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time

            # report losses and save samples each 60 iterations
            m = 60
            i += 1
            if i % m == 0:
                rec_loss /= m
                kl_loss /= m
                print(blend_factor)
                print('\n[%d/%d] - ptime: %.2f, rec loss: %.9f, KL loss: %.9f' % (
                    (epoch + 1), train_epoch, per_epoch_ptime, rec_loss, kl_loss))
                rec_loss = 0
                kl_loss = 0
                with torch.no_grad():
                    vae.eval()
                    x_rec = vae(x, x_prev, lod, blend_factor)
                    resultsample = torch.cat([x, x_rec]) * 0.5 + 0.5
                    resultsample = resultsample.cpu()
                    save_image(resultsample.view(-1, 3, needed_resolution, needed_resolution),
                               'results_rec/sample_' + str(epoch) + "_" + str(i) + '.png')
                    #x_rec = vae.decode(sample1)
                    #resultsample = x_rec * 0.5 + 0.5
                    #resultsample = resultsample.cpu()
                    #save_image(resultsample.view(-1, 3, im_size, im_size),
                    #           'results_gen/sample_' + str(epoch) + "_" + str(i) + '.png')

        del batches
        del data_train
        save_model(vae, "VAEmodel_tmp.pkl")
    print("Training finish!... save training results")
    save_model(vae, "VAEmodel.pkl")

if __name__ == '__main__':
    main(True)
