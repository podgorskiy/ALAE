from __future__ import print_function
import torch.utils.data
from scipy import misc
from torch import optim
from torchvision.utils import save_image
from net import *
from torch.autograd import Variable
import numpy as np
import pickle
import time
import random
import os
import dlutils
#dlutils.use_cuda = False
from dlutils import batch_provider
from dlutils.pytorch.cuda_helper import *
from torch.nn import functional as F
import math

im_size = 64


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    #BCE = F.binary_cross_entropy(recon_x * 0.5 + 0.5, x * 0.5 + 0.5)
    BCE = torch.mean((recon_x - x)**2)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))
    return BCE, KLD * 0.05


def process_batch(batch):
    label = [x[0] for x in batch]
    data = [misc.imresize(x[1], [im_size, im_size]).transpose((2, 0, 1)) for x in batch]
    
    y = torch.from_numpy(np.asarray(label)).type(LongTensor).cuda()
    x = torch.from_numpy(np.asarray(data, dtype=np.float32)).cuda() / 127.5 - 1.
    x = x.view(-1, 3, im_size, im_size)
    return x, y


def compose(f, b):
    i = f[:, :3]
    m = f[:, 3:4]
    return i * m + b * (1.0 - m)


def main(folding_id, total_classes, inliner_classes, folds=5):
    batch_size = 128
    z1size = 256
    z2size = 64
    data_train = []
    data_valid = []

    for i in range(folds):
        if True:#i != folding_id:
            with open('F:/DATASETS/celeba/' + 'data_fold_%d.pkl' % i, 'rb') as pkl: #'F:/DATASETS/celeba/' +
                fold = pickle.load(pkl)
            if False:#len(data_valid) == 0:
                data_valid = fold
            else:
                data_train += fold

    outlier_classes = []
    for i in range(total_classes):
        if i not in inliner_classes:
            outlier_classes.append(i)

    # keep only train classes
    #data_train = [x for x in data_train if x[0] in inliner_classes]

    print("Train set size:", len(data_train))

    GF = VAE(zsize=z1size, layer_count=4)
    GF.cuda()
    GF.train()
    GF.weight_init(mean=0, std=0.02)

    lr = 0.001

    G_optimizer = optim.Adam(GF.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
 
    train_epoch = 300

    sample1 = torch.randn(128, z1size).view(-1, z1size, 1, 1)

    for epoch in range(train_epoch):
        GF.train()

        random.shuffle(data_train)

        batches = batch_provider(data_train, batch_size, process_batch, report_progress=True)

        Rtrain_loss = 0
        KLtrain_loss = 0

        epoch_start_time = time.time()

        if (epoch + 1) % 100 == 0:
            G_optimizer.param_groups[0]['lr'] /= 4
            print("learning rate change!")

        i = 0
        for x, y in batches:
            GF.train()
            x_ = x
            GF.zero_grad()
            rec, mu, logvar = GF(x)

            loss_re, loss_kl = loss_function(rec, x, mu, logvar)
            (loss_re + loss_kl).backward()
            G_optimizer.step()
            Rtrain_loss += loss_re.item()
            KLtrain_loss += loss_kl.item()

            #############################################

            directory = 'results_rec'+str(inliner_classes[0])
            os.makedirs(directory, exist_ok=True)
            directory = 'results_gen'+str(inliner_classes[0])
            os.makedirs(directory, exist_ok=True)


            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time

            m = 40
            i += 1
            if i % m == 0:
                Rtrain_loss /= (m)
                KLtrain_loss /= (m)
                print('[%d/%d] - ptime: %.2f, Rloss: %.9f, KLloss: %.9f' % (
                (epoch + 1), train_epoch, per_epoch_ptime, Rtrain_loss, KLtrain_loss))
                Rtrain_loss = 0
                KLtrain_loss = 0
                with torch.no_grad():
                    GF.eval()
                    x_fake, _, _ = GF(x_)
                    # x_fake = compose(x_foreground, x_background)
                    resultsample = torch.cat([x_, x_fake]) * 0.5 + 0.5
                    resultsample = resultsample.cpu()
                    save_image(resultsample.view(-1, 3, im_size, im_size),
                               'results_rec' + str(inliner_classes[0]) + '/sample_' + str(epoch) + "_" + str(i) + '.png')
                    x_fake = GF.decode(sample1)
                    # x_fake = compose(x_foreground, x_background)
                    resultsample = x_fake * 0.5 + 0.5
                    resultsample = resultsample.cpu()
                    save_image(resultsample.view(-1, 3, im_size, im_size),
                               'results_gen' + str(inliner_classes[0]) + '/sample_' + str(epoch) + "_" + str(i) + '.png')
                    # save_image(x_for_pred.view(-1, 3, im_size, im_size), 'results'+str(inliner_classes[0])+'/x_for_pred_' + str(epoch) + '.png')


        # m = len(batches)
        # Rtrain_loss /= (m)
        # KLtrain_loss /= (m)
        # print('[%d/%d] - ptime: %.2f, Rloss: %.9f, KLloss: %.9f' % (
        # (epoch + 1), train_epoch, per_epoch_ptime, Rtrain_loss, KLtrain_loss))
        # Rtrain_loss = 0
        # KLtrain_loss = 0
        # with torch.no_grad():
        #     GF.eval()
        #     x_fake, _, _ = GF(x_)
        #     # x_fake = compose(x_foreground, x_background)
        #     resultsample = torch.cat([x_, x_fake]) * 0.5 + 0.5
        #     resultsample = resultsample.cpu()
        #     save_image(resultsample.view(-1, 3, im_size, im_size),
        #                'results_rec' + str(inliner_classes[0]) + '/sample_' + str(epoch) + "_" + str(i) + '.png')
        #     x_fake = GF.decode(sample1)
        #     # x_fake = compose(x_foreground, x_background)
        #     resultsample = x_fake * 0.5 + 0.5
        #     resultsample = resultsample.cpu()
        #     save_image(resultsample.view(-1, 3, im_size, im_size),
        #                'results_gen' + str(inliner_classes[0]) + '/sample_' + str(epoch) + "_" + str(i) + '.png')
        #     # save_image(x_for_pred.view(-1, 3, im_size, im_size), 'results'+str(inliner_classes[0])+'/x_for_pred_' + str(epoch) + '.png')


    print("Training finish!... save training results")
    torch.save(GF.state_dict(), "VAEmodel_%d.pkl" %(folding_id))

if __name__ == '__main__':
    main(0, 10, [0])
