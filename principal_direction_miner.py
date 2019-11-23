# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Linear Separability (LS)."""
import tensorflow as tf
import torch
import dnnlib
import dnnlib.tflib
import dnnlib.tflib as tflib
import pickle
from dataloader import *
import scipy.linalg

from checkpointer import Checkpointer
from scheduler import ComboMultiStepLR

from dlutils import batch_provider
from dlutils.pytorch.cuda_helper import *
from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
from model_z_gan import Model
import argparse
import logging
import sys
import lreq
from skimage.transform import resize
from tqdm import tqdm

from launcher import run
from PIL import Image
from matplotlib import pyplot as plt
import utils
from net import *

from collections import defaultdict
import numpy as np
import sklearn.svm
import tensorflow as tf
import dnnlib.tflib as tflib

from metrics import metric_base
from training import misc

dnnlib.tflib.init_tf()
tf_config     = {'rnd.np_random_seed': 1000}



#----------------------------------------------------------------------------

classifier_urls = [
    'https://drive.google.com/uc?id=1Q5-AI6TwWhCVM7Muu4tBM7rp5nG_gmCX', # celebahq-classifier-00-male.pkl
    'https://drive.google.com/uc?id=1Q5c6HE__ReW2W8qYAXpao68V1ryuisGo', # celebahq-classifier-01-smiling.pkl
    'https://drive.google.com/uc?id=1Q7738mgWTljPOJQrZtSMLxzShEhrvVsU', # celebahq-classifier-02-attractive.pkl
    'https://drive.google.com/uc?id=1QBv2Mxe7ZLvOv1YBTLq-T4DS3HjmXV0o', # celebahq-classifier-03-wavy-hair.pkl
    'https://drive.google.com/uc?id=1QIvKTrkYpUrdA45nf7pspwAqXDwWOLhV', # celebahq-classifier-04-young.pkl
    'https://drive.google.com/uc?id=1QJPH5rW7MbIjFUdZT7vRYfyUjNYDl4_L', # celebahq-classifier-05-5-o-clock-shadow.pkl
    'https://drive.google.com/uc?id=1QPZXSYf6cptQnApWS_T83sqFMun3rULY', # celebahq-classifier-06-arched-eyebrows.pkl
    'https://drive.google.com/uc?id=1QPgoAZRqINXk_PFoQ6NwMmiJfxc5d2Pg', # celebahq-classifier-07-bags-under-eyes.pkl
    'https://drive.google.com/uc?id=1QQPQgxgI6wrMWNyxFyTLSgMVZmRr1oO7', # celebahq-classifier-08-bald.pkl
    'https://drive.google.com/uc?id=1QcSphAmV62UrCIqhMGgcIlZfoe8hfWaF', # celebahq-classifier-09-bangs.pkl
    'https://drive.google.com/uc?id=1QdWTVwljClTFrrrcZnPuPOR4mEuz7jGh', # celebahq-classifier-10-big-lips.pkl
    'https://drive.google.com/uc?id=1QgvEWEtr2mS4yj1b_Y3WKe6cLWL3LYmK', # celebahq-classifier-11-big-nose.pkl
    'https://drive.google.com/uc?id=1QidfMk9FOKgmUUIziTCeo8t-kTGwcT18', # celebahq-classifier-12-black-hair.pkl
    'https://drive.google.com/uc?id=1QthrJt-wY31GPtV8SbnZQZ0_UEdhasHO', # celebahq-classifier-13-blond-hair.pkl
    'https://drive.google.com/uc?id=1QvCAkXxdYT4sIwCzYDnCL9Nb5TDYUxGW', # celebahq-classifier-14-blurry.pkl
    'https://drive.google.com/uc?id=1QvLWuwSuWI9Ln8cpxSGHIciUsnmaw8L0', # celebahq-classifier-15-brown-hair.pkl
    'https://drive.google.com/uc?id=1QxW6THPI2fqDoiFEMaV6pWWHhKI_OoA7', # celebahq-classifier-16-bushy-eyebrows.pkl
    'https://drive.google.com/uc?id=1R71xKw8oTW2IHyqmRDChhTBkW9wq4N9v', # celebahq-classifier-17-chubby.pkl
    'https://drive.google.com/uc?id=1RDn_fiLfEGbTc7JjazRXuAxJpr-4Pl67', # celebahq-classifier-18-double-chin.pkl
    'https://drive.google.com/uc?id=1RGBuwXbaz5052bM4VFvaSJaqNvVM4_cI', # celebahq-classifier-19-eyeglasses.pkl
    'https://drive.google.com/uc?id=1RIxOiWxDpUwhB-9HzDkbkLegkd7euRU9', # celebahq-classifier-20-goatee.pkl
    'https://drive.google.com/uc?id=1RPaNiEnJODdr-fwXhUFdoSQLFFZC7rC-', # celebahq-classifier-21-gray-hair.pkl
    'https://drive.google.com/uc?id=1RQH8lPSwOI2K_9XQCZ2Ktz7xm46o80ep', # celebahq-classifier-22-heavy-makeup.pkl
    'https://drive.google.com/uc?id=1RXZM61xCzlwUZKq-X7QhxOg0D2telPow', # celebahq-classifier-23-high-cheekbones.pkl
    'https://drive.google.com/uc?id=1RgASVHW8EWMyOCiRb5fsUijFu-HfxONM', # celebahq-classifier-24-mouth-slightly-open.pkl
    'https://drive.google.com/uc?id=1RkC8JLqLosWMaRne3DARRgolhbtg_wnr', # celebahq-classifier-25-mustache.pkl
    'https://drive.google.com/uc?id=1RqtbtFT2EuwpGTqsTYJDyXdnDsFCPtLO', # celebahq-classifier-26-narrow-eyes.pkl
    'https://drive.google.com/uc?id=1Rs7hU-re8bBMeRHR-fKgMbjPh-RIbrsh', # celebahq-classifier-27-no-beard.pkl
    'https://drive.google.com/uc?id=1RynDJQWdGOAGffmkPVCrLJqy_fciPF9E', # celebahq-classifier-28-oval-face.pkl
    'https://drive.google.com/uc?id=1S0TZ_Hdv5cb06NDaCD8NqVfKy7MuXZsN', # celebahq-classifier-29-pale-skin.pkl
    'https://drive.google.com/uc?id=1S3JPhZH2B4gVZZYCWkxoRP11q09PjCkA', # celebahq-classifier-30-pointy-nose.pkl
    'https://drive.google.com/uc?id=1S3pQuUz-Jiywq_euhsfezWfGkfzLZ87W', # celebahq-classifier-31-receding-hairline.pkl
    'https://drive.google.com/uc?id=1S6nyIl_SEI3M4l748xEdTV2vymB_-lrY', # celebahq-classifier-32-rosy-cheeks.pkl
    'https://drive.google.com/uc?id=1S9P5WCi3GYIBPVYiPTWygrYIUSIKGxbU', # celebahq-classifier-33-sideburns.pkl
    'https://drive.google.com/uc?id=1SANviG-pp08n7AFpE9wrARzozPIlbfCH', # celebahq-classifier-34-straight-hair.pkl
    'https://drive.google.com/uc?id=1SArgyMl6_z7P7coAuArqUC2zbmckecEY', # celebahq-classifier-35-wearing-earrings.pkl
    'https://drive.google.com/uc?id=1SC5JjS5J-J4zXFO9Vk2ZU2DT82TZUza_', # celebahq-classifier-36-wearing-hat.pkl
    'https://drive.google.com/uc?id=1SDAQWz03HGiu0MSOKyn7gvrp3wdIGoj-', # celebahq-classifier-37-wearing-lipstick.pkl
    'https://drive.google.com/uc?id=1SEtrVK-TQUC0XeGkBE9y7L8VXfbchyKX', # celebahq-classifier-38-wearing-necklace.pkl
    'https://drive.google.com/uc?id=1SF_mJIdyGINXoV-I6IAxHB_k5dxiF6M-', # celebahq-classifier-39-wearing-necktie.pkl
]

#----------------------------------------------------------------------------

def prob_normalize(p):
    p = np.asarray(p).astype(np.float32)
    assert len(p.shape) == 2
    return p / np.sum(p)

def mutual_information(p):
    p = prob_normalize(p)
    px = np.sum(p, axis=1)
    py = np.sum(p, axis=0)
    result = 0.0
    for x in range(p.shape[0]):
        p_x = px[x]
        for y in range(p.shape[1]):
            p_xy = p[x][y]
            p_y = py[y]
            if p_xy > 0.0:
                result += p_xy * np.log2(p_xy / (p_x * p_y)) # get bits as output
    return result

def entropy(p):
    p = prob_normalize(p)
    result = 0.0
    for x in range(p.shape[0]):
        for y in range(p.shape[1]):
            p_xy = p[x][y]
            if p_xy > 0.0:
                result -= p_xy * np.log2(p_xy)
    return result

def conditional_entropy(p):
    # H(Y|X) where X corresponds to axis 0, Y to axis 1
    # i.e., How many bits of additional information are needed to where we are on axis 1 if we know where we are on axis 0?
    p = prob_normalize(p)
    y = np.sum(p, axis=0, keepdims=True) # marginalize to calculate H(Y)
    return max(0.0, entropy(y) - mutual_information(p)) # can slip just below 0 due to FP inaccuracies, clean those up.

#----------------------------------------------------------------------------

def parse_tfrecord_np(record):
    ex = tf.train.Example()
    ex.ParseFromString(record)
    shape = ex.features.feature['shape'].int64_list.value
    data = ex.features.feature['data'].bytes_list.value[0]
    dlat = ex.features.feature['dlat'].bytes_list.value[0]
    lat = ex.features.feature['lat'].bytes_list.value[0]
    return np.fromstring(data, np.uint8).reshape(shape), np.fromstring(dlat, np.float32), np.fromstring(lat, np.float32)


class LS:
    def __init__(self, cfg, percent_keep, attrib_indices, minibatch_gpu):
        self.percent_keep = percent_keep
        self.attrib_indices = attrib_indices
        self.minibatch_size = minibatch_gpu
        self.cfg = cfg

    def evaluate(self, logger, mapping, decoder, lod, attrib_idx):
        result_expr = []

        rnd = np.random.RandomState(5)

        with tf.Graph().as_default(), tf.Session() as sess:
            ds = tf.data.TFRecordDataset("generated_data.000")
            ds = ds.batch(self.minibatch_size)
            batch = ds.make_one_shot_iterator().get_next()

            classifier = misc.load_pkl(classifier_urls[attrib_idx])

            i = 0
            while True:
                try:
                    records = sess.run(batch)
                    images = []
                    dlats = []
                    lats = []
                    for r in records:
                        im, dlat, lat = parse_tfrecord_np(r)

                        # plt.imshow(im.transpose(1, 2, 0), interpolation='nearest')
                        # plt.show()

                        images.append(im)
                        dlats.append(dlat)
                        lats.append(lat)
                    images = np.stack(images)
                    dlats = np.stack(dlats)
                    lats = np.stack(lats)
                    logits = classifier.run(images, None, num_gpus=2, assume_frozen=True)
                    logits = torch.tensor(logits)
                    predictions = torch.softmax(torch.cat([logits, -logits], dim=1), dim=1)

                    result_dict = dict(latents=lats, dlatents=dlats)
                    result_dict[attrib_idx] = predictions.cpu().numpy()
                    result_expr.append(result_dict)
                    i += 1
                except tf.errors.OutOfRangeError:
                    break

        results = {key: np.concatenate([value[key] for value in result_expr], axis=0) for key in result_expr[0].keys()}

        np.save("wspace_att_%d" % attrib_idx, results)

        exit()

        conditional_entropies = defaultdict(list)
        idx = attrib_idx
        for attrib_idx in [idx]: # self.attrib_indices:
            pruned_indices = list(range(results['latents'].shape[0]))
            pruned_indices = sorted(pruned_indices, key=lambda i: -np.max(results[attrib_idx][i]))
            keep = int(results['latents'].shape[0] * self.percent_keep)
            print('Keeping: %d' % keep)
            pruned_indices = pruned_indices[:keep]

            # Fit SVM to the remaining samples.
            svm_targets = np.argmax(results[attrib_idx][pruned_indices], axis=1)
            for space in ['latents', 'dlatents']:
                svm_inputs = results[space][pruned_indices]
                try:
                    svm = sklearn.svm.LinearSVC()
                    svm.fit(svm_inputs, svm_targets)
                    svm.score(svm_inputs, svm_targets)
                    svm_outputs = svm.predict(svm_inputs)
                except:
                    svm_outputs = svm_targets  # assume perfect prediction

                # Calculate conditional entropy.
                p = [[np.mean([case == (row, col) for case in zip(svm_outputs, svm_targets)]) for col in (0, 1)] for row in
                     (0, 1)]
                conditional_entropies[space].append(conditional_entropy(p))

        scores = {key: 2 ** np.sum(values) for key, values in conditional_entropies.items()}

        logger.info("latents Z  Result = %f" % (scores['latents']))
        logger.info("dlatents W Result = %f" % (scores['dlatents']))


        # # Calculate conditional entropy for each attribute.
        # conditional_entropies = defaultdict(list)
        # for attrib_idx in self.attrib_indices:
        #     # Prune the least confident samples.
        #     pruned_indices = list(range(self.num_samples))
        #     pruned_indices = sorted(pruned_indices, key=lambda i: -np.max(results[attrib_idx][i]))
        #     pruned_indices = pruned_indices[:self.num_keep]
        #
        #     # Fit SVM to the remaining samples.
        #     svm_targets = np.argmax(results[attrib_idx][pruned_indices], axis=1)
        #     for space in ['latents', 'dlatents']:
        #         svm_inputs = results[space][pruned_indices]
        #         try:
        #             svm = sklearn.svm.LinearSVC()
        #             svm.fit(svm_inputs, svm_targets)
        #             svm.score(svm_inputs, svm_targets)
        #             svm_outputs = svm.predict(svm_inputs)
        #         except:
        #             svm_outputs = svm_targets # assume perfect prediction
        #
        #         # Calculate conditional entropy.
        #         p = [[np.mean([case == (row, col) for case in zip(svm_outputs, svm_targets)]) for col in (0, 1)] for row in (0, 1)]
        #         conditional_entropies[space].append(conditional_entropy(p))
        #
        # # Calculate separability scores.
        # scores = {key: 2**np.sum(values) for key, values in conditional_entropies.items()}
        #
        # logger.info("latents Z  Result = %f" % (scores['latents']))
        # logger.info("dlatents W Result = %f" % (scores['dlatents']))


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
    model.requires_grad_(False)

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

    checkpointer.load()

    model.eval()

    layer_count = cfg.MODEL.LAYER_COUNT

    logger.info("Evaluating LS metric")

    decoder = nn.DataParallel(decoder)

    with torch.no_grad():
        ppl = LS(cfg, percent_keep=0.5, attrib_indices=range(40), minibatch_gpu=4)
        ppl.evaluate(logger, mapping_fl, decoder, cfg.DATASET.MAX_RESOLUTION_LEVEL - 2, 19)


if __name__ == "__main__":
    gpu_count = 1
    run(sample, get_cfg_defaults(), description='StyleGAN', default_config='configs/experiment_ffhq_z.yaml',
        world_size=gpu_count, write_log=False)
