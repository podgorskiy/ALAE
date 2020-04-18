# Copyright 2019-2020 Stanislav Pidhorskyi
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

from yacs.config import CfgNode as CN


_C = CN()

_C.NAME = ""
_C.PPL_CELEBA_ADJUSTMENT = False
_C.OUTPUT_DIR = "results"

_C.DATASET = CN()
_C.DATASET.PATH = 'celeba/data_fold_%d_lod_%d.pkl'
_C.DATASET.PATH_TEST = ''
_C.DATASET.FFHQ_SOURCE = '/data/datasets/ffhq-dataset/tfrecords/ffhq/ffhq-r%02d.tfrecords'
_C.DATASET.PART_COUNT = 1
_C.DATASET.PART_COUNT_TEST = 1
_C.DATASET.SIZE = 70000
_C.DATASET.SIZE_TEST = 10000
_C.DATASET.FLIP_IMAGES = True
_C.DATASET.SAMPLES_PATH = 'dataset_samples/faces/realign128x128'

_C.DATASET.STYLE_MIX_PATH = 'style_mixing/test_images/set_celeba/'

_C.DATASET.MAX_RESOLUTION_LEVEL = 10

_C.MODEL = CN()

_C.MODEL.LAYER_COUNT = 6
_C.MODEL.START_CHANNEL_COUNT = 64
_C.MODEL.MAX_CHANNEL_COUNT = 512
_C.MODEL.LATENT_SPACE_SIZE = 256
_C.MODEL.DLATENT_AVG_BETA = 0.995
_C.MODEL.TRUNCATIOM_PSI = 0.7
_C.MODEL.TRUNCATIOM_CUTOFF = 8
_C.MODEL.STYLE_MIXING_PROB = 0.9
_C.MODEL.MAPPING_LAYERS = 5
_C.MODEL.CHANNELS = 3
_C.MODEL.GENERATOR = "GeneratorDefault"
_C.MODEL.ENCODER = "EncoderDefault"
_C.MODEL.MAPPING_TO_LATENT = "MappingToLatent"
_C.MODEL.MAPPING_FROM_LATENT = "MappingFromLatent"
_C.MODEL.Z_REGRESSION = False

_C.TRAIN = CN()

_C.TRAIN.EPOCHS_PER_LOD = 15

_C.TRAIN.BASE_LEARNING_RATE = 0.0015
_C.TRAIN.ADAM_BETA_0 = 0.0
_C.TRAIN.ADAM_BETA_1 = 0.99
_C.TRAIN.LEARNING_DECAY_RATE = 0.1
_C.TRAIN.LEARNING_DECAY_STEPS = []
_C.TRAIN.TRAIN_EPOCHS = 110

_C.TRAIN.LOD_2_BATCH_8GPU = [512, 256, 128,   64,   32,    32]
_C.TRAIN.LOD_2_BATCH_4GPU = [512, 256, 128,   64,   32,    16]
_C.TRAIN.LOD_2_BATCH_2GPU = [256, 256, 128,   64,   32,    16]
_C.TRAIN.LOD_2_BATCH_1GPU = [128, 128, 128,   64,   32,    16]


_C.TRAIN.SNAPSHOT_FREQ = [300, 300, 300, 100, 50, 30, 20, 20, 10]

_C.TRAIN.REPORT_FREQ = [100, 80, 60, 30, 20, 10, 10, 5, 5]

_C.TRAIN.LEARNING_RATES = [0.002]


def get_cfg_defaults():
    return _C.clone()
