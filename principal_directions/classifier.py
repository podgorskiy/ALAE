# Copyright 2019-2020 Stanislav Pidhorskyi
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.


import dnnlib.tflib
import pickle

_cache_dir = 'cache'
dnnlib.tflib.init_tf()


def load_pkl(file_or_url):
    file = None
    if dnnlib.util.is_url(file_or_url):
        file = dnnlib.util.open_url(file_or_url, cache_dir=_cache_dir)
    else:
        file = open(file_or_url, 'rb')
    return pickle.load(file, encoding='latin1')


classifier_urls = [
    'https://drive.google.com/uc?id=1Q5-AI6TwWhCVM7Muu4tBM7rp5nG_gmCX',  # celebahq-classifier-00-male.pkl
    'https://drive.google.com/uc?id=1Q5c6HE__ReW2W8qYAXpao68V1ryuisGo',  # celebahq-classifier-01-smiling.pkl
    'https://drive.google.com/uc?id=1Q7738mgWTljPOJQrZtSMLxzShEhrvVsU',  # celebahq-classifier-02-attractive.pkl
    'https://drive.google.com/uc?id=1QBv2Mxe7ZLvOv1YBTLq-T4DS3HjmXV0o',  # celebahq-classifier-03-wavy-hair.pkl
    'https://drive.google.com/uc?id=1QIvKTrkYpUrdA45nf7pspwAqXDwWOLhV',  # celebahq-classifier-04-young.pkl
    'https://drive.google.com/uc?id=1QJPH5rW7MbIjFUdZT7vRYfyUjNYDl4_L',  # celebahq-classifier-05-5-o-clock-shadow.pkl
    'https://drive.google.com/uc?id=1QPZXSYf6cptQnApWS_T83sqFMun3rULY',  # celebahq-classifier-06-arched-eyebrows.pkl
    'https://drive.google.com/uc?id=1QPgoAZRqINXk_PFoQ6NwMmiJfxc5d2Pg',  # celebahq-classifier-07-bags-under-eyes.pkl
    'https://drive.google.com/uc?id=1QQPQgxgI6wrMWNyxFyTLSgMVZmRr1oO7',  # celebahq-classifier-08-bald.pkl
    'https://drive.google.com/uc?id=1QcSphAmV62UrCIqhMGgcIlZfoe8hfWaF',  # celebahq-classifier-09-bangs.pkl
    'https://drive.google.com/uc?id=1QdWTVwljClTFrrrcZnPuPOR4mEuz7jGh',  # celebahq-classifier-10-big-lips.pkl
    'https://drive.google.com/uc?id=1QgvEWEtr2mS4yj1b_Y3WKe6cLWL3LYmK',  # celebahq-classifier-11-big-nose.pkl
    'https://drive.google.com/uc?id=1QidfMk9FOKgmUUIziTCeo8t-kTGwcT18',  # celebahq-classifier-12-black-hair.pkl
    'https://drive.google.com/uc?id=1QthrJt-wY31GPtV8SbnZQZ0_UEdhasHO',  # celebahq-classifier-13-blond-hair.pkl
    'https://drive.google.com/uc?id=1QvCAkXxdYT4sIwCzYDnCL9Nb5TDYUxGW',  # celebahq-classifier-14-blurry.pkl
    'https://drive.google.com/uc?id=1QvLWuwSuWI9Ln8cpxSGHIciUsnmaw8L0',  # celebahq-classifier-15-brown-hair.pkl
    'https://drive.google.com/uc?id=1QxW6THPI2fqDoiFEMaV6pWWHhKI_OoA7',  # celebahq-classifier-16-bushy-eyebrows.pkl
    'https://drive.google.com/uc?id=1R71xKw8oTW2IHyqmRDChhTBkW9wq4N9v',  # celebahq-classifier-17-chubby.pkl
    'https://drive.google.com/uc?id=1RDn_fiLfEGbTc7JjazRXuAxJpr-4Pl67',  # celebahq-classifier-18-double-chin.pkl
    'https://drive.google.com/uc?id=1RGBuwXbaz5052bM4VFvaSJaqNvVM4_cI',  # celebahq-classifier-19-eyeglasses.pkl
    'https://drive.google.com/uc?id=1RIxOiWxDpUwhB-9HzDkbkLegkd7euRU9',  # celebahq-classifier-20-goatee.pkl
    'https://drive.google.com/uc?id=1RPaNiEnJODdr-fwXhUFdoSQLFFZC7rC-',  # celebahq-classifier-21-gray-hair.pkl
    'https://drive.google.com/uc?id=1RQH8lPSwOI2K_9XQCZ2Ktz7xm46o80ep',  # celebahq-classifier-22-heavy-makeup.pkl
    'https://drive.google.com/uc?id=1RXZM61xCzlwUZKq-X7QhxOg0D2telPow',  # celebahq-classifier-23-high-cheekbones.pkl
    'https://drive.google.com/uc?id=1RgASVHW8EWMyOCiRb5fsUijFu-HfxONM',  # celebahq-classifier-24-mouth-slightly-open.pkl
    'https://drive.google.com/uc?id=1RkC8JLqLosWMaRne3DARRgolhbtg_wnr',  # celebahq-classifier-25-mustache.pkl
    'https://drive.google.com/uc?id=1RqtbtFT2EuwpGTqsTYJDyXdnDsFCPtLO',  # celebahq-classifier-26-narrow-eyes.pkl
    'https://drive.google.com/uc?id=1Rs7hU-re8bBMeRHR-fKgMbjPh-RIbrsh',  # celebahq-classifier-27-no-beard.pkl
    'https://drive.google.com/uc?id=1RynDJQWdGOAGffmkPVCrLJqy_fciPF9E',  # celebahq-classifier-28-oval-face.pkl
    'https://drive.google.com/uc?id=1S0TZ_Hdv5cb06NDaCD8NqVfKy7MuXZsN',  # celebahq-classifier-29-pale-skin.pkl
    'https://drive.google.com/uc?id=1S3JPhZH2B4gVZZYCWkxoRP11q09PjCkA',  # celebahq-classifier-30-pointy-nose.pkl
    'https://drive.google.com/uc?id=1S3pQuUz-Jiywq_euhsfezWfGkfzLZ87W',  # celebahq-classifier-31-receding-hairline.pkl
    'https://drive.google.com/uc?id=1S6nyIl_SEI3M4l748xEdTV2vymB_-lrY',  # celebahq-classifier-32-rosy-cheeks.pkl
    'https://drive.google.com/uc?id=1S9P5WCi3GYIBPVYiPTWygrYIUSIKGxbU',  # celebahq-classifier-33-sideburns.pkl
    'https://drive.google.com/uc?id=1SANviG-pp08n7AFpE9wrARzozPIlbfCH',  # celebahq-classifier-34-straight-hair.pkl
    'https://drive.google.com/uc?id=1SArgyMl6_z7P7coAuArqUC2zbmckecEY',  # celebahq-classifier-35-wearing-earrings.pkl
    'https://drive.google.com/uc?id=1SC5JjS5J-J4zXFO9Vk2ZU2DT82TZUza_',  # celebahq-classifier-36-wearing-hat.pkl
    'https://drive.google.com/uc?id=1SDAQWz03HGiu0MSOKyn7gvrp3wdIGoj-',  # celebahq-classifier-37-wearing-lipstick.pkl
    'https://drive.google.com/uc?id=1SEtrVK-TQUC0XeGkBE9y7L8VXfbchyKX',  # celebahq-classifier-38-wearing-necklace.pkl
    'https://drive.google.com/uc?id=1SF_mJIdyGINXoV-I6IAxHB_k5dxiF6M-',  # celebahq-classifier-39-wearing-necktie.pkl
]

classifier_id_name = {
    0: "gender",
    1: "smiling",
    2: "attractive",
    3: "wavy-hair",
    4: "young",
    5: "5-o-clock-shadow",
    6: "arched-eyebrows",
    7: "bags-under-eyes",
    8: "bald",
    9: "bangs",
    10: "big-lips",
    11: "big-nose",
    12: "black-hair",
    13: "blond-hair",
    14: "blurry",
    15: "brown-hair",
    16: "bushy-eyebrows",
    17: "chubby",
    18: "double-chin",
    19: "eyeglasses",
    20: "goatee",
    21: "gray-hair",
    22: "heavy-makeup",
    23: "high-cheekbones",
    24: "mouth-slightly-open",
    25: "mustache",
    26: "narrow-eyes",
    27: "no-beard",
    28: "oval-face",
    29: "pale-skin",
    30: "pointy-nose",
    31: "receding-hairline",
    32: "rosy-cheeks",
    33: "sideburns",
    34: "straight-hair",
    35: "wearing-earrings",
    36: "wearing-hat",
    37: "wearing-lipstick",
    38: "wearing-necklace",
    39: "wearing-necktie",
}


def make_classifier(attrib_idx):
    classifier = load_pkl(classifier_urls[attrib_idx])
    return classifier

