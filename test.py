import random
import cv2
import json
import numpy as np
import copy

from lib.helper.logger import logger
from tensorpack.dataflow import DataFromGenerator,BatchData, MultiProcessPrefetchData



from lib.dataset.augmentor.augmentation import Rotate_aug,\
                                                Affine_aug,\
                                                Mirror,\
                                                Padding_aug,\
                                                Img_dropout,Random_crop



from lib.dataset.augmentor.visual_augmentation import ColorDistort,pixel_jitter

from train_config import config as cfg

def _map_func( dp, is_training):
    """Data augmentation function."""
    ####customed here

    fname = dp[0]
    label = dp[1]

    image = cv2.imread(fname, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    label = np.array(label, dtype=np.float)

    if is_training:
        if random.uniform(0, 1) > 0.5:
            print('sss')
            image = Random_crop(image, shrink=0.5)
        if random.uniform(0, 1) > 0.5:
            image, _ = Mirror(image, label=None, symmetry=None)
        if random.uniform(0, 1) > 0.0:
            angle = random.uniform(-20, 20)
            image, _ = Rotate_aug(image, label=None, angle=angle)

        if random.uniform(0, 1) > 0.5:
            strength = random.uniform(0, 30)
            image, _ = Affine_aug(image, strength=strength, label=None)


        if random.uniform(0, 1) > 0.5:
            image = pixel_jitter(image, 15)

        if random.uniform(0, 1) > 0.5:
            if random.uniform(0, 1) > 0.5:
                ksize = random.choice([3, 5, 9, 11])
                image = cv2.GaussianBlur(image, (ksize, ksize), 1.5)
            if random.uniform(0, 1) > 0.5:
                ksize = random.choice([3, 5, 9, 11])
                image = cv2.medianBlur(image, ksize)
            if random.uniform(0, 1) > 0.5:
                ksize = random.choice([3, 5, 9, 11])
                image = cv2.blur(image, (ksize, ksize))

        if random.uniform(0, 1) > 0.5:
            image = pixel_jitter(image, 15)
        # if random.uniform(0, 1) > 0.5:
        #     image = Img_dropout(image, 0.2)

        # if random.uniform(0, 1) > 0.5:
        #     image = Padding_aug(image, 0.3)

        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST,
                          cv2.INTER_LANCZOS4]
        interp_method = random.choice(interp_methods)

        image = cv2.resize(image, (cfg.MODEL.win, cfg.MODEL.hin), interpolation=interp_method)

    image = cv2.resize(image, (cfg.MODEL.win, cfg.MODEL.hin), interpolation=cv2.INTER_NEAREST)
    #######head pose

    if cfg.MODEL.channel == 1:
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = np.expand_dims(image, -1)

    image = np.transpose(image, axes=[2, 0, 1])

    label = label.reshape([-1]).astype(np.float32)
    image = image.astype(np.float32)


    return image, label



img_path='/media/lz/ssd_2/kaggle/data/images/Train_155704.png'
label= [79, 1, 4]


for i in range(100):

    image,label=_map_func([img_path,label],True)

    image=image.astype(np.uint8)
    image=np.transpose(image,[1,2,0])

    cv2.imshow('ss',image)
    cv2.waitKey(0)