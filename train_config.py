

import os
import numpy as np
from easydict import EasyDict as edict

config = edict()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config.TRAIN = edict()
#### below are params for dataiter
config.TRAIN.process_num = 8
config.TRAIN.prefetch_size = 30
############

config.TRAIN.num_gpu = 1
config.TRAIN.batch_size = 128
config.TRAIN.log_interval = 10                  ##10 iters for a log msg
config.TRAIN.epoch = 150

config.TRAIN.lr_value_every_epoch = [0.00001,0.0001,0.001,0.0001,0.00001,0.000001,0.0000001]          ####no use
config.TRAIN.lr_decay_every_epoch = [1,2,80,100,120,140]

config.TRAIN.weight_decay_factor = 1.e-4                                    ####l2
config.TRAIN.vis=False                                                      #### if to check the training data

config.TRAIN.vis_mixcut=False
config.TRAIN.mix_precision=False                                            ##use mix precision to speedup, tf1.14 at least
config.TRAIN.opt='Adam'                                                     ##Adam or SGD

config.MODEL = edict()
config.MODEL.model_path = './models/'                                        ## save directory
config.MODEL.hin = 137 ##137                                               # input size during training , 128,160,   depends on
config.MODEL.win = 236 ## 236
config.MODEL.channel = 3


config.MODEL.pretrained_model=None
config.DATA = edict()

config.DATA.root_path=''
config.DATA.train_txt_path='train.txt'
config.DATA.val_txt_path='val.txt'

############the model is trained with RGB mode
config.DATA.PIXEL_MEAN = [0.485, 0.456, 0.406]             ###rgb
config.DATA.PIXEL_STD = [0.229, 0.224, 0.225]

config.DATA.base_extend_range=[0.1,0.2]                 ###extand
config.DATA.scale_factor=[0.7,1.35]                     ###scales


config.MODEL.focal_loss=False
config.MODEL.ohem =False

config.MODEL.cut_mix_beta=1.