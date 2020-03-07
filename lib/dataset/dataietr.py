

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


class data_info(object):
    def __init__(self,img_root,ann_file):
        self.ann_file=ann_file
        self.root_path = img_root
        self.metas=[]

        self.load_anns()

    def load_anns(self):
        with open(self.ann_file, 'r') as f:
            image_label_list = f.readlines()

        self.metas=image_label_list

            ###some change can be made here

        logger.info('the datasets contains %d samples'%(len(self.metas)))


    def get_all_sample(self):
        random.shuffle(self.metas)
        return self.metas
#
#
class DataIter():
    def __init__(self,img_root_path='',ann_file=None,training_flag=True):

        self.shuffle=True
        self.training_flag=training_flag
        self.num_gpu = cfg.TRAIN.num_gpu
        self.batch_size = cfg.TRAIN.batch_size
        self.process_num = cfg.TRAIN.process_num
        self.prefetch_size = cfg.TRAIN.prefetch_size

        self.generator = HWGDataIter(img_root_path, ann_file, self.training_flag)

        self.ds=self.build_iter()

        self.size = self.__len__()


    def parse_file(self,im_root_path,ann_file):

        raise NotImplementedError("you need implemented the parse func for your data")


    def build_iter(self):

        ds = DataFromGenerator(self.generator)
        ds = BatchData(ds, self.batch_size)
        ds = MultiProcessPrefetchData(ds, self.prefetch_size, self.process_num)
        ds.reset_state()
        ds = ds.get_data()
        return ds

    # def __iter__(self):
    #     for i in range(self.size):
    #         one_batch = next(self.ds)
    #         return one_batch[0], one_batch[1]

    def __call__(self, *args, **kwargs):



        for i in range(self.size):
            one_batch=next(self.ds)
            return one_batch[0],one_batch[1]




    def __len__(self):
        return len(self.generator)//self.batch_size

    def _map_func(self,dp,is_training):

        raise NotImplementedError("you need implemented the map func for your data")




class HWGDataIter():
    def __init__(self, img_root_path='', ann_file=None, training_flag=True,shuffle=True):



        self.eye_close_thres=0.02
        self.mouth_close_thres = 0.02
        self.big_mouth_open_thres = 0.08
        self.training_flag = training_flag
        self.shuffle = shuffle


        self.raw_data_set_size = None     ##decided by self.parse_file


        self.color_augmentor = ColorDistort()


        self.lst = self.parse_file(img_root_path, ann_file)







    def __call__(self, *args, **kwargs):
        idxs = np.arange(len(self.lst))

        # while True:
        if self.shuffle:
            np.random.shuffle(idxs)
        for k in idxs:
            yield self._map_func(self.lst[k], self.training_flag)

    def __len__(self):
        assert self.raw_data_set_size is not None

        return self.raw_data_set_size

    def balance(self,anns):
        res_anns = copy.deepcopy(anns)

        lar_count = 0
        for ann in anns:

            ### 300w  balance,  according to keypoints
            if ann['keypoints'] is not None:

                label = ann['keypoints']
                label = np.array(label, dtype=np.float).reshape((-1, 2))
                bbox = ann['bbox']
                bbox_width = bbox[2] - bbox[0]
                bbox_height = bbox[3] - bbox[1]

                if bbox_width < 50 or bbox_height < 50:
                    res_anns.remove(ann)

                left_eye_close = np.sqrt(
                    np.square(label[37, 0] - label[41, 0]) +
                    np.square(label[37, 1] - label[41, 1])) / bbox_height < self.eye_close_thres \
                    or np.sqrt(np.square(label[38, 0] - label[40, 0]) +
                               np.square(label[38, 1] - label[40, 1])) / bbox_height < self.eye_close_thres
                right_eye_close = np.sqrt(
                    np.square(label[43, 0] - label[47, 0]) +
                    np.square(label[43, 1] - label[47, 1])) / bbox_height <  self.eye_close_thres \
                    or np.sqrt(np.square(label[44, 0] - label[46, 0]) +
                               np.square(label[44, 1] - label[46, 1])) / bbox_height < self.eye_close_thres
                if left_eye_close or right_eye_close:
                    for i in range(10):
                        res_anns.append(ann)
                ###half face
                if np.sqrt(np.square(label[36, 0] - label[45, 0]) +
                           np.square(label[36, 1] - label[45, 1])) / bbox_width < 0.5:
                    for i in range(20):
                        res_anns.append(ann)

                if np.sqrt(np.square(label[62, 0] - label[66, 0]) +
                           np.square(label[62, 1] - label[66, 1])) / bbox_height > 0.15:
                    for i in range(20):
                        res_anns.append(ann)

                if np.sqrt(np.square(label[62, 0] - label[66, 0]) +
                           np.square(label[62, 1] - label[66, 1])) / cfg.MODEL.hin > self.big_mouth_open_thres:
                    for i in range(50):
                        res_anns.append(ann)
                ##########eyes diff aug
                if left_eye_close and not right_eye_close:
                    for i in range(40):
                        res_anns.append(ann)
                    lar_count += 1
                if not left_eye_close and right_eye_close:
                    for i in range(40):
                        res_anns.append(ann)
                    lar_count += 1

            # elif ann['attr'] is not None:
            #
            #     ###celeba data,
            #     if ann['attr'][0]>0:
            #         for i in range(10):
            #             res_anns.append(ann)





        logger.info('befor balance the dataset contains %d images' % (len(anns)))
        logger.info('after balanced the datasets contains %d samples' % (len(res_anns)))

        random.shuffle(res_anns)
        return res_anns

    def parse_file(self,im_root_path,ann_file):
        '''
        :return:
        '''
        logger.info("[x] Get dataset from {}".format(im_root_path))

        ann_info = data_info(im_root_path, ann_file)
        all_samples = ann_info.get_all_sample()
        self.raw_data_set_size=len(all_samples)
        # balanced_samples = self.balance(all_samples)
        return all_samples


    def _map_func(self,dp,is_training):
        """Data augmentation function."""
        ####customed here


        cur_data_info=dp.rstrip().split(' ')
        fname= cur_data_info[0]
        label=cur_data_info[1:4]
        label=[int(x) for x in label]
        # attr =dp['attr']

        #### 300W
        image = cv2.imread(fname, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = np.array(label, dtype=np.float)


        if is_training:
            if random.uniform(0, 1) > 0.5:
                image=Random_crop(image,shrink=0.2)
            # if random.uniform(0, 1) > 0.5.:
            #     image, _ = Mirror(image, label=None, symmetry=None)
            if random.uniform(0, 1) > 0.0:
                angle = random.uniform(-30, 30)
                image, _ = Rotate_aug(image, label=None, angle=angle)

            if random.uniform(0, 1) > 0.5:
                strength = random.uniform(0, 50)
                image, _ = Affine_aug(image, strength=strength, label=None)

            if random.uniform(0, 1) > 0.5:
                image=self.color_augmentor(image)
            if random.uniform(0, 1) > 0.5:
                image=pixel_jitter(image,15)
            if random.uniform(0, 1) > 0.5:
                image = Img_dropout(image, 0.2)

            if random.uniform(0, 1) > 0.5:
                image = Padding_aug(image, 0.3)

        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST,
                          cv2.INTER_LANCZOS4]
        interp_method = random.choice(interp_methods)

        image = cv2.resize(image, (cfg.MODEL.win, cfg.MODEL.hin), interpolation=interp_method)

        #######head pose

        if cfg.MODEL.channel==1:
            image = image.astype(np.uint8)
            image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            image=np.expand_dims(image,-1)


        image = np.transpose(image,axes=[2,0,1])

        label = label.reshape([-1]).astype(np.float32)
        image = image.astype(np.float32)

        if not cfg.TRAIN.vis :
            image=image/255.
        return image, label



