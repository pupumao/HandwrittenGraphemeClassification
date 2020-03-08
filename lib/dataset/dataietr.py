

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



        for line in image_label_list:
            cur_data_info = line.rstrip().split(' ')
            fname = cur_data_info[0]
            label = cur_data_info[1:4]
            label = [int(x) for x in label]


            self.metas.append([fname,label])

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

    def balance(self,anns,training_flag=False):

        def static_data_distribution(map):
            map = list(map)
            cnt_map = {}
            for i in range(len(map)):
                if map[i] in cnt_map:
                    cnt_map[map[i]] += 1
                else:
                    cnt_map[map[i]] = 1

            sorted_cnt_map = sorted(cnt_map.items(), key=lambda a: a[0])

            for item in sorted_cnt_map:
                print(item[0], item[1])

            return cnt_map


        res_anns = copy.deepcopy(anns)

        ### step 1  statics

        STATIC_LABELS = []
        for ann in anns:

            fname = ann[0]
            label = ann[1]
            STATIC_LABELS.append(label)

        STATIC_LABELS = np.array(STATIC_LABELS)

        print('before balance')
        print('the first class distribution')
        s_dict1=static_data_distribution(STATIC_LABELS[:, 0])
        print('the first class distribution')
        s_dict2=static_data_distribution(STATIC_LABELS[:, 1])
        print('the first class distribution')
        s_dict3=static_data_distribution(STATIC_LABELS[:, 2])




        STATIC_LABELS=[]
        for ann in anns:

            fname = ann[0]
            label = ann[1]
            STATIC_LABELS.append(label)

            ## aug training set
            if training_flag:
                if s_dict1[label[0]]<500:
                    res_anns.append(ann)

                if s_dict1[label[0]]<200:
                    for _ in range(2):
                        res_anns.append(ann)

                ## second class
                if label[1]==3:
                    res_anns.append(ann)
                if label[1]==5 or label[1]==6 or label[1]==8 or label[1]==10:
                    for _ in range(4):
                        res_anns.append(ann)

                ## third class
                if label[2] == 1:
                    for _ in range(2):
                        res_anns.append(ann)


                if  label[2]==3:
                    for _ in range(20):
                        res_anns.append(ann)
                if  label[2]==6:
                    for _ in range(10):
                        res_anns.append(ann)


        STATIC_LABELS=np.array(STATIC_LABELS)

        print('before balance')
        print('the first class distribution')
        static_data_distribution(STATIC_LABELS[:,0])
        print('the first class distribution')
        static_data_distribution(STATIC_LABELS[:, 1])
        print('the first class distribution')
        static_data_distribution(STATIC_LABELS[:, 2])

        STATIC_LABELS_after = []
        for ann in res_anns:

            fname = ann[0]
            label = ann[1]
            STATIC_LABELS_after.append(label)


        STATIC_LABELS_after = np.array(STATIC_LABELS_after)

        print('after balance')
        print('the first class distribution')
        static_data_distribution(STATIC_LABELS_after[:, 0])
        print('the first class distribution')
        static_data_distribution(STATIC_LABELS_after[:, 1])
        print('the first class distribution')
        static_data_distribution(STATIC_LABELS_after[:, 2])


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
        all_samples = self.balance(all_samples,self.training_flag)
        return all_samples


    def _map_func(self,dp,is_training):
        """Data augmentation function."""
        ####customed here


        fname= dp[0]
        label=dp[1]

        image = cv2.imread(fname, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = np.array(label, dtype=np.float)


        if is_training:
            if random.uniform(0, 1) > 0.5:
                image=Random_crop(image,shrink=0.4)
            # if random.uniform(0, 1) > 0.5.:
            #     image, _ = Mirror(image, label=None, symmetry=None)

            if random.uniform(0, 1) > 0.0:
                angle = random.uniform(-20, 20)
                image, _ = Rotate_aug(image, label=None, angle=angle)

            if random.uniform(0, 1) > 0.5:
                strength = random.uniform(0, 30)
                image, _ = Affine_aug(image, strength=strength, label=None)

            # if random.uniform(0, 1) > 0.5:
            #     image=self.color_augmentor(image)
            # if random.uniform(0, 1) > 0.5:
            #     image=pixel_jitter(image,15)

            if random.uniform(0, 1) > 0.5:
                if random.uniform(0, 1) > 0.5:
                    ksize = random.choice([3, 5, 9, 11])
                    image = cv2.GaussianBlur(image,(ksize,ksize),1.5)
                if random.uniform(0, 1) > 0.5:
                    ksize = random.choice([3, 5, 9, 11])
                    image = cv2.medianBlur(image, ksize)
                if random.uniform(0, 1) > 0.5:
                    ksize = random.choice([3, 5, 9, 11])
                    image = cv2.blur(image, (ksize,ksize))

            # if random.uniform(0, 1) > 0.5:
            #     image = pixel_jitter(image, 15)
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



