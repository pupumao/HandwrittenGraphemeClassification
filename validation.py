from lib.dataset.dataietr import DataIter
from train_config import config
from lib.core.model.ShuffleNet_Series.ShuffleNetV2.network import ShuffleNetV2
from lib.core.model.semodel.SeResnet import se_resnet50,se_resnext50_32x4d
import torch
import time
import argparse
import sklearn.metrics
from tqdm import tqdm
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
from train_config import config as cfg
cfg.TRAIN.batch_size=1
cfg.TRAIN.process_num=1
ds = DataIter(cfg.DATA.root_path,cfg.DATA.val_txt_path,True)



def vis(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###build model


    model=se_resnext50_32x4d()

    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()





    cls1_pre_list=[]
    cls1_true_list = []

    cls2_pre_list = []
    cls2_true_list = []

    cls3_pre_list = []
    cls3_true_list = []
    for step in tqdm(range(ds.size)):

        images, labels = ds()


        cls1_true_list.append(labels[0][0])
        cls2_true_list.append(labels[0][1])
        cls3_true_list.append(labels[0][2])

        img_show = np.array(images)

        # img_show=np.transpose(img_show[0],axes=[1,2,0])
        #
        # images=torch.from_numpy(images)
        # images=images.to(device)
        #
        # start=time.time()
        #
        # logit1, logit2, logit3 = model(images)
        # res1 = torch.softmax(logit1,1)
        # res2 = torch.softmax(logit2,1)
        # res3 = torch.softmax(logit3,1)
        #
        #
        # res1=res1.cpu().detach().numpy()[0]
        #
        # res2 = res2.cpu().detach().numpy()[0]
        # res3 = res3.cpu().detach().numpy()[0]
        # cls1_pre_list.append(np.argmax(res1))
        # cls2_pre_list.append(np.argmax(res2))
        # cls3_pre_list.append(np.argmax(res3))


        #print(res)

        img_show=(img_show[0]*255).astype(np.uint8)
        img_show=np.transpose(img_show,[1,2,0])
        img_show=cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)



        cv2.imshow('tmp',img_show)
        cv2.waitKey(0)


    score1=sklearn.metrics.recall_score(
        cls1_true_list, cls1_pre_list, average='macro')
    score2 = sklearn.metrics.recall_score(
        cls2_true_list, cls2_pre_list, average='macro')
    score3 = sklearn.metrics.recall_score(
        cls3_true_list, cls3_pre_list, average='macro')

    final_score = np.average([score1,score2,score3], weights=[2, 1, 1])


    print('cu score is %5f'%final_score)

def load_checkpoint(net, checkpoint,device):
    # from collections import OrderedDict
    #
    # temp = OrderedDict()
    # if 'state_dict' in checkpoint:
    #     checkpoint = dict(checkpoint['state_dict'])
    # for k in checkpoint:
    #     k2 = 'module.'+k if not k.startswith('module.') else k
    #     temp[k2] = checkpoint[k]

    net.load_state_dict(torch.load(checkpoint,map_location=device), strict=True)
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Start train.')

    parser.add_argument('--model', dest='model', type=str, default=None, \
                        help='the model to use')

    args = parser.parse_args()



    vis(args.model)




