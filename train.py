from lib.helper.logger import logger
from lib.core.base_trainer.net_work import Train
from lib.dataset.dataietr import DataIter
from lib.core.model.ShuffleNet_Series.ShuffleNetV2.network import ShuffleNetV2

from lib.core.model.semodel.SeResnet import se_resnet50
import cv2
import numpy as np

from train_config import config as cfg
import setproctitle

logger.info('The trainer start')

setproctitle.setproctitle("face*_*_")


def main():







    ###build dataset
    train_ds = DataIter(cfg.DATA.root_path, cfg.DATA.train_txt_path, True)
    test_ds = DataIter(cfg.DATA.root_path, cfg.DATA.val_txt_path, False)


    ###build trainer
    trainer = Train(train_ds=train_ds,val_ds=test_ds)

    trainer.load_weight()
    if cfg.TRAIN.vis:
        for step in range(train_ds.size):

            images, labels=train_ds()

            for i in range(images.shape[0]):
                example_image=np.array(images[i],dtype=np.uint8)
                example_image=np.transpose(example_image,[1,2,0])
                example_label=np.array(labels[i])

                _h, _w, _ = example_image.shape

                print(example_label.shape)

                cv2.imshow('example',example_image)
                cv2.waitKey(0)







    ### train
    trainer.custom_loop()

if __name__=='__main__':
    main()