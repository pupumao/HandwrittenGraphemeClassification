#-*-coding:utf-8-*-

import sklearn.metrics

import time
import os
import torch.nn as nn
from train_config import config as cfg
#from lib.dataset.dataietr import DataIter


from lib.helper.logger import logger

from lib.core.model.ShuffleNet_Series.ShuffleNetV2.utils import accuracy, AvgrageMeter, CrossEntropyLabelSmooth, save_checkpoint, get_lastest_model, get_parameters
from lib.core.model.loss.focal_loss import FocalLoss
from lib.core.model.head.simple_head import Header

from lib.core.model.ShuffleNet_Series.ShuffleNetV2.network import ShuffleNetV2

from lib.core.model.semodel.SeResnet import se_resnet50

import torch.nn as nn

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


class Train(object):
  """Train class.
  Args:
    epochs: Number of epochs
    enable_function: If True, wraps the train_step and test_step in tf.function
    model: Densenet model.
    batch_size: Batch size.
    strategy: Distribution strategy in use.
  """

  def __init__(self,train_ds,val_ds):
    self.epochs = cfg.TRAIN.epoch
    self.batch_size = cfg.TRAIN.batch_size
    self.l2_regularization=cfg.TRAIN.weight_decay_factor




    self.device = torch.device("cuda")

    self.model = se_resnet50().to(self.device)





    if 'Adam' in cfg.TRAIN.opt:

      self.optimizer = torch.optim.Adam(get_parameters(self.model),
                                     lr=0.001,
                                     weight_decay=cfg.TRAIN.weight_decay_factor)

    else:
      self.optimizer  = torch.optim.SGD(get_parameters(self.model),
                                lr=0.001,
                                momentum=0.99,
                                weight_decay=cfg.TRAIN.weight_decay_factor)
    ###control vars
    self.iter_num=0

    self.lr_decay_every_epoch =cfg.TRAIN.lr_decay_every_epoch
    self.lr_val_every_epoch = cfg.TRAIN.lr_value_every_epoch


    self.train_ds=train_ds

    self.val_ds = val_ds

    # self.scheduler = torch.optim.lr_scheduler.MultiStepLR( self.optimizer, milestones=[100,120,140], gamma=0.1)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR( self.optimizer, 50)
    self.scheduler = GradualWarmupScheduler( self.optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler_cosine)

    self.losser= torch.nn.CrossEntropyLoss()
    if cfg.MODEL.focal_loss:
        self.losser1=FocalLoss(class_num=168)
        self.losser2 = FocalLoss(class_num=11)
        self.losser3 = FocalLoss(class_num=7)
  def loss_function(self,predicts,labels):

    labels=labels.long()



    predict1 = predicts[0]
    predict2 = predicts[1]
    predict3 = predicts[2]

    label1=labels[:,0]
    label2=labels[:,1]
    label3=labels[:,2]


    if cfg.MODEL.focal_loss:
        loss1 = self.losser1(predict1, label1)
        loss2 = self.losser2(predict2, label2)

        loss3 = self.losser3(predict3, label3)
    else:
        loss1 = self.losser(predict1, label1)
        loss2 = self.losser(predict2, label2)

        loss3 = self.losser(predict3, label3)

    # res1 = torch.softmax(predict1, 1)
    # res2 = torch.softmax(predict2, 1)
    # res3 = torch.softmax(predict3, 1)
    res1 = predict1
    res2 = predict2
    res3 = predict3
    res1 = torch.argmax(res1, 1)
    correct1= (res1 == label1).sum().float()

    res2 = torch.argmax(res2, 1)
    correct2 = (res2 == label2).sum().float()

    res3 = torch.argmax(res3, 1)
    correct3 = (res3 == label3).sum().float()

    return loss1 , \
           loss2 , \
           loss3,\
           correct1/self.batch_size,\
           correct2/self.batch_size,\
           correct3/self.batch_size



  def custom_loop(self):
    """Custom training and testing loop.
    Args:
      train_dist_dataset: Training dataset created using strategy.
      test_dist_dataset: Testing dataset created using strategy.
      strategy: Distribution strategy.
    Returns:
      train_loss, train_accuracy, test_loss, test_accuracy
    """

    def distributed_train_epoch(epoch_num):
      total_loss = 0.0
      num_train_batches = 0.0

      self.model.train()
      for step in range(self.train_ds.size):

        start=time.time()



        images, target = self.train_ds()

        images_torch = torch.from_numpy(images)
        target_torch = torch.from_numpy(target)

        data, target = images_torch.to(self.device), target_torch.to(self.device)


        output1,output2,output3 = self.model(data)


        loss1,loss2,loss3,acc1,acc2,acc3 = self.loss_function([output1,output2,output3], target)

        current_loss=loss1+loss2+loss3
        self.optimizer.zero_grad()
        current_loss.backward()
        self.optimizer.step()

        total_loss += current_loss
        num_train_batches += 1
        self.iter_num+=1
        time_cost_per_batch=time.time()-start

        images_per_sec=cfg.TRAIN.batch_size/time_cost_per_batch


        if self.iter_num%cfg.TRAIN.log_interval==0:
          logger.info('epoch_num: %d, '
                      'iter_num: %d, '
                      'loss1: %.6f, '
                      'acc1:  %.6f, '
                      'loss2: %.6f, '
                      'acc2:  %.6f, '
                      'loss3: %.6f, '
                      'acc3:  %.6f, '
                      'loss_value: %.6f,  '
                      'speed: %d images/sec ' % (epoch_num,
                                                 self.iter_num,
                                                 loss1,
                                                 acc1,
                                                 loss2,
                                                 acc2,
                                                 loss3,
                                                 acc3,
                                                 current_loss,
                                                 images_per_sec))

      return total_loss, num_train_batches

    def distributed_test_epoch(epoch_num):
      total_loss=0.
      total_acc1=0.
      total_acc2 = 0.
      total_acc3 = 0.
      num_test_batches = 0.0
      self.model.eval()
      with torch.no_grad():
        for i in range(self.val_ds.size):
          images, target = self.val_ds()
          images_torch = torch.from_numpy(images)
          target_torch = torch.from_numpy(target)

          data, target = images_torch.to(self.device), target_torch.to(self.device)

          output1, output2, output3 = self.model(data)

          loss1, loss2, loss3, acc1, acc2, acc3 = self.loss_function([output1, output2, output3], target)

          cur_loss=loss1+loss2+loss3
          total_loss+=cur_loss
          total_acc1+=acc1
          total_acc2 += acc2
          total_acc3 += acc3
          num_test_batches += 1
      return total_loss,\
             total_acc1,\
             total_acc2,\
             total_acc3, \
             num_test_batches


    for epoch in range(self.epochs):
      self.scheduler.step()
      for param_group in self.optimizer.param_groups:
        lr=param_group['lr']
      logger.info('learning rate: [%f]' %(lr))
      start=time.time()

      train_total_loss, num_train_batches = distributed_train_epoch(epoch)
      test_total_loss,test_total_acc1,test_total_acc2,test_total_acc3, num_test_batches = distributed_test_epoch(epoch)



      time_consume_per_epoch=time.time()-start
      training_massage = 'Epoch: %d, ' \
                         'Train Loss: %.6f, ' \
                         'Test Loss: %.6f ' \
                         'Test acc1: %.6f '\
                         'Test acc2: %.6f '\
                         'Test acc3: %.6f '\
                         'Time consume: %.2f'%(epoch,
                                               train_total_loss / num_train_batches,
                                               test_total_loss / num_test_batches,
                                               test_total_acc1 / num_test_batches,
                                               test_total_acc2 / num_test_batches,
                                               test_total_acc3 / num_test_batches,
                                               time_consume_per_epoch)

      logger.info(training_massage)


      #### save the model every end of epoch
      current_model_saved_name='./model/epoch_%d_val_loss%.6f.pth'%(epoch,test_total_loss / num_test_batches)

      logger.info('A model saved to %s' % current_model_saved_name)

      if not os.access(cfg.MODEL.model_path,os.F_OK):
        os.mkdir(cfg.MODEL.model_path)


      torch.save(self.model.state_dict(),current_model_saved_name)
      # save_checkpoint({
      #           'state_dict': self.model.state_dict(),
      #           },iters=epoch,tag=current_model_saved_name)



    return (train_total_loss / num_train_batches,
            test_total_loss / num_test_batches)


  def load_weight(self):
      if cfg.MODEL.pretrained_model is not None:
          self.model.load_state_dict(torch.load(cfg.MODEL.pretrained_model,map_location=self.device),strict=False)


