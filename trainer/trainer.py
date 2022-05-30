import os
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as f
from tqdm import tqdm
from utils.generals import AverageMeter, iou, dice, increment_path
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    _scales_ = [0.75, 1, 1.25]

    def __init__(
            self,
            model,
            train_loader,
            test_loader,
            optimizer,
            lr_scheduler,
            criterion,
            use_tensorboard,
            epochs,
            device,
            save_dir,
            exist_ok=False,
            max_norm=0
    ):
        super(Trainer, self).__init__()
        self.model = nn.DataParallel(model).to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.use_tensorboard = use_tensorboard
        self.device = device
        self.max_norm = max_norm
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.save_dir = increment_path(save_dir, exist_ok)
        if self.use_tensorboard:
            self.writer = SummaryWriter(save_dir)
        self.global_step = 0
        self.save_epoch = 0

    def _train_one_epoch(self, epoch):
        self.model.train()
        self.criterion.train()
        dice_lis, iou_lis = [], []
        pdar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.epochs}')
        bce_metric = AverageMeter()
        tve_metric = AverageMeter()
        iou_metric = AverageMeter()
        dice_metric = AverageMeter()
        for images_, labels_ in pdar:
            images_ = images_.to(self.device)
            labels_ = labels_.to(self.device)
            for scale in self._scales_:
                self.global_step += 1
                if scale != 1:
                    old_size = images_.shape[-1]
                    new_size = int(round(old_size * scale / 32) * 32)
                    images = f.upsample(images_, size=(new_size, new_size), mode="bilinear", align_corners=True)
                    labels = f.upsample(labels_, size=(new_size, new_size), mode="bilinear", align_corners=True)
                else:
                    images, labels = images_, labels_

                outputs = self.model(images)
                bce_loss, tve_loss = self.criterion(outputs, labels)
                loss = bce_loss + tve_loss

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.optimizer.step()
                self.lr_scheduler.step()

                iou_lis = iou(outputs, labels, iou_lis)
                dice_lis = dice(outputs, labels, dice_lis)

                bce_metric.update(bce_loss.item())
                tve_metric.update(tve_loss.item())
                iou_metric.update(np.array(iou_lis).mean())
                dice_metric.update(np.array(dice_lis).mean())

                pdar.set_postfix(bce=bce_metric.avg, tve=tve_metric.avg, lr=self.optimizer.param_groups[0]['lr'],
                                 iou=iou_metric.avg, dice=dice_metric.avg)
                if self.use_tensorboard:
                    self.writer.add_scalar('lr', self.optimizer.param_groups[0]["lr"], self.global_step)
        return bce_metric.avg, tve_metric.avg, np.array(iou_lis).mean(), np.array(dice_lis).mean()

    def _evaluation(self):
        return self.evaluation(self.model, self.test_loader, self.device, self.criterion)

    @staticmethod
    @torch.no_grad()
    def evaluation(model, test_loader, device, criterion):
        model.eval()
        criterion.eval()
        bce_metric = AverageMeter()
        tve_metric = AverageMeter()
        iou_metric = AverageMeter()
        dice_metric = AverageMeter()
        iou_lis, dice_lis = [], []
        pdar = tqdm(test_loader, desc='Evaluation')
        for images, labels in pdar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            bce_loss, tve_loss = criterion(outputs, labels)

            iou_lis = iou(outputs, labels, iou_lis)
            dice_lis = dice(outputs, labels, dice_lis)

            bce_metric.update(bce_loss.item())
            tve_metric.update(tve_loss.item())
            iou_metric.update(np.array(iou_lis).mean())
            dice_metric.update(np.array(dice_lis).mean())
            pdar.set_postfix(bce=bce_metric.avg, tve=tve_metric.avg, iou=iou_metric.avg, dice=dice_metric.avg)
        return bce_metric.avg, tve_metric.avg, np.array(iou_lis).mean(), np.array(dice_lis).mean()

    def train(self):
        best_iou = 0
        best_dice = 0
        for epoch in range(self.epochs):
            bce_loss, tve_loss, train_iou, train_dice = self._train_one_epoch(epoch)
            tags = ["train/bce_loss", "train/tve_loss", "train/iou", "train/dice"]
            values = [bce_loss, tve_loss, train_iou, train_dice]
            if self.use_tensorboard:
                for tag, value in zip(tags, values):
                    self.writer.add_scalar(tag, value, epoch)
            if epoch > self.save_epoch:
                test_bce, test_tve, test_iou, dice_iou = self._evaluation()
                tags = ["test/bce_loss", "test/tve_loss", "test/iou", "test/dice"]
                values = [test_bce, test_tve, test_iou, dice_iou]
                if self.use_tensorboard:
                    for tag, value in zip(tags, values):
                        self.writer.add_scalar(tag, value, epoch)
                ckpt = {
                    'epoch': epoch,
                    'best_iou': test_iou,
                    'best_dice': dice_iou,
                    'model_state_dict': self.model.module.state_dict() if self.model.module else self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'lr_scheduler': self.lr_scheduler.state_dict(),
                }
                if (best_iou + best_dice) < (test_iou + dice_iou):
                    best_iou = test_iou
                    best_dice = dice_iou
                    self.save_model(ckpt, os.path.join(self.save_dir, 'best_model.pth'))
                self.save_model(ckpt, os.path.join(self.save_dir, 'last_model.pth'))

    @staticmethod
    def save_model(ckpt, path):
        torch.save(ckpt, path)

    @staticmethod
    def load_model(path):
        ckpt = torch.load(path)
        return ckpt
