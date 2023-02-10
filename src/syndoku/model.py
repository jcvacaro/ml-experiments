from argparse import ArgumentParser
import os
import torch
import torchvision
import torchvision.models.detection
import pytorch_lightning as pl
import mlflow
import mlflow.pytorch

import common
from data import SyndokuData

class SyndokuDetection(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group('syndoku-detection-model')
        parser.add_argument('--weights', default='FasterRCNN_ResNet50_FPN_Weights.COCO_V1', help='The model weights')
        parser.add_argument('--weights-backbone', default='ResNet50_Weights.IMAGENET1K_V1', help='The model backbone weights')
        parser.add_argument("--trainable-backbone-layers", default=None, type=int, help='number of trainable layers of backbone')
        parser.add_argument('--rpn-score-thresh', default=None, type=float, help='rpn score threshold for faster-rcnn')
        return parent_parser

    def __init__(self, args, data_module):
        super().__init__()
        self.save_hyperparameters(args, logger=False)
        self.data = data_module
        self.model = self._create_model()

    def _create_model(self):
        kwargs = {"trainable_backbone_layers": self.hparams.trainable_backbone_layers}
        if "rcnn" in self.hparams.model:
            if self.hparams.rpn_score_thresh is not None:
                kwargs["rpn_score_thresh"] = self.hparams.rpn_score_thresh
        return torchvision.models.get_model(
            self.hparams.model,
            # weights=self.hparams.weights,
            weights=None,
            weights_backbone=self.hparams.weights_backbone,
            num_classes=self.data.num_classes + 1,  # background class
            **kwargs
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        model_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = common.train.get_optimizer(self.hparams, self.model, model_params)
        lr_scheduler = common.train.get_lr_scheduler(self.hparams, optimizer)
        return optimizer if lr_scheduler is None else ([optimizer], [lr_scheduler])
