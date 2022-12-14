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
from vision.references.detection import coco_eval, coco_utils

class SyndokuDetection(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group('syndoku-detection-model')
        parser.add_argument('--rpn-score-thresh', default=None, type=float, help='rpn score threshold for faster-rcnn')
        parser.add_argument("--trainable-backbone-layers", default=None, type=int, help='number of trainable layers of backbone')
        return parent_parser

    def __init__(self, args, data_module):
        super().__init__()
        self.save_hyperparameters(args)
        self.data = data_module
        self.model = self.create_model()

    def create_model(self):
        kwargs = {"trainable_backbone_layers": self.hparams.trainable_backbone_layers}
        if "rcnn" in self.hparams.model and self.hparams.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = self.hparams.rpn_score_thresh
        return torchvision.models.get_model(
            self.hparams['model'],
            num_classes=self.data.num_classes + 1, 
            **kwargs
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = list(image.to(self.device) for image in images)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log('train_loss', loss)
        return loss

    def log_metrics(self, metrics_prefix):
        metrics = {}
        def avg_map(i): return 'map' if i<6 else 'mar'
        area_map = {0:'all', 8:'all', 3:'small', 9:'small', 4:'medium', 10:'medium', 5:'large', 11:'large'}
        eval = self.coco_evaluator.coco_eval['bbox']
        for i in (0,3,4,5,8,9,10,11): # average metrics mAP/mAR for small/medium/large areas
            metrics.update({f'metric/{metrics_prefix}_{avg_map(i)}_{area_map[i]}' : eval.stats[i]})
        self.log_dict(metrics)
        return metrics

    def val_test_step(self, batch, batch_idx, dataset_name, dataset):
        cpu_device = torch.device('cpu')
        if batch_idx == 0: # init
            self.coco = coco_utils.get_coco_api_from_dataset(dataset)
            self.iou_types = ["bbox"]
            self.coco_evaluator = coco_eval.CocoEvaluator(self.coco, self.iou_types)
        images, targets = batch
        images = list(image.to(self.device) for image in images)
        outputs = self.model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        res = {target['image_id'].item(): output for target, output in zip(targets, outputs)}
        self.coco_evaluator.update(res)
        return {}

    def val_test_epoch_end(self, outputs, metrics_prefix):
        self.coco_evaluator.synchronize_between_processes()
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()
        return self.log_metrics(metrics_prefix)

    # def validation_step(self, batch, batch_idx):
    #     return self.val_test_step(batch, batch_idx, 'val_dataset', self.data.val_dataset())

    # def validation_epoch_end(self, outputs):
    #     return self.val_test_epoch_end(outputs, metrics_prefix='val')

    def configure_optimizers(self):
        model_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = common.train.get_optimizer(self.hparams, self.model, model_params)
        lr_scheduler = common.train.get_lr_scheduler(self.hparams, optimizer)
        return optimizer if lr_scheduler is None else ([optimizer], [lr_scheduler])
