from argparse import ArgumentParser
import os
import torch
import torchvision
import torchvision.models.detection
import pytorch_lightning as pl
import mlflow
import mlflow.pytorch

import common
import common.args
import common.train
from data import SyndokuData
from model import SyndokuDetection

def add_argparse_args(parser):
    common.args.add_argparse_args(parser)
    SyndokuDetection.add_argparse_args(parser)
    pl.Trainer.add_argparse_args(parser)
    return parser

def train(args, experiment_name='default', callbacks=[]):
    mlflow.pytorch.autolog(log_models=args.save_model)
    mlf_logger = pl.loggers.MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri='file:./mlruns',
    )
    common.train.add_lightning_callbacks(args, callbacks)
    data = SyndokuData(args)
    model = SyndokuDetection(args, data)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=mlf_logger)
    trainer.fit(model, data)
    # return trainer.callback_metrics

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_argparse_args(parser)
    args = parser.parse_args()
    train(args)
