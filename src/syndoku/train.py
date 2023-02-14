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

def train(args, callbacks=[]):
    pl.seed_everything(args.seed, workers=True)
    mlflow.pytorch.autolog(log_models=args.save_model)
    mlf_logger = pl.loggers.MLFlowLogger(
        experiment_name=args.experiment_name,
        run_id=args.run_id,
        tracking_uri='file:./mlruns',
    )
    common.train.add_lightning_callbacks(args, callbacks)
    data = SyndokuData(args)
    model = SyndokuDetection(args, data)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=mlf_logger, deterministic=True)
    trainer.fit(model, data)
    return trainer.callback_metrics

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_argparse_args(parser)
    args = parser.parse_args()
    train(args)
