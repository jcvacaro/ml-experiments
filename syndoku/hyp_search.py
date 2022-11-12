import argparse
from pathlib import Path
import numpy as np
import torch
import optuna
from optuna.integration import PyTorchLightningPruningCallback

import guia_mlutils as G
from train import train, add_argparse_args

METRIC = 'metric/val_map_all'

def create_args(save_model=False):
    parser = argparse.ArgumentParser(description=__doc__)
    add_argparse_args(parser)
    G.args.add_optuna_argparse_args(parser)
    args = parser.parse_args()
    args.save_model = save_model
    return args

def run(trial, args, metric=METRIC):
    callbacks = [PyTorchLightningPruningCallback(trial, monitor=metric)] if args.study_pruning else []
    metrics = train(args, experiment_name=args.study, callbacks=callbacks)
    return -metrics[metric]

def objective_augmentation(trial):
    args = create_args()
    args.dataset_dir = '/data'
    #args.model = 'fcn_resnet50'
    args.augment_contrast = trial.suggest_uniform('augment_contrast', 0, 1) > 0.5
    args.augment_saturation = trial.suggest_uniform('augment_saturation', 0, 1) > 0.5
    args.augment_hflip = trial.suggest_uniform('augment_hflip', 0, 1) > 0.5
    args.augment_vflip = trial.suggest_uniform('augment_vflip', 0, 1) > 0.5
    args.augment_rotate = trial.suggest_uniform('augment_rotate', 0, 1) > 0.5
    return run(trial, args)

def objective_augmentation_photometric(trial):
    args = create_args()
    args.dataset_dir = '/data'
    args.augment_contrast = trial.suggest_uniform('augment_contrast', 0, 1) > 0.5
    args.augment_brightness = trial.suggest_uniform('augment_brightness', 0, 1) > 0.5
    args.augment_saturation = trial.suggest_uniform('augment_saturation', 0, 1) > 0.5
    return run(trial, args)

if __name__ == '__main__':
    args = create_args()
    if args.study_dir:
        Path(args.study_dir).mkdir(parents=True, exist_ok=True)
    pruner = optuna.pruners.MedianPruner() if args.study_pruning else optuna.pruners.NopPruner()
    study = optuna.create_study(
        study_name=args.study, 
        storage=f'sqlite:///{args.study_dir}/{args.study}.db',
        load_if_exists=True, 
        pruner=pruner
    )
    study.optimize(globals()[args.study], n_trials=args.study_trials)
