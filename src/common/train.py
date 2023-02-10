import random
import numpy as np
import torch
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

def get_optimizer(args, model, params):
    if args.optimizer == 'sgd':
        return torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.optimizer == 'adam':
        return torch.optim.Adam(params, lr=args.lr)
    raise ValueError('not supported optimizer', args.optimizer)

def get_lr_scheduler(args, optimizer):
    if args.lr_scheduler == 'steplr':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    if args.lr_scheduler == 'multisteplr':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    return None

def add_lightning_callbacks(args, callbacks):
    callbacks.append(LearningRateMonitor())
    if args.early_stopping:
        callbacks.append(EarlyStopping(
            monitor=args.es_monitor,
            mode=args.es_mode,
            verbose=args.es_verbose,
            patience=args.es_patience,
        ))
