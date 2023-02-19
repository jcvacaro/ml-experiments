import random
import numpy as np
import torch
import torchvision

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

# https://pytorch.org/docs/stable/notes/randomness.html
def seed_everything(args):
    if args.seed >= 0:
        print('applying deterministic settings ...')
        torch.manual_seed(args.seed)
        # torch.use_deterministic_algorithms(True)
        random.seed(args.seed)
        np.random.seed(args.seed)
