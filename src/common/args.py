import argparse

def add_optuna_argparse_args(parent_parser):
    parser = parent_parser.add_argument_group('optuna')
    parser.add_argument('--study', default='study_0', help='The optuna study and objective function name to be optimized')
    parser.add_argument('--study-dir', default='studies', help='The output directory for storing optuna results')
    parser.add_argument('--study-trials', default=3, type=int, help='The number of trials for optuna')
    parser.add_argument('--study-pruning', action='store_true', help='Activate the pruning feature')
    return parent_parser

def add_training_argparse_args(parent_parser):
    parser = parent_parser.add_argument_group('training')
    parser.add_argument('-s', '--seed', default=33, type=int, help='Fix seed for the experiments')
    parser.add_argument("--discard-model", dest="save_model", help="Do not save the model", action="store_false")
    return parent_parser

def add_model_argparse_args(parent_parser):
    parser = parent_parser.add_argument_group('model')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='The model name')
    return parent_parser

def add_dataset_argparse_args(parent_parser):
    parser = parent_parser.add_argument_group('dataset')
    parser.add_argument('--dataset', help='The dataset type')
    parser.add_argument('--dataset-dir', help='Dataset root directory')
    parser.add_argument('--dataset-train-subset', default=-1, type=int, help='Select a subset of the full training dataset')
    parser.add_argument('--dataset-val-split', default=0.1, type=float, help='Validation split considering the training dataset')
    parser.add_argument('-b', '--batch-size', default=4, type=int, help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    return parent_parser

def add_scheduler_argparse_args(parent_parser):
    parser = parent_parser.add_argument_group('scheduler')
    parser.add_argument('--lr-scheduler', choices=['none', 'steplr', 'multisteplr'], default='none', help='The lr scheduler')
    parser.add_argument('--lr-step-size', default=3, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[3, 7], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    return parent_parser

def add_optimizer_argparse_args(parent_parser):
    parser = parent_parser.add_argument_group('scheduler')
    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='adam', help='The optimizer')
    parser.add_argument('--lr', default=3e-4, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float, metavar='W', help='weight decay (default: 1e-5)', dest='weight_decay')
    return parent_parser

def add_augmentation_argparse_args(parent_parser):
    parser = parent_parser.add_argument_group('augmentations')
    parser.add_argument("--augment-hflip", help="Horizontal flip augmentation", action="store_true")
    parser.add_argument('--hflip-prob', default=0.5, type=float, help='Probability of applying the augmentation')
    parser.add_argument("--augment-vflip", help="Vertical flip augmentation", action="store_true")
    parser.add_argument('--vflip-prob', default=0.5, type=float, help='Probability of applying the augmentation')
    parser.add_argument("--augment-rotate", help="Rotation augmentation", action="store_true")
    parser.add_argument('--rotate-prob', default=0.5, type=float, help='Probability of applying the augmentation')
    parser.add_argument('--rotate-angle', default=90, type=int, help='The rotation angle')
    parser.add_argument("--augment-contrast", help="Image contrast augmentation", action="store_true")
    parser.add_argument('--contrast-prob', default=0.5, type=float, help='Probability for applying contrast augmentation')
    parser.add_argument('--contrast-min', default=0.5, type=float, help='Min value for contrast adjustment')
    parser.add_argument('--contrast-max', default=1.5, type=float, help='Max value for contrast adjustment')
    parser.add_argument("--augment-brightness", help="Image brightness augmentation", action="store_true")
    parser.add_argument('--brightness-prob', default=0.5, type=float, help='Probability for applying brightness augmentation')
    parser.add_argument('--brightness-min', default=0.9, type=float, help='Min value for brightness adjustment')
    parser.add_argument('--brightness-max', default=1.1, type=float, help='Max value for brightness adjustment')
    parser.add_argument("--augment-saturation", help="Image saturation augmentation", action="store_true")
    parser.add_argument('--saturation-prob', default=0.5, type=float, help='Probability for applying saturation augmentation')
    parser.add_argument('--saturation-min', default=0.5, type=float, help='Min value for saturation adjustment')
    parser.add_argument('--saturation-max', default=1.5, type=float, help='Max value for saturation adjustment')
    parser.add_argument("--augment-all", help="Apply all augmentations for training", action="store_true")
    return parent_parser

def add_early_stopping_argparse_args(parent_parser):
    parser = parent_parser.add_argument_group('early stopping')
    parser.add_argument("--early-stopping", help="Early stopping callback", action="store_true")
    parser.add_argument("--es_monitor", type=str, default="metric/val_map_all", help="Early stopping monitor parameter")
    parser.add_argument("--es_mode", type=str, default="min", help="Early stopping mode parameter")
    parser.add_argument("--es_verbose", type=bool, default=True, help="Early stopping verbose parameter")
    parser.add_argument("--es_patience", type=int, default=3, help="Early stopping patience parameter")
    return parent_parser

def add_argparse_args(parser):
    add_model_argparse_args(parser)
    add_dataset_argparse_args(parser)
    add_scheduler_argparse_args(parser)
    add_optimizer_argparse_args(parser)
    add_augmentation_argparse_args(parser)
    add_training_argparse_args(parser)
    add_early_stopping_argparse_args(parser)
    return parser

def update_arg_parser(parser):
    parser.add_argument('-s', '--seed', default=33, type=int, help='Fix seed for the experiments')

    # dataset
    parser.add_argument('--dataset', help='The dataset type')
    parser.add_argument('--train-dataset', help='Dataset for training')
    parser.add_argument('--val-dataset', help='Dataset for validation')
    parser.add_argument('--test-dataset', help='Dataset for testing')
    parser.add_argument('--train-dataset-subset', default=-1, type=int, help='Select a subset of the full training dataset')
    parser.add_argument('--val-dataset-split', default=0.1, type=float, help='Validation split considering the training dataset')

    parser.add_argument('--model', help='The model name')
    parser.add_argument('--device', default='cuda:0', help='The device')
    parser.add_argument('-b', '--batch-size', default=8, type=int, help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--input-size', default=256, type=int, help='Value to resize the input image')
    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='adam', help='The optimizer')
    parser.add_argument('--lr-scheduler', choices=['none', 'steplr', 'multisteplr'], default='none', help='The lr scheduler')
    parser.add_argument('--lr', default=3e-4, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float, metavar='W', help='weight decay (default: 1e-5)', dest='weight_decay')
    parser.add_argument('--lr-step-size', default=3, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[3, 7], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    parser.add_argument("--discard-model", dest="save_model", help="Do not save the model", action="store_false")
    parser.add_argument("--pretrained", help="Use pre-trained models from the modelzoo", action="store_true")

    # augmentation flags
    parser.add_argument("--augment-hflip", help="Horizontal flip augmentation", action="store_true")
    parser.add_argument('--hflip-prob', default=0.5, type=float, help='Probability of applying the augmentation')
    parser.add_argument("--augment-vflip", help="Vertical flip augmentation", action="store_true")
    parser.add_argument('--vflip-prob', default=0.5, type=float, help='Probability of applying the augmentation')
    parser.add_argument("--augment-rotate", help="Rotation augmentation", action="store_true")
    parser.add_argument('--rotate-prob', default=0.5, type=float, help='Probability of applying the augmentation')
    parser.add_argument("--augment-contrast", help="Image contrast augmentation", action="store_true")
    parser.add_argument('--min-contrast', default=0.5, type=float, help='Min value for contrast adjustment')
    parser.add_argument('--max-contrast', default=1.5, type=float, help='Max value for contrast adjustment')
    parser.add_argument("--augment-saturation", help="Image saturation augmentation", action="store_true")
    parser.add_argument('--min-saturation', default=0.5, type=float, help='Min value for saturation adjustment')
    parser.add_argument('--max-saturation', default=1.5, type=float, help='Max value for saturation adjustment')
    parser.add_argument("--augment-all", help="Apply all augmentations for training", action="store_true")
