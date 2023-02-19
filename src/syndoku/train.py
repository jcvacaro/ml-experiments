from argparse import ArgumentParser
import math
import mlflow
import mlflow.pytorch
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import common
import common.args
import common.train
from data import SyndokuData
import model as SyndokuModel

def add_argparse_args(parser):
    common.args.add_argparse_args(parser)
    SyndokuModel.add_argparse_args(parser)
    return parser

def train(args):
    mlflow.log_params(args.__dict__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    common.train.seed_everything(args)
    data = SyndokuData(args)
    model = SyndokuModel.create_model(args, data).to(device)
    model_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = common.train.get_optimizer(args, model, model_params)
    lr_scheduler = common.train.get_lr_scheduler(args, optimizer)

    start_epoch = 0
    if args.run_id:
        checkpoint = mlflow.pytorch.load_model(f'runs:/{args.run_id}/checkpoint')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if lr_scheduler:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1

    print('Start training using device:', device)
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(args, model, optimizer, data.train_dataloader(), device, epoch)
        if lr_scheduler:
            lr_scheduler.step()

        # evaluate after every epoch
        metrics = evaluate(args, model, data.val_dataloader(), device)
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v), step=epoch)
        print(f'[Epoch={epoch}] train_loss:{train_loss}; map:{metrics["map"]};')

        # save checkpoint
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler else None,
            "epoch": epoch,
        }
        mlflow.pytorch.log_model(model, 'checkpoint')

    # done!
    print('training done')

def train_one_epoch(args, model, optimizer, data_loader, device, epoch):
    model.train()

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    loss_total, loss_count = 0.0, 0.0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        loss_count += len(images)
        loss_total += loss_value

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(losses)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

    return loss_total / loss_count

@torch.inference_mode()
def evaluate(args, model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision()
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(images)
        metric.update(outputs, targets)
    return metric.compute()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_argparse_args(parser)
    args = parser.parse_args()
    with mlflow.start_run(run_id=args.run_id):
        train(args)
