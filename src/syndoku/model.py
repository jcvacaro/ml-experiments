import torchvision

def add_argparse_args(parent_parser):
    parser = parent_parser.add_argument_group('syndoku-detection-model')
    parser.add_argument('--weights', default='FasterRCNN_ResNet50_FPN_Weights.COCO_V1', help='The model weights')
    parser.add_argument('--weights-backbone', default='ResNet50_Weights.IMAGENET1K_V1', help='The model backbone weights')
    parser.add_argument("--trainable-backbone-layers", default=None, type=int, help='number of trainable layers of backbone')
    parser.add_argument('--rpn-score-thresh', default=None, type=float, help='rpn score threshold for faster-rcnn')
    return parent_parser

def create_model(args, data):
    kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers}
    if "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh
    return torchvision.models.get_model(
        args.model,
        # weights=args.weights,
        weights=None,
        weights_backbone=args.weights_backbone,
        num_classes=data.num_classes + 1,  # background class
        **kwargs
    )
