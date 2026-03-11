import argparse
import models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--data', type=str, metavar='DIR', default='data',
                    help='path to dataset')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet101',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--img_size', default=448, type=int, metavar='N',help='input image size')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-pf', '--prefetch_factor', default=4, type=int, metavar='N',
                    help='number of samples loaded in advance by each worker (default: 2)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--weight-decay', default=0.05, type=float,
                    metavar='W', help='weight decay (default: 0.05)',
                    dest='weight_decay')
parser.add_argument('--cos', '--cosine_lr', default=1, type=int,
                    metavar='COS', help='lr decay by decay', dest='cos')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('-p', '--print-freq', default=500, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--use_amp', type=bool, default=False,
                    help='Use Automatic Mixed Precision (AMP) training. Set to False to use FP32.')
parser.add_argument('--grad_accum', default=1, type=int,help='number of gradient accumulation steps')

# ============================================================================
#
# ============================================================================
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw", can be "sgd")')
parser.add_argument('--adamw_lr', type=float, default=5e-4, metavar='LR',
                    help='learning rate for AdamW (default: 1e-4)')
parser.add_argument('--adamw_betas', type=float, nargs=2, default=[0.9, 0.999], metavar='BETA',
                    help='betas for AdamW optimizer')
parser.add_argument('--adamw_eps', type=float, default=1e-8, metavar='EPSILON',
                    help='epsilon for AdamW optimizer')
parser.add_argument('--clip-grad', type=float, default=1.0,
                    help='Max gradient norm (default: 5.0)')
parser.add_argument('--use_sam', action='store_true',
                    help='Use Sharpness-Aware Minimization (SAM) optimizer.')
parser.add_argument('--sam_rho', type=float, default=0.02,
                    help='SAM rho hyperparameter (default: 0.05)')
parser.add_argument('--lrwarmup_epo', type=int, default=5,
                    help='Number of warmup epochs for the learning rate')

# ============================================================================
#
# ============================================================================
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--eval_freq', default=1, type=int, help='frequency for evaluation')
parser.add_argument('--pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=3, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--ngpus_per_node', default=2, type=int, help='number of GPUs per node')
parser.add_argument('--nprocs_per_node', default=2, type=int, help='number of processes per node')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# ============================================================================
#
# ============================================================================
parser.add_argument('--log_base',
                    default='./resultslocal_resnet_1108', type=str, metavar='PATH',
                    help='path to save logs (default: none)')
parser.add_argument('--method_name', default='Ours', type=str,help='name of the method')

# ============================================================================
#
# ============================================================================
parser.add_argument ('--dataset', type=str, default="sewer-ml", help = '')
parser.add_argument ('--classes_num', type=int, default=17, help = 'number of epoch for lambda to decay')
# parser.add_argument ('--classes_num', type=int, default=80, help = 'number of epoch for lambda to decay')
parser.add_argument('--classes_names', type=list, default= ['RB', 'OB', 'PF', 'DE', 'FS', 'IS', 'RO', 'IN', 'AF', 'BE', 'FO', 'GR', 'PH', 'PB', 'OS', 'OP', 'OK'], help='list of classes names')
# parser.add_argument('--classes_names', type=list, default= ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
#     'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
#     'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
#     'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
#     'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
#     'skateboard', 'surfboard', 'tennis racket',
#     'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
#     'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#     'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
#     'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
#     'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
#     'hair drier', 'toothbrush'], help='list of classes names')
parser.add_argument ('--n_feature', type=int, default=48, help = 'number of pre-saved features')
parser.add_argument ('--feature_dim', type=int, default=512, help = 'the dim of each feature')
parser.add_argument ('--n_levels', type=int, default=1, help = 'number of global table levels')
parser.add_argument('--sum', type=bool, default=True, help='sum or concat')
parser.add_argument('--concat', type=int, default=1, help='sum or concat')
parser.add_argument('--min_scale', type=float, default=0.8, help='')
parser.add_argument('--presave_ratio', type=float, default=0.9, help='the ratio for presaving features')

# ============================================================================
#
# ============================================================================
parser.add_argument ('--second_lambda', type=float, default=0.5, help = 'weight lambda for second order moment loss')
parser.add_argument ('--third_lambda', type=float, default=0.25, help = 'weight lambda for second order moment loss')
parser.add_argument('--proto_momentum', type=float, default=0.99, help='momentum for proto updating')
parser.add_argument('--proto_write_start', type=int, default=5, help='Epoch to start writing to prototype memory.')
parser.add_argument('--update_thresh', type=float, default=0.85, help='Confidence threshold for updating prototypes.')
parser.add_argument('--proto_topq_warm', type=float, default=0.30, help='Top-q quantile of positive samples for warmup batch prototypes.')
parser.add_argument('--proto_min_pos', type=int, default=1, help='Minimum positive samples required to form a batch prototype.')
parser.add_argument('--proto_sim_gate', type=float, default=0.0, help='Similarity gate threshold; 0 means disabled.')
parser.add_argument('--proto_min_count_for_sim', type=int, default=5, help='Min count for a class to enable similarity gating.')
parser.add_argument('--proto_cap_per_class', type=int, default=8, help='Max number of features per class per batch to update memory.')

# ============================================================================
# SCU/CAU
# ============================================================================
parser.add_argument('--scu_proto_blend_start', type=int, default=10, help='SCU blend start epoch.')
parser.add_argument('--scu_proto_blend_end', type=int, default=30, help='SCU blend end epoch.')
parser.add_argument('--cau_proto_blend_start', type=int, default=10, help='CAU blend start epoch.')
parser.add_argument('--cau_proto_blend_end', type=int, default=30, help='CAU blend end epoch.')
parser.add_argument('--prior_smooth_alpha', type=float, default=1.0, help='Alpha for prior smoothing (Dirichlet/Laplace).')
parser.add_argument('--delay_scu_epochs', type=int, default=10, help='Epoch to start delay scu.')
parser.add_argument('--delay_cau_epochs', type=int, default=10, help='Epoch to start delay cau.')
parser.add_argument('--causal_tau', type=float, default=0.5, help='Temperature coefficient for causal attention mechanism.')

# ============================================================================
#
# ============================================================================
parser.add_argument("--sanity", type=str, default="person,dog", help="Comma-separated class names to check, e.g. 'person,dog'. If set, validate() will print a single-sample sanity report on rank0.")
