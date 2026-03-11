import math
import os
import random
import datetime
import json
import warnings
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.cuda.amp import GradScaler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandAugment
import timm
from timm.utils import ModelEmaV2
from timm.loss import AsymmetricLossMultiLabel
from utilis.helper import _resolve_timm_backbone_name, SimpleTensorEMA, set_optimizer
import models
from ops.config import parser
from training.schedule import lr_setter
from training.train import train
from training.validate import validate
from utilis.meters import AverageMeter
from utilis.saving import save_checkpoint
from sewer_dataset import SewerMLDataset, collate_fn
from models.resnet_with_table import GenericBackboneWithTable
from utilis.sam import SAM

best_mAP = 0

warnings.filterwarnings(
    "ignore",
    message="Palette images with Transparency expressed in bytes should be converted to RGBA"
)

def main():
    args = parser.parse_args()
    if args.dataset == "sewer-ml":
        args.classes_num = 17
    elif args.dataset.lower() in ["coco", "mscoco", "ms-coco"]:
        args.classes_num = 80

    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{now}_{args.method_name}"
    args.log_path = os.path.join(args.log_base, args.dataset, exp_name)
    os.makedirs(args.log_path, exist_ok=True)

    args.log_file_txt = os.path.join(args.log_path, "log.txt")
    args.results_file_csv = os.path.join(args.log_path, "results.csv")
    print(f"[INFO] experiment log saved in: {args.log_path}")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True


    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.gpu is not None and not args.distributed:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()

    main_worker(ngpus_per_node, args)



def main_worker(ngpus_per_node, args):


    global best_mAP

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if "LOCAL_RANK" in os.environ and args.gpu is None:
            args.gpu = int(os.environ["LOCAL_RANK"])
        if "RANK" in os.environ:
            args.rank = int(os.environ["RANK"])
        else:
            args.rank = args.gpu if args.gpu is not None else 0

        torch.cuda.set_device(args.gpu)
        dist.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    else:
        args.rank = 0

    if args.rank == 0:
        config_path = os.path.join(args.log_path, "config.json")
        try:
            with open(config_path, 'w') as f:
                config_dict = vars(args)
                serializable_config = {}
                for k, v in config_dict.items():
                    if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                        serializable_config[k] = v
                    else:
                        serializable_config[k] = str(v)
                json.dump(serializable_config, f, indent=2)
            print(f"Config saved in {config_path}")
        except Exception as e:
            print(f"[WARN] saving config.json failed: {e}")

    header = "epoch,mAP,CP,CR,CF1,CP_threshold,OP,OR,OF1,OP_threshold,best_f1\n"
    try:
        with open(args.results_file_csv, 'w') as f:
            f.write(header)
    except Exception as e:
        print(f"[WARN] create results.csv failed: {e}")

    # ==================== 1. dataset setup ====================
    if args.rank == 0:
        print("==> Preparing dataset...")
    img_size = args.img_size

    if args.dataset.lower() in ["sewer-ml", "sewerml", "sewer"]:
        # ========== SewerML ==========
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
        ])

        eval_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
        ])

        annotations_dir = '/path/to/your/SewerML/annotations'
        data_dir = '/path/to/your/SewerML/Data/Root'

        train_csv_path = os.path.join(annotations_dir, 'SewerML_Train.csv')
        val_csv_path = os.path.join(annotations_dir, 'SewerML_Val.csv')
        test_csv_path = os.path.join(annotations_dir, 'SewerML_Test.csv')

        train_dataset = SewerMLDataset(csv_file=train_csv_path, root_dir=data_dir, split='Train', transform=train_transform, classes_names = args.classes_names)
        val_dataset = SewerMLDataset(csv_file=val_csv_path, root_dir=data_dir, split='Val', transform=eval_transform,
                                     classes_names=args.classes_names)
        test_dataset = SewerMLDataset(csv_file=test_csv_path, root_dir=data_dir, split='Test', transform=eval_transform,
                                      classes_names=args.classes_names)

        num_classes = train_dataset.num_classes
        assert num_classes == len(args.classes_names)
        used_collate = collate_fn
    else:
        raise ValueError(f"unsupported: {args.dataset}")
    if args.rank == 0:
        print(f"having {num_classes} labels/classes。")
        print("args.arch =", args.arch)
    # ==================== 2. model setup ====================
    if args.rank == 0:
        print("==> Building model...")
    if args.arch.startswith('swin_') or args.arch.startswith('vit_'):
        resolved = _resolve_timm_backbone_name(args.arch)
        print(f"=> using timm features_only backbone '{resolved}' (pretrained={args.pretrained})")
        backbone = timm.create_model(
            resolved, pretrained=args.pretrained, features_only=True, out_indices=[-1], img_size=img_size, pretrained_cfg_overlay=dict(input_size=(3, img_size, img_size)),
        )
        pe = getattr(backbone, 'patch_embed', None) or getattr(getattr(backbone, 'model', None), 'patch_embed', None)
        if pe is not None:
            pe.img_size = (img_size, img_size)
            if hasattr(pe, 'strict_img_size'):
                pe.strict_img_size = False
        try:
            feat_dim = backbone.feature_info.channels()[-1]
        except Exception:
            if args.arch.startswith('swin_large'):
                feat_dim = 1536  # Swin-Large
                print(f"WARNING: fallback feat_dim={feat_dim} for Swin-Large")
            elif args.arch.startswith('swin_base'):
                feat_dim = 1024  # Swin-Base
                print(f"WARNING: fallback feat_dim={feat_dim} for Swin-Base")
            else:
                feat_dim = 1024
                print(f"WARNING: unknown swin arch, fallback feat_dim={feat_dim}")

        model = GenericBackboneWithTable(backbone, feat_dim, num_classes, args)

    else:
        if args.pretrained:
            if args.rank == 0:
                print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True, args=args, num_classes=args.classes_num)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=False, args=args, num_classes=args.classes_num)


    # ==================== 3. GPU setup ====================
    if args.distributed and (args.gpu is None):
        if "LOCAL_RANK" in os.environ:
            args.gpu = int(os.environ["LOCAL_RANK"])

    if args.distributed:
        assert args.gpu is not None, "Please set the LOCAL_RANK"
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)

        args.batch_size = int(args.batch_size / ngpus_per_node)

        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            output_device=args.gpu,
            find_unused_parameters=False,
            broadcast_buffers=False,
            gradient_as_bucket_view=True
        )
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    real_model = model.module if hasattr(model, "module") else model

    def _params(m):
        trainable_params = []
        if m is not None:
            for p in m.parameters():
                p.requires_grad_(True)
                trainable_params.append(p)
        return trainable_params

    use_adamw = args.opt.lower() == "adamw"
    base_lr_main = args.adamw_lr if use_adamw else args.lr

    if args.arch.startswith('swin_large'):
        mult = 0.05
    else:
        mult = 0.1
    lr_mult = {
        "backbone": mult,
        "classifier": 1.0,
        "causal": 0.5,
    }

    param_groups = []

    if hasattr(real_model, "backbone"):
        if args.arch.startswith('vit_'):
            p = _params(real_model.backbone)
            if p:
                param_groups.append({
                    "params": p,
                    "lr": 1e-5,
                    "weight_decay": 1e-2,
                    "name": "backbone",
                })
        elif args.arch.startswith('swin_'):
            p = _params(real_model.backbone)
            if p:
                param_groups.append({
                    "params": p,
                    "lr": base_lr_main * lr_mult["backbone"],
                    "weight_decay": args.weight_decay,
                    "name": "backbone",
                })
        else:
            p = _params(real_model.backbone)
            if p:
                param_groups.append({
                    "params": p,
                    "lr": base_lr_main * lr_mult["backbone"],
                    "weight_decay": args.weight_decay,
                    "name": "backbone",
                })

    combined_classifier_params = []
    if hasattr(real_model, "fc1"): combined_classifier_params.extend(_params(real_model.fc1))
    if hasattr(real_model, "prototype_attention_modules"): combined_classifier_params.extend(
        _params(real_model.prototype_attention_modules))
    if combined_classifier_params: param_groups.append(
        {"params": combined_classifier_params, "lr": base_lr_main * lr_mult["classifier"],
         "weight_decay": args.weight_decay, "name": "classifier_proto"})

    delay_causal_start_epoch = getattr(args, "delay_cau_epochs", 10)
    initial_causal_lr = 0.0 if args.start_epoch < delay_causal_start_epoch else base_lr_main * lr_mult["causal"]
    target_causal_lr = base_lr_main * lr_mult["causal"]
    if hasattr(real_model, "causal_module"):
        p = _params(real_model.causal_module)
    else:
        p = []
    if p: param_groups.append(
        {"params": p, "lr": initial_causal_lr, "weight_decay": args.weight_decay, "name": "causal",
         "base_lr": target_causal_lr})
    use_amp = args.use_amp and torch.cuda.is_available()
    scaler = torch.amp.GradScaler(enabled=use_amp)
    if not use_amp and args.rank == 0:
        print("INFO: Automatic Mixed Precision (AMP) is DISABLED. Training will use FP32.")

    ema_device = torch.device(f"cuda:{args.gpu}") if (args.gpu is not None and torch.cuda.is_available()) else None
    ema_decay = getattr(args, "ema_decay", 0.9997)
    model_ema = SimpleTensorEMA(real_model, decay=ema_decay, device=ema_device)

    # ==================== 4. loss function setup ====================
    class_weights = train_dataset.class_weights
    with torch.no_grad():
        class_weights = torch.clamp(class_weights, min=1.0, max=10.0)
    device = torch.device(f"cuda:{args.gpu}" if (
                args.gpu is not None and torch.cuda.is_available()) else "cuda" if torch.cuda.is_available() else "cpu")

    if class_weights is not None:
        class_weights = class_weights.to(device)

    if args.rank == 0:
        print("Using class weights to initialize ASL.")
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights).to(device)

    criterion_train = AsymmetricLossMultiLabel(
        gamma_pos=0.0, gamma_neg=4.0, clip=0.05, disable_torch_grad_focal_loss=True)

    if use_adamw:
        if args.rank == 0:
            print("=> using AdamW as base optimizer CLASS")

        base_optimizer_class = torch.optim.AdamW

        optimizer_kwargs = {
            'betas': tuple(args.adamw_betas),
            'eps': args.adamw_eps,
            'weight_decay': args.weight_decay
        }
    else:
        if args.rank == 0:
            print("=> using SGD as base optimizer CLASS")

        base_optimizer_class = torch.optim.SGD

        optimizer_kwargs = {
            'momentum': args.momentum,
            'weight_decay': args.weight_decay
        }

    if args.rank == 0:
        print(f"=> Wrapping with SAM (rho={args.sam_rho})")

    optimizer = SAM(param_groups,
                    base_optimizer_class,
                    rho=args.sam_rho,
                    **optimizer_kwargs)

    for pg in optimizer.param_groups:
        if "base_lr" not in pg:
            pg["base_lr"] = pg["lr"]
        if args.rank == 0: print(
            f"[PG Defined] {pg.get('name', '?')}: initial_lr={pg['lr']:.6g}, base_lr={pg['base_lr']:.6g}, n_params={len(pg['params'])}")

    t_max_cosine = args.epochs - args.lrwarmup_epo

    eta_min_cosine = args.adamw_lr * 0.01

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer.base_optimizer,
        T_max=t_max_cosine,
        eta_min=eta_min_cosine
    )

    if args.rank == 0:
        print(f"INFO: Using CosineAnnealingLR scheduler starting after epoch {args.lrwarmup_epo}.")
        print(f"      T_max={t_max_cosine}, eta_min={eta_min_cosine:.2e}")


    # ==================== 5. Checkpoint ====================
    if args.resume:
        assert os.path.isfile(args.resume), f"no checkpoint at {args.resume}"
        map_loc = f'cuda:{args.gpu}' if args.gpu is not None else 'cpu'

        if args.rank == 0 or not args.distributed:
            print(f"=> loading checkpoint '{args.resume}'")

        checkpoint = torch.load(args.resume, map_location=map_loc)

        args.start_epoch = int(checkpoint.get('epoch', 0))
        best_f1 = float(checkpoint.get('best_f1', 0.0))

        missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
        if (args.rank == 0 or not args.distributed) and (missing or unexpected):
            print(f"=> [warn] missing keys: {missing}, unexpected keys: {unexpected}")

        optimizer.load_state_dict(checkpoint['optimizer'])

        if 'ema_state_dict' in checkpoint:
            model_ema.load_state_dict(checkpoint['ema_state_dict'])
            if args.rank == 0:
                print("=> loaded EMA model state.")
        else:
            if args.rank == 0:
                print("=> [warn] EMA model state not found in checkpoint, re-initializing.")

        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])

        for pg in optimizer.param_groups:
            group_name = pg.get("name", "?")
            target_lr = 0.0
            if group_name == "backbone":
                target_lr = base_lr_main * lr_mult["backbone"]
            elif group_name == "classifier_proto":
                target_lr = base_lr_main * lr_mult["classifier"]
            elif group_name == "causal":
                target_lr = base_lr_main * lr_mult["causal"]
            pg.setdefault('base_lr', target_lr)

        if args.distributed:
            dist.barrier()

        if args.rank == 0 or not args.distributed:
            print(f"=> loaded checkpoint '{args.resume}' (epoch {args.start_epoch})")

    # ==================== 6. DataLoader ====================
    if args.distributed:
        print("initializing distributed sampler")
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, prefetch_factor=args.prefetch_factor, persistent_workers=True,
        pin_memory=True, sampler=train_sampler, drop_last=True, collate_fn=used_collate
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, prefetch_factor=args.prefetch_factor, persistent_workers=True,
        pin_memory=True, sampler=val_sampler, drop_last=False, collate_fn=used_collate
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, prefetch_factor=args.prefetch_factor, persistent_workers=True,
        pin_memory=True, sampler=test_sampler, drop_last=False, collate_fn=used_collate
    )


    # ==================== 7. training ====================
    log_dir = os.path.dirname(args.log_path)
    print('tensorboard dir {}'.format(log_dir))
    tensor_writer = None
    if args.rank == 0:
        tensor_writer = SummaryWriter(log_dir)
    if args.evaluate:
        validate(test_loader, model, criterion, 0, True, args, tensor_writer)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        lr_scale = 1.0
        if epoch < args.lrwarmup_epo:
            lr_scale = float(epoch + 1) / float(args.lrwarmup_epo)

        for pg in optimizer.param_groups:
            group_name = pg.get("name", "?")
            target_base_lr = pg.get("base_lr", pg["lr"])

            if group_name == "causal" and epoch == delay_causal_start_epoch and pg["lr"] == 0.0:
                if args.rank == 0: print(
                    f"==> Activating LR for group '{group_name}' to {target_base_lr:.6g} at epoch {epoch}")
                pg["lr"] = target_base_lr
            if epoch < args.lrwarmup_epo:
                pg['lr'] = target_base_lr * lr_scale
            elif epoch == args.lrwarmup_epo and pg['lr'] != target_base_lr:
                if args.rank == 0 and pg['lr'] > 0:
                    print(f"==> Setting LR for group '{group_name}' to base_lr {target_base_lr:.6g} after warmup")
                pg['lr'] = target_base_lr

        train(train_loader, model, criterion_train, optimizer, epoch, args,
              tensor_writer=tensor_writer, model_ema=model_ema, grad_accum=getattr(args, "grad_accum", 1),
              scaler=scaler, scheduler=scheduler)
        eval_model = model_ema.module if model_ema is not None else model

        val_metrics = validate(val_loader, eval_model, criterion, epoch, False, args, tensor_writer)
        val_mAP = val_metrics.get('mAP', 0.0)

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            test_metrics = validate(test_loader, eval_model, criterion, epoch, True, args, tensor_writer)
            test_mAP = test_metrics.get('mAP', 0.0)
            if tensor_writer is not None and args.rank == 0:
                tensor_writer.add_scalar('metrics/test_mAP', test_mAP, epoch)
        if epoch >= args.lrwarmup_epo:
            scheduler.step()

        if args.rank == 0:
            is_best = val_mAP > best_mAP
            best_f1 = max(val_mAP, best_mAP)

            try:
                with open(args.results_file_csv, 'a') as f:

                    f.write(f"{epoch},"
                            f"{val_metrics.get('mAP', 0):.4f},"
                            f"{val_metrics.get('CP', 0):.4f},"
                            f"{val_metrics.get('CR', 0):.4f},"
                            f"{val_metrics.get('CF1', 0):.4f},"
                            f"{val_metrics.get('CP_threshold', 0):.2f},"
                            f"{val_metrics.get('OP', 0):.4f},"
                            f"{val_metrics.get('OR', 0):.4f},"
                            f"{val_metrics.get('OF1', 0):.4f},"
                            f"{val_metrics.get('OP_threshold', 0):.2f},"
                            f"{best_f1:.4f}\n")
            except Exception as e:
                print(f"[WARN]  results.csv failed: {e}")

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'ema_state_dict': model_ema.state_dict(),
                'best_f1': best_f1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.log_path, epoch)

    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
