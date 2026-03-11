from PIL import Image
import timm
import copy
import torch
from .sam import *

def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def _resolve_timm_backbone_name(arch: str) -> str:

    if arch in timm.list_models(arch):
        return arch

    alias_map = {
        "swin_base": "swin_base_patch4_window7_224",
        # "vit_l": "vit_large_patch16_224",
        "swin_large_22k": "swin_large_patch4_window12_384.in22k_ft_in1k"
        # "swin_tiny": "swin_tiny_patch4_window7_224",
        # "swin_small": "swin_small_patch4_window7_224",
        # "swin_large": "swin_large_patch4_window7_224",
    }
    if arch in alias_map:
        return alias_map[arch]

    if arch.startswith("vit_"):
        candidates = timm.list_models(f"{arch}*")
        if candidates:
            for pref in ("patch16_224", "224", "patch32_384", "384"):
                for m in candidates:
                    if pref in m:
                        return m
            return candidates[0]
    if arch.startswith("swin_"):
        candidates = timm.list_models(f"{arch}*")
        if candidates:
            for pref in ("patch4_window7_224", "224", "256", "384"):
                for m in candidates:
                    if pref in m:
                        return m
            return candidates[0]

    raise RuntimeError(f" '{arch}'  timm  error")

def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def set_optimizer(model, args):

    if args.opt == 'adam':
        parameters = add_weight_decay(model, args.weight_decay)

        if args.use_sam and SAM is not None:
            print(f"INFO: Wrapping Adam optimizer with SAM (rho={args.sam_rho}).")
            base_optimizer = torch.optim.Adam
            optimizer = SAM(params=parameters, base_optimizer=base_optimizer, rho=args.sam_rho, lr=args.lr, weight_decay=0)
        else:
            if args.use_sam and SAM is None:
                print("ERROR: --use_sam was specified but sam.py could not be imported.")
            optimizer = torch.optim.Adam(params=parameters, lr=args.lr, weight_decay=0)


    elif args.opt == 'adamw':
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if p.requires_grad]},
        ]

        if args.use_sam and SAM is not None:
            print(f"INFO: Wrapping AdamW optimizer with SAM (rho={args.sam_rho}).")
            base_optimizer = getattr(torch.optim, 'AdamW')
            optimizer = SAM(
                params=param_dicts,
                base_optimizer=base_optimizer,
                rho=args.sam_rho,
                lr=args.lr,
                betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay
            )
        else:
            if args.use_sam and SAM is None:
                print("ERROR: --use_sam was specified but sam.py could not be imported.")
            optimizer = getattr(torch.optim, 'AdamW')(
                param_dicts,
                args.lr,
                betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay
            )

    return optimizer

class SimpleTensorEMA:
    def __init__(self, model, decay=0.999, device=None, copy_buffers=True):

        self.decay = float(decay)
        self.copy_buffers = bool(copy_buffers)
        with torch.no_grad():
            self.ema = copy.deepcopy(model).eval()
            if device is not None:
                self.ema = self.ema.to(device)
            for p in self.ema.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        ema_params = dict(self.ema.named_parameters())
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in ema_params:
                continue
            p_ema = ema_params[name]
            if p_ema.dtype != p.dtype:
                p_ema.data = p_ema.data.to(dtype=p.dtype)
            if p_ema.device != p.device:
                p_ema.data = p_ema.data.to(device=p.device)
            p_ema.mul_(d).add_(p.data, alpha=1.0 - d)

        if self.copy_buffers:
            ema_bufs = dict(self.ema.named_buffers())
            for name, b in model.named_buffers():
                if name in ema_bufs:
                    if ema_bufs[name].device != b.device:
                        ema_bufs[name].data = ema_bufs[name].data.to(b.device)
                    if ema_bufs[name].dtype != b.dtype:
                        ema_bufs[name].data = ema_bufs[name].data.to(b.dtype)
                    ema_bufs[name].copy_(b)

    @property
    def module(self):
        return self.ema

    def state_dict(self):
        return {
            "ema": self.ema.state_dict(),
            "decay": self.decay,
            "copy_buffers": self.copy_buffers,
        }

    def load_state_dict(self, state):
        self.ema.load_state_dict(state["ema"], strict=False)
        self.decay = float(state.get("decay", self.decay))
        self.copy_buffers = bool(state.get("copy_buffers", self.copy_buffers))

