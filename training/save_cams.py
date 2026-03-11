import os
import random
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
import torch.distributed as dist
from PIL import Image
import numpy as np
import torchvision.transforms as T

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib not found. Heatmap overlays will be disabled.")
    plt = None



def _percentile_norm(x: torch.Tensor, lo=2.0, hi=98.0, eps=1e-8):

    vlo = torch.quantile(x.float(), lo / 100.0)
    vhi = torch.quantile(x.float(), hi / 100.0)
    x = (x.float() - vlo) / (vhi - vlo + eps)
    return x.clamp(0, 1)


def _minmax_norm(x: torch.Tensor, eps=1e-8):

    x = x.float()
    x = x - x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    x = x / (x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + eps)
    return x.clamp(0, 1)


def _overlay_heatmap_on_image(img_tensor: torch.Tensor, heatmap: torch.Tensor, alpha=0.5):

    if plt is None:
        raise ImportError("matplotlib is required for heatmap overlay.")

    cmap = plt.get_cmap('jet')
    hm = heatmap.detach().cpu().numpy()
    hm_color = (np.array(cmap(hm))[:, :, :3] * 255).astype(np.uint8)

    base = (img_tensor.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)

    base_pil = Image.fromarray(base)
    hm_pil = Image.fromarray(hm_color)

    blended = Image.blend(base_pil.convert("RGBA"), hm_pil.convert("RGBA"), alpha=alpha).convert("RGB")
    return blended


def compute_vanilla_cam(fmap: torch.Tensor, fc_weight: torch.Tensor, class_idx: int):

    B, C, H, W = fmap.shape
    w = fc_weight[class_idx].view(1, C, 1, 1)
    cam = (fmap.float() * w.float()).sum(dim=1)
    return cam


def compute_proto_cam(fmap: torch.Tensor, proto: torch.Tensor):

    B, C, H, W = fmap.shape
    f = fmap.view(B, C, -1).permute(0, 2, 1)

    f_norm = F.normalize(f.float(), dim=-1)
    p_norm = F.normalize(proto.view(1, 1, C).float(), dim=-1)

    sim = torch.einsum('bhc,khc->bhk', f_norm, p_norm.expand(B, -1, -1))

    sim = sim.squeeze(-1)
    return sim.view(B, H, W)


def save_debug_cams(image_paths, fmap, logits, y, core, epoch, out_dir,
                    num_images=10, per_image_topk=3, args=None, use_proto=True):

    try:
        is_dist = (getattr(args, "distributed", False) and dist.is_initialized())
        is_main = (not is_dist) or dist.get_rank() == 0
    except Exception:
        is_main, is_dist = True, False

    if not is_main or plt is None:
        return

    os.makedirs(out_dir, exist_ok=True)
    B, C, Hf, Wf = fmap.shape
    K = logits.shape[1]

    fc_weight = None
    if hasattr(core, 'classifier') and hasattr(core.classifier, 'weight'):
        fc_weight = core.classifier.weight.detach()
    elif hasattr(core, 'head') and hasattr(core.head, 'weight'):
        fc_weight = core.head.weight.detach()

    idx_all = list(range(B))
    random.shuffle(idx_all)
    sel_idx = idx_all[:min(num_images, B)]

    with torch.no_grad():
        probs = logits.sigmoid()

    MODEL_INPUT_SIZE = args.img_size
    vis_transform = T.Compose([
        T.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])


    for i, b in enumerate(sel_idx):

        original_path = image_paths[b]
        try:
            img_pil = Image.open(original_path).convert('RGB')
        except Exception as e:
            print(f"Warning: [save_cams]  {original_path}. Error: {e}")
            continue

        img_tensor_clean = vis_transform(img_pil)

        fmap_b = fmap[b:b + 1]

        pos_cls_indices = (y[b] > 0).nonzero(as_tuple=True)[0].tolist()

        true_label_names = []
        if args and hasattr(args, 'classes_names') and args.classes_names:
            for idx in pos_cls_indices:
                if 0 <= idx < len(args.classes_names):
                    true_label_names.append(args.classes_names[idx].replace(' ', '_'))


        if not true_label_names:
            true_labels_str = "NoTrueLabels"
        else:
            true_labels_str = "+".join(true_label_names)

        topk_pred = torch.topk(probs[b], k=min(per_image_topk, K)).indices.tolist()

        show_classes = list(dict.fromkeys(pos_cls_indices + topk_pred))

        base_dir = os.path.join(out_dir, f"ep{epoch:03d}_idx{i:02d}_b{b:04d}")
        os.makedirs(base_dir, exist_ok=True)

        base_filename = os.path.basename(original_path)
        filename_without_ext = os.path.splitext(base_filename)[0]

        new_image_filename = f"{filename_without_ext}-{true_labels_str}.jpg"

        to_pil_image(img_tensor_clean.cpu()).save(os.path.join(base_dir, new_image_filename))

        for c in show_classes:
            prob_str = f"p{probs[b, c]:.3f}".replace(".", "_")

            class_name_str = "unknown"
            if args and hasattr(args, 'classes_names') and args.classes_names and 0 <= c < len(args.classes_names):
                class_name_str = args.classes_names[c]
            safe_class_name = class_name_str.replace(' ', '_').replace('/', '_')

            if fc_weight is not None:
                cam = compute_vanilla_cam(fmap_b, fc_weight, c)[0]
                cam_n = _percentile_norm(cam)
                cam_up = F.interpolate(cam_n[None, None], size=img_tensor_clean.shape[-2:], mode='bilinear',
                                       align_corners=False)[0, 0]
                overlay = _overlay_heatmap_on_image(img_tensor_clean, cam_up, alpha=0.5)

                cam_filename = f"vanilla_c{c}_{safe_class_name}_{prob_str}.jpg"
                overlay.save(os.path.join(base_dir, cam_filename))

            if use_proto and hasattr(core, "memory") and hasattr(core.memory, "protos"):
                proto = core.memory.protos[c].detach()
                pcam = compute_proto_cam(fmap_b, proto)[0]

                pcam_n = _percentile_norm(pcam)
                pcam_up = F.interpolate(pcam_n[None, None], size=img_tensor_clean.shape[-2:], mode='bilinear',
                                        align_corners=False)[0, 0]

                overlay = _overlay_heatmap_on_image(img_tensor_clean, pcam_up, alpha=0.5)

                proto_cam_filename = f"proto_c{c}_{safe_class_name}_{prob_str}.jpg"
                overlay.save(os.path.join(base_dir, proto_cam_filename))