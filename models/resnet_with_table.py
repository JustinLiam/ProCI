import torch
import torch.nn as nn
from .model_utilis import load_state_dict_from_url
import timm
import math
from .memory import ProtoMemory
import torch.nn.functional as F
from causal import CausalInterventionModule

__all__ = ['ResNet_with_table', 'resnet18_with_table', 'resnet34_with_table', 'resnet50_with_table', 'tresnet_l', 'convnext_base', 'swin_base', 'resnet101']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet18_with_table': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet50_with_table': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
    'tresnet_l': 'https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/tresnet/tresnet_l.pth'
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def log_softmax_sim(z, p, tau: float = 0.1):
    z = F.normalize(z, p=2, dim=-1)
    p = F.normalize(p, p=2, dim=-1)
    sims = torch.einsum('...d,nd->...n', z, p) / max(tau, 1e-6)
    return sims - torch.logsumexp(sims, dim=-1, keepdim=True)

class PrototypeAttention(nn.Module):
    def __init__(self, feature_dim, attention_dim=128):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, features):
        attention_logits = self.attention_net(features)
        attention_weights = F.softmax(attention_logits, dim=0)
        weighted_prototype = (features * attention_weights).sum(dim=0)
        return weighted_prototype, attention_weights.squeeze()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_with_table(nn.Module):

    def __init__(self, block, layers, num_classes=17, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, args=None):
        super(ResNet_with_table, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * block.expansion, num_classes, bias=False)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)





    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def compute_class_features_from_map(self, map):

        B, C, H, W = map.shape
        W_fc = self.fc1.weight
        CAM = torch.einsum('kc,bchw->bkhw', W_fc, map)
        A = CAM.clamp_min(0)
        denom = A.sum(dim=(2,3), keepdim=True).clamp_min(1e-6)
        A = A / denom
        Z = (map.unsqueeze(1) * A.unsqueeze(2)).sum(dim=(3,4))
        return Z

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        fmap=x
        x = self.avgpool(x)
        t=x
        x = torch.flatten(x, 1)
        flatten_features = x
        logits = self.fc1(x)


        return logits, flatten_features,t,fmap

    def forward(self, x):
        return self._forward_impl(x)


class GenericBackboneWithTable(nn.Module):

    def __init__(self, backbone: nn.Module, feat_dim: int, classes_num: int, args):
        super().__init__()
        self.backbone = backbone

        if hasattr(self.backbone, "feature_info") and \
                hasattr(self.backbone, "features_only") and \
                getattr(self.backbone, "features_only"):
            try:
                inferred_dim = self.backbone.feature_info.channels()[-1]
                if inferred_dim is not None:
                    feat_dim = inferred_dim
            except Exception:
                pass

        self.num_features = feat_dim
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.num_features, classes_num, bias=True)



        classes_names = getattr(args, "classes_names", None)
        if classes_names is None:
            classes_names = list(range(classes_num))
        assert len(classes_names) == classes_num, \
            f"class_names length {len(classes_names)} must equal classes_num={classes_num}"
        self.memory = ProtoMemory(classes=classes_names, D=self.num_features)

        self.causal_module = CausalInterventionModule(feature_dim=self.num_features)

        self.prototype_attention_modules = nn.ModuleList(
            [PrototypeAttention(feature_dim=self.num_features) for _ in range(classes_num)]
        )


    def compute_class_features_from_map(
            self,
            fmap: torch.Tensor,
            *,
            relu_cam: bool = True,
            normalize: str = "softmax",
            detach_map: bool = False,
    ) -> torch.Tensor:

        fmap_in = fmap.detach() if detach_map else fmap
        W_fc = self.fc1.weight

        cam = torch.einsum('kc,bchw->bkhw', W_fc, fmap_in)

        if relu_cam:
            cam = F.relu(cam, inplace=False).float()
        else:
            cam = cam.float()

        B, K, H, W = cam.shape

        if normalize == "softmax":
            a = cam.view(B, K, H * W)
            a = a - a.max(dim=-1, keepdim=True).values
            a = torch.softmax(a, dim=-1).view(B, K, H, W)
        elif normalize == "sigmoid":
            a = torch.sigmoid(cam)
            a = a / (a.sum(dim=(2, 3), keepdim=True) + 1e-6)
        else:
            denom = cam.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            a = cam / denom

        Z = (fmap_in.float().unsqueeze(1) * a.unsqueeze(2)).sum(dim=(3, 4))
        return Z

    def _forward_features_only(self, x: torch.Tensor) -> torch.Tensor:

        feats = self.backbone(x)
        if isinstance(feats, (list, tuple)):
            fmap = feats[-1]
        else:
            fmap = feats
        if fmap.dim() != 4:
            raise RuntimeError(f"features_only  4D fmap,  {fmap.dim()}D, shape={tuple(fmap.shape)}")
        return fmap

    def forward(self, x):

        out = self.backbone(x)
        if isinstance(out, (list, tuple)):
            if len(out) == 0:
                raise RuntimeError("backbone(x) ")
            fmap = out[-1]  # 取最后一层 [B,C,H,W]
        elif isinstance(out, dict):
            if len(out) == 0:
                raise RuntimeError("backbone(x) ")
            last_key = list(out.keys())[-1]
            fmap = out[last_key]
        else:
            fmap = out

        if fmap.dim() != 4:
            raise RuntimeError(f" 4D fmap， {fmap.dim()}D，shape={tuple(fmap.shape)}")

        if (fmap.shape[-1] == self.num_features and fmap.shape[1] <= 64 and fmap.shape[2] <= 64) \
                or (fmap.shape[-1] > fmap.shape[1] and fmap.shape[-1] > fmap.shape[2]):

            fmap = fmap.permute(0, 3, 1, 2).contiguous()

        t = self.avgpool(fmap)
        flatten_features = torch.flatten(t, 1)
        assert flatten_features.size(1) == self.num_features, \
            f": got {flatten_features.size(1)}, expect {self.num_features}"
        logits = self.fc1(flatten_features)
        return logits, flatten_features, t, fmap

    def forward_with_protos(
            self,
            images,
            y,
            ema_probs_override=None,
            epoch: int = 0,
            cfg=None,
    ):
        logits, flatten, t, fmap = self.forward(images)

        with torch.no_grad():
            probs = logits.detach().sigmoid()
            if ema_probs_override is not None:
                probs = ema_probs_override

        Z = self.compute_class_features_from_map(fmap)
        K = y.shape[1]
        device = Z.device
        batch_protos = torch.zeros_like(self.memory.protos)
        updated_mask_loss = torch.zeros(K, dtype=torch.bool, device=device)
        updated_mask_write = torch.zeros(K, dtype=torch.bool, device=device)
        write_counts = torch.zeros(K, dtype=torch.long, device=device)

        old_protos_norm = F.normalize(self.memory.protos.detach(), p=2, dim=-1)

        thr = cfg["thr"]
        proto_topq_warm = cfg["proto_topq_warm"]
        proto_min_pos = cfg["proto_min_pos"]
        proto_sim_gate = cfg["proto_sim_gate"]
        proto_min_count_sim = cfg["proto_min_count_sim"]
        cap_k = cfg["cap_k"]
        proto_write_start = cfg["proto_write_start"]

        for k in range(K):
            pos_idx_all = (y[:, k] > 0).nonzero(as_tuple=True)[0]

            if epoch < proto_write_start:
                if pos_idx_all.numel() > 0:
                    pos_p = probs[pos_idx_all, k]
                    n_keep = max(1, int(math.ceil(proto_topq_warm * pos_p.numel())))
                    top_sel = pos_p.topk(n_keep).indices
                    keep_idx = pos_idx_all[top_sel]
                    class_k_features = Z[keep_idx, k, :]
                    with torch.no_grad():
                        batch_prototype_k, _ = self.prototype_attention_modules[k](class_k_features)
                    batch_protos[k] = batch_prototype_k
                    updated_mask_loss[k] = True
                continue

            gated_mask = (y[:, k] > 0) & (probs[:, k] > thr)
            cand_idx = gated_mask.nonzero(as_tuple=True)[0]

            if proto_sim_gate > 0.0 and self.memory.counts[k].item() >= proto_min_count_sim and cand_idx.numel() > 0:
                zcand = Z[cand_idx, k, :]
                logP_all = log_softmax_sim(zcand, old_protos_norm, tau=cfg.get("sim_tau", 0.3))
                logP_k = logP_all[:, k]
                cand_idx = cand_idx[logP_k.exp() > proto_sim_gate]

            cap_this = int(cap_k[k].item())
            num_cand = int(cand_idx.numel())
            if cand_idx.numel() > cap_this:
                k_sel = min(cap_this, num_cand)
                if k_sel < num_cand:
                    cand_p = probs[cand_idx, k].flatten()
                    top_sel = cand_p.topk(k_sel).indices
                    cand_idx = cand_idx[top_sel]

            if cand_idx.numel() >= max(1, proto_min_pos):
                class_k_features = Z[cand_idx, k, :]
                batch_prototype_k, _ = self.prototype_attention_modules[k](class_k_features)
                batch_protos[k] = batch_prototype_k
                updated_mask_loss[k] = True
                updated_mask_write[k] = True
                write_counts[k] = cand_idx.numel()

        return {
            "logits": logits,
            "fmap": fmap,
            "Z": Z,
            "batch_protos": batch_protos,
            "updated_mask_loss": updated_mask_loss,
            "updated_mask_write": updated_mask_write,
            "write_counts": write_counts,
        }


def tresnet_l(pretrained: bool = False, progress: bool = True, **kwargs):

    if timm is None:
        raise ImportError("  `pip install timm`。")

    classes_num = kwargs.get("classes_num")
    args = kwargs.get("args", None)

    backbone = timm.create_model("tresnet_l",
                                 pretrained=pretrained,
                                 classes_num=0,
                                 global_pool='')

    feat_dim = getattr(backbone, "num_features", None)
    if feat_dim is None:
        raise RuntimeError("num_features")

    model = GenericBackboneWithTable(backbone=backbone,
                                     feat_dim=feat_dim,
                                     classes_num=classes_num,
                                     args=args)
    return model


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet_with_table(block, layers, **kwargs)
    if pretrained:

        pretrained_state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model

def resnet101(pretrained: bool = False, progress: bool = True, **kwargs):

    if timm is None:
        raise ImportError("`pip install timm`")

    num_classes = kwargs.get("num_classes")
    args = kwargs.get("args", None)

    backbone = timm.create_model("resnet101",
                                 pretrained=pretrained,
                                 num_classes=0,
                                 global_pool='')

    feat_dim = getattr(backbone, "num_features", 2048)

    model = GenericBackboneWithTable(backbone=backbone,
                                     feat_dim=feat_dim,
                                     classes_num=num_classes,
                                     args=args)
    return model


def resnet18_with_table(pretrained=False, progress=True, **kwargs):

    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34_with_table(pretrained=False, progress=True, **kwargs):

    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50_with_table(pretrained=False, progress=True, **kwargs):

    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def convnext_base(pretrained: bool = False, progress: bool = True, **kwargs):

    if timm is None:
        raise ImportError(" `pip install timm`")

    num_classes = kwargs.get("classes_num")
    args = kwargs.get("args", None)

    backbone = timm.create_model("convnext_base",
                                 pretrained=pretrained,
                                 num_classes=0,
                                 global_pool='')

    feat_dim = getattr(backbone, "num_features", 1024)
    model = GenericBackboneWithTable(backbone=backbone,
                                     feat_dim=feat_dim,
                                     num_classes=num_classes,
                                     args=args)
    return model


def swin_base(pretrained: bool = False, progress: bool = True, **kwargs):

    if timm is None:
        raise ImportError(" `pip install timm`")

    num_classes = kwargs.get("classes_num")
    args = kwargs.get("args", None)

    backbone = timm.create_model("swin_base_patch4_window7_224",
                                 pretrained=pretrained,
                                 num_classes=0,
                                 global_pool='')

    feat_dim = getattr(backbone, "num_features", 1024)
    model = GenericBackboneWithTable(backbone=backbone,
                                     feat_dim=feat_dim,
                                     num_classes=num_classes,
                                     args=args)
    return model
