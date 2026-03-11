import time
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.distributed as dist
import torch.utils.data.distributed
from utilis.matrix import accuracy
from utilis.meters import AverageMeter, ProgressMeter
from utilis.metrics import evaluation

LabelWeightDict = {"RB":1.00,"OB":0.5518,"PF":0.2896,"DE":0.1622,"FS":0.6419,"IS":0.1847,"RO":0.3559,"IN":0.3131,"AF":0.0811,"BE":0.2275,"FO":0.2477,"GR":0.0901,"PH":0.4167,"PB":0.4167,"OS":0.9009,"OP":0.3829,"OK":0.4396}


def _ddp_concat_tensor_cpu(t: torch.Tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return t.cpu()
    ws = dist.get_world_size()
    rank = dist.get_rank()
    t_cpu = t.detach().cpu()
    obj_list = [None for _ in range(ws)]
    dist.all_gather_object(obj_list, t_cpu.numpy())
    if rank == 0:
        arrs = [np.asarray(o) for o in obj_list if o is not None]
        if len(arrs) == 0:
            return None
        return torch.from_numpy(np.concatenate(arrs, axis=0))
    else:
        return None

def _safe_np_div(num, den):
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    out = np.zeros_like(num, dtype=np.float64)
    mask = den != 0
    out[mask] = num[mask] / den[mask]
    return out

def _avg_precision_np(scores, labels):
    order = np.argsort(scores)[::-1]
    y = labels[order].astype(np.int32)
    pos = int(y.sum())
    if pos == 0:
        return np.nan
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1)
    rec  = tp / pos
    mrec = np.concatenate([[0.], rec, [1.]])
    mpre = np.concatenate([[0.], prec, [0.]])
    for i in range(mpre.size - 1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1]))

def _log_diagnostics_to_tb(writer, probs_np, tgts_np, epoch, class_names=None, topk_pr=5, ece_bins=20):

    N, C = probs_np.shape
    Ng = tgts_np.sum(0)
    ap = np.zeros(C, dtype=np.float64)
    for c in range(C):
        ap[c] = _avg_precision_np(probs_np[:, c], tgts_np[:, c])
    valid = Ng > 0
    mAP_valid = float(np.nanmean(ap[valid])) if valid.any() else 0.0
    writer.add_histogram('hist/ap_per_class', np.nan_to_num(ap, nan=0.0), epoch)
    writer.add_scalar('map/mAP_valid', mAP_valid, epoch)
    if valid.any():
        q1, q2 = np.quantile(Ng[valid], [0.33, 0.66])
        head = (Ng >= q2) & valid
        mid  = (Ng <  q2) & (Ng >= q1) & valid
        tail = (Ng <  q1) & valid
        writer.add_scalar('map/head', float(np.nanmean(ap[head])) if head.any() else 0.0, epoch)
        writer.add_scalar('map/mid',  float(np.nanmean(ap[mid]))  if mid.any()  else 0.0, epoch)
        writer.add_scalar('map/tail', float(np.nanmean(ap[tail])) if tail.any() else 0.0, epoch)

    topk = np.argsort(Ng)[::-1][:min(topk_pr, C)]
    for c in topk:
        writer.add_pr_curve(f'pr_curve/class_{c}', tgts_np[:, c].astype(bool), probs_np[:, c], global_step=epoch)

    brier = float(np.mean((probs_np - tgts_np) ** 2))
    writer.add_scalar('calibration/brier', brier, epoch)
    edges = np.linspace(0, 1, ece_bins + 1)
    ece = 0.0
    for b in range(ece_bins):
        m = (probs_np >= edges[b]) & (probs_np < edges[b+1])
        cnt = int(m.sum())
        if cnt > 0:
            acc  = float(((probs_np[m] >= 0.5) == tgts_np[m]).mean())
            conf = float(probs_np[m].mean())
            ece += (cnt / (N * C)) * abs(acc - conf)
    writer.add_scalar('calibration/ece', float(ece), epoch)

    take = min(2000, N)
    idx = np.random.permutation(N)[:take]
    pos_scores = probs_np[idx][tgts_np[idx]==1]
    neg_scores = probs_np[idx][tgts_np[idx]==0]
    if pos_scores.size > 0:
        writer.add_histogram('scores/pos', pos_scores, epoch)
    if neg_scores.size > 0:
        writer.add_histogram('scores/neg', neg_scores, epoch)

    th = 0.5
    pred_bin = (probs_np >= th).astype(np.int32)
    writer.add_scalar('density/predicted_per_img@0.5', float(pred_bin.sum(1).mean()), epoch)
    writer.add_scalar('density/true_per_img', float(tgts_np.sum(1).mean()), epoch)

    scores_all = probs_np.reshape(-1)
    labels_all = tgts_np.reshape(-1)
    order = np.argsort(scores_all)[::-1]
    y = labels_all[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    tpr = tp / max(1, tp[-1]); fpr = fp / max(1, fp[-1])
    auc = float(np.trapz(tpr, fpr))
    writer.add_scalar('auc/micro', auc, epoch)

def _is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

@torch.no_grad()
def run_sanity_check(val_loader, model, class_names, want_names=("person","dog"),
                     device=None, topk_show=10, args=None):
    if class_names is None:
        if args.rank == 0:
            print("[Sanity] class_names is None; skip.")
        return
    name2idx = {n: i for i, n in enumerate(class_names)}
    missing = [w for w in want_names if w not in name2idx]
    if missing:
        if args.rank == 0:
            print(f"[Sanity] classes not found in dataset: {missing}")
        return

    idxs = [name2idx[w] for w in want_names]
    model.eval()

    for batch in val_loader:
        images, targets = None, None
        if isinstance(batch, dict):
            images = (batch.get("images") or batch.get("img") or batch.get("image"))
            targets = (batch.get("targets") or batch.get("target") or
                       batch.get("labels") or batch.get("y"))
        elif isinstance(batch, (list, tuple)):
            if len(batch) >= 2:
                images, targets = batch[0], batch[1]
            else:
                continue
        else:
            try:
                images, targets, *_ = batch
            except Exception:
                continue

        if images is None or targets is None:
            continue

        if device is not None and torch.is_tensor(images):
            images  = images.to(device, non_blocking=True)
        if device is not None and torch.is_tensor(targets):
            targets = targets.to(device, non_blocking=True).float()

        out = model(images)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        probs = logits.sigmoid()

        if torch.is_tensor(targets):
            mask = (targets[:, idxs] > 0.5).all(dim=1)
            if mask.any():
                j = mask.nonzero(as_tuple=False)[0].item()
                p = probs[j].detach().cpu().numpy()
                y = targets[j].detach().cpu().numpy()
                topk = p.argsort()[::-1][:topk_show]
                if args.rank == 0:
                    print("[Sanity] Found one sample containing:", want_names)
                    print("[Sanity] GT positives:",
                          [class_names[i] for i in range(len(y)) if y[i] > 0.5])
                    print("[Sanity] Top-{} preds:".format(topk_show),
                          [(class_names[i], float(p[i]), int(y[i] > 0.5)) for i in topk])
                return

    print("[Sanity] No sample found that contains:", want_names)


def validate_single(val_loader, model, criterion, epoch=0, test=True, args=None, tensor_writer=None):
    if test:
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5],
            prefix='Test: ')
    else:
        batch_time = AverageMeter('val Time', ':6.3f')
        losses = AverageMeter('val Loss', ':.4e')
        top1 = AverageMeter('Val Acc@1', ':6.2f')
        top5 = AverageMeter('Val Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5],
            prefix='Val: ')

    model.eval()


    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(val_loader):
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                images, target, _ = batch
            else:
                images, target = batch
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                if target is not None:
                    target = target.cuda(args.gpu, non_blocking=True)
            output, cfeatures,t,map= model(images)

            loss = criterion(output, target)

            if target is not None:
                loss = criterion(output, target)
                acc1, acc5 = accuracy(output, target, topk=(1, 10))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                method_name = args.log_path.split('/')[-2]
                progress.display(i, method_name)
                progress.write_log(i, args.log_path)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        with open(args.log_path, 'a') as f1:
            f1.writelines(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                          .format(top1=top1, top5=top5))
        if test:
            tensor_writer.add_scalar('loss/test', loss.item(), epoch)
            tensor_writer.add_scalar('ACC@1/test', top1.avg, epoch)
            tensor_writer.add_scalar('ACC@5/test', top5.avg, epoch)
        else:
            tensor_writer.add_scalar('loss/val', loss.item(), epoch)
            tensor_writer.add_scalar('ACC@1/val', top1.avg, epoch)
            tensor_writer.add_scalar('ACC@5/val', top5.avg, epoch)

    return top1.avg

def validate(val_loader, model, criterion, epoch=0, test=True, args=None, tensor_writer=None):
    model.eval()

    try:
        _batch = next(iter(val_loader))
    except StopIteration:
        print("[validate] Empty val_loader; skip guards.")
        _batch = None

    if _batch is not None:
        if isinstance(_batch, dict):
            sample_imgs = (_batch.get("images") or _batch.get("img") or _batch.get("image"))
            _sample_tgts = (_batch.get("targets") or _batch.get("target") or
                            _batch.get("labels") or _batch.get("y"))
            if sample_imgs is None or _sample_tgts is None:
                raise KeyError("[validate] Unknown dict keys for images/targets in val batch.")
        elif isinstance(_batch, (list, tuple)):
            if len(_batch) < 2:
                raise ValueError(f"[validate] Batch has length {len(_batch)} < 2.")
            sample_imgs, _sample_tgts = _batch[:2]
        else:
            try:
                sample_imgs, _sample_tgts, *_ = _batch
            except Exception as e:
                raise TypeError(f"[validate] Unexpected batch type: {type(_batch)}") from e

        if getattr(args, "gpu", None) is not None:
            sample_imgs = sample_imgs.cuda(args.gpu, non_blocking=True)

        _out = model(sample_imgs)
        if isinstance(_out, (tuple, list)):
            _out = _out[0]
        assert _out.shape[1] == getattr(val_loader.dataset, "num_classes", _out.shape[1]), \
            f"Model out dim {_out.shape[1]} != dataset classes {getattr(val_loader.dataset, 'num_classes', 'UNK')}"

    if _is_main_process():
        ds_names = getattr(val_loader.dataset, "classes_names", None)
        if ds_names is not None:
            print("[COCO-ML] Class order (head):", ds_names[:10], " ...")

        try:
            from config import classes_names as cfg_names
            if ds_names is not None and cfg_names is not None and ds_names != cfg_names:
                mismatch = [(i, a, b) for i, (a, b) in enumerate(zip(ds_names, cfg_names)) if a != b]
                print(
                    f"[WARN] dataset vs config class order mismatch at positions: {mismatch[:5]} (total {len(mismatch)})")
        except Exception:
            pass

    if getattr(args, "sanity", "") and _is_main_process():
        want = [s.strip() for s in args.sanity.split(",") if s.strip()]
        device = torch.device(f"cuda:{args.gpu}") if getattr(args, "gpu", None) is not None else torch.device("cpu")
        run_sanity_check(
            val_loader=val_loader,
            model=model,
            class_names=getattr(val_loader.dataset, "classes_names", None),
            want_names=want,
            device=device,
            topk_show=10,
            args = args
        )

    batch_time = AverageMeter('val Time' if not test else 'Time', ':6.3f')
    losses     = AverageMeter('val Loss' if not test else 'Loss', ':.4e')

    outs_local = []
    tgts_local = []
    names_local = []

    distributed = getattr(args, "distributed", False) and dist.is_initialized()
    rank        = dist.get_rank()  if distributed else 0
    world_size  = dist.get_world_size() if distributed else 1

    if getattr(args, "method_name", None):
        method_name = args.method_name
    elif getattr(args, "log_path", None):
        try:
            method_name = args.log_path.rstrip('/').split('/')[-2]
        except Exception:
            method_name = args.log_path
    else:
        method_name = "N/A"

    end = time.time()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                images, target, filenames = batch
            else:
                images, target = batch
                filenames = [None] * (images.size(0) if torch.is_tensor(images) else len(images))

            if getattr(args, "gpu", None) is not None and torch.is_tensor(images):
                images = images.cuda(args.gpu, non_blocking=True)
                if target is not None:
                    target = target.cuda(args.gpu, non_blocking=True).float()
            else:
                if target is not None and torch.is_tensor(target):
                    target = target.float()

            output, cfeatures, t, fmap = model(images)

            probs = output.detach().sigmoid().cpu()
            outs_local.append(probs)
            names_local.extend(list(filenames))

            has_label_batch = (target is not None)
            if has_label_batch:
                loss = criterion(output, target)
                losses.update(loss.item(), probs.size(0))
                tg    = target.detach().cpu().float()
                tgts_local.append(tg)

            if torch.is_tensor(images) and images.is_cuda:
                torch.cuda.synchronize(device=images.device)
            now = time.time()
            batch_time.update(now - end)
            end = now

            if i % getattr(args, "print_freq", 50) == 0 and rank == 0:
                msg = (f"{'Test' if test else 'Val'}: "
                       f"[{i}/{len(val_loader)}]  Time {batch_time.val:.3f} ({batch_time.avg:.3f})")
                if has_label_batch:
                    msg += f"  Loss {losses.val:.4e} ({losses.avg:.4e})"
                msg += f"  method name {method_name}"
                print(msg)

    scores_cat = torch.cat(outs_local, dim=0) if len(outs_local) > 0 else torch.zeros(0, 0)
    scores_all = _ddp_concat_tensor_cpu(scores_cat)

    if len(tgts_local) > 0:
        tgts_cat = torch.cat(tgts_local, dim=0)
        tgts_all = _ddp_concat_tensor_cpu(tgts_cat)
    else:
        tgts_all = None

    if distributed:
        obj_list = [None for _ in range(world_size)]
        dist.all_gather_object(obj_list, names_local)
        if rank == 0:
            names_all = []
            for o in obj_list:
                if o is not None:
                    names_all.extend(o)
    else:
        names_all = names_local

    if rank == 0:
        if scores_all is None or scores_all.numel() == 0:
            print(" * No samples collected.")
            if tensor_writer is not None:
                tensor_writer.add_scalar('loss/val' if not test else 'loss/test', losses.avg, epoch)
            return {}

        scores_np = scores_all.numpy()
        has_imp_normal = args.dataset.lower() in ["sewer-ml", "sewerml", "sewer"]
        ds = getattr(val_loader, "dataset", None)
        weights = None
        if ds is not None and hasattr(ds, "class_weights") and ds.class_weights is not None:
            w = ds.class_weights
            weights = w.detach().cpu().numpy() if torch.is_tensor(w) else np.asarray(w, dtype=np.float32)
        if has_imp_normal and 'LabelWeightDict' in globals():
            weights = np.array([LabelWeightDict.get(name, 1.0) for name in args.classes_names], dtype=np.float32)
        if weights is None:
            weights = np.ones(scores_np.shape[1], dtype=np.float32)


        if tgts_all is not None and tgts_all.numel() > 0:
            targets_np = tgts_all.numpy()
            assert scores_np.shape == targets_np.shape, f"shape mismatch: {scores_np.shape} vs {targets_np.shape}"

            thresholds = np.arange(0.5, 0.95, 0.05).tolist()

            all_metrics = []
            for th in thresholds:
                new_m, main_m, aux_m = evaluation(scores=scores_np, targets=targets_np, weights=weights, threshold=th)
                main_m['threshold'] = th
                all_metrics.append((main_m, aux_m, new_m))

            best_map = all_metrics[0][0]['mAP']

            best_cf1_idx = np.argmax([m['CF1'] for m, _, _ in all_metrics])
            best_cf1_metrics, aux_cf1, _ = all_metrics[best_cf1_idx]

            best_of1_idx = np.argmax([m['OF1'] for m, _, _ in all_metrics])
            best_of1_metrics, aux_of1, _ = all_metrics[best_of1_idx]

            best_f2_idx = -1
            if has_imp_normal:
                best_f2_idx = np.argmax([n['F2'] for _, _, n in all_metrics])

            final_metrics = {
                'mAP': best_map,
                'CP': best_cf1_metrics['CP'],
                'CR': best_cf1_metrics['CR'],
                'CF1': best_cf1_metrics['CF1'],
                'CP_threshold': best_cf1_metrics['threshold'],
                'OP': best_of1_metrics['OP'],
                'OR': best_of1_metrics['OR'],
                'OF1': best_of1_metrics['OF1'],
                'OP_threshold': best_of1_metrics['threshold'],
            }

            if has_imp_normal:
                best_f2_metrics, _, best_new_f2_metrics = all_metrics[best_f2_idx]
                final_metrics['F2_CIW'] = best_new_f2_metrics['F2']
                final_metrics['F2_CIW_threshold'] = best_f2_metrics['threshold']

                best_f1n_idx = np.argmax([n['F1_Normal'] for _, _, n in all_metrics])
                final_metrics['F1_Normal'] = all_metrics[best_f1n_idx][2]['F1_Normal']
                final_metrics['F1_Normal_threshold'] = all_metrics[best_f1n_idx][0]['threshold']

            split = 'test' if test else 'val'
            log_msg = (
                f" * [{split.upper()}] mAP {final_metrics['mAP']:.3f}  "
                f"CP {final_metrics['CP']:.3f} (th={final_metrics['CP_threshold']:.2f})  "
                f"CR {final_metrics['CR']:.3f}  CF1 {final_metrics['CF1']:.3f}  "
                f"OP {final_metrics['OP']:.3f} (th={final_metrics['OP_threshold']:.2f})  "
                f"OR {final_metrics['OR']:.3f}  OF1 {final_metrics['OF1']:.3f}"
            )

            if has_imp_normal and 'F2_CIW' in final_metrics:
                log_msg += f"  F2_CIW {final_metrics['F2_CIW']:.3f} (th={final_metrics['F2_CIW_threshold']:.2f})"
                log_msg += f"  F1_Normal {final_metrics['F1_Normal']:.3f}  (th={final_metrics['F1_Normal_threshold']:.2f})"

            print(log_msg)
            if tensor_writer is not None:
                tensor_writer.add_scalar(f'loss/{split}', losses.avg, epoch)

                tensor_writer.add_scalar(f'metrics/{split}/mAP', final_metrics['mAP'], epoch)
                tensor_writer.add_scalar(f'metrics/{split}/CP_at_best_CF1', final_metrics['CP'], epoch)
                tensor_writer.add_scalar(f'metrics/{split}/CR_at_best_CF1', final_metrics['CR'], epoch)
                tensor_writer.add_scalar(f'metrics/{split}/CF1_best', final_metrics['CF1'], epoch)
                tensor_writer.add_scalar(f'metrics/{split}/OP_at_best_OF1', final_metrics['OP'], epoch)
                tensor_writer.add_scalar(f'metrics/{split}/OR_at_best_OF1', final_metrics['OR'], epoch)
                tensor_writer.add_scalar(f'metrics/{split}/OF1_best', final_metrics['OF1'], epoch)
                tensor_writer.add_scalar(f'metrics/{split}/threshold_for_CF1', final_metrics['CP_threshold'], epoch)
                tensor_writer.add_scalar(f'metrics/{split}/threshold_for_OF1', final_metrics['OP_threshold'], epoch)

                if has_imp_normal and 'F2_CIW' in final_metrics:
                    tensor_writer.add_scalar(f'metrics/{split}/F2_CIW_best', final_metrics['F2_CIW'], epoch)
                    tensor_writer.add_scalar(f'metrics/{split}/threshold_for_F2_CIW', final_metrics['F2_CIW_threshold'],
                                             epoch)
                    tensor_writer.add_scalar(f'metrics/{split}/F1_Normal_best', final_metrics['F1_Normal'], epoch)

                if 'P_class' in aux_cf1:
                    tensor_writer.add_histogram(f'hist/{split}/Class_Precision', np.array(aux_cf1['P_class']), epoch)
                if 'R_class' in aux_cf1:
                    tensor_writer.add_histogram(f'hist/{split}/Class_Recall', np.array(aux_cf1['R_class']), epoch)
                if 'F1_class' in aux_cf1:
                    tensor_writer.add_histogram(f'hist/{split}/Class_F1', np.array(aux_cf1['F1_class']), epoch)

                _log_diagnostics_to_tb(tensor_writer, probs_np=scores_np, tgts_np=targets_np, epoch=epoch,
                                       class_names=getattr(val_loader.dataset, "classes_names", None))

            return final_metrics
        else:
            return {}
    else:
        return {}