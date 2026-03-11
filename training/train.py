import os
import sys
import random
import shutil
import time
import math
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import contextlib
import torch.nn.parallel
import torch.optim
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
from torch.autograd import Variable
from utilis.matrix import accuracy
from utilis.meters import AverageMeter, ProgressMeter
from timm.loss import AsymmetricLossMultiLabel
from .save_cams import compute_proto_cam, _overlay_heatmap_on_image, save_debug_cams


def ramp(e, a, b):
    if e <= a: return 0.0
    if e >= b: return 1.0
    return (e - a) / float(b - a)


def log_softmax_sim(z, p, tau: float = 0.1):

    z = F.normalize(z, p=2, dim=-1)
    p = F.normalize(p, p=2, dim=-1)

    sims = torch.einsum('...d,nd->...n', z, p) / max(tau, 1e-6)

    return sims - torch.logsumexp(sims, dim=-1, keepdim=True)


def _set_trainable(m, flag: bool):
    if m is None: return
    for p in m.parameters():
        p.requires_grad_(flag)


def touch_params_zero_loss(module: nn.Module):
    z = 0.0
    for p in module.parameters():
        if p.requires_grad:
            z = z + p.float().sum() * 0.0
    if not torch.is_tensor(z):
        z = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
    return z


def train(train_loader, model, criterion, optimizer, epoch, args,
          tensor_writer=None, model_ema=None, grad_accum=1, scaler=None, scheduler=None):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    ema_forward_interval = getattr(args, "ema_forward_interval", 10)
    ema_forward_micro = getattr(args, "ema_forward_micro", 8)
    global_step = 0

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    is_main = (not getattr(args, "distributed", False)) or (dist.is_initialized() and dist.get_rank() == 0)

    proto_write_start = getattr(args, "proto_write_start", 5)
    proto_topq_warm = getattr(args, "proto_topq_warm", 0.30)
    proto_min_pos = getattr(args, "proto_min_pos", 1)
    proto_sim_gate = getattr(args, "proto_sim_gate", 0.0)
    proto_min_count_sim = getattr(args, "proto_min_count_for_sim", 5)
    proto_cap_per_class = getattr(args, "proto_cap_per_class", 8)

    scu_proto_blend_start = getattr(args, "scu_proto_blend_start", 0)
    scu_proto_blend_end = getattr(args, "scu_proto_blend_end", 12)
    cau_proto_blend_start = getattr(args, "cau_proto_blend_start", 5)
    cau_proto_blend_end = getattr(args, "cau_proto_blend_end", 15)
    prior_smooth_alpha = getattr(args, "prior_smooth_alpha", 1.0)
    delay_scu_epochs = getattr(args, "delay_scu_epochs", 10)
    delay_causal_epochs = getattr(args, "delay_cau_epochs", 10)

    causal_criterion = AsymmetricLossMultiLabel(gamma_pos=0.0, gamma_neg=4.0, clip=0.05, disable_torch_grad_focal_loss=True)

    model.train()

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    end = time.time()
    optimizer.zero_grad(set_to_none=True)
    ema_core = None
    if model_ema is not None:
        ema_core = getattr(model_ema, "ema", None)
        if ema_core is None:
            ema_core = getattr(model_ema, "module", None)
        if ema_core is None:
            ema_core = model_ema

        ema_core.eval()
        ema_core.to(device)
        for _p in ema_core.parameters():
            _p.requires_grad_(False)

    amp_enabled = scaler.is_enabled() if scaler is not None else False

    for i, (images, target, image_paths) in enumerate(train_loader):

        if i == 0 and is_main:
            print(f"\n[ProCI DEBUG] Epoch {epoch} starting. grad_accum is set to: {grad_accum}\n")

        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)

        y = target.to(device, non_blocking=True)
        if y.dtype != torch.float32:
            y = y.float()

        core = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

        K = y.shape[1]


        base_cap = getattr(args, "proto_cap_per_class", 8)
        counts_all = core.memory.counts.detach()
        median_c = counts_all.median().clamp(min=1)
        ratio = (counts_all.clamp(min=1).float() / median_c.float())
        cap_k = torch.clamp(
            (base_cap * ratio.pow(-0.5)).round(),
            4, 12
        ).to(torch.long)

        thr_hi, thr_lo = getattr(args, "update_thresh", 0.85), 0.60
        thr = thr_hi - (thr_hi - thr_lo) * ramp(epoch, 5, 20)


        with torch.amp.autocast(device_type='cuda', enabled=amp_enabled):
            logits, flatten, t, fmap = model(images)

            with torch.no_grad():
                probs = logits.detach().sigmoid()
                if (ema_core is not None) and (global_step % ema_forward_interval == 0):
                    _micro = min(images.size(0), ema_forward_micro)
                    ema_out = ema_core(images[:_micro])
                    ema_logits = ema_out[0] if isinstance(ema_out, (list, tuple)) else ema_out
                    probs[:_micro] = ema_logits.sigmoid()

            proto_cfg = dict(
                thr=thr,
                proto_topq_warm=proto_topq_warm,
                proto_min_pos=proto_min_pos,
                proto_sim_gate=proto_sim_gate,
                proto_min_count_sim=proto_min_count_sim,
                cap_k=cap_k,
                proto_write_start=proto_write_start,
                sim_tau=getattr(args, "sim_tau", 0.3),
            )

            fwd = core.forward_with_protos(
                images=images,
                y=y,
                ema_probs_override=probs,
                epoch=epoch,
                cfg=proto_cfg,
            )

            logits = fwd["logits"]
            fmap = fwd["fmap"]
            Z = fwd["Z"]
            batch_protos = fwd["batch_protos"]
            updated_mask_loss = fwd["updated_mask_loss"]
            updated_mask_write = fwd["updated_mask_write"]
            write_counts = fwd["write_counts"]


            old_protos_detached = core.memory.protos.detach()


            protos_for_loss = torch.where(updated_mask_loss.unsqueeze(1), batch_protos, old_protos_detached)


            C_batch = F.normalize(protos_for_loss, p=2, dim=-1)
            C_mem = F.normalize(core.memory.protos.detach(), p=2, dim=-1)

            w_cau = ramp(epoch, cau_proto_blend_start, cau_proto_blend_end)
            confounder_dict = F.normalize((1.0 - w_cau) * C_batch + w_cau * C_mem, p=2, dim=-1)


            m0, m1 = 0.80, getattr(args, "proto_momentum", 0.98)
            m = m0 + (m1 - m0) * ramp(epoch, 5, 20)

            counts_delta = torch.zeros(K, dtype=torch.long, device=device)

            with torch.no_grad():
                for k in range(K):
                    if updated_mask_write[k]:
                        updated_proto = m * old_protos_detached[k] + (1 - m) * batch_protos[k]

                        core.memory.protos[k] = updated_proto

                        counts_delta[k] = write_counts[k]


            with torch.no_grad():
                update_mask_long = updated_mask_write.long()
                batch_protos_sync = batch_protos.detach().clone()
                mask_cnt = update_mask_long.clone()
                write_cnt = write_counts.clone()
                if getattr(args, "distributed", False) and dist.is_initialized():
                    skip_flag = torch.tensor([0 if updated_mask_write.any() else 1],
                                             device=images.device, dtype=torch.int64)
                    dist.all_reduce(skip_flag, op=dist.ReduceOp.MAX)
                    if skip_flag.item() == 0:
                        dist.all_reduce(batch_protos_sync, op=dist.ReduceOp.SUM)
                        dist.all_reduce(mask_cnt, op=dist.ReduceOp.SUM)
                        dist.all_reduce(write_cnt, op=dist.ReduceOp.SUM)

                denom = mask_cnt.clamp(min=1).unsqueeze(1).to(batch_protos_sync.dtype)
                global_avg_target = batch_protos_sync / denom
                apply_mask = (mask_cnt > 0).unsqueeze(1)

                P_old = core.memory.protos.detach()
                P_updated = m * P_old + (1.0 - m) * global_avg_target
                P_final = torch.where(apply_mask, F.normalize(P_updated, p=2, dim=-1), P_old)
                core.memory.protos[:] = P_final
                core.memory.counts += write_cnt

            with torch.no_grad():
                counts_f = core.memory.counts.float()
                alpha = prior_smooth_alpha
                prior_prob = (counts_f + alpha) / (counts_f.sum() + alpha * counts_f.numel())
            B, _, D = Z.shape


            loss_cls = criterion(logits, y)

            pos_mask = (y > 0).unsqueeze(-1)
            Z_masked = Z * pos_mask

            Z_sum = Z_masked.sum(dim=1, dtype=torch.float32)
            pos_counts = pos_mask.sum(dim=1, dtype=torch.float32).clamp(min=1)
            Z_global = Z_sum / pos_counts
            Z = F.normalize(Z.float(), p=2, dim=-1)
            Z_global = F.normalize(Z_global, p=2, dim=-1)
            cause_feat_expanded = Z_global.unsqueeze(1).expand(-1, K, -1)
            query_feat_expanded = Z
            cause_flat = cause_feat_expanded.reshape(-1, D)
            query_flat = query_feat_expanded.reshape(-1, D)

            causal_logit_flat = core.causal_module(
                cause_flat,
                query_flat,
                confounder_dict,
                prior_prob,
                tau=args.causal_tau
            )

            causal_logit_vector = causal_logit_flat.view(B, K)


            y_target = y

            loss_causal = causal_criterion(
                causal_logit_vector,
                y_target
            )


            scu_w = ramp(epoch, delay_scu_epochs, delay_scu_epochs + 20)
            cau_w = ramp(epoch, delay_causal_epochs, delay_causal_epochs + 20)

            current_second_lambda = args.second_lambda * scu_w
            current_third_lambda = args.third_lambda * cau_w

            if epoch == delay_causal_epochs:
                iter_warmup_steps = float(getattr(args, "cau_iter_warmup", 500))
                iter_ramp = min(1.0, i / iter_warmup_steps)
                next_epoch_cau_w = ramp(epoch + 1, delay_causal_epochs, delay_causal_epochs + 20)
                target_lambda = args.third_lambda * next_epoch_cau_w
                current_third_lambda = target_lambda * iter_ramp

            if is_main and i % args.print_freq == 0:
                print(
                    f"  [DEBUG Weights] epoch={epoch}, cau_w={cau_w:.4f}, current_third_lambda={current_third_lambda:.6f}")


            Zn = F.normalize(Z, p=2, dim=-1)

            Pn_batch = F.normalize(protos_for_loss, p=2, dim=-1)
            Pn_mem = F.normalize(core.memory.protos.detach(), p=2, dim=-1)
            w_scu = ramp(epoch, scu_proto_blend_start, scu_proto_blend_end)
            Pn_mix = F.normalize((1.0 - w_scu) * Pn_batch + w_scu * Pn_mem, p=2, dim=-1)

            logP = log_softmax_sim(Zn, Pn_mix, tau=getattr(args, "sim_tau", 0.3))
            pos_mask = (y > 0)
            if pos_mask.any():
                diag = logP.diagonal(dim1=1, dim2=2)
                loss_scu = -diag.masked_select(pos_mask).mean()
            else:
                loss_scu = logP.sum() * 0.0

            loss_total_unscaled = (loss_cls + current_second_lambda * loss_scu + current_third_lambda * loss_causal)

            dummy_loss = touch_params_zero_loss(core.causal_module)
            dummy_loss += touch_params_zero_loss(core.prototype_attention_modules)
            loss_total_unscaled = loss_total_unscaled + dummy_loss


            losses.update(loss_total_unscaled.item(), images.size(0))

        is_sync_step = (i + 1) % grad_accum == 0
        is_last_step_of_epoch = (i + 1) == len(train_loader)

        if args.distributed and not is_sync_step and not is_last_step_of_epoch:
            sync_context = model.no_sync()
        else:
            sync_context = contextlib.nullcontext()

        if args.use_sam:
            if amp_enabled:
                print("ERROR: SAM (davda54) is not compatible with AMP (GradScaler). Disabling AMP.")
                amp_enabled = False
            if grad_accum > 1:
                print("ERROR: SAM (davda54) is not compatible with grad_accum > 1. Setting to 1.")
                grad_accum = 1

            loss_total_unscaled.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)

            optimizer.first_step(zero_grad=True)

            with torch.amp.autocast(device_type='cuda', enabled=amp_enabled):

                fwd_smooth = core.forward_with_protos(
                    images=images, y=y, ema_probs_override=probs, epoch=epoch, cfg=proto_cfg,
                )
                logits_smooth = fwd_smooth["logits"]
                Z_smooth = fwd_smooth["Z"]
                batch_protos_smooth = fwd_smooth["batch_protos"]
                updated_mask_loss_smooth = fwd_smooth["updated_mask_loss"]

                pos_mask_bk = (y > 0)
                pos_mask_bkc = pos_mask_bk.unsqueeze(-1)

                protos_for_loss_smooth = torch.where(
                    updated_mask_loss_smooth.unsqueeze(1), batch_protos_smooth, old_protos_detached
                )

                loss_cls_smooth = criterion(logits_smooth, y)

                Zn_smooth = F.normalize(Z_smooth, p=2, dim=-1)
                Pn_batch_smooth = F.normalize(protos_for_loss_smooth, p=2, dim=-1)
                Pn_mix_smooth = F.normalize((1.0 - w_scu) * Pn_batch_smooth + w_scu * Pn_mem, p=2, dim=-1)
                logP_smooth = log_softmax_sim(Zn_smooth, Pn_mix_smooth, tau=getattr(args, "sim_tau", 0.3))

                if pos_mask_bk.any():
                    diag_smooth = logP_smooth.diagonal(dim1=1, dim2=2)
                    loss_scu_smooth = -diag_smooth.masked_select(pos_mask_bk).mean()
                else:
                    loss_scu_smooth = logP_smooth.sum() * 0.0

                C_batch_smooth = F.normalize(protos_for_loss_smooth, p=2, dim=-1)
                confounder_dict_smooth = F.normalize((1.0 - w_cau) * C_batch_smooth + w_cau * C_mem, p=2, dim=-1)

                Z_masked_smooth = Z_smooth.float() * pos_mask_bkc.float()
                Z_sum_smooth = Z_masked_smooth.sum(dim=1, dtype=torch.float32)
                Z_global_smooth = F.normalize(Z_sum_smooth / pos_counts, p=2, dim=-1)

                cause_feat_expanded_smooth = Z_global_smooth.unsqueeze(1).expand(-1, K, -1)
                query_feat_expanded_smooth = F.normalize(Z_smooth.float(), p=2, dim=-1)

                cause_flat_smooth = cause_feat_expanded_smooth.reshape(-1, D)
                query_flat_smooth = query_feat_expanded_smooth.reshape(-1, D)

                causal_logit_flat_smooth = core.causal_module(
                    cause_flat_smooth, query_flat_smooth, confounder_dict_smooth, prior_prob, tau=args.causal_tau
                )
                causal_logit_vector_smooth = causal_logit_flat_smooth.view(B, K)
                loss_causal_smooth = causal_criterion(causal_logit_vector_smooth, y)

                dummy_loss_smooth = touch_params_zero_loss(core.causal_module)
                dummy_loss_smooth += touch_params_zero_loss(core.prototype_attention_modules)

                loss_total_smooth = (
                        loss_cls_smooth + current_second_lambda * loss_scu_smooth + current_third_lambda * loss_causal_smooth
                )
                loss_total_smooth = loss_total_smooth + dummy_loss_smooth

            loss_total_smooth.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)

            optimizer.second_step(zero_grad=True)

            global_step += 1

            if model_ema is not None:
                core = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                model_ema.update(core)

        else:
            with sync_context:
                loss = loss_total_unscaled / grad_accum
                if amp_enabled and scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if is_sync_step or is_last_step_of_epoch:
                if args.clip_grad > 0:
                    if amp_enabled and scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)

                if amp_enabled and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()


                global_step += 1
                optimizer.zero_grad(set_to_none=True)

                if model_ema is not None:
                    core = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                    model_ema.update(core)

        batch_time.update(time.time() - end)
        end = time.time()

        if getattr(args, "method_name", None):
            method_name = args.method_name
        elif getattr(args, "log_path", None):
            try:
                method_name = args.log_path.rstrip('/').split('/')[-2]
            except Exception:
                method_name = args.log_path
        else:
            method_name = "N/A"
        if i % args.print_freq == 0 and is_main:
            progress.display(i, method_name)
            progress.write_log(i, args.log_file_txt)

        if is_main and i == 0 and ((epoch + 1) % 10 == 0):
            try:
                core_eval = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                out_dir = os.path.join(os.path.dirname(getattr(args, "log_path", "./logs")), "cams")

                with torch.no_grad():
                    save_debug_cams(
                        image_paths=image_paths,
                        fmap=fmap.detach(),
                        logits=logits.detach(),
                        y=y.detach(),
                        core=core_eval,
                        epoch=epoch,
                        out_dir=out_dir,
                        num_images=getattr(args, "cam_num_images", 6),
                        per_image_topk=getattr(args, "cam_topk", 3),
                        args=args,
                        use_proto=getattr(args, "cam_use_proto", True)
                    )

                if i % args.print_freq == 0:
                    print(f"  [VISUALS] Successfully saved CAMs to {out_dir}")

            except Exception as e:
                print(f"\n[WARNING] Failed to save CAM visualizations for epoch {epoch}.")
                print(f"  Error Type: {type(e).__name__}")
                print(f"  Error Details: {e}")
                print("  Training will continue...\n")

    remainder = (i + 1) % grad_accum
    if remainder != 0:
        scale_up = grad_accum / float(remainder)

        if scaler is not None:
            scaler.unscale_(optimizer)

        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(scale_up)

        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)

        if amp_enabled and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        global_step += 1
        optimizer.zero_grad(set_to_none=True)

        if model_ema is not None:  
            core = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            model_ema.update(core)



