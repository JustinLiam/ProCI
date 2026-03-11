import numpy as np

def average_precision(scores, target, max_k = None):
    
    assert scores.shape == target.shape, "The input and targets do not have the same shape"
    assert scores.ndim == 1, "The input has dimension {}, but expected it to be 1D".format(scores.shape)

    indices = np.argsort(scores, axis=0)[::-1]

    total_cases = np.sum(target)

    if max_k is None:
        max_k = len(indices)
    else:
        max_k = min(int(max_k), len(indices))

    pos_count = 0.0
    total_count = 0.0
    precision_at_i = 0.0

    for i in range(max_k):
        label = target[indices[i]]
        total_count += 1
        if label == 1:
            pos_count += 1
            precision_at_i += pos_count / total_count
        if pos_count == total_cases:
            break
        
    if pos_count > 0:
        precision_at_i /= pos_count
    else:
        precision_at_i = 0
    return precision_at_i

def _safe_div(num, den):
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den != 0)

def micro_f1(Ng, Np, Nc):
    mF1 = (2 * np.sum(Nc)) / (np.sum(Np) + np.sum(Ng))

    return mF1

def macro_f1(Ng, Np, Nc):
    n_class = len(Ng)
    precision_k = _safe_div(Nc, Np)
    recall_k = _safe_div(Nc, Ng)
    denom = (precision_k + recall_k)
    F1_k = np.divide(2.0 * precision_k * recall_k, denom,
                     out=np.zeros_like(denom, dtype=float),
                     where=denom != 0)
    MF1 = float(np.mean(F1_k)) if n_class > 0 else 0.0

    return precision_k, recall_k, F1_k, MF1


def overall_metrics(Ng, Np, Nc):
    OP = _safe_div(np.sum(Nc), np.sum(Np))
    OR = _safe_div(np.sum(Nc), np.sum(Ng))
    denom = (OP + OR)
    OF1 = (2.0 * OP * OR / denom) if denom != 0 else 0.0
    return float(OP), float(OR), float(OF1)


def per_class_metrics(Ng, Np, Nc):
    n_class = len(Ng)
    Pk = _safe_div(Nc, Np)
    Rk = _safe_div(Nc, Ng)
    CP = float(np.mean(Pk)) if n_class > 0 else 0.0
    CR = float(np.mean(Rk)) if n_class > 0 else 0.0
    denom = (CP + CR)
    CF1 = (2.0 * CP * CR / denom) if denom != 0 else 0.0
    return CP, CR, CF1

def mean_average_precision(ap):
    return float(np.mean(ap)) if len(ap) > 0 else 0.0

def exact_match_accuracy(scores, targets, threshold = 0.5):
    targets_bin = (targets == 1).astype(int)

    n_examples, n_class = scores.shape
    binary_mat = np.equal(targets_bin, (scores >= threshold).astype(int))
    row_sums = binary_mat.sum(axis=1)
    EMAcc = float(np.sum(row_sums == n_class) / n_examples)

    return EMAcc

def class_weighted_f2(Ng, Np, Nc, weights, threshold=0.5):
    assert len(weights) == len(Ng) == len(Np) == len(Nc), "weights / Ng / Np / Nc length mismatch"

    precision_k = _safe_div(Nc, Np)
    recall_k = _safe_div(Nc, Ng)
    denom = 4.0 * precision_k + recall_k
    F2_k = np.divide(5.0 * precision_k * recall_k, denom,
                     out=np.zeros_like(denom, dtype=float),
                     where=denom != 0)

    ciwF2 = float(np.sum(F2_k * weights) / np.sum(weights))

    return ciwF2, F2_k


def evaluation(scores, targets, weights=None, threshold=0.5):
    assert scores.shape == targets.shape, \
        "The input and targets do not have the same size: Input: {} - Targets: {}".format(scores.shape, targets.shape)

    n_examples, n_class = scores.shape

    if weights is None:
        weights = np.ones(n_class, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        assert len(weights) == n_class, "weights must match number of explicit classes"

    targets_bin = (targets == 1).astype(int)
    pred_bin = (scores >= threshold).astype(int)

    Nc = np.zeros(n_class + 1, dtype=float)
    Np = np.zeros(n_class + 1, dtype=float)
    Ng = np.zeros(n_class + 1, dtype=float)

    ap = np.zeros(n_class, dtype=float)

    for k in range(n_class):
        tmp_scores  = scores[:, k]
        tmp_targets = targets_bin[:, k]
        Ng[k] = np.sum(tmp_targets == 1)
        Np[k] = np.sum(pred_bin[:, k] == 1)
        Nc[k] = np.sum(tmp_targets * pred_bin[:, k])
        ap[k] = average_precision(tmp_scores, tmp_targets)

    pred_is_norm = (pred_bin.sum(axis=1) == 0).astype(int)
    gt_is_norm   = (targets_bin.sum(axis=1) == 0).astype(int)
    Ng[-1] = np.sum(gt_is_norm == 1)
    Np[-1] = np.sum(pred_is_norm == 1)
    Nc[-1] = np.sum(gt_is_norm * pred_is_norm)

    use_normal_in_macro = False
    if use_normal_in_macro:
        Ng_macro, Np_macro, Nc_macro = Ng, Np, Nc
    else:
        Ng_macro, Np_macro, Nc_macro = Ng[:-1], Np[:-1], Nc[:-1]

    OP, OR, OF1 = overall_metrics(Ng, Np, Nc)
    mF1 = micro_f1(Ng, Np, Nc)

    CP, CR, CF1 = per_class_metrics(Ng_macro, Np_macro, Nc_macro)
    precision_k, recall_k, F1_k, MF1 = macro_f1(Ng_macro, Np_macro, Nc_macro)

    EMAcc = exact_match_accuracy(scores, targets)

    mAP = mean_average_precision(ap)

    P_norm = _safe_div(Nc[-1], Np[-1])
    R_norm = _safe_div(Nc[-1], Ng[-1])
    denom  = P_norm + R_norm
    F1_norm = (2.0 * P_norm * R_norm / denom) if denom != 0 else 0.0

    assert len(weights) == n_class, "weights must match number of explicit classes"
    F2, F2_k = class_weighted_f2(Ng[:-1], Np[:-1], Nc[:-1], weights)

    new_metrics = {
        "F2": F2,
        "F2_class": list(F2_k) + [
            (5 * _safe_div(Nc[-1], Np[-1]) * _safe_div(Nc[-1], Ng[-1])) /
            max(1e-12, 4 * _safe_div(Nc[-1], Np[-1]) + _safe_div(Nc[-1], Ng[-1]))
        ],
        "F1_Normal": float(F1_norm)
    }

    main_metrics = {
        "OP": float(OP), "OR": float(OR), "OF1": float(OF1),
        "CP": float(CP), "CR": float(CR), "CF1": float(CF1),
        "MF1": float(MF1), "mF1": float(mF1),
        "EMAcc": float(EMAcc), "mAP": float(mAP)
    }

    auxillery_metrics = {
        "P_class": list(precision_k),
        "R_class": list(recall_k),
        "F1_class": list(F1_k),
        "AP": list(ap),
        "Np": list(Np),
        "Nc": list(Nc),
        "Ng": list(Ng)
    }

    return new_metrics, main_metrics, auxillery_metrics