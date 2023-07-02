import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve
from math import sqrt


def calculate_U(y_true, y_score, positive=1):
    n1 = np.sum(y_true == positive)
    n0 = len(y_score) - n1

    ## Calculate the rank for each observation
    # Get the order: The index of the score at each rank from 0 to n
    order = np.argsort(y_score)
    # Get the rank: The rank of each score at the indices from 0 to n
    rank = np.argsort(order)
    # Python starts at 0, but statistical ranks at 1, so add 1 to every rank
    rank += 1
    rank[6] = 4.5
    rank[2] = 4.5
    # If the rank for target observations is higher than expected for a random model,
    # then a possible reason could be that our model ranks target observations higher
    U1 = np.sum(rank[y_true == 1]) - (n1 * (n1 + 1)) / 2.
    U0 = np.sum(rank[y_true == 0]) - (n0 * (n0 + 1)) / 2.

    # Formula for the relation between AUC and the U statistic
    AUC1 = U1 / (n1 * n0)
    AUC0 = U0 / (n1 * n0)

    return U1, AUC1, U0, AUC0


def roc_auc_ci(y_true, y_score, positive=1):
    AUC = roc_auc_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    # Cortes, Corinna, and Mehryar Mohri. "Confidence intervals for the area under the ROC curve."
    # Advances in neural information processing systems 17 (2005): 305-312.
    Q1 = AUC / (2 - AUC)
    Q2 = 2 * AUC ** 2 / (1 + AUC)
    SE_AUC = sqrt((AUC * (1 - AUC) + (N1 - 1) * (Q1 - AUC ** 2) + (N2 - 1) * (Q2 - AUC ** 2)) / (N1 * N2))
    lower = AUC - 1.96 * SE_AUC
    upper = AUC + 1.96 * SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return AUC, lower, upper


def get_roc_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def get_bastic_statistics(value_lst):
    arr = np.array([x for x in value_lst if x == x])
    Q2 = np.median(arr)
    Q1, Q3 = np.quantile(arr, .25), np.quantile(arr, .75)
    mean, min, max = np.mean(arr), np.min(arr), np.max(arr)
    nan_ratio = (len(value_lst) - len(arr)) / float(len(value_lst))
    result_dict = {
        "Q1": Q1,
        "Q2": Q2,
        "Q3": Q3,
        "mean": mean,
        "min": min,
        "max": max,
        "nan_ratio": nan_ratio,
    }
    return result_dict


def get_roc_analysis(y_true, y_score, positive=1, negative=0):
    auroc, lower, upper = roc_auc_ci(y_true, y_score)
    positive_lst = y_score[y_true == positive]
    negative_lst = y_score[y_true == negative]
    U_val, p_val = mannwhitneyu(positive_lst, negative_lst)
    threshold = get_roc_threshold(y_true, y_score)
    result_dict = {
        "AUROC": auroc,
        "CI_lower": lower,
        "CI_upper": upper,
        "p-value": p_val,
        "threshold": threshold
    }
    return result_dict


def run_statistic(y_true, y_score, positive=1, negative=0):
    y_true, y_score = np.array(y_true), np.array(y_score)
    positive_lst = y_score[y_true == positive]
    negative_lst = y_score[y_true == negative]

    positive_basic = get_bastic_statistics(positive_lst)
    negative_basic = get_bastic_statistics(negative_lst)
    roc_result = get_roc_analysis(y_true, y_score)
    threshold = roc_result['threshold']
    accuracy, sensitivity, specificity, ppv, npv = get_binary_metric(y_true, y_score, threshold)

    final_result = dict()
    for key, value in positive_basic.items():
        new_key = 'pos_%s' % key
        final_result[new_key] = value
    for key, value in negative_basic.items():
        new_key = 'neg_%s' % key
        final_result[new_key] = value
    for key, value in roc_result.items():
        final_result[key] = value
    final_result['accuracy'] = accuracy
    final_result['sensitivity'] = sensitivity
    final_result['specificity'] = specificity
    final_result['PPV'] = ppv
    final_result['NPV'] = npv
    return final_result


def get_mae(original, reconstructed):
    return np.mean(np.abs(original - reconstructed))


def get_rmse(original, reconstructed):
    return np.sqrt(((original - reconstructed)**2).mean())


def get_prd(original, reconstructed):
    return np.sqrt(((original - reconstructed)**2).sum()/(original**2).sum()) * 100.0


def get_prdn(original, reconstructed):
    avg = original.mean()
    return np.sqrt(((original - reconstructed)**2).sum()/((original-avg)**2).sum()) * 100.0


def get_snr(original, reconstructed):
    avg = original.mean()
    return np.log(((original - avg)**2).sum()/((original-reconstructed)**2).sum()) * 10.0


def get_binary_metric(y_real, y_pred, threshold=0.5):
    tp = tn = fp = fn = 0
    for i in range(len(y_real)):
        actual = y_real[i]
        predict = y_pred[i]
        if predict >= threshold and actual >= 1:
            tp += 1
        elif predict < threshold and actual == 0:
            tn += 1
        elif predict >= threshold and actual == 0:
            fp += 1
        elif predict < threshold and actual >= 1:
            fn += 1

    accuracy = (tp + tn) / (tp + tn + fn + fp) if (tp + tn + fn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    return accuracy, sensitivity, specificity, ppv, npv
