import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             roc_auc_score)
from torch import nn
from transformers import EvalPrediction


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(preds, p.label_ids)
    return result


def multi_label_metrics(predictions, labels, threshold=0.5):
    sigmoid = nn.Sigmoid()
    proba = sigmoid(torch.tensor(predictions))
    y_pred = (proba > threshold).numpy().astype(int)
    y_true = labels.astype(int)
    f1_micro_average = f1_score(y_true, y_pred, average="micro")
    roc_auc = roc_auc_score(y_true, proba, average="micro")
    pr_auc = average_precision_score(y_true, proba, average="micro")
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {
        "f1": f1_micro_average,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": accuracy,
    }
    return metrics