import os
from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.metrics import roc_curve as sk_roc_curve
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.multiclass import type_of_target
from tqdm.auto import tqdm


def k_fold_roc_curve(
    model_outputs: Dict[str, Union[np.ndarray, torch.Tensor]],
    model_name: str,
    num_classes: int = 8,
    average: str = "macro",
    legend_key: str = "Fold",
    show_mean_and_std: bool = True,
) -> None:
    """Plots the ROC and PRC curves for a k-fold cross validation

    Args:
        model_outputs (Dict[str, Union[np.ndarray, torch.Tensor]]): A dictionary
            containing the model outputs for each fold. Each entry in the
            dictionary should be a dictionary containing the keys "y" and "proba"
            which are the ground truth labels and the predicted probabilities
            respectively.
        model_name (str): The name of the model.
        num_classes (int, optional): Number of classification targets. Defaults
            to 8.
        average (str, optional): The type of averaging to use when computing the
            ROC and PRC charts. Defaults to "macro".
        legend_key (str, optional): The label to use for the fold number in the
            plot legend. Defaults to "Fold".
        show_mean_and_std (bool, optional): Shows the mean and standard
            deviation of the ROC and PRC curves across all folds in the plot.
            Defaults to True.

    Raises:
        NotImplementedError: If the average parameter is not "macro" or "weighted".
    """
    fig, ax = plt.subplots(2, 1, figsize=(8.27, 11.69), dpi=100)
    tprs, aurocs, tpr_threshes = [], [], []
    fpr_mean = np.linspace(0, 1, 1000)
    precisions, auprcs, recall_threshes = [], [], []
    recall_mean = np.linspace(0, 1, 1000)
    for fold_idx, fold in enumerate(tqdm(model_outputs)):
        y_type = type_of_target(fold["y"])
        if y_type == "multiclass":
            roc_label_binarizer = LabelBinarizer().fit(fold["y"])
            y_onehot_test = roc_label_binarizer.transform(fold["y"])
        else:
            y_onehot_test = fold["y"]
        intermediate_tprs, intermediate_tpr_threshes = [], []
        intermediate_precisions, intermediate_recall_threshes = [], []
        for i in range(num_classes):
            if np.sum(y_onehot_test, axis=0)[i] == 0:
                continue

            # ROC Curve
            fpr, tpr, tpr_thresh = sk_roc_curve(
                y_onehot_test[:, i], fold["y_prob"][:, i]
            )
            intermediate_tpr_threshes.append(tpr_thresh[np.abs(tpr - 0.85).argmin()])
            tpr_interp = np.interp(fpr_mean, fpr, tpr)
            tpr_interp[0] = 0.0
            intermediate_tprs.append(tpr_interp)

            # Precision-Recall Curve
            precision, recall, recall_thresh = precision_recall_curve(
                y_onehot_test[:, i], fold["y_prob"][:, i]
            )
            prec_interp = np.interp(recall_mean, recall[::-1], precision[::-1])
            intermediate_precisions.append(prec_interp)
            intermediate_recall_threshes.append(
                recall_thresh[np.abs(recall - 0.85).argmin()]
            )

        auroc = roc_auc_score(y_onehot_test, fold["y_prob"], average=average)
        auprc = average_precision_score(y_onehot_test, fold["y_prob"], average=average)

        if average == "macro":
            tprs.append(np.mean(intermediate_tprs, axis=0))
            aurocs.append(auroc)
            tpr_threshes.append(np.mean(intermediate_tpr_threshes))
            precisions.append(np.mean(intermediate_precisions, axis=0))
            auprcs.append(auprc)
            recall_threshes.append(np.mean(intermediate_recall_threshes))
        elif average == "weighted":
            class_distributions = np.sum(y_onehot_test, axis=0)
            if len(class_distributions) < num_classes:
                class_distributions = np.append(
                    class_distributions,
                    np.zeros(num_classes - len(class_distributions)),
                )
            class_distributions = class_distributions / np.sum(class_distributions)
            aurocs.append(auroc)
            tprs.append(np.array(intermediate_tprs).T @ class_distributions)
            precisions.append(np.array(intermediate_precisions).T @ class_distributions)
            auprcs.append(auprc)
        else:
            raise NotImplementedError

        ax[0].plot(
            fpr_mean,
            tprs[-1],
            label=f"ROC {legend_key} {fold_idx + 1} (AUC = {aurocs[-1]:.4f})",
            alpha=0.3,
        )
        ax[1].plot(
            recall_mean,
            precisions[-1],
            label=f"PRC {legend_key} {fold_idx + 1} (AUC = {auprcs[-1]:.4f})",
            alpha=0.3,
        )

    ax[0].set_xlabel("False Positive Rate")
    ax[0].set_ylabel("True Positive Rate")
    ax[0].set_title(f"Reciver Operating Characteristic Curve ({model_name})")
    ax[0].set_ylim(-0.1, 1.1)

    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title(f"Precision-Recall Curve ({model_name})")
    ax[1].set_ylim(-0.1, 1.1)

    if show_mean_and_std:
        tpr_mean = np.mean(tprs, axis=0)
        tpr_mean[-1] = 1.0
        auroc_mean = auc(fpr_mean, tpr_mean)
        auroc_std = np.std(aurocs)
        ax[0].plot(
            fpr_mean,
            tpr_mean,
            label=r"Mean ROC (AUC = %0.4f $\pm$ %0.4f)" % (auroc_mean, auroc_std),
            lw=2,
            alpha=0.8,
        )
        tpr_std = np.std(tprs, axis=0)
        ax[0].fill_between(
            fpr_mean,
            np.maximum(tpr_mean - tpr_std, 0),
            np.minimum(tpr_mean + tpr_std, 1),
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
            color="grey",
        )

        # PRC
        prec_mean = np.mean(precisions, axis=0)
        auprc_mean = auc(recall_mean, prec_mean)
        auprc_std = np.std(auprcs)
        ax[1].plot(
            recall_mean,
            prec_mean,
            label=r"Mean PRC (AUC = %0.4f $\pm$ %0.4f)" % (auprc_mean, auprc_std),
            lw=2,
            alpha=0.8,
        )
        prec_std = np.std(precisions, axis=0)
        ax[1].fill_between(
            recall_mean,
            np.maximum(prec_mean - prec_std, 0),
            np.minimum(prec_mean + prec_std, 1),
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
            color="grey",
        )

    fig.suptitle(f"ROC and PRC Curves for {model_name}, average={average}")
    ax[0].legend()
    ax[1].legend()
    plt.show()
    return fig


def godbole_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = None
) -> Union[np.ndarray, list]:
    """Calculates the accuracy score for each class or its average
        (Godbole and Sarawagi, 2004)

    Args:
        y_true (np.ndarray): Binarized ground truth labels.
        y_pred (np.ndarray): Binarized predicted labels.
        average (str, optional): Determines which form of averaging is used.
            Defaults to None.

    Returns:
        np.ndarray: The accuracy score for each class
    """
    numerator = np.logical_and(y_pred, y_true).astype(int).sum(axis=1)
    denominator = np.logical_or(y_pred, y_true).astype(int).sum(axis=1)
    if average is None:
        total = []
        for i in range(y_pred.shape[1]):
            indices = np.where(y_true[:, i] == 1)[0]
            total.append(
                np.sum(numerator[indices] / denominator[indices]) / len(indices)
            )
        return total
    elif average == "macro":
        # Take a simple mean of the accuracies
        total = np.sum(numerator / denominator) / len(y_pred)
    elif average == "weighted":
        # Weight by class distribution
        total = np.sum(numerator / denominator)
        total *= np.sum(y_true, axis=0) / len(y_pred)
    return total
