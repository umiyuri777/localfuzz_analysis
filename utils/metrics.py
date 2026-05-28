"""
二値分類の評価指標（混同行列、適合率・再現率・F値・AUC）の計算と表示を行うモジュール。
"""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def print_confusion_matrix(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    title: str = "混同行列",
) -> np.ndarray:
    """
    混同行列を計算し、表示して返す。

    Args:
        y_true: 正解ラベル。
        y_pred: 予測ラベル。
        title: 表示タイトル。

    Returns:
        混同行列（2x2）。
    """
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n【{title}】")
    print(cm)
    return cm


def calculate_binary_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    y_pred_proba: Optional[Union[np.ndarray, pd.Series]] = None,
    zero_division: Union[int, str] = 0,
) -> dict:
    """
    二値分類の評価指標（Accuracy, Precision, Recall, F1, AUC）を計算する。

    Args:
        y_true: 正解ラベル。
        y_pred: 予測ラベル。
        y_pred_proba: 正クラス（1）の予測確率。AUC計算に使用。Noneの場合はAUCを計算しない。
        zero_division: Precision/Recall/F1 でゼロ除算時の扱い（0 または 'nan'）。

    Returns:
        accuracy, precision, recall, f1, (auc) を格納した辞書。
    """
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=zero_division),
        "recall": recall_score(y_true, y_pred, zero_division=zero_division),
        "f1": f1_score(y_true, y_pred, zero_division=zero_division),
    }
    if y_pred_proba is not None:
        try:
            result["auc"] = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            result["auc"] = float("nan")
    return result


def print_binary_metrics(metrics: dict, title: str = "評価指標") -> None:
    """
    calculate_binary_metrics の戻り値を整形して表示する。

    Args:
        metrics: calculate_binary_metrics の戻り値（または同形式の辞書）。
        title: 表示タイトル。
    """
    print(f"\n【{title}】")
    print(f"Accuracy:  {metrics.get('accuracy', 0):.4f}")
    print(f"Precision: {metrics.get('precision', 0):.4f}")
    print(f"Recall:    {metrics.get('recall', 0):.4f}")
    print(f"F1-Score:  {metrics.get('f1', 0):.4f}")
    if "auc" in metrics and not np.isnan(metrics["auc"]):
        print(f"AUC:       {metrics['auc']:.4f}")


def f1_from_confusion_matrix(cm: np.ndarray) -> Tuple[float, float, float]:
    """
    2x2混同行列から F1, Precision, Recall を計算する。
    sklearn の confusion_matrix は [[TN, FP], [FN, TP]] の並びを前提とする。

    Args:
        cm: 2x2 の混同行列。

    Returns:
        (f1, precision, recall)
    """
    if cm.shape != (2, 2):
        raise ValueError("2値分類用です。混同行列は 2x2 である必要があります。")
    TN, FP = cm[0, 0], cm[0, 1]
    FN, TP = cm[1, 0], cm[1, 1]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall


def compute_roc_curve(
    y_true: Union[np.ndarray, pd.Series],
    y_pred_proba: Union[np.ndarray, pd.Series],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    ROC曲線用の FPR, TPR, 閾値 および AUC を計算する。

    Returns:
        (fpr, tpr, thresholds, auc)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    return fpr, tpr, thresholds, auc_score
