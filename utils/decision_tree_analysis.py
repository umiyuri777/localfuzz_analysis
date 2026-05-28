"""
決定木分析の共通クラス。

- 混同行列
- 10分割交差検証（層化）
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .data_loader import BUG_PREDICTION_FEATURE_NAMES
from .metrics import (
    calculate_binary_metrics,
    print_binary_metrics,
    print_confusion_matrix,
)
from .preprocessing import bug_prediction_feature_names, build_bug_prediction_column_transformer

DEFAULT_FEATURE_NAMES = list(BUG_PREDICTION_FEATURE_NAMES)
DEFAULT_N_SPLITS = 10
DEFAULT_SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


class DecisionTreeAnalyzer:
    """
    決定木の学習・混同行列・10分割交差検証を扱うクラス。
    """

    def __init__(
        self,
        feature_names: Optional[list] = None,
        target_name: str = "bug_detected_any",
        random_state: int = 42,
        n_splits: int = DEFAULT_N_SPLITS,
        include_tree: bool = True,
        **tree_kwargs: Any,
    ):
        """
        Args:
            feature_names: 説明変数の列名。
            target_name: 目的変数の列名。
            random_state: 乱数シード。
            n_splits: 交差検証の分割数（デフォルト10）。
            include_tree: True なら tree 列を含む（tree-different）。False は speedUpItem 用。
            **tree_kwargs: DecisionTreeClassifier に渡す引数（例: max_depth=None, min_samples_leaf=0.05）。
        """
        self.include_tree = include_tree
        self.feature_names = feature_names or bug_prediction_feature_names(
            include_tree=include_tree,
        )
        self.target_name = target_name
        self.random_state = random_state
        self.n_splits = n_splits
        self.tree_kwargs = tree_kwargs or {"max_depth": None, "min_samples_leaf": 0.05}

        self.pipeline: Optional[Pipeline] = None
        self.X_train_: Optional[pd.DataFrame] = None
        self.X_test_: Optional[pd.DataFrame] = None
        self.y_train_: Optional[pd.Series] = None
        self.y_test_: Optional[pd.Series] = None

    def build_pipeline(self, model_step_name: str = "model") -> Pipeline:
        """前処理 + DecisionTreeClassifier の Pipeline を構築する。"""
        self.pipeline = Pipeline([
            (
                "preprocess",
                build_bug_prediction_column_transformer(include_tree=self.include_tree),
            ),
            (
                model_step_name,
                DecisionTreeClassifier(random_state=self.random_state, **self.tree_kwargs),
            ),
        ])
        return self.pipeline

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> "DecisionTreeAnalyzer":
        if self.pipeline is None:
            self.build_pipeline()
        X_use = X[self.feature_names] if isinstance(X, pd.DataFrame) else X
        self.pipeline.fit(X_use, y)
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("fit を先に実行してください。")
        X_use = X[self.feature_names] if isinstance(X, pd.DataFrame) else X
        return self.pipeline.predict(X_use)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("fit を先に実行してください。")
        X_use = X[self.feature_names] if isinstance(X, pd.DataFrame) else X
        return self.pipeline.predict_proba(X_use)[:, 1]

    def set_train_test(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        self.X_train_ = X_train
        self.X_test_ = X_test
        self.y_train_ = y_train
        self.y_test_ = y_test

    def evaluate(
        self,
        X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_test: Optional[Union[pd.Series, np.ndarray]] = None,
        title: str = "決定木",
    ) -> dict:
        """混同行列と評価指標を表示し、指標の辞書を返す。"""
        X = X_test if X_test is not None else self.X_test_
        y = y_test if y_test is not None else self.y_test_
        if X is None or y is None:
            raise ValueError("X_test と y_test を渡すか、set_train_test で設定してください。")

        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)
        print_confusion_matrix(y, y_pred, title=f"{title} - 混同行列")
        metrics = calculate_binary_metrics(y, y_pred, y_pred_proba=y_pred_proba)
        print_binary_metrics(metrics, title=title)
        return metrics

    def cross_validate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        scoring: Optional[Dict[str, str]] = None,
        return_estimator: bool = False,
        title: str = "決定木 10分割交差検証",
    ) -> dict:
        """
        10分割層化交差検証を実行し、結果を表示して返す。
        """
        if self.pipeline is None:
            self.build_pipeline()
        X_use = X[self.feature_names] if isinstance(X, pd.DataFrame) else X
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        scoring = scoring or DEFAULT_SCORING

        cv_results = cross_validate(
            self.pipeline,
            X_use,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            return_estimator=return_estimator,
            n_jobs=-1,
        )

        print("=" * 70)
        print(f"【{title}】")
        print("=" * 70)
        print("\n【全体の統計】")
        print("-" * 70)
        for key in scoring:
            test_key = f"test_{key}"
            if test_key not in cv_results:
                continue
            vals = cv_results[test_key]
            mean_v = vals.mean()
            std_v = vals.std()
            print(f"{key}: 平均 {mean_v:.4f} ± {std_v:.4f}, 範囲 [{vals.min():.4f}, {vals.max():.4f}]")
        print("\n【各フォールドの詳細結果】")
        print("-" * 70)
        headers = ["Fold"] + list(scoring.keys())
        print("  ".join(f"{h:>12}" for h in headers))
        print("-" * 70)
        for i in range(self.n_splits):
            row = [f"{i+1}"] + [f"{cv_results[f'test_{k}'][i]:.4f}" for k in scoring]
            print("  ".join(f"{v:>12}" for v in row))
        return cv_results


def build_decision_tree_pipeline(
    *,
    include_tree: bool = True,
    model_step_name: str = "model",
    random_state: int = 42,
    **tree_kwargs: Any,
) -> Pipeline:
    """決定木の学習用 Pipeline を構築して返す（fit 前）。"""
    analyzer = DecisionTreeAnalyzer(
        include_tree=include_tree,
        random_state=random_state,
        **tree_kwargs,
    )
    return analyzer.build_pipeline(model_step_name=model_step_name)
