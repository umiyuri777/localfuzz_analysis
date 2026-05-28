"""
ロジスティック回帰分析の共通クラス。

- 混同行列
- 適合率・再現率・F値・AUC
- オッズ比
- ロジット式の表示
- ロジスティック回帰曲線の描画
"""

from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_loader import BUG_PREDICTION_FEATURE_NAMES
from .metrics import (
    calculate_binary_metrics,
    compute_roc_curve,
    print_binary_metrics,
    print_confusion_matrix,
)
from .preprocessing import (
    bug_prediction_feature_names,
    build_bug_prediction_column_transformer,
)


DEFAULT_FEATURE_NAMES = list(BUG_PREDICTION_FEATURE_NAMES)


class LogisticRegressionAnalyzer:
    """
    ロジスティック回帰の学習・評価・オッズ比・ロジット式・ROC曲線を一括で扱うクラス。
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        target_name: str = "bug_detected",
        random_state: int = 42,
        include_tree: bool = True,
        **logistic_kwargs,
    ):
        """
        Args:
            feature_names: 説明変数の列名。None の場合は include_tree に応じたデフォルトを使用。
            target_name: 目的変数の列名。
            random_state: 再現性のための乱数シード。
            include_tree: True なら tree 列を含む（tree-different）。False は speedUpItem 用。
            **logistic_kwargs: LogisticRegression に渡す追加引数（例: max_iter=1000, C=1.0）。
        """
        self.include_tree = include_tree
        self.feature_names = feature_names or bug_prediction_feature_names(
            include_tree=include_tree,
        )
        self.target_name = target_name
        self.random_state = random_state
        self.logistic_kwargs = logistic_kwargs

        self.pipeline: Optional[Pipeline] = None
        self.scaler_: Optional[StandardScaler] = None
        self.model_: Optional[LogisticRegression] = None
        self.transformed_feature_names_: Optional[List[str]] = None
        self.X_train_: Optional[pd.DataFrame] = None
        self.X_test_: Optional[pd.DataFrame] = None
        self.y_train_: Optional[pd.Series] = None
        self.y_test_: Optional[pd.Series] = None

    def build_pipeline(self, model_step_name: str = "model") -> Pipeline:
        """前処理（数値標準化 + cpNum_dir one-hot）+ LogisticRegression の Pipeline を構築する。"""
        self.pipeline = Pipeline([
            (
                "preprocess",
                build_bug_prediction_column_transformer(include_tree=self.include_tree),
            ),
            (
                model_step_name,
                LogisticRegression(
                    random_state=self.random_state,
                    **self.logistic_kwargs,
                ),
            ),
        ])
        return self.pipeline

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> "LogisticRegressionAnalyzer":
        """
        説明変数 X と目的変数 y でパイプラインを学習する。
        fit 前に build_pipeline が未呼び出しの場合は自動で構築する。
        """
        if self.pipeline is None:
            self.build_pipeline()
        X_df = X[self.feature_names] if isinstance(X, pd.DataFrame) else X
        self.pipeline.fit(X_df, y)
        preprocess = self.pipeline.named_steps["preprocess"]
        self.scaler_ = preprocess.named_transformers_["num"]
        model_step = next(
            name for name in self.pipeline.named_steps if name != "preprocess"
        )
        self.model_ = self.pipeline.named_steps[model_step]
        self.transformed_feature_names_ = list(preprocess.get_feature_names_out())
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """予測ラベルを返す。"""
        if self.pipeline is None:
            raise RuntimeError("fit を先に実行してください。")
        X_df = X[self.feature_names] if isinstance(X, pd.DataFrame) else X
        return self.pipeline.predict(X_df)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """正クラス（1）の確率を返す。"""
        if self.pipeline is None:
            raise RuntimeError("fit を先に実行してください。")
        X_df = X[self.feature_names] if isinstance(X, pd.DataFrame) else X
        return self.pipeline.predict_proba(X_df)[:, 1]

    def set_train_test(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        """train_test_split の結果を保持し、評価メソッドで利用する。"""
        self.X_train_ = X_train
        self.X_test_ = X_test
        self.y_train_ = y_train
        self.y_test_ = y_test

    def evaluate(
        self,
        X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_test: Optional[Union[pd.Series, np.ndarray]] = None,
        title: str = "ロジスティック回帰",
    ) -> dict:
        """
        テストデータで混同行列と評価指標を表示し、指標の辞書を返す。
        X_test, y_test が未指定の場合は set_train_test で設定した値を使用する。
        """
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

    def get_odds_ratios(self) -> pd.DataFrame:
        """
        各説明変数のオッズ比（exp(係数)）を DataFrame で返す。
        fit 済みのモデル（標準化済み係数）を使用する。
        """
        if self.model_ is None:
            raise RuntimeError("fit を先に実行してください。")
        coef = self.model_.coef_[0]
        odds_ratios = np.exp(coef)
        names = self.transformed_feature_names_ or self.feature_names
        return pd.DataFrame({
            "特徴量": names,
            "係数": coef,
            "オッズ比": odds_ratios,
        })

    def get_logit_formula(
        self,
        use_original_scale: bool = True,
        scaler_mean_scale: Optional[tuple] = None,
    ) -> tuple:
        """
        ロジット式の文字列と係数（切片・各特徴量）を返す。

        Args:
            use_original_scale: True のとき元の変数に対する係数に変換して表示。
            scaler_mean_scale: (mean_array, scale_array) を渡すとその値で変換。
                未指定で use_original_scale が True のときは self.scaler_ を使用。

        Returns:
            (式の説明文字列, intercept, coef_array)
        """
        if self.model_ is None or self.scaler_ is None:
            raise RuntimeError("fit を先に実行してください。")

        intercept = self.model_.intercept_[0]
        coef = self.model_.coef_[0]
        names = self.transformed_feature_names_ or self.feature_names

        if use_original_scale:
            if scaler_mean_scale is not None:
                mean_, scale_ = scaler_mean_scale
            else:
                mean_ = self.scaler_.mean_
                scale_ = self.scaler_.scale_
            n_num = len(mean_)
            coef_num = coef[:n_num]
            coef_other = coef[n_num:]
            coef = np.concatenate([coef_num / scale_, coef_other])
            intercept = intercept - np.sum(coef_num * mean_ / scale_)
            var_suffix = ""
        else:
            var_suffix = "_scaled"

        formula = "logit(P) = β₀ + " + " + ".join(
            f"β_{i+1}·{name}{var_suffix}" for i, name in enumerate(names)
        )
        return formula, intercept, coef

    def print_logit_formula(
        self,
        use_original_scale: bool = True,
        title: str = "ロジスティック回帰式",
    ) -> None:
        """
        ロジット式を標準出力に表示する。
        """
        formula, intercept, coef = self.get_logit_formula(use_original_scale=use_original_scale)
        print("=" * 70)
        print(f"【{title}】")
        print("=" * 70)
        print(f"\n{formula}")
        print("\nまたは確率の形式:")
        names = self.transformed_feature_names_ or self.feature_names
        linear = f"{intercept:.6f} + " + " + ".join(f"{c:.6f}·{n}" for c, n in zip(coef, names))
        print(f"P = 1 / (1 + exp(-({linear})))")
        print("\n係数:")
        names = self.transformed_feature_names_ or self.feature_names
        print(f"β₀ (切片) = {intercept:.6f}")
        for i, name in enumerate(names):
            print(f"β_{i+1} ({name}) = {coef[i]:.6f}")
        print("\n【具体的な回帰式（数値代入）】")
        parts = [f"{intercept:.6f}"]
        for i, name in enumerate(names):
            parts.append(f"{coef[i]:.6f}·{name}")
        print("logit(P) = " + " + ".join(parts))

    def plot_roc_curve(
        self,
        X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_test: Optional[Union[pd.Series, np.ndarray]] = None,
        title: str = "ROC Curve - ロジスティック回帰",
        figsize: tuple = (8, 6),
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """
        ROC曲線を描画する。
        """
        X = X_test if X_test is not None else self.X_test_
        y = y_test if y_test is not None else self.y_test_
        if X is None or y is None:
            raise ValueError("X_test と y_test を渡すか、set_train_test で設定してください。")

        y_pred_proba = self.predict_proba(X)
        fpr, tpr, _, auc_score = compute_roc_curve(y, y_pred_proba)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_score:.4f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random (AUC = 0.5000)")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_logistic_curve(
        self,
        feature_index: int = 0,
        X_fixed: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        n_points: int = 200,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        figsize: tuple = (8, 5),
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """
        1つの説明変数を変化させ、その他を固定したときのロジスティック回帰曲線（確率）を描画する。

        Args:
            feature_index: 横軸に取る説明変数のインデックス（0: tree, 1: cpNum, ...）。
            X_fixed: 他変数の固定値。行は1行で、列は feature_names の順。
                未指定の場合は学習データの平均（または0）で固定。
            n_points: 曲線の点数。
            title: グラフタイトル。
            xlabel: 横軸ラベル。未指定のときは feature_names[feature_index]。
        """
        if self.model_ is None or self.scaler_ is None:
            raise RuntimeError("fit を先に実行してください。")

        name = self.feature_names[feature_index]
        if X_fixed is None:
            fixed = np.zeros(len(self.feature_names))
            if self.X_train_ is not None and isinstance(self.X_train_, pd.DataFrame):
                for i, col in enumerate(self.feature_names):
                    if col in self.X_train_.columns:
                        fixed[i] = self.X_train_[col].mean()
        else:
            fixed = np.asarray(X_fixed).flatten()
            if len(fixed) != len(self.feature_names):
                fixed = np.resize(fixed, len(self.feature_names))

        x_min = 0
        x_max = 1000
        if self.X_train_ is not None and isinstance(self.X_train_, pd.DataFrame) and name in self.X_train_.columns:
            x_min = self.X_train_[name].min()
            x_max = self.X_train_[name].max()
        x_line = np.linspace(x_min, x_max, n_points)

        X_line = np.tile(fixed, (n_points, 1))
        X_line[:, feature_index] = x_line
        X_line_df = pd.DataFrame(X_line, columns=self.feature_names)
        proba = self.predict_proba(X_line_df)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        ax.plot(x_line, proba, color="darkorange", lw=2)
        ax.set_xlabel(xlabel or name, fontsize=12)
        ax.set_ylabel("P(bug_detected=1)", fontsize=12)
        ax.set_title(title or f"ロジスティック回帰曲線（{name}）", fontsize=14, fontweight="bold")
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig


def build_logistic_regression_pipeline(
    *,
    include_tree: bool = True,
    model_step_name: str = "model",
    random_state: int = 42,
    **logistic_kwargs,
) -> Pipeline:
    """ロジスティック回帰の学習用 Pipeline を構築して返す（fit 前）。"""
    analyzer = LogisticRegressionAnalyzer(
        include_tree=include_tree,
        random_state=random_state,
        **logistic_kwargs,
    )
    return analyzer.build_pipeline(model_step_name=model_step_name)
