"""
バグ予測モデル用の前処理（ColumnTransformer）。

tree-different: tree + cpNum + cpNum_range（標準化）, cpNum_dir（one-hot）
speedUpItem:    cpNum + cpNum_range（標準化）, cpNum_dir（one-hot）— tree なし
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_loader import (
    BUG_PREDICTION_FEATURE_NAMES,
    SPEEDUP_BUG_PREDICTION_FEATURE_NAMES,
)

CPNUM_DIR_QUADRANT_CATEGORIES = [1, 2, 3, 4]


def bug_prediction_feature_names(*, include_tree: bool) -> list[str]:
    """データセット種別に応じた説明変数の列名を返す。"""
    if include_tree:
        return list(BUG_PREDICTION_FEATURE_NAMES)
    return list(SPEEDUP_BUG_PREDICTION_FEATURE_NAMES)


def build_bug_prediction_column_transformer(*, include_tree: bool = True) -> ColumnTransformer:
    """
    cpNum_dir を名義尺度（4象限）として one-hot 化し、数値列を標準化する。

    Args:
        include_tree: True なら tree 列も数値特徴量に含める（tree-different 用）。
    """
    if include_tree:
        numeric_features = ["tree", "cpNum", "cpNum_range"]
    else:
        numeric_features = ["cpNum", "cpNum_range"]
    categorical_features = ["cpNum_dir"]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(
                    categories=[CPNUM_DIR_QUADRANT_CATEGORIES],
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_speedup_bug_column_transformer() -> ColumnTransformer:
    """speedUpItem 用（tree なし）の前処理。後方互換のエイリアス。"""
    return build_bug_prediction_column_transformer(include_tree=False)
