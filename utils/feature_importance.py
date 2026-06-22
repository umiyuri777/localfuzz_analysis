"""
特徴量重要度の集計と LaTeX 表出力の共通ユーティリティ。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

DEFAULT_DECIMALS = 4


def normalized_feature_name(name: str) -> str:
    """ColumnTransformer が付与する接頭辞を外した特徴量名を返す。"""
    if name.startswith("num__"):
        return name[len("num__") :]
    if name.startswith("cat__"):
        return name[len("cat__") :]
    return name


def is_cpnum_dir_feature(name: str) -> bool:
    """cpNum_dir の one-hot 列かどうかを判定する。"""
    normalized = normalized_feature_name(name)
    return normalized == "cpNum_dir" or normalized.startswith("cpNum_dir_")


def latex_feature_name(name: str) -> str:
    """特徴量名を LaTeX 表のセル用に整形する。"""
    name = normalized_feature_name(name)
    if name.startswith("cpNum_dir_"):
        level = name.split("_", 2)[-1]
        return rf"\mathbb{{1}}[\mathrm{{cpNum\_dir}}{{=}}{level}]"
    latex_names = {
        "tree": r"tree",
        "cpNum": r"cpNum",
        "cpNum_range": r"cpNum\_range",
    }
    return latex_names.get(name, name.replace("_", r"\_"))


def format_latex_value(value: float, decimals: int = DEFAULT_DECIMALS) -> str:
    """数値を LaTeX 表のセル用に整形する。NaN のときは --- を返す。"""
    if value is None or np.isnan(value):
        return "---"
    return f"{value:.{decimals}f}"


def get_model_step_name(pipeline: Pipeline) -> str:
    """前処理以外のパイプラインステップ名（学習器）を返す。"""
    return next(name for name in pipeline.named_steps if name != "preprocess")


def _build_importance_stats(
    feature_names: list[str],
    importances: np.ndarray,
    *,
    exclude_cpnum_dir: bool,
) -> list[dict[str, float | str]]:
    stats: list[dict[str, float | str]] = []
    for idx, feature in enumerate(feature_names):
        if exclude_cpnum_dir and is_cpnum_dir_feature(feature):
            continue
        fold_values = importances[:, idx]
        stats.append({
            "feature": feature,
            "mean": float(np.mean(fold_values)),
            "std": float(np.std(fold_values)),
            "min": float(np.min(fold_values)),
            "max": float(np.max(fold_values)),
        })
    stats.sort(key=lambda row: row["mean"], reverse=True)
    return stats


def extract_fold_importances_from_cv(
    cv_results: dict[str, Any],
    step_name: str,
    *,
    exclude_cpnum_dir: bool = True,
) -> dict[str, list[float]]:
    """交差検証結果から各特徴量の fold 別重要度を辞書で返す。"""
    n_folds = len(cv_results["estimator"])
    feature_names: list[str] | None = None
    importances: np.ndarray | None = None

    for fold_idx, estimator in enumerate(cv_results["estimator"]):
        preprocess = estimator.named_steps["preprocess"]
        fold_feature_names = list(preprocess.get_feature_names_out())
        if feature_names is None:
            feature_names = fold_feature_names
            importances = np.zeros((n_folds, len(feature_names)))
        elif fold_feature_names != feature_names:
            raise ValueError("前処理後の特徴量名が fold 間で一致しません。")

        model = estimator.named_steps[step_name]
        assert importances is not None
        importances[fold_idx] = model.feature_importances_

    assert feature_names is not None
    assert importances is not None

    result: dict[str, list[float]] = {}
    for idx, feature in enumerate(feature_names):
        if exclude_cpnum_dir and is_cpnum_dir_feature(feature):
            continue
        normalized = normalized_feature_name(feature)
        result[normalized] = [float(v) for v in importances[:, idx]]

    return result


def compute_feature_importance_stats_from_cv(
    cv_results: dict[str, Any],
    step_name: str,
    *,
    exclude_cpnum_dir: bool = True,
) -> list[dict[str, float | str]]:
    """交差検証結果から各 fold の特徴量重要度を集計する。"""
    n_folds = len(cv_results["estimator"])
    feature_names: list[str] | None = None
    importances: np.ndarray | None = None

    for fold_idx, estimator in enumerate(cv_results["estimator"]):
        preprocess = estimator.named_steps["preprocess"]
        fold_feature_names = list(preprocess.get_feature_names_out())
        if feature_names is None:
            feature_names = fold_feature_names
            importances = np.zeros((n_folds, len(feature_names)))
        elif fold_feature_names != feature_names:
            raise ValueError("前処理後の特徴量名が fold 間で一致しません。")

        model = estimator.named_steps[step_name]
        assert importances is not None
        importances[fold_idx] = model.feature_importances_

    assert feature_names is not None
    assert importances is not None
    return _build_importance_stats(
        feature_names,
        importances,
        exclude_cpnum_dir=exclude_cpnum_dir,
    )


def compute_feature_importance_stats_from_pipeline(
    pipeline: Pipeline,
    step_name: str | None = None,
    *,
    exclude_cpnum_dir: bool = True,
) -> list[dict[str, float | str]]:
    """学習済みパイプライン1本から特徴量重要度を返す（単一学習の集計形式）。"""
    model_step = step_name or get_model_step_name(pipeline)
    preprocess = pipeline.named_steps["preprocess"]
    feature_names = list(preprocess.get_feature_names_out())
    model = pipeline.named_steps[model_step]
    importances = np.asarray(model.feature_importances_, dtype=float).reshape(1, -1)
    return _build_importance_stats(
        feature_names,
        importances,
        exclude_cpnum_dir=exclude_cpnum_dir,
    )


def format_latex_all_importance_table(
    model_importances: list[tuple[str, list[dict[str, float | str]]]],
    caption: str,
    label: str,
    *,
    decimals: int = DEFAULT_DECIMALS,
) -> str:
    """複数手法の特徴量重要度を LaTeX 表形式の文字列に整形する。"""
    lines = [
        r"\begin{table}[tb]",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        r"    \centering",
        r"    \begin{tabular}{llrrrr}",
        r"        \hline",
        r"        手法 & 特徴量 & 平均重要度 & 標準偏差 & 最小値 & 最大値 \\\hline \hline",
    ]

    for model_name, importance_rows in model_importances:
        n_features = len(importance_rows)
        for row_idx, row in enumerate(importance_rows):
            if row_idx == 0:
                method_cell = rf"\multirow{{{n_features}}}{{*}}{{{model_name}}}         "
            else:
                method_cell = " "

            suffix = r" \\\hline" if row_idx == n_features - 1 else r" \\"
            lines.append(
                f"        {method_cell}"
                f"& {latex_feature_name(str(row['feature']))} "
                f"& {format_latex_value(float(row['mean']), decimals)}"
                f" & {format_latex_value(float(row['std']), decimals)}"
                f" & {format_latex_value(float(row['min']), decimals)}"
                f" & {format_latex_value(float(row['max']), decimals)}{suffix}"
            )

    lines.extend([
        r"    \end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)
