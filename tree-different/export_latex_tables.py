"""
tree-different/task0, task1, task2 について、
4手法の10分割交差検証結果を LaTeX 表形式で一括出力する。

出力:
- 全シナリオ横断の評価結果表（適合率・再現率・F値・AUC）
- ベースライン・各手法の比較表（適合率・再現率・F値・正解率・AUC）
- 各手法の評価結果（適合率・再現率・F値・AUC）
- ロジスティック回帰以外の特徴量重要度
- ロジスティック回帰式（標準化数値＋cpNum_dir の one-hot）
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_loader import (  # noqa: E402
    BUG_PREDICTION_FEATURE_NAMES,
    collect_data_per_run,
    parse_directory_name,
)
from utils.decision_tree_analysis import build_decision_tree_pipeline  # noqa: E402
from utils.gradient_boosting_analysis import build_gradient_boosting_pipeline  # noqa: E402
from utils.logistic_regression_analysis import (  # noqa: E402
    FittedLogisticCoefficients,
    build_logistic_regression_pipeline,
    fit_logistic_regression,
)
from utils.feature_importance import (  # noqa: E402
    compute_feature_importance_stats_from_cv,
    extract_fold_importances_from_cv,
    format_latex_all_importance_table,
    format_latex_value,
    latex_feature_name,
)
from utils.random_forest_analysis import build_random_forest_pipeline  # noqa: E402

RANDOM_STATE = 42
N_SPLITS = 10
METRIC_DECIMALS = 4
LOGISTIC_COEF_DECIMALS = 3
LOGISTIC_INTERCEPT_DECIMALS = 3
LOGISTIC_TERMS_ON_FIRST_LINE = 2

METRIC_ROWS = [
    ("precision", "適合率"),
    ("recall", "再現率"),
    ("f1", "F値"),
    ("auc", "AUC"),
]

SCORING = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "accuracy": "accuracy",
    "roc_auc": "roc_auc",
}

CLASS_LABELS_JA = {
    1: "不具合発見する",
    0: "不具合発見しない",
}

BASELINE_MODEL_NAME = "BL"
BASELINE_DISPLAY_NAME = "ベースライン（常にバグ発見）"

MODEL_SPECS: list[tuple[str, str, str]] = [
    ("logistic", "ロジスティック回帰", "lr"),
    ("tree", "決定木", "dt"),
    ("rf", "ランダムフォレスト", "rf"),
    ("gb", "勾配ブースティング", "gb"),
]

CONSOLE_MODEL_ORDER = [BASELINE_MODEL_NAME] + [name for _, name, _ in MODEL_SPECS]


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    description: str
    dataset_macro: str
    metrics_caption: str
    metrics_label: str
    importance_caption: str
    importance_label: str
    logistic_equation_label: str
    standardization_label: str


TASK_CONFIGS: dict[str, TaskConfig] = {
    "task0": TaskConfig(
        task_id="task0",
        description="1回の実行でバグ発見（per_run / bug_detected）",
        dataset_macro=r"\testfirst",
        metrics_caption=r"各手法の評価結果（\testfirst）",
        metrics_label="tab:all_single_metrics_prospect",
        importance_caption=r"各手法における特徴量重要度（\testfirst）",
        importance_label="tab:all_importance",
        logistic_equation_label="eq:logistic_single_scaled",
        standardization_label="eq:standardization_1",
    ),
    "task1": TaskConfig(
        task_id="task1",
        description="5回中1回でもバグ発見（bug_detected_any）",
        dataset_macro=r"\testsecond",
        metrics_caption=r"各手法の評価結果（\testsecond）",
        metrics_label="tab:all_single_metrics_prospect_task1",
        importance_caption=r"各手法における特徴量重要度（\testsecond）",
        importance_label="tab:all_importance_task1",
        logistic_equation_label="eq:logistic_multi_scaled_prospect",
        standardization_label="eq:standardization_task1",
    ),
    "task2": TaskConfig(
        task_id="task2",
        description="5回全てバグ発見（bug_detected_all）",
        dataset_macro=r"\testthird",
        metrics_caption=r"各手法の評価結果（\testthird）",
        metrics_label="tab:all_single_metrics_prospect_task2",
        importance_caption=r"各手法における特徴量重要度（\testthird）",
        importance_label="tab:all_importance_task2",
        logistic_equation_label="eq:logistic_all_scaled_prospect",
        standardization_label="eq:standardization_task2",
    ),
}

BOX_PLOT_MODEL_ORDER = [BASELINE_MODEL_NAME, "LR", "DT", "RF", "GB"]
COMBINED_TABLE_DECIMALS = 2
TASK_ORDER = ["task0", "task1", "task2"]
IMPORTANCE_BOX_PLOT_MODEL_ORDER = ["DT", "RF", "GB"]
IMPORTANCE_BOX_PLOT_FEATURE_ORDER = [
    "tree",
    "cpNum",
    "cpNum_range",
    "cpNum_dir_2",
    "cpNum_dir_3",
    "cpNum_dir_4",
]


def _default_logs_root() -> Path:
    return Path(__file__).resolve().parent / "Logs"


def _format_value(value: float, decimals: int = METRIC_DECIMALS) -> str:
    return format_latex_value(value, decimals)


def _format_latex_metric(value: float | None, decimals: int = METRIC_DECIMALS) -> str:
    """評価指標を LaTeX 表のセル用に整形する。NaN のときは --- を返す。"""
    if value is None or np.isnan(value):
        return "---"
    return f"{value:.{decimals}f}"


def format_latex_baseline_result(
    metrics: dict,
    *,
    caption: str,
    label: str,
    decimals: int = METRIC_DECIMALS,
) -> str:
    """ベースラインの評価指標を LaTeX 表形式の文字列に整形する。"""
    rows = [
        ("適合率", metrics["precision"]),
        ("再現率", metrics["recall"]),
        ("F値", metrics["f1"]),
        ("AUC", metrics.get("auc")),
    ]
    body_lines = []
    for i, (name, value) in enumerate(rows):
        suffix = r" \\ \hline" if i == len(rows) - 1 else r" \\"
        body_lines.append(f"    {name} & {_format_latex_metric(value, decimals)}{suffix}")
    body = "\n".join(body_lines)
    return "\n".join([
        r"\begin{table}[t]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \begin{tabular}{l|r} \hline",
        r"    評価指標 & 値 \\ \hline \hline",
        body,
        r"  \end{tabular}",
        r"\end{table}",
    ])


def format_latex_dataset_distribution(
    y: pd.Series,
    *,
    caption: str,
    label: str,
) -> str:
    """目的変数 y のクラス分布を LaTeX 表形式の文字列に整形する。"""
    counts = y.value_counts()
    n_positive = int(counts.get(1, 0))
    n_negative = int(counts.get(0, 0))
    return "\n".join([
        r"\begin{table}[t]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \begin{tabular}{l|r} \hline",
        r"    分類 & 数 \\ \hline \hline",
        f"    {CLASS_LABELS_JA[1]} & {n_positive}\\\\",
        f"    {CLASS_LABELS_JA[0]} & {n_negative}\\\\ \\hline",
        r"  \end{tabular}",
        r"\end{table}",
    ])


def format_latex_comparison_table(
    results: list[tuple[str, dict]],
    target_label: str,
    *,
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """アルゴリズム比較結果を LaTeX 表形式の文字列に整形する。"""
    if caption is None:
        caption = rf"アルゴリズム比較（目的: {target_label}）"
    if label is None:
        label = "tab:tree_different_model_comparison"

    lines = [
        r"\begin{table}[H]",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        r"    \centering",
        r"    \begin{tabular}{lrrrr}",
        r"        \hline",
        r"        アルゴリズム & 適合率 & 再現率 & F値 & AUC \\",
        r"        \hline \hline",
    ]
    for name, metrics in results:
        lines.append(
            f"        {name} & "
            f"{_format_latex_metric(metrics['precision'])} & "
            f"{_format_latex_metric(metrics['recall'])} & "
            f"{_format_latex_metric(metrics['f1'])} & "
            f"{_format_latex_metric(metrics.get('auc'))} \\\\"
        )
    lines.extend([
        r"        \hline",
        r"    \end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def _transformed_feature_to_latex_term(feature_name: str) -> str:
    """
    ColumnTransformer 出力の特徴量名を LaTeX の説明変数表記に変換する。

    - num__tree など → tree_{std}（標準化済み数値）
    - cat__cpNum_dir_2 など → 名義ダミー \\mathbb{1}[\\mathrm{cpNum\\_dir}{=}2]
    """
    if feature_name.startswith("num__"):
        base = feature_name[len("num__") :]
        return rf"{latex_feature_name(base)}_{{std}}"
    if feature_name.startswith("cat__"):
        rest = feature_name[len("cat__") :]
        if rest.startswith("cpNum_dir_"):
            level = rest.split("_", 2)[-1]
            return rf"\mathbb{{1}}[\mathrm{{cpNum\_dir}}{{=}}{level}]"
        return rest.replace("_", r"\_")
    return feature_name.replace("_", r"\_")


def _format_latex_logistic_coef_term(
    coef: float,
    var_latex: str,
    decimals: int = LOGISTIC_COEF_DECIMALS,
) -> str:
    sign = "+" if coef >= 0 else "-"
    return rf"{sign} {abs(coef):.{decimals}f}\, {var_latex}"


def _format_latex_mean(mean: float) -> str:
    if abs(mean - round(mean, 1)) < 1e-9:
        return f"{mean:.1f}"
    return f"{mean:.2f}"


def _format_latex_scale(scale: float) -> str:
    if scale >= 10:
        return f"{scale:.1f}"
    return f"{scale:.2f}"


def format_latex_logistic_scaled_equation(
    coefficients: FittedLogisticCoefficients,
    *,
    label: str = "eq:logistic_single_scaled",
    coef_decimals: int = LOGISTIC_COEF_DECIMALS,
    intercept_decimals: int = LOGISTIC_INTERCEPT_DECIMALS,
    terms_on_first_line: int = LOGISTIC_TERMS_ON_FIRST_LINE,
) -> str:
    """標準化数値＋cpNum_dir ダミーを用いたロジスティック回帰式を LaTeX で整形する。"""
    terms = [
        _format_latex_logistic_coef_term(
            coef,
            _transformed_feature_to_latex_term(name),
            coef_decimals,
        )
        for coef, name in zip(coefficients.coefficients, coefficients.feature_names)
    ]
    z_first = (
        f"{coefficients.intercept:.{intercept_decimals}f} "
        + " ".join(terms[:terms_on_first_line])
    )
    remaining = terms[terms_on_first_line:]

    lines = [
        r"\begin{equation}",
        f"\\label{{{label}}}",
        r"\begin{aligned}",
        r"  P &= \frac{1}{1 + \exp(-z)}, \\",
        rf"  z &= {z_first} \\",
    ]
    for term in remaining:
        lines.append(rf"    &\quad {term} \\")
    lines[-1] = lines[-1].rstrip(" \\")

    lines.extend([
        r"\end{aligned}",
        r"\end{equation}",
    ])
    return "\n".join(lines)


def format_latex_standardization_equation(
    coefficients: FittedLogisticCoefficients,
    *,
    label: str = "eq:standardization_1",
) -> str:
    """数値説明変数の標準化式（平均・標準偏差）を LaTeX で整形する。"""
    stat_lines = []
    for name, mean, scale in coefficients.numeric_standardization:
        var = latex_feature_name(name)
        stat_lines.append(
            rf"  {var} &: \mu={_format_latex_mean(mean)},\ "
            rf"\sigma \approx {_format_latex_scale(scale)}, \\"
        )

    ref = coefficients.reference_cpnum_dir

    return "\n".join([
        r"\begin{equation}",
        f"\\label{{{label}}}",
        r"\begin{aligned}",
        r"  x_{std} &= \frac{x - \mu}{\sigma}, \\",
        *stat_lines,
        rf"  \mathrm{{cpNum\_dir}} &: \text{{象限 {ref} を基準カテゴリとする}} \\",
        r"    &\quad \text{one-hot 符号化}",
        r"\end{aligned}",
        r"\end{equation}",
    ])


def format_latex_logistic_equations(
    coefficients: FittedLogisticCoefficients,
    *,
    logistic_label: str = "eq:logistic_single_scaled",
    standardization_label: str = "eq:standardization_1",
) -> str:
    """ロジスティック回帰式と標準化式の LaTeX を連結して返す。"""
    return "\n\n".join([
        format_latex_logistic_scaled_equation(
            coefficients,
            label=logistic_label,
        ),
        format_latex_standardization_equation(
            coefficients,
            label=standardization_label,
        ),
    ])


def _bug_row_to_category(bug_row: list[str]) -> str:
    if bug_row == ["timeout"]:
        return "timeout"
    if bug_row == ["null"]:
        return "normal"
    return "bug"


def collect_data_for_tree_analysis(logs_root: Path, verbose: bool = False) -> pd.DataFrame:
    """task1/task2 ノートブックと同じ条件で集約データを収集する。"""
    data_records = []
    logs_path = Path(logs_root)

    for tree_dir in logs_path.glob("tree=*"):
        tree_value = int(tree_dir.name.split("=")[1])

        for param_dir in tree_dir.iterdir():
            if not param_dir.is_dir():
                continue

            cpnum, cpnum_range, cpnum_dir = parse_directory_name(param_dir.name)
            if cpnum is None or cpnum_range is None or cpnum_dir is None:
                continue

            detected_bugs_path = param_dir / "detected_bugs.csv"
            exe_time_path = param_dir / "exe_time.csv"
            if not detected_bugs_path.exists() or not exe_time_path.exists():
                continue

            bug_results = []
            with open(detected_bugs_path, "r", encoding="utf-8") as bug_f:
                for bug_row in csv.reader(bug_f):
                    bug_results.append(_bug_row_to_category(bug_row))

            if len(bug_results) < 5:
                continue

            first_5_results = bug_results[:5]
            data_records.append({
                "tree": tree_value,
                "cpNum": cpnum,
                "cpNum_range": cpnum_range,
                "cpNum_dir": cpnum_dir,
                "bug_detected_any": 1 if "bug" in first_5_results else 0,
                "bug_detected_all": 1 if all(r == "bug" for r in first_5_results) else 0,
            })

    df = pd.DataFrame(data_records)
    if verbose and len(df) > 0:
        print(f"データ収集完了: {len(df)}件のレコード")
        print(df["bug_detected_any"].value_counts())
        print(df["bug_detected_all"].value_counts())
    return df


def load_task_dataset(task_id: str, logs_root: Path, verbose: bool = False):
    if task_id == "task0":
        df = collect_data_per_run(
            logs_root=str(logs_root),
            include_exe_time=False,
            verbose=verbose,
        )
        if len(df) == 0:
            raise ValueError(f"データが0件です: {logs_root}")
        X = df[BUG_PREDICTION_FEATURE_NAMES]
        y = df["bug_detected"]
        return X, y

    df = collect_data_for_tree_analysis(logs_root, verbose=verbose)
    if len(df) == 0:
        raise ValueError(f"データが0件です: {logs_root}")
    target = "bug_detected_any" if task_id == "task1" else "bug_detected_all"
    X = df[BUG_PREDICTION_FEATURE_NAMES]
    y = df[target]
    return X, y


def build_pipeline(model_key: str, task_id: str) -> tuple[Pipeline, str]:
    _ = task_id  # 将来 task 別ハイパーパラメータ用
    builders = {
        "logistic": ("lr", build_logistic_regression_pipeline),
        "tree": ("dt", build_decision_tree_pipeline),
        "rf": ("rf", build_random_forest_pipeline),
        "gb": ("gb", build_gradient_boosting_pipeline),
    }
    if model_key not in builders:
        raise ValueError(f"未知のモデル: {model_key}")
    step_name, builder = builders[model_key]
    return builder(
        include_tree=True,
        model_step_name=step_name,
        random_state=RANDOM_STATE,
    ), step_name


def run_cross_validation(X, y, pipeline):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=SCORING,
        return_train_score=False,
        return_estimator=True,
        n_jobs=-1,
    )


def summarize_scores(scores: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }


def _cv_test_scores(cv_results: dict, metric_key: str) -> np.ndarray:
    cv_metric = "roc_auc" if metric_key == "auc" else metric_key
    return cv_results[f"test_{cv_metric}"]


def extract_fold_scores(cv_results: dict) -> dict[str, list[float]]:
    """各評価指標の fold 別スコアを辞書で返す。"""
    result = {
        metric_key: [float(v) for v in _cv_test_scores(cv_results, metric_key)]
        for metric_key, _ in METRIC_ROWS
    }
    result["accuracy"] = [float(v) for v in cv_results["test_accuracy"]]
    return result


def compute_baseline_fold_scores(X, y) -> dict[str, list[float]]:
    """常にバグ発見と予測するベースラインの fold 別スコアを返す。"""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    y_array = np.asarray(y)
    fold_scores = {
        metric_key: []
        for metric_key, _ in METRIC_ROWS
    }
    fold_scores["accuracy"] = []
    fold_scores["auc"] = []

    for _, test_idx in cv.split(X, y_array):
        y_test = y_array[test_idx]
        y_pred = np.ones(len(y_test), dtype=int)
        y_proba = np.ones(len(y_test), dtype=float)
        fold_scores["precision"].append(
            float(precision_score(y_test, y_pred, zero_division=0))
        )
        fold_scores["recall"].append(
            float(recall_score(y_test, y_pred, zero_division=0))
        )
        fold_scores["f1"].append(
            float(f1_score(y_test, y_pred, zero_division=0))
        )
        fold_scores["accuracy"].append(float(accuracy_score(y_test, y_pred)))
        try:
            fold_scores["auc"].append(float(roc_auc_score(y_test, y_proba)))
        except ValueError:
            fold_scores["auc"].append(float("nan"))

    return fold_scores


def fold_scores_to_mean_metrics(fold_scores: dict[str, list[float]]) -> dict[str, float]:
    """fold 別スコアから compare_models 形式の指標辞書（平均値）を生成する。"""
    return {
        metric_key: float(np.mean(scores))
        for metric_key, scores in fold_scores.items()
    }


def build_console_results(
    fold_scores_by_model: dict[str, dict[str, list[float]]],
) -> list[tuple[str, dict[str, float]]]:
    """コンソール比較表用の (モデル名, 平均指標) リストを返す。"""
    display_names = {BASELINE_MODEL_NAME: BASELINE_DISPLAY_NAME}
    results: list[tuple[str, dict[str, float]]] = []
    for model_name in CONSOLE_MODEL_ORDER:
        if model_name not in fold_scores_by_model:
            continue
        display_name = display_names.get(model_name, model_name)
        results.append((
            display_name,
            fold_scores_to_mean_metrics(fold_scores_by_model[model_name]),
        ))
    return results


def _to_box_plot_model_name(model_name: str) -> str:
    """plot_metrics_boxplot.py 用の短いモデル名に変換する。"""
    mapping = {
        "ロジスティック回帰": "LR",
        "決定木": "DT",
        "ランダムフォレスト": "RF",
        "勾配ブースティング": "GB",
        BASELINE_MODEL_NAME: BASELINE_MODEL_NAME,
    }
    return mapping.get(model_name, model_name)


def build_box_plot_fold_scores(
    fold_scores: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, list[float]]]:
    """plot_metrics_boxplot.py 用にモデル名を揃え、定義順に並べ替える。"""
    metric_keys = {metric_key for metric_key, _ in METRIC_ROWS}
    renamed = {
        _to_box_plot_model_name(model_name): {
            metric_key: scores[metric_key]
            for metric_key in metric_keys
            if metric_key in scores
        }
        for model_name, scores in fold_scores.items()
    }
    return {
        model_name: renamed[model_name]
        for model_name in BOX_PLOT_MODEL_ORDER
        if model_name in renamed
    }


def build_box_plot_fold_importances(
    fold_importances: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, list[float]]]:
    """plot_metrics_boxplot.py 用に特徴量重要度の fold データを整形する。"""
    renamed = {
        _to_box_plot_model_name(model_name): features
        for model_name, features in fold_importances.items()
    }
    ordered: dict[str, dict[str, list[float]]] = {}
    for model_name in IMPORTANCE_BOX_PLOT_MODEL_ORDER:
        if model_name not in renamed:
            continue
        feature_scores = renamed[model_name]
        ordered[model_name] = {
            feature_name: feature_scores[feature_name]
            for feature_name in IMPORTANCE_BOX_PLOT_FEATURE_ORDER
            if feature_name in feature_scores
        }
    return ordered


def _format_python_float_list(values: list[float]) -> list[str]:
    return [f"{value:.{METRIC_DECIMALS}f}" for value in values]


def format_python_cv_fold_scores_entry(
    model_scores: dict[str, dict[str, list[float]]],
    *,
    indent: int = 4,
) -> list[str]:
    """1 task 分の CV fold スコアを Python 辞書リテラル行として整形する。"""
    pad = " " * indent
    lines: list[str] = []

    for model_idx, (model_name, metrics) in enumerate(model_scores.items()):
        lines.append(f'{pad}"{model_name}": {{')
        metric_items = list(metrics.items())
        for metric_idx, (metric_key, scores) in enumerate(metric_items):
            formatted_scores = ", ".join(_format_python_float_list(scores))
            comma = "," if metric_idx < len(metric_items) - 1 else ""
            lines.append(f'{pad}    "{metric_key}": [{formatted_scores}]{comma}')
        model_comma = "," if model_idx < len(model_scores) - 1 else ""
        lines.append(f"{pad}}}{model_comma}")

    return lines


def format_python_cv_fold_scores_block(
    all_fold_scores: dict[str, dict[str, dict[str, list[float]]]],
    *,
    variable_name: str = "CV_FOLD_SCORES",
) -> str:
    """
    plot_metrics_boxplot.py へ貼り付けやすい Python 辞書ブロックを生成する。

    task0 / task1 / task2 を 1 つの dict にまとめた形式で出力する。
    """
    lines = [
        "# " + "=" * 72,
        f"# 貼り付け用: plot_metrics_boxplot.py の {variable_name} にコピー",
        "# " + "=" * 72,
        f"{variable_name}: dict[str, dict[str, dict[str, list[float]]]] = {{",
    ]

    task_items = list(all_fold_scores.items())
    for task_idx, (task_id, model_scores) in enumerate(task_items):
        lines.append(f'    "{task_id}": {{')
        lines.extend(format_python_cv_fold_scores_entry(model_scores, indent=8))
        task_comma = "," if task_idx < len(task_items) - 1 else ""
        lines.append(f"    }}{task_comma}")

    lines.append("}")
    return "\n".join(lines)


def format_python_cv_fold_importances_block(
    all_fold_importances: dict[str, dict[str, dict[str, list[float]]]],
    *,
    variable_name: str = "CV_FOLD_IMPORTANCES",
) -> str:
    """plot_metrics_boxplot.py へ貼り付けやすい特徴量重要度ブロックを生成する。"""
    return format_python_cv_fold_scores_block(
        all_fold_importances,
        variable_name=variable_name,
    )


def format_latex_all_metrics_table(
    model_metrics: list[tuple[str, dict[str, dict[str, float]]]],
    caption: str,
    label: str,
) -> str:
    lines = [
        r"\begin{table}[tb]",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        r"    \centering",
        r"    \begin{tabular}{llcccc}",
        r"        \hline",
        r"        手法 & 評価指標 & 平均値 & 標準偏差 & 最小値 & 最大値 \\\hline \hline",
    ]

    n_metric_rows = len(METRIC_ROWS)
    for model_name, metrics in model_metrics:
        for row_idx, (metric_key, metric_label) in enumerate(METRIC_ROWS):
            summary = metrics[metric_key]
            if row_idx == 0:
                method_cell = rf"\multirow{{{n_metric_rows}}}{{*}}{{{model_name}}} "
            else:
                method_cell = " "

            suffix = r" \\\hline" if row_idx == n_metric_rows - 1 else r" \\"
            lines.append(
                f"        {method_cell}"
                f"& {metric_label}"
                f"& {_format_value(summary['mean'])}"
                f" & {_format_value(summary['std'])}"
                f" & {_format_value(summary['min'])}"
                f" & {_format_value(summary['max'])}{suffix}"
            )

    lines.extend([
        r"    \end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def _format_combined_table_value(value: float, decimals: int = COMBINED_TABLE_DECIMALS) -> str:
    if value is None or np.isnan(value):
        return "---"
    return f"{value:.{decimals}f}"


def _metric_stats_from_folds(scores: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(scores, dtype=float)
    return float(np.mean(arr)), float(np.max(arr)), float(np.min(arr))


def _normalize_fold_scores_for_combined_table(
    fold_scores: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, list[float]]]:
    """全 task 横断表用に BL / LR / ... の短いモデル名へ揃える。"""
    return {
        _to_box_plot_model_name(model_name): scores
        for model_name, scores in fold_scores.items()
    }


def format_latex_combined_all_tasks_metrics_table(
    all_task_fold_scores: dict[str, dict[str, dict[str, list[float]]]],
    *,
    caption: str = r"各手法の評価結果（モデル構築プロセス）",
    label: str = "tab:model_all_metrics",
) -> str:
    """task0/1/2 を横並びにした table* 形式の評価結果表（AUC 含む）を生成する。"""
    n_metric_rows = len(METRIC_ROWS)
    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{2.5pt}",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \begin{tabular}{@{}l|l|r|rrr|rrr|rrr|rrr@{}} \hline",
        r"    & & BL & \multicolumn{3}{c|}{LR} & \multicolumn{3}{c|}{DT} & \multicolumn{3}{c|}{RF} & \multicolumn{3}{c}{GB} \\",
        r"    シナリオ & 指標 & & 平均 & 最大 & 最小 & 平均 & 最大 & 最小 & 平均 & 最大 & 最小 & 平均 & 最大 & 最小 \\ \hline \hline",
    ]

    for task_idx, task_id in enumerate(TASK_ORDER):
        if task_id not in all_task_fold_scores:
            continue
        config = TASK_CONFIGS[task_id]
        scores_by_model = _normalize_fold_scores_for_combined_table(
            all_task_fold_scores[task_id]
        )

        for row_idx, (metric_key, metric_label) in enumerate(METRIC_ROWS):
            if row_idx == 0:
                scenario_cell = rf"\multirow{{{n_metric_rows}}}{{*}}{{{config.dataset_macro}}}"
            else:
                scenario_cell = " "

            row_parts = [scenario_cell, metric_label]

            bl_mean, _, _ = _metric_stats_from_folds(scores_by_model["BL"][metric_key])
            row_parts.append(_format_combined_table_value(bl_mean))

            for model_name in BOX_PLOT_MODEL_ORDER[1:]:
                mean, max_value, min_value = _metric_stats_from_folds(
                    scores_by_model[model_name][metric_key]
                )
                row_parts.extend([
                    _format_combined_table_value(mean),
                    _format_combined_table_value(max_value),
                    _format_combined_table_value(min_value),
                ])

            is_last_metric_row = row_idx == n_metric_rows - 1
            suffix = r" \\ \hline" if is_last_metric_row else r" \\"
            lines.append(f"    {' & '.join(row_parts)}{suffix}")

    lines.extend([
        r"  \end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def build_logistic_latex(
    X,
    y,
    config: TaskConfig,
) -> str:
    _, coefficients = fit_logistic_regression(
        X,
        y,
        include_tree=True,
        random_state=RANDOM_STATE,
        model_step_name="lr",
    )
    return format_latex_logistic_equations(
        coefficients,
        logistic_label=config.logistic_equation_label,
        standardization_label=config.standardization_label,
    )


def _print_task_report(
    config: TaskConfig,
    y: pd.Series,
    results: list[tuple[str, dict[str, float]]],
    metrics_table: str,
    importance_table: str,
    logistic_latex: str,
) -> None:
    """1 task 分の評価結果を compare_models.py と同様に標準出力する。"""
    baseline_metrics = results[0][1]

    print("\n【LaTeX形式：ベースライン評価結果・データセット分布】")
    print("-" * 88)
    print(format_latex_baseline_result(
        baseline_metrics,
        caption=rf"ベースラインの評価結果 ({config.dataset_macro})",
        label=f"tab:baseline_result_{config.task_id}",
    ))
    print()
    print(format_latex_dataset_distribution(
        y,
        caption=rf"データセットの分布（{config.dataset_macro}）",
        label=f"data_set_{config.task_id}",
    ))

    print("\n" + "=" * 88)
    print(f"【アルゴリズム比較】適合率・再現率・F値・正解率・AUC（目的: {config.description}）")
    print("=" * 88)
    header = f"{'アルゴリズム':<24} {'適合率':>10} {'再現率':>10} {'F値':>10} {'正解率':>10} {'AUC':>10}"
    print(header)
    print("-" * 88)
    for name, metrics in results:
        auc_str = (
            f"{metrics['auc']:>10.4f}"
            if "auc" in metrics and not np.isnan(metrics["auc"])
            else f"{'N/A':>10}"
        )
        row = (
            f"{name:<24} "
            f"{metrics['precision']:>10.4f} "
            f"{metrics['recall']:>12.4f} "
            f"{metrics['f1']:>14.4f} "
            f"{metrics['accuracy']:>11.4f} "
            f"{auc_str}"
        )
        print(row)
    print("=" * 88)

    print("\n【LaTeX形式の比較結果表】")
    print("-" * 88)
    print(format_latex_comparison_table(
        results,
        config.description,
        label=f"tab:model_comparison_{config.task_id}",
    ))

    print("\n【LaTeX形式：各手法の評価結果】")
    print("-" * 88)
    print(metrics_table)
    print("\n【LaTeX形式：各手法の特徴量重要度】")
    print("-" * 88)
    print(importance_table)
    print("\n【LaTeX形式：ロジスティック回帰式】")
    print("-" * 88)
    print(logistic_latex)


def evaluate_task(
    task_id: str,
    logs_root: Path,
    verbose: bool = False,
) -> tuple[
    str,
    str,
    str,
    dict[str, dict[str, list[float]]],
    dict[str, dict[str, list[float]]],
    pd.Series,
    list[tuple[str, dict[str, float]]],
]:
    config = TASK_CONFIGS[task_id]
    X, y = load_task_dataset(task_id, logs_root, verbose=verbose)

    model_metrics: list[tuple[str, dict[str, dict[str, float]]]] = []
    model_importances: list[tuple[str, list[dict[str, float | str]]]] = []
    fold_scores: dict[str, dict[str, list[float]]] = {
        BASELINE_MODEL_NAME: compute_baseline_fold_scores(X, y),
    }
    fold_importances: dict[str, dict[str, list[float]]] = {}

    for model_key, model_name, _ in MODEL_SPECS:
        pipeline, step_name = build_pipeline(model_key, task_id)
        cv_results = run_cross_validation(X, y, pipeline)

        metrics = {
            metric_key: summarize_scores(_cv_test_scores(cv_results, metric_key))
            for metric_key, _ in METRIC_ROWS
        }
        model_metrics.append((model_name, metrics))
        fold_scores[model_name] = extract_fold_scores(cv_results)

        if model_key != "logistic":
            importance_stats = compute_feature_importance_stats_from_cv(
                cv_results,
                step_name,
                exclude_cpnum_dir=False,
            )
            model_importances.append((model_name, importance_stats))
            fold_importances[model_name] = extract_fold_importances_from_cv(
                cv_results,
                step_name,
                exclude_cpnum_dir=False,
            )

    metrics_table = format_latex_all_metrics_table(
        model_metrics,
        caption=config.metrics_caption,
        label=config.metrics_label,
    )
    importance_table = format_latex_all_importance_table(
        model_importances,
        caption=config.importance_caption,
        label=config.importance_label,
    )
    logistic_latex = build_logistic_latex(X, y, config)
    console_results = build_console_results(fold_scores)
    return (
        metrics_table,
        importance_table,
        logistic_latex,
        fold_scores,
        fold_importances,
        y,
        console_results,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="各 task の4手法評価結果を LaTeX 表形式で出力する",
    )
    parser.add_argument(
        "--task",
        choices=["task0", "task1", "task2", "all"],
        default="all",
        help="対象 task（default: all）",
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=None,
        help="Logs ディレクトリ（default: tree-different/Logs）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="LaTeX をファイル出力するディレクトリ（未指定時は標準出力のみ）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="データ収集時の統計を表示する",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs_root = args.logs_root or _default_logs_root()
    task_ids = list(TASK_CONFIGS.keys()) if args.task == "all" else [args.task]

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    all_fold_scores: dict[str, dict[str, dict[str, list[float]]]] = {}
    all_fold_importances: dict[str, dict[str, dict[str, list[float]]]] = {}
    all_task_fold_scores: dict[str, dict[str, dict[str, list[float]]]] = {}

    for task_id in task_ids:
        config = TASK_CONFIGS[task_id]
        print("=" * 88)
        print(f"【{task_id}】{config.description}")
        print("=" * 88)

        (
            metrics_table,
            importance_table,
            logistic_latex,
            fold_scores,
            fold_importances,
            y,
            console_results,
        ) = evaluate_task(
            task_id,
            logs_root,
            verbose=args.verbose,
        )
        print(f"データ件数: {len(y)} 件")
        all_task_fold_scores[task_id] = fold_scores
        all_fold_scores[task_id] = build_box_plot_fold_scores(fold_scores)
        all_fold_importances[task_id] = build_box_plot_fold_importances(fold_importances)

        _print_task_report(
            config,
            y,
            console_results,
            metrics_table,
            importance_table,
            logistic_latex,
        )
        print()

        if args.output_dir is not None:
            metrics_path = args.output_dir / f"{task_id}_metrics.tex"
            importance_path = args.output_dir / f"{task_id}_importance.tex"
            logistic_path = args.output_dir / f"{task_id}_logistic.tex"
            metrics_path.write_text(metrics_table + "\n", encoding="utf-8")
            importance_path.write_text(importance_table + "\n", encoding="utf-8")
            logistic_path.write_text(logistic_latex + "\n", encoding="utf-8")
            print(f"保存しました: {metrics_path}")
            print(f"保存しました: {importance_path}")
            print(f"保存しました: {logistic_path}")
            print()

    if all(task_id in all_task_fold_scores for task_id in TASK_ORDER):
        combined_table = format_latex_combined_all_tasks_metrics_table(all_task_fold_scores)
        print("=" * 88)
        print("【LaTeX形式：全シナリオ横断の評価結果表】")
        print("=" * 88)
        print(combined_table)
        print()

        if args.output_dir is not None:
            combined_path = args.output_dir / "all_tasks_metrics.tex"
            combined_path.write_text(combined_table + "\n", encoding="utf-8")
            print(f"保存しました: {combined_path}")
            print()

    if all_fold_scores:
        print("=" * 88)
        print("【貼り付け用：plot_metrics_boxplot.py の CV_FOLD_SCORES】")
        print("=" * 88)
        print(format_python_cv_fold_scores_block(all_fold_scores))
        print()

        if args.output_dir is not None:
            boxplot_data_path = args.output_dir / "cv_fold_scores.py"
            boxplot_data_path.write_text(
                format_python_cv_fold_scores_block(all_fold_scores) + "\n",
                encoding="utf-8",
            )
            print(f"保存しました: {boxplot_data_path}")
            print()

    if all_fold_importances:
        print("=" * 88)
        print("【貼り付け用：plot_metrics_boxplot.py の CV_FOLD_IMPORTANCES】")
        print("=" * 88)
        print(format_python_cv_fold_importances_block(all_fold_importances))
        print()

        if args.output_dir is not None:
            importance_data_path = args.output_dir / "cv_fold_importances.py"
            importance_data_path.write_text(
                format_python_cv_fold_importances_block(all_fold_importances) + "\n",
                encoding="utf-8",
            )
            print(f"保存しました: {importance_data_path}")
            print()


if __name__ == "__main__":
    main()
