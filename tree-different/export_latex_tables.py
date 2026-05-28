"""
tree-different/task0, task1, task2 について、
4手法の10分割交差検証結果を LaTeX 表形式で一括出力する。

出力:
- 各手法の評価結果（適合率・再現率・F値）
- ロジスティック回帰以外の特徴量重要度
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
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
from utils.logistic_regression_analysis import build_logistic_regression_pipeline  # noqa: E402
from utils.random_forest_analysis import build_random_forest_pipeline  # noqa: E402

RANDOM_STATE = 42
N_SPLITS = 10
METRIC_DECIMALS = 4

METRIC_ROWS = [
    ("precision", "適合率"),
    ("recall", "再現率"),
    ("f1", "F値"),
]

SCORING = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    description: str
    dataset_macro: str
    metrics_caption: str
    metrics_label: str
    importance_caption: str
    importance_label: str


TASK_CONFIGS: dict[str, TaskConfig] = {
    "task0": TaskConfig(
        task_id="task0",
        description="1回の実行でバグ発見（per_run / bug_detected）",
        dataset_macro=r"\testfirst",
        metrics_caption=r"各手法の評価結果（\testfirst）",
        metrics_label="tab:all_single_metrics_prospect",
        importance_caption=r"各手法における特徴量重要度（\testfirst）",
        importance_label="tab:all_importance",
    ),
    "task1": TaskConfig(
        task_id="task1",
        description="5回中1回でもバグ発見（bug_detected_any）",
        dataset_macro=r"\testsecond",
        metrics_caption=r"各手法の評価結果（\testsecond）",
        metrics_label="tab:all_single_metrics_prospect_task1",
        importance_caption=r"各手法における特徴量重要度（\testsecond）",
        importance_label="tab:all_importance_task1",
    ),
    "task2": TaskConfig(
        task_id="task2",
        description="5回全てバグ発見（bug_detected_all）",
        dataset_macro=r"\testthird",
        metrics_caption=r"各手法の評価結果（\testthird）",
        metrics_label="tab:all_single_metrics_prospect_task2",
        importance_caption=r"各手法における特徴量重要度（\testthird）",
        importance_label="tab:all_importance_task2",
    ),
}

MODEL_SPECS: list[tuple[str, str, str]] = [
    ("logistic", "ロジスティック回帰", "lr"),
    ("tree", "決定木", "dt"),
    ("rf", "ランダムフォレスト", "rf"),
    ("gb", "勾配ブースティング", "gb"),
]


def _default_logs_root() -> Path:
    return Path(__file__).resolve().parent / "Logs"


def _format_value(value: float, decimals: int = METRIC_DECIMALS) -> str:
    if value is None or np.isnan(value):
        return "---"
    return f"{value:.{decimals}f}"


def latex_feature_name(name: str) -> str:
    # ColumnTransformer が付与する接頭辞を外す（表示用）
    if name.startswith("num__"):
        name = name[len("num__") :]
    elif name.startswith("cat__"):
        name = name[len("cat__") :]

    latex_names = {
        "tree": r"tree",
        "cpNum": r"cpNum",
        "cpNum_range": r"cpNum\_range",
        "cpNum_dir": r"cpNum\_dir",
    }
    if name.startswith("cpNum_dir_"):
        # 例: cpNum_dir_1 -> cpNum_dir=1
        suffix = name.split("_", 2)[-1]
        return rf"cpNum\_dir={suffix}"
    return latex_names.get(name, name.replace("_", r"\_"))


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


def compute_feature_importance_stats(
    cv_results,
    step_name: str,
) -> list[dict[str, float | str]]:
    n_folds = len(cv_results["estimator"])
    feature_names: list[str] | None = None
    importances: np.ndarray | None = None

    for fold_idx, estimator in enumerate(cv_results["estimator"]):
        preprocess = estimator.named_steps["preprocess"]
        fold_feature_names = list(preprocess.get_feature_names_out())
        if feature_names is None:
            feature_names = fold_feature_names
            importances = np.zeros((n_folds, len(feature_names)))
        else:
            # 念のため（カテゴリ固定なので通常一致するはず）
            if fold_feature_names != feature_names:
                raise ValueError("前処理後の特徴量名がfold間で一致しません。")

        model = estimator.named_steps[step_name]
        importances[fold_idx] = model.feature_importances_

    stats = []
    assert feature_names is not None
    assert importances is not None
    for idx, feature in enumerate(feature_names):
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


def format_latex_all_importance_table(
    model_importances: list[tuple[str, list[dict[str, float | str]]]],
    caption: str,
    label: str,
) -> str:
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
                f"& {_format_value(float(row['mean']))}"
                f" & {_format_value(float(row['std']))}"
                f" & {_format_value(float(row['min']))}"
                f" & {_format_value(float(row['max']))}{suffix}"
            )

    lines.extend([
        r"    \end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def evaluate_task(task_id: str, logs_root: Path, verbose: bool = False) -> tuple[str, str]:
    config = TASK_CONFIGS[task_id]
    X, y = load_task_dataset(task_id, logs_root, verbose=verbose)

    model_metrics: list[tuple[str, dict[str, dict[str, float]]]] = []
    model_importances: list[tuple[str, list[dict[str, float | str]]]] = []

    for model_key, model_name, _ in MODEL_SPECS:
        pipeline, step_name = build_pipeline(model_key, task_id)
        cv_results = run_cross_validation(X, y, pipeline)

        metrics = {
            metric_key: summarize_scores(cv_results[f"test_{metric_key}"])
            for metric_key, _ in METRIC_ROWS
        }
        model_metrics.append((model_name, metrics))

        if model_key != "logistic":
            importance_stats = compute_feature_importance_stats(
                cv_results,
                step_name,
            )
            model_importances.append((model_name, importance_stats))

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
    return metrics_table, importance_table


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

    for task_id in task_ids:
        config = TASK_CONFIGS[task_id]
        print("=" * 88)
        print(f"【{task_id}】{config.description}")
        print("=" * 88)

        metrics_table, importance_table = evaluate_task(
            task_id,
            logs_root,
            verbose=args.verbose,
        )

        print("\n【LaTeX形式：各手法の評価結果】")
        print("-" * 88)
        print(metrics_table)
        print("\n【LaTeX形式：各手法の特徴量重要度】")
        print("-" * 88)
        print(importance_table)
        print()

        if args.output_dir is not None:
            metrics_path = args.output_dir / f"{task_id}_metrics.tex"
            importance_path = args.output_dir / f"{task_id}_importance.tex"
            metrics_path.write_text(metrics_table + "\n", encoding="utf-8")
            importance_path.write_text(importance_table + "\n", encoding="utf-8")
            print(f"保存しました: {metrics_path}")
            print(f"保存しました: {importance_path}")
            print()


if __name__ == "__main__":
    main()
