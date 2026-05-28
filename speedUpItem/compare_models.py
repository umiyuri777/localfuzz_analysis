"""
speedUpItem: 4つのアルゴリズム（ロジスティック回帰・決定木・ランダムフォレスト・勾配ブースティング）を
一括実行し、適合率・再現率・F値・正解率を表形式で比較する。
"""

import argparse
from pathlib import Path

import numpy as np

# プロジェクトルートをパスに追加して utils をインポート
import sys
if (project_root := Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.data_loader import load_speedup_bug_dataset
from utils.decision_tree_analysis import build_decision_tree_pipeline
from utils.gradient_boosting_analysis import build_gradient_boosting_pipeline
from utils.logistic_regression_analysis import build_logistic_regression_pipeline
from utils.metrics import calculate_binary_metrics
from utils.random_forest_analysis import build_random_forest_pipeline

RANDOM_STATE = 42
SPEEDUP_DIR = Path(__file__).resolve().parent
TARGET_LABELS = {
    "bug_detected_any": "5回中1回でもバグ",
    "bug_detected_all": "5回全てバグ",
    "per_run": "1回の実行でバグ",
}

# 各モデルのデフォルトハイパーパラメータ（target 指定で上書き可能）
DEFAULT_MODEL_PARAMS = {
    "logistic": {"random_state": RANDOM_STATE},
    "tree": {"random_state": RANDOM_STATE},
    "rf": {"random_state": RANDOM_STATE},
    "gb": {"random_state": RANDOM_STATE},
}

# target × アルゴリズムごとの予測閾値
# （predict_proba の正クラス確率がこの値以上で 1）。None のときは predict() のデフォルト 0.5 を使用
THRESHOLDS_BY_TARGET = {
    "bug_detected_any": {
        "ロジスティック回帰": None,
        "決定木": None,
        "ランダムフォレスト": None,
        "勾配ブースティング": None,
    },
    "bug_detected_all": {
        "ロジスティック回帰": None,
        "決定木": None,
        "ランダムフォレスト": None,
        "勾配ブースティング": None,
    },
    "per_run": {
        "ロジスティック回帰": None,
        "決定木": None,
        "ランダムフォレスト": None,
        "勾配ブースティング": None,
    },
}
# THRESHOLDS_BY_TARGET = {
#     "bug_detected_any": {
#         "ロジスティック回帰": 0.010,
#         "決定木": 0.490,
#         "ランダムフォレスト": 0.010,
#         "勾配ブースティング": 0.520,
#     },
#     "bug_detected_all": {
#         "ロジスティック回帰": 0.190,
#         "決定木": 0.040,
#         "ランダムフォレスト": 0.250,
#         "勾配ブースティング": 0.280,
#     },
#     "per_run": {
#         "ロジスティック回帰": 0.5,
#         "決定木": 0.340,
#         "ランダムフォレスト": 0.400,
#         "勾配ブースティング": 0.450,
#     },
# }

# target ごとのハイパーパラメータ上書き（キーはモデル名、値は DEFAULT にマージする dict）
TARGET_HYPERPARAMS = {
    "bug_detected_any": {
        "logistic": {"class_weight": None, "max_iter": 1000, "penalty": "l2", "C": np.float64(0.001)}, 
        "tree": {"max_depth": None, "min_samples_leaf": 0.05, "criterion": "gini", "ccp_alpha": 0.0}, 
        "rf": {"max_depth": None, "min_samples_leaf": 0.05, "n_estimators": 100, "n_jobs": -1, "max_features": "sqrt"},
        "gb": {'learning_rate': 0.1, 'max_depth': 3, 'min_samples_leaf': 1, 'n_estimators': 50}
    },
    "bug_detected_all": {
        "logistic": {"class_weight": "balanced", "max_iter": 1000, "penalty": "l1", "C": np.float64(0.01), "solver": "liblinear"}, 
        "tree": {"max_depth": None, "min_samples_leaf": 0.01, "criterion": "gini", "ccp_alpha": 0.0, "class_weight": "balanced"}, 
        "rf": {"max_depth": None, "n_estimators": 200, "n_jobs": -1, "max_features": "sqrt", "min_samples_leaf": 1, "class_weight": "balanced"},
        "gb": {'learning_rate': 0.01, 'max_depth': 3, 'min_samples_leaf': 1, 'n_estimators': 50}
    },
    "per_run": {
        "logistic": {"max_iter": 1000}, 
        "tree": {"max_depth": None, "min_samples_leaf": 0.05}, 
        "rf": {"max_depth": None, "min_samples_leaf": 0.05, "n_estimators": 100, "n_jobs": -1},
        "gb": {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 100, "min_samples_leaf": 5}
    },
}


METRIC_DECIMALS = 4

# 目的変数のクラスラベル（LaTeX表用）
CLASS_LABELS_JA = {
    1: "不具合発見する",
    0: "不具合発見しない",
}


def format_latex_baseline_result(
    metrics: dict,
    caption: str | None = None,
    label: str | None = None,
    dataset_macro: str = r"\testfirst",
    decimals: int = METRIC_DECIMALS,
) -> str:
    """ベースラインの評価指標を LaTeX 表形式の文字列に整形する。"""
    if caption is None:
        caption = rf"ベースラインの評価結果 ({dataset_macro})"
    if label is None:
        label = "tab:baseline_result_1"

    rows = [
        ("適合率", metrics["precision"]),
        ("再現率", metrics["recall"]),
        ("F値", metrics["f1"]),
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
    y,
    caption: str | None = None,
    label: str | None = None,
    dataset_macro: str = r"\testfirst",
) -> str:
    """目的変数 y のクラス分布を LaTeX 表形式の文字列に整形する。"""
    counts = y.value_counts()
    n_positive = int(counts.get(1, 0))
    n_negative = int(counts.get(0, 0))

    if caption is None:
        caption = rf"データセットの分布（{dataset_macro}）"
    if label is None:
        label = "data_set_1"

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


def _format_latex_metric(value: float, decimals: int = METRIC_DECIMALS) -> str:
    """評価指標を LaTeX 表のセル用に整形する。NaN のときは --- を返す。"""
    if value is None or np.isnan(value):
        return "---"
    return f"{value:.{decimals}f}"


def format_latex_comparison_table(
    results: list[tuple[str, dict]],
    target_label: str,
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """アルゴリズム比較結果を LaTeX 表形式の文字列に整形する。"""
    if caption is None:
        caption = rf"アルゴリズム比較（目的: {target_label}）"
    if label is None:
        label = "tab:speedup_model_comparison"

    lines = [
        r"\begin{table}[H]",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        r"    \centering",
        r"    \begin{tabular}{lrrr}",
        r"        \hline",
        r"        アルゴリズム & 適合率 & 再現率 & F値 \\",
        r"        \hline \hline",
    ]
    for name, metrics in results:
        lines.append(
            f"        {name} & "
            f"{_format_latex_metric(metrics['precision'])} & "
            f"{_format_latex_metric(metrics['recall'])} & "
            f"{_format_latex_metric(metrics['f1'])} \\\\"
        )
    lines.extend([
        r"        \hline",
        r"    \end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def _build_models(target: str):
    """target に応じたハイパーパラメータで、全モデルを前処理付き Pipeline として返す。"""
    def params(model_name: str):
        # return {**DEFAULT_MODEL_PARAMS[model_name], **TARGET_HYPERPARAMS.get(target, {}).get(model_name, {})}
        return DEFAULT_MODEL_PARAMS[model_name]

    return [
        (
            "ロジスティック回帰",
            build_logistic_regression_pipeline(include_tree=False, **params("logistic")),
        ),
        (
            "決定木",
            build_decision_tree_pipeline(include_tree=False, **params("tree")),
        ),
        (
            "ランダムフォレスト",
            build_random_forest_pipeline(include_tree=False, **params("rf")),
        ),
        (
            "勾配ブースティング",
            build_gradient_boosting_pipeline(include_tree=False, **params("gb")),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="4アルゴリズムで不具合発見を予測し比較")
    parser.add_argument(
        "--target",
        choices=["bug_detected_any", "bug_detected_all", "per_run"],
        default="bug_detected_any",
        help="目的変数（default: bug_detected_any）",
    )
    args = parser.parse_args()
    target_label = TARGET_LABELS[args.target]

    train_dir = str(SPEEDUP_DIR / "tree=500")
    test_dir = str(SPEEDUP_DIR / "Logs")

    X_train, y_train = load_speedup_bug_dataset(
        train_dir, target=args.target, verbose=True, tree_value=500
    )
    X_test, y_test = load_speedup_bug_dataset(test_dir, target=args.target, verbose=True)

    # ベースライン: どのパラメータでもバグ発見すると予測する（常に正例と予測）
    y_pred_baseline = np.ones(len(y_test), dtype=int)
    y_proba_baseline = np.ones(len(y_test), dtype=float)
    metrics_baseline = calculate_binary_metrics(
        y_test, y_pred_baseline, y_pred_proba=y_proba_baseline
    )
    results = [("ベースライン（常にバグ発見）", metrics_baseline)]

    print("\n【LaTeX形式：ベースライン評価結果・テストデータセット分布】")
    print("-" * 88)
    print(format_latex_baseline_result(metrics_baseline))
    print()
    print(format_latex_dataset_distribution(y_test))

    models = _build_models(args.target)
    thresholds_for_target = THRESHOLDS_BY_TARGET.get(args.target, {})

    for name, model in models:
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        threshold = thresholds_for_target.get(name)
        if threshold is not None:
            y_pred = (proba >= threshold).astype(int)
        else:
            y_pred = model.predict(X_test)
        metrics = calculate_binary_metrics(y_test, y_pred, y_pred_proba=proba)
        results.append((name, metrics))

    # 表形式で表示
    print("\n" + "=" * 88)
    print(f"【アルゴリズム比較】適合率・再現率・F値・正解率・AUC（目的: {target_label}）")
    print("=" * 88)
    header = f"{'アルゴリズム':<24} {'適合率':>10} {'再現率':>10} {'F値':>10} {'正解率':>10} {'AUC':>10}"
    print(header)
    print("-" * 88)
    for name, m in results:
        auc_str = f"{m['auc']:>10.4f}" if "auc" in m and not np.isnan(m["auc"]) else f"{'N/A':>10}"
        row = f"{name:<24} {m['precision']:>10.4f} {m['recall']:>12.4f} {m['f1']:>14.4f} {m['accuracy']:>11.4f} {auc_str}"
        print(row)
    print("=" * 88)

    print("\n【LaTeX形式の比較結果表】")
    print("-" * 88)
    print(format_latex_comparison_table(results, target_label))


if __name__ == "__main__":
    main()
