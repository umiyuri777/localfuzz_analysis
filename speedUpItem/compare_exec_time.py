"""
speedUpItem: 4つの予測モデルが「不具合発見する」と予測したパラメータのみ実行した場合の
exe_time 合計を、全パラメータ実行時（ベースライン）の合計と target ごとに比較する。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SPEEDUP_DIR = Path(__file__).resolve().parent
project_root = SPEEDUP_DIR.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(SPEEDUP_DIR) not in sys.path:
    sys.path.insert(0, str(SPEEDUP_DIR))

from compare_models import (  # noqa: E402
    BASELINE_MODEL_NAME,
    TARGET_LABELS,
    TARGET_ORDER,
    THRESHOLDS_BY_TARGET,
    _build_models,
)
from utils.data_loader import (  # noqa: E402
    SPEEDUP_BUG_PREDICTION_FEATURE_NAMES,
    collect_data_aggregated_flat,
    collect_data_per_run_flat,
    load_speedup_bug_dataset,
)

RUNS_PER_PARAM_DIR = 5
METRIC_DECIMALS = 4


def load_exe_times_by_dir(logs_root: Path) -> dict[str, list[int]]:
    """各パラメータディレクトリの exe_time.csv を dir_name キーで読み込む。"""
    exe_times: dict[str, list[int]] = {}

    for param_dir in sorted(logs_root.iterdir()):
        if not param_dir.is_dir():
            continue

        exe_time_path = param_dir / "exe_time.csv"
        if not exe_time_path.exists():
            continue

        with open(exe_time_path, "r", encoding="utf-8") as time_f:
            rows = list(csv.reader(time_f))

        times: list[int] = []
        for row in rows[:RUNS_PER_PARAM_DIR]:
            try:
                times.append(int(row[0]))
            except (ValueError, IndexError):
                times.append(0)

        exe_times[param_dir.name] = times

    return exe_times


def compute_baseline_total(exe_times_by_dir: dict[str, list[int]]) -> int:
    """全パラメータ・全実行の exe_time 合計を返す。"""
    return sum(sum(times) for times in exe_times_by_dir.values())


def _load_test_logs_df(target: str) -> pd.DataFrame:
    """テストデータを dir_name 付きで読み込む。"""
    test_dir = str(SPEEDUP_DIR / "Logs")
    if target == "per_run":
        return collect_data_per_run_flat(logs_root=test_dir, verbose=False)
    return collect_data_aggregated_flat(logs_root=test_dir, verbose=False)


def _predict_positive_mask(
    target: str,
    logs_df: pd.DataFrame,
    model_name: str,
    model,
) -> np.ndarray:
    """モデルが「不具合発見する（=1）」と予測した行のマスクを返す。"""
    train_dir = str(SPEEDUP_DIR / "tree=500")
    X_train, y_train = load_speedup_bug_dataset(
        train_dir, target=target, verbose=False, tree_value=500
    )
    X_test = logs_df[SPEEDUP_BUG_PREDICTION_FEATURE_NAMES].copy()

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    threshold = THRESHOLDS_BY_TARGET.get(target, {}).get(model_name)
    if threshold is not None:
        return (proba >= threshold).astype(bool)
    return model.predict(X_test).astype(bool)


def sum_exe_time_for_predictions(
    target: str,
    logs_df: pd.DataFrame,
    positive_mask: np.ndarray,
    exe_times_by_dir: dict[str, list[int]],
) -> int:
    """予測陽性に対応する exe_time の合計を返す。"""
    total = 0

    if target == "per_run":
        for i, row in enumerate(logs_df.itertuples(index=False)):
            if not positive_mask[i]:
                continue
            times = exe_times_by_dir.get(row.dir_name)
            if times is None:
                continue
            run_index = int(row.run_index)
            if 1 <= run_index <= len(times):
                total += times[run_index - 1]
        return total

    for i, row in enumerate(logs_df.itertuples(index=False)):
        if not positive_mask[i]:
            continue
        times = exe_times_by_dir.get(row.dir_name)
        if times is None:
            continue
        total += sum(times)

    return total


def compute_reduction_rate(model_total: int, baseline_total: int) -> float:
    """削減率 (1 - model_total / baseline_total) を返す。baseline が 0 のとき NaN。"""
    if baseline_total == 0:
        return float("nan")
    return 1.0 - (model_total / baseline_total)


def evaluate_exec_time_for_target(
    target: str,
    exe_times_by_dir: dict[str, list[int]],
    baseline_total: int,
) -> list[dict[str, float | int | str]]:
    """1 target について各モデルの exe_time 集計結果を返す。"""
    logs_df = _load_test_logs_df(target)
    results: list[dict[str, float | int | str]] = []

    baseline_entry = {
        "model": BASELINE_MODEL_NAME,
        "total_frames": baseline_total,
        "reduction_rate": 0.0,
        "predicted_positive_count": len(logs_df) if target == "per_run" else len(logs_df),
    }
    results.append(baseline_entry)

    for model_name, model in _build_models(target):
        positive_mask = _predict_positive_mask(target, logs_df, model_name, model)
        model_total = sum_exe_time_for_predictions(
            target, logs_df, positive_mask, exe_times_by_dir
        )
        results.append({
            "model": model_name,
            "total_frames": model_total,
            "reduction_rate": compute_reduction_rate(model_total, baseline_total),
            "predicted_positive_count": int(positive_mask.sum()),
        })

    return results


def _format_number(value: float | int) -> str:
    if isinstance(value, float) and np.isnan(value):
        return "---"
    if isinstance(value, float):
        return f"{value:.{METRIC_DECIMALS}f}"
    return f"{value:,}"


def print_console_report(
    target: str,
    results: list[dict[str, float | int | str]],
    baseline_total: int,
) -> None:
    """コンソールに比較表を出力する。"""
    target_label = TARGET_LABELS[target]
    print("\n" + "=" * 96)
    print(f"【実行時間比較】目的: {target_label}（{target}）")
    print("=" * 96)
    print(f"ベースライン合計（全実行）: {baseline_total:,} フレーム")
    print("-" * 96)
    header = (
        f"{'モデル':<24} {'予測陽性数':>12} "
        f"{'合計フレーム数':>16} {'削減率':>12}"
    )
    print(header)
    print("-" * 96)

    for row in results:
        reduction = row["reduction_rate"]
        reduction_str = (
            f"{reduction:>12.{METRIC_DECIMALS}f}"
            if isinstance(reduction, float) and not np.isnan(reduction)
            else f"{'---':>12}"
        )
        print(
            f"{row['model']:<24} "
            f"{row['predicted_positive_count']:>12,} "
            f"{row['total_frames']:>16,} "
            f"{reduction_str}"
        )
    print("=" * 96)


def format_latex_exec_time_table(
    target: str,
    results: list[dict[str, float | int | str]],
) -> str:
    """LaTeX 形式の比較表を返す。"""
    target_label = TARGET_LABELS[target]
    label_suffix = target.replace("_", "-")
    lines = [
        r"\begin{table}[H]",
        f"    \\caption{{予測モデルによる実行時間削減比較（目的: {target_label}）}}",
        f"    \\label{{tab:speedup_exec_time_{label_suffix}}}",
        r"    \centering",
        r"    \begin{tabular}{lrrr}",
        r"        \hline",
        r"        モデル & 予測陽性数 & 合計フレーム数 & 削減率 \\",
        r"        \hline \hline",
    ]

    for row in results:
        reduction = row["reduction_rate"]
        reduction_str = (
            _format_number(float(reduction))
            if isinstance(reduction, float) and not np.isnan(reduction)
            else "---"
        )
        lines.append(
            f"        {row['model']} & "
            f"{row['predicted_positive_count']:,} & "
            f"{row['total_frames']:,} & "
            f"{reduction_str} \\\\"
        )

    lines.extend([
        r"        \hline",
        r"    \end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def results_to_dataframe(
    target: str,
    results: list[dict[str, float | int | str]],
    baseline_total: int,
) -> pd.DataFrame:
    """結果を DataFrame に変換する。"""
    df = pd.DataFrame(results)
    df.insert(0, "target", target)
    df.insert(1, "baseline_total_frames", baseline_total)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="予測モデルによる実行時間削減をベースラインと比較"
    )
    parser.add_argument(
        "--target",
        choices=["bug_detected_any", "bug_detected_all", "per_run"],
        default=None,
        help="省略時は3種類すべて評価",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SPEEDUP_DIR / "output",
        help="CSV 出力先",
    )
    args = parser.parse_args()

    logs_root = SPEEDUP_DIR / "Logs"
    exe_times_by_dir = load_exe_times_by_dir(logs_root)
    baseline_total = compute_baseline_total(exe_times_by_dir)

    print(f"読み込み完了: {len(exe_times_by_dir)} パラメータディレクトリ")
    print(f"ベースライン合計: {baseline_total:,} フレーム")

    targets = [args.target] if args.target else TARGET_ORDER
    all_dfs: list[pd.DataFrame] = []

    for target in targets:
        results = evaluate_exec_time_for_target(
            target, exe_times_by_dir, baseline_total
        )
        print_console_report(target, results, baseline_total)
        print("\n【LaTeX形式の比較結果表】")
        print("-" * 96)
        print(format_latex_exec_time_table(target, results))
        all_dfs.append(results_to_dataframe(target, results, baseline_total))

    if len(all_dfs) > 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / "exec_time_comparison.csv"
        pd.concat(all_dfs, ignore_index=True).to_csv(output_path, index=False)
        print(f"\n保存しました: {output_path}")


if __name__ == "__main__":
    main()
