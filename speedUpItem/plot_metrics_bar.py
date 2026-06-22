"""
per_run / bug_detected_any / bug_detected_all の適合率・再現率・F値を
1 枚の棒グラフに描画する。

train=tree=500, test=Logs の hold-out 評価結果を可視化する。
交差検証を行っていないため、箱ひげ図ではなくグループ化棒グラフを使う。

使い方:
1. compare_models.py を実行し、末尾の METRICS_SCORES ブロックを下記に貼り付ける
2. ラベルなどの設定を必要に応じて変更する
3. python plot_metrics_bar.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

# =============================================================================
# 貼り付け用データ（compare_models.py の出力をここにコピー）
# =============================================================================
METRICS_SCORES: dict[str, dict[str, dict[str, float]]] = {
    "per_run": {
        "BL": {
            "precision": 0.72,
            "recall": 1.00,
            "f1": 0.84
        },
        "LR": {
            "precision": 0.81,
            "recall": 0.80,
            "f1": 0.81
        },
        "DT": {
            "precision": 0.85,
            "recall": 0.81,
            "f1": 0.83
        },
        "RF": {
            "precision": 0.85,
            "recall": 0.81,
            "f1": 0.83
        },
        "GB": {
            "precision": 0.85,
            "recall": 0.81,
            "f1": 0.83
        }
    },
    "bug_detected_any": {
        "BL": {
            "precision": 0.94,
            "recall": 1.00,
            "f1": 0.97
        },
        "LR": {
            "precision": 0.94,
            "recall": 1.00,
            "f1": 0.97
        },
        "DT": {
            "precision": 0.94,
            "recall": 1.00,
            "f1": 0.97
        },
        "RF": {
            "precision": 0.94,
            "recall": 1.00,
            "f1": 0.97
        },
        "GB": {
            "precision": 0.94,
            "recall": 0.99,
            "f1": 0.96
        }
    },
    "bug_detected_all": {
        "BL": {
            "precision": 0.42,
            "recall": 1.00,
            "f1": 0.59
        },
        "LR": {
            "precision": 0.67,
            "recall": 0.57,
            "f1": 0.62
        },
        "DT": {
            "precision": 0.56,
            "recall": 0.83,
            "f1": 0.67
        },
        "RF": {
            "precision": 0.54,
            "recall": 0.81,
            "f1": 0.65
        },
        "GB": {
            "precision": 0.54,
            "recall": 0.81,
            "f1": 0.65
        }
    }
}

# =============================================================================
# 描画設定（文言はここで変更）
# =============================================================================
MODEL_ORDER = ["BL", "LR", "DT", "RF", "GB"]
MODEL_LABELS = {
    "BL": "BL",
    "LR": "LR",
    "DT": "DT",
    "RF": "RF",
    "GB": "GB",
}

TARGET_ORDER = ["per_run", "bug_detected_any", "bug_detected_all"]
TARGET_LABELS = {
    "per_run": "単回",
    "bug_detected_any": "1回以上",
    "bug_detected_all": "全回",
}

METRIC_ORDER = ["precision", "recall", "f1"]
METRIC_LABELS = {
    "precision": "適合率",
    "recall": "再現率",
    "f1": "F値",
}

FIGURE_TITLE = "各モデルの評価結果(適用プロセス)"
YLABEL = "スコア"
YLIM = (0.0, 1.05)

FIGURE_SIZE = (15, 5)
METRIC_COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
BAR_ALPHA = 0.85
GROUP_GAP = 1.0
BAR_WIDTH = 0.22
BAR_LABEL_FONTSIZE = 8
BAR_LABEL_ROTATION = 45
BAR_LABEL_Y = -0.02
MODEL_LABEL_FONTSIZE = 15
MODEL_LABEL_Y = -0.15

OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_BASENAME = "metrics_bar_all"
OUTPUT_FORMATS = ("png",)

JAPANESE_FONTS = [
    "Hiragino Sans",
    "Hiragino Kaku Gothic Pro",
    "Yu Gothic",
    "Meiryo",
    "Noto Sans CJK JP",
]


def configure_matplotlib() -> None:
    """日本語表示とマイナス記号の設定。"""
    plt.rcParams["axes.unicode_minus"] = False
    for font in JAPANESE_FONTS:
        try:
            plt.rcParams["font.family"] = font
            break
        except OSError:
            continue


def _validate_scores(
    metrics_scores: dict[str, dict[str, dict[str, float]]],
) -> None:
    missing_targets = [target for target in TARGET_ORDER if target not in metrics_scores]
    if missing_targets:
        raise ValueError(f"METRICS_SCORES に target が不足しています: {missing_targets}")

    for target in TARGET_ORDER:
        target_scores = metrics_scores[target]
        for model_name in MODEL_ORDER:
            if model_name not in target_scores:
                available = ", ".join(sorted(target_scores))
                raise ValueError(
                    f"{target} にモデル '{model_name}' がありません。"
                    f" 利用可能: {available}"
                )
            for metric_key in METRIC_ORDER:
                if metric_key not in target_scores[model_name]:
                    raise ValueError(
                        f"{target} / {model_name} に指標 '{metric_key}' がありません"
                    )


def _build_grouped_bar_data(
    target_scores: dict[str, dict[str, float]],
) -> tuple[
    list[float],
    list[float],
    list[str],
    list[float],
    list[str],
    list[str],
]:
    """1 target 分のグループ化棒グラフ用データを生成する。"""
    n_metrics = len(METRIC_ORDER)
    group_stride = n_metrics + GROUP_GAP

    heights: list[float] = []
    positions: list[float] = []
    bar_colors: list[str] = []
    bar_metric_keys: list[str] = []
    group_centers: list[float] = []

    for group_idx, model_name in enumerate(MODEL_ORDER):
        group_start = group_idx * group_stride
        group_centers.append(group_start + (n_metrics - 1) / 2)

        for metric_idx, metric_key in enumerate(METRIC_ORDER):
            positions.append(group_start + metric_idx)
            heights.append(target_scores[model_name][metric_key])
            bar_colors.append(METRIC_COLORS[metric_idx])
            bar_metric_keys.append(metric_key)

    tick_labels = [MODEL_LABELS.get(model, model) for model in MODEL_ORDER]
    return heights, positions, bar_colors, group_centers, tick_labels, bar_metric_keys


def _add_bar_metric_labels(
    ax: plt.Axes,
    positions: list[float],
    metric_keys: list[str],
) -> None:
    """各棒の直下に指標ラベルを付ける（白黒印刷でも区別可能にする）。"""
    label_transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for pos, metric_key in zip(positions, metric_keys):
        ax.text(
            pos,
            BAR_LABEL_Y,
            METRIC_LABELS[metric_key],
            ha="right",
            va="top",
            rotation=BAR_LABEL_ROTATION,
            rotation_mode="anchor",
            fontsize=BAR_LABEL_FONTSIZE,
            transform=label_transform,
        )


def _add_model_labels(
    ax: plt.Axes,
    group_centers: list[float],
    model_labels: list[str],
) -> None:
    """アルゴリズム名を指標ラベルの下に表示する。"""
    label_transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for pos, label in zip(group_centers, model_labels):
        ax.text(
            pos,
            MODEL_LABEL_Y,
            label,
            ha="center",
            va="top",
            fontsize=MODEL_LABEL_FONTSIZE,
            transform=label_transform,
        )


def plot_metrics_bar(
    metrics_scores: dict[str, dict[str, dict[str, float]]],
    *,
    output_dir: Path = OUTPUT_DIR,
    show: bool = False,
) -> Path:
    """全アルゴリズムを各 target サブプロット内に横並びで描画する。"""
    _validate_scores(metrics_scores)

    configure_matplotlib()
    fig, axes = plt.subplots(1, len(TARGET_ORDER), figsize=FIGURE_SIZE, sharey=True)
    if len(TARGET_ORDER) == 1:
        axes = [axes]

    for ax, target in zip(axes, TARGET_ORDER):
        heights, positions, bar_colors, group_centers, tick_labels, bar_metric_keys = (
            _build_grouped_bar_data(metrics_scores[target])
        )

        ax.bar(
            positions,
            heights,
            width=BAR_WIDTH,
            color=bar_colors,
            alpha=BAR_ALPHA,
            edgecolor="white",
            linewidth=0.5,
        )

        ax.set_xticks([])
        _add_bar_metric_labels(ax, positions, bar_metric_keys)
        _add_model_labels(ax, group_centers, tick_labels)
        target_label = TARGET_LABELS.get(target, target)
        ax.set_title(target_label, fontsize=12, fontweight="bold")
        ax.set_ylim(*YLIM)
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel(YLABEL, fontsize=12)
    fig.suptitle(FIGURE_TITLE, fontsize=14, fontweight="bold")

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for fmt in OUTPUT_FORMATS:
        output_path = output_dir / f"{OUTPUT_BASENAME}.{fmt}"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        saved_paths.append(output_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_paths[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3種類の目的変数について全アルゴリズムの評価指標を棒グラフで描画する",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="画像の保存先ディレクトリ",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="保存後にウィンドウで表示する",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not METRICS_SCORES:
        raise SystemExit(
            "METRICS_SCORES が空です。"
            " compare_models.py の出力を貼り付けてから実行してください。"
        )

    output_path = plot_metrics_bar(
        METRICS_SCORES,
        output_dir=args.output_dir,
        show=args.show,
    )
    print(f"保存しました: {output_path}")


if __name__ == "__main__":
    main()
