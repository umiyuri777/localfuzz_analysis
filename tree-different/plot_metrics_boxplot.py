"""
task0 / task1 / task2 の適合率・再現率・F値を 1 枚の箱ひげ図に描画する。

各 task のサブプロット内に、全アルゴリズムを横並びで表示する。
各アルゴリズムのグループ内に適合率・再現率・F値の 3 箱を並べる。

使い方:
1. export_latex_tables.py を実行し、出力された CV_FOLD_SCORES ブロックを下記に貼り付ける
2. ラベルなどの設定を必要に応じて変更する
3. python plot_metrics_boxplot.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

# =============================================================================
# 貼り付け用データ（export_latex_tables.py の出力をここにコピー）
# =============================================================================
CV_FOLD_SCORES: dict[str, dict[str, dict[str, list[float]]]] = {
    "task0": {
        "BL": {
            "precision": [0.5769, 0.5769, 0.5769, 0.5759, 0.5759, 0.5759, 0.5759, 0.5759, 0.5759, 0.5759],
            "recall": [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            "f1": [0.7317, 0.7317, 0.7317, 0.7309, 0.7309, 0.7309, 0.7309, 0.7309, 0.7309, 0.7309]
        },
        "LR": {
            "precision": [0.7936, 0.7955, 0.7852, 0.8328, 0.7947, 0.7937, 0.7614, 0.7933, 0.7811, 0.8048],
            "recall": [0.8700, 0.8989, 0.8860, 0.9212, 0.9084, 0.8971, 0.8826, 0.8762, 0.8891, 0.8617],
            "f1": [0.8300, 0.8440, 0.8326, 0.8748, 0.8477, 0.8423, 0.8176, 0.8327, 0.8316, 0.8323]
        },
        "DT": {
            "precision": [0.8744, 0.8767, 0.8778, 0.9085, 0.8820, 0.8932, 0.8430, 0.8806, 0.8715, 0.8853],
            "recall": [0.8266, 0.8443, 0.8299, 0.8778, 0.8650, 0.8601, 0.8376, 0.8537, 0.8505, 0.8312],
            "f1": [0.8498, 0.8602, 0.8531, 0.8929, 0.8734, 0.8763, 0.8403, 0.8669, 0.8609, 0.8574]
        },
        "RF": {
            "precision": [0.8744, 0.8767, 0.8778, 0.9085, 0.8820, 0.8932, 0.8430, 0.8806, 0.8715, 0.8853],
            "recall": [0.8266, 0.8443, 0.8299, 0.8778, 0.8650, 0.8601, 0.8376, 0.8537, 0.8505, 0.8312],
            "f1": [0.8498, 0.8602, 0.8531, 0.8929, 0.8734, 0.8763, 0.8403, 0.8669, 0.8609, 0.8574]
        },
        "GB": {
            "precision": [0.8744, 0.8767, 0.8778, 0.9085, 0.8820, 0.8932, 0.8430, 0.8806, 0.8715, 0.8853],
            "recall": [0.8266, 0.8443, 0.8299, 0.8778, 0.8650, 0.8601, 0.8376, 0.8537, 0.8505, 0.8312],
            "f1": [0.8498, 0.8602, 0.8531, 0.8929, 0.8734, 0.8763, 0.8403, 0.8669, 0.8609, 0.8574]
        }
    },
    "task1": {
        "BL": {
            "precision": [0.8472, 0.8472, 0.8472, 0.8472, 0.8426, 0.8426, 0.8426, 0.8426, 0.8426, 0.8426],
            "recall": [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            "f1": [0.9173, 0.9173, 0.9173, 0.9173, 0.9146, 0.9146, 0.9146, 0.9146, 0.9146, 0.9146]
        },
        "LR": {
            "precision": [0.8472, 0.8472, 0.8472, 0.8472, 0.8426, 0.8426, 0.8426, 0.8465, 0.8426, 0.8426],
            "recall": [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            "f1": [0.9173, 0.9173, 0.9173, 0.9173, 0.9146, 0.9146, 0.9146, 0.9169, 0.9146, 0.9146]
        },
        "DT": {
            "precision": [0.8472, 0.8472, 0.8472, 0.8472, 0.8426, 0.8426, 0.8426, 0.8426, 0.8426, 0.8426],
            "recall": [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            "f1": [0.9173, 0.9173, 0.9173, 0.9173, 0.9146, 0.9146, 0.9146, 0.9146, 0.9146, 0.9146]
        },
        "RF": {
            "precision": [0.8472, 0.8472, 0.8472, 0.8472, 0.8426, 0.8426, 0.8426, 0.8426, 0.8426, 0.8426],
            "recall": [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            "f1": [0.9173, 0.9173, 0.9173, 0.9173, 0.9146, 0.9146, 0.9146, 0.9146, 0.9146, 0.9146]
        },
        "GB": {
            "precision": [0.8429, 0.8565, 0.8443, 0.8551, 0.8469, 0.8502, 0.8443, 0.8483, 0.8469, 0.8510],
            "recall": [0.9672, 0.9781, 0.9781, 0.9672, 0.9725, 0.9670, 0.9835, 0.9835, 0.9725, 0.9725],
            "f1": [0.9008, 0.9133, 0.9063, 0.9077, 0.9054, 0.9049, 0.9086, 0.9109, 0.9054, 0.9077]
        }
    },
    "task2": {
        "BL": {
            "precision": [0.3565, 0.3565, 0.3565, 0.3519, 0.3519, 0.3519, 0.3519, 0.3519, 0.3519, 0.3519],
            "recall": [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            "f1": [0.5256, 0.5256, 0.5256, 0.5205, 0.5205, 0.5205, 0.5205, 0.5205, 0.5205, 0.5205]
        },
        "LR": {
            "precision": [0.6420, 0.5556, 0.6265, 0.6567, 0.5542, 0.6301, 0.6338, 0.5059, 0.5753, 0.5978],
            "recall": [0.6753, 0.6494, 0.6753, 0.5789, 0.6053, 0.6053, 0.5921, 0.5658, 0.5526, 0.7237],
            "f1": [0.6582, 0.5988, 0.6500, 0.6154, 0.5786, 0.6174, 0.6122, 0.5342, 0.5638, 0.6548]
        },
        "DT": {
            "precision": [0.6239, 0.6016, 0.6063, 0.6552, 0.7053, 0.5846, 0.7234, 0.6606, 0.6327, 0.6283],
            "recall": [0.8831, 1.0000, 1.0000, 1.0000, 0.8816, 1.0000, 0.8947, 0.9474, 0.8158, 0.9342],
            "f1": [0.7312, 0.7512, 0.7549, 0.7917, 0.7836, 0.7379, 0.8000, 0.7784, 0.7126, 0.7513]
        },
        "RF": {
            "precision": [0.6040, 0.6161, 0.6179, 0.6383, 0.6824, 0.6050, 0.7111, 0.6700, 0.6289, 0.6460],
            "recall": [0.7922, 0.8961, 0.9870, 0.7895, 0.7632, 0.9474, 0.8421, 0.8816, 0.8026, 0.9605],
            "f1": [0.6854, 0.7302, 0.7600, 0.7059, 0.7205, 0.7385, 0.7711, 0.7614, 0.7052, 0.7725]
        },
        "GB": {
            "precision": [0.6373, 0.6053, 0.6179, 0.6300, 0.6818, 0.6000, 0.6848, 0.6559, 0.6311, 0.6435],
            "recall": [0.8442, 0.8961, 0.9870, 0.8289, 0.7895, 0.9474, 0.8289, 0.8026, 0.8553, 0.9737],
            "f1": [0.7263, 0.7225, 0.7600, 0.7159, 0.7317, 0.7347, 0.7500, 0.7219, 0.7263, 0.7749]
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

TASK_ORDER = ["task0", "task1", "task2"]
TASK_LABELS = {
    "task0": "単回",
    "task1": "1回以上",
    "task2": "全回",
}

METRIC_ORDER = ["precision", "recall", "f1"]
METRIC_LABELS = {
    "precision": "適合率",
    "recall": "再現率",
    "f1": "F値",
}

FIGURE_TITLE = "各モデルの評価結果(モデル構築プロセス)"
YLABEL = "スコア"
YLIM = (0.0, 1.02)  # 1.0 ちょうどの箱ひげが上端で見えなくなるのを防ぐ

FIGURE_SIZE = (15, 5)
METRIC_COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
BOX_ALPHA = 0.7
GROUP_GAP = 0.4
METRIC_SPACING = 0.3
BOX_WIDTH = 0.2
BOX_LABEL_FONTSIZE = 8
BOX_LABEL_ROTATION = 45
BOX_LABEL_Y = -0.02
MODEL_LABEL_FONTSIZE = 15
MODEL_LABEL_Y = -0.15

OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_BASENAME = "metrics_model"
OUTPUT_FORMATS = ("png",)  # 日本語ラベル利用時は pdf はフォント設定が必要

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
    cv_fold_scores: dict[str, dict[str, dict[str, list[float]]]],
) -> None:
    missing_tasks = [task_id for task_id in TASK_ORDER if task_id not in cv_fold_scores]
    if missing_tasks:
        raise ValueError(f"CV_FOLD_SCORES に task が不足しています: {missing_tasks}")

    for task_id in TASK_ORDER:
        task_scores = cv_fold_scores[task_id]
        for model_name in MODEL_ORDER:
            if model_name not in task_scores:
                available = ", ".join(sorted(task_scores))
                raise ValueError(
                    f"{task_id} にモデル '{model_name}' がありません。"
                    f" 利用可能: {available}"
                )
            for metric_key in METRIC_ORDER:
                if metric_key not in task_scores[model_name]:
                    raise ValueError(
                        f"{task_id} / {model_name} に指標 '{metric_key}' がありません"
                    )


def _build_grouped_boxplot_data(
    task_scores: dict[str, dict[str, list[float]]],
) -> tuple[
    list[list[float]],
    list[float],
    list[str],
    list[float],
    list[str],
    list[str],
]:
    """1 task 分のグループ化箱ひげ図用データを生成する。"""
    n_metrics = len(METRIC_ORDER)
    group_stride = (n_metrics - 1) * METRIC_SPACING + GROUP_GAP + METRIC_SPACING

    box_data: list[list[float]] = []
    positions: list[float] = []
    box_colors: list[str] = []
    box_metric_keys: list[str] = []
    group_centers: list[float] = []

    for group_idx, model_name in enumerate(MODEL_ORDER):
        group_start = group_idx * group_stride
        group_centers.append(group_start + (n_metrics - 1) * METRIC_SPACING / 2)

        for metric_idx, metric_key in enumerate(METRIC_ORDER):
            positions.append(group_start + metric_idx * METRIC_SPACING)
            box_data.append(task_scores[model_name][metric_key])
            box_colors.append(METRIC_COLORS[metric_idx])
            box_metric_keys.append(metric_key)

    tick_labels = [MODEL_LABELS.get(model, model) for model in MODEL_ORDER]
    return box_data, positions, box_colors, group_centers, tick_labels, box_metric_keys


def _add_box_metric_labels(
    ax: plt.Axes,
    positions: list[float],
    metric_keys: list[str],
) -> None:
    """各箱ひげ図の直下に指標ラベルを付ける（白黒印刷でも区別可能にする）。"""
    label_transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for pos, metric_key in zip(positions, metric_keys):
        ax.text(
            pos,
            BOX_LABEL_Y,
            METRIC_LABELS[metric_key],
            ha="right",
            va="top",
            rotation=BOX_LABEL_ROTATION,
            rotation_mode="anchor",
            fontsize=BOX_LABEL_FONTSIZE,
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


def plot_metrics_boxplot(
    cv_fold_scores: dict[str, dict[str, dict[str, list[float]]]],
    *,
    output_dir: Path = OUTPUT_DIR,
    show: bool = False,
) -> Path:
    """全アルゴリズムを各 task サブプロット内に横並びで描画する。"""
    _validate_scores(cv_fold_scores)

    configure_matplotlib()
    fig, axes = plt.subplots(1, len(TASK_ORDER), figsize=FIGURE_SIZE, sharey=True)
    if len(TASK_ORDER) == 1:
        axes = [axes]

    for ax, task_id in zip(axes, TASK_ORDER):
        box_data, positions, box_colors, group_centers, tick_labels, box_metric_keys = (
            _build_grouped_boxplot_data(cv_fold_scores[task_id])
        )

        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=BOX_WIDTH,
            patch_artist=True,
            manage_ticks=False,
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(BOX_ALPHA)

        ax.set_xticks([])
        _add_box_metric_labels(ax, positions, box_metric_keys)
        _add_model_labels(ax, group_centers, tick_labels)
        task_label = TASK_LABELS.get(task_id, task_id)
        ax.set_title(task_label, fontsize=12, fontweight="bold")
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
        description="task0〜task2 の全アルゴリズム評価指標を箱ひげ図で描画する",
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
    if not CV_FOLD_SCORES:
        raise SystemExit(
            "CV_FOLD_SCORES が空です。"
            " export_latex_tables.py の出力を貼り付けてから実行してください。"
        )

    output_path = plot_metrics_boxplot(
        CV_FOLD_SCORES,
        output_dir=args.output_dir,
        show=args.show,
    )
    print(f"保存しました: {output_path}")


if __name__ == "__main__":
    main()
