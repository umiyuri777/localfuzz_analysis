"""
per_run / bug_detected_any / bug_detected_all の適合率・再現率・F値と
特徴量重要度を 1 枚の棒グラフに描画する。

train=tree=500, test=Logs の hold-out 評価結果を可視化する。
交差検証を行っていないため、箱ひげ図ではなくグループ化棒グラフを使う。

使い方:
1. compare_models.py を実行し、末尾の METRICS_SCORES / FEATURE_IMPORTANCES ブロックを下記に貼り付ける
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
            "precision": 0.7200,
            "recall": 1.0000,
            "f1": 0.8372
        },
        "LR": {
            "precision": 0.8113,
            "recall": 0.8000,
            "f1": 0.8056
        },
        "DT": {
            "precision": 0.8464,
            "recall": 0.8111,
            "f1": 0.8284
        },
        "RF": {
            "precision": 0.8464,
            "recall": 0.8111,
            "f1": 0.8284
        },
        "GB": {
            "precision": 0.8464,
            "recall": 0.8111,
            "f1": 0.8284
        }
    },
    "bug_detected_any": {
        "BL": {
            "precision": 0.9400,
            "recall": 1.0000,
            "f1": 0.9691
        },
        "LR": {
            "precision": 0.9400,
            "recall": 1.0000,
            "f1": 0.9691
        },
        "DT": {
            "precision": 0.9400,
            "recall": 1.0000,
            "f1": 0.9691
        },
        "RF": {
            "precision": 0.9400,
            "recall": 1.0000,
            "f1": 0.9691
        },
        "GB": {
            "precision": 0.9394,
            "recall": 0.9894,
            "f1": 0.9637
        }
    },
    "bug_detected_all": {
        "BL": {
            "precision": 0.4200,
            "recall": 1.0000,
            "f1": 0.5915
        },
        "LR": {
            "precision": 0.6667,
            "recall": 0.5714,
            "f1": 0.6154
        },
        "DT": {
            "precision": 0.5645,
            "recall": 0.8333,
            "f1": 0.6731
        },
        "RF": {
            "precision": 0.5397,
            "recall": 0.8095,
            "f1": 0.6476
        },
        "GB": {
            "precision": 0.5397,
            "recall": 0.8095,
            "f1": 0.6476
        }
    }
}

# =============================================================================
# 貼り付け用データ（compare_models.py の出力をここにコピー）
# =============================================================================
FEATURE_IMPORTANCES: dict[str, dict[str, dict[str, float]]] = {
    "per_run": {
        "DT": {
            "cpNum": 0.3351,
            "cpNum_range": 0.6642,
            "cpNum_dir_2": 0.0002,
            "cpNum_dir_3": 0.0000,
            "cpNum_dir_4": 0.0005
        },
        "RF": {
            "cpNum": 0.2805,
            "cpNum_range": 0.7127,
            "cpNum_dir_2": 0.0020,
            "cpNum_dir_3": 0.0020,
            "cpNum_dir_4": 0.0027
        },
        "GB": {
            "cpNum": 0.3342,
            "cpNum_range": 0.6589,
            "cpNum_dir_2": 0.0010,
            "cpNum_dir_3": 0.0030,
            "cpNum_dir_4": 0.0029
        }
    },
    "bug_detected_any": {
        "DT": {
            "cpNum": 0.2634,
            "cpNum_range": 0.7183,
            "cpNum_dir_2": 0.0159,
            "cpNum_dir_3": 0.0017,
            "cpNum_dir_4": 0.0006
        },
        "RF": {
            "cpNum": 0.3049,
            "cpNum_range": 0.6197,
            "cpNum_dir_2": 0.0288,
            "cpNum_dir_3": 0.0236,
            "cpNum_dir_4": 0.0231
        },
        "GB": {
            "cpNum": 0.2837,
            "cpNum_range": 0.6622,
            "cpNum_dir_2": 0.0218,
            "cpNum_dir_3": 0.0061,
            "cpNum_dir_4": 0.0261
        }
    },
    "bug_detected_all": {
        "DT": {
            "cpNum": 0.3702,
            "cpNum_range": 0.6184,
            "cpNum_dir_2": 0.0018,
            "cpNum_dir_3": 0.0013,
            "cpNum_dir_4": 0.0082
        },
        "RF": {
            "cpNum": 0.3367,
            "cpNum_range": 0.6384,
            "cpNum_dir_2": 0.0059,
            "cpNum_dir_3": 0.0066,
            "cpNum_dir_4": 0.0125
        },
        "GB": {
            "cpNum": 0.3716,
            "cpNum_range": 0.6141,
            "cpNum_dir_2": 0.0022,
            "cpNum_dir_3": 0.0023,
            "cpNum_dir_4": 0.0098
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

IMPORTANCE_MODEL_ORDER = ["DT", "RF", "GB"]
IMPORTANCE_MODEL_LABELS = {
    "DT": "DT",
    "RF": "RF",
    "GB": "GB",
}

TARGET_ORDER = ["per_run", "bug_detected_any", "bug_detected_all"]
TARGET_LABELS = {
    "per_run": "Single",
    "bug_detected_any": "Partial",
    "bug_detected_all": "All",
}

METRIC_ORDER = ["precision", "recall", "f1"]
METRIC_LABELS = {
    "precision": "適合率",
    "recall": "再現率",
    "f1": "F値",
}

FEATURE_ORDER = [
    "cpNum",
    "cpNum_range",
    "cpNum_dir_2",
    "cpNum_dir_3",
    "cpNum_dir_4",
]
FEATURE_LABELS = {
    "cpNum": "C",
    "cpNum_range": "D",
    "cpNum_dir_2": "E=2",
    "cpNum_dir_3": "E=3",
    "cpNum_dir_4": "E=4",
}

FIGURE_TITLE = "各モデルの評価結果(適用プロセス)"
IMPORTANCE_FIGURE_TITLE = "各モデルの特徴量重要度(適用プロセス)"
YLABEL = "スコア"
IMPORTANCE_YLABEL = "重要度"
YLIM = (0.0, 1.05)

FIGURE_SIZE = (15, 5)
# 白黒印刷向けグレースケール（明→暗）
METRIC_COLORS = ["#C8C8C8", "#909090", "#585858"]
FEATURE_COLORS = ["#C0C0C0", "#909090", "#707070", "#505050", "#303030"]
BAR_ALPHA = 0.85
BAR_EDGE_COLOR = "black"
GROUP_GAP = 1.0
BAR_WIDTH = 0.22
BAR_LABEL_FONTSIZE = 8
BAR_LABEL_ROTATION = 90
BAR_LABEL_Y = -0.02
BAR_VALUE_LABEL_FONTSIZE = 7
BAR_VALUE_LABEL_OFFSET = 0.015
BAR_VALUE_LABEL_FORMAT = "{:.2f}"
MODEL_LABEL_FONTSIZE = 15
MODEL_LABEL_Y = -0.15

OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_BASENAME = "metrics_apply"
IMPORTANCE_OUTPUT_BASENAME = "importance_apply"
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


def _validate_grouped_scores(
    grouped_scores: dict[str, dict[str, dict[str, float]]],
    *,
    target_order: list[str],
    group_order: list[str],
    item_order: list[str],
    data_name: str,
) -> None:
    missing_targets = [target for target in target_order if target not in grouped_scores]
    if missing_targets:
        raise ValueError(f"{data_name} に target が不足しています: {missing_targets}")

    for target in target_order:
        target_scores = grouped_scores[target]
        for group_name in group_order:
            if group_name not in target_scores:
                available = ", ".join(sorted(target_scores))
                raise ValueError(
                    f"{target} にグループ '{group_name}' がありません。"
                    f" 利用可能: {available}"
                )
            for item_key in item_order:
                if item_key not in target_scores[group_name]:
                    raise ValueError(
                        f"{target} / {group_name} に項目 '{item_key}' がありません"
                    )


def _build_grouped_bar_data(
    group_scores: dict[str, dict[str, float]],
    *,
    group_order: list[str],
    item_order: list[str],
    item_colors: list[str],
    group_labels: dict[str, str],
) -> tuple[
    list[float],
    list[float],
    list[str],
    list[float],
    list[str],
    list[str],
]:
    """1 target 分のグループ化棒グラフ用データを生成する。"""
    n_items = len(item_order)
    group_stride = n_items + GROUP_GAP

    heights: list[float] = []
    positions: list[float] = []
    bar_colors: list[str] = []
    bar_item_keys: list[str] = []
    group_centers: list[float] = []

    for group_idx, group_name in enumerate(group_order):
        group_start = group_idx * group_stride
        group_centers.append(group_start + (n_items - 1) / 2)

        for item_idx, item_key in enumerate(item_order):
            positions.append(group_start + item_idx)
            heights.append(group_scores[group_name][item_key])
            bar_colors.append(item_colors[item_idx])
            bar_item_keys.append(item_key)

    tick_labels = [group_labels.get(group_name, group_name) for group_name in group_order]
    return heights, positions, bar_colors, group_centers, tick_labels, bar_item_keys


def _add_bar_item_labels(
    ax: plt.Axes,
    positions: list[float],
    item_keys: list[str],
    item_labels: dict[str, str],
) -> None:
    """各棒の直下に項目ラベルを付ける（白黒印刷でも区別可能にする）。"""
    label_transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for pos, item_key in zip(positions, item_keys):
        ax.text(
            pos,
            BAR_LABEL_Y,
            item_labels[item_key],
            ha="right",
            va="top",
            rotation=BAR_LABEL_ROTATION,
            rotation_mode="anchor",
            fontsize=BAR_LABEL_FONTSIZE,
            transform=label_transform,
        )


def _add_bar_value_labels(
    ax: plt.Axes,
    bars: list[object],
    *,
    y_offset: float,
    value_format: str,
    show_zero: bool,
) -> None:
    """各棒の上に数値ラベルを付ける。0 値でも表示する。"""
    y_min, y_max = ax.get_ylim()
    # y_max 側でラベルが切れないように、上限を少し引いておく。
    y_cap = y_max - (y_offset / 2)

    for bar in bars:
        height = float(bar.get_height())
        if not show_zero and abs(height) < 1e-12:
            continue

        label = value_format.format(height)
        x = bar.get_x() + bar.get_width() / 2
        y = height + y_offset
        if y > y_cap:
            y = y_cap

        ax.text(
            x,
            y,
            label,
            ha="center",
            va="bottom",
            fontsize=BAR_VALUE_LABEL_FONTSIZE,
            color="black",
            clip_on=False,
        )


def _add_model_labels(
    ax: plt.Axes,
    group_centers: list[float],
    model_labels: list[str],
) -> None:
    """アルゴリズム名を項目ラベルの下に表示する。"""
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


def _plot_grouped_bar_chart(
    grouped_scores: dict[str, dict[str, dict[str, float]]],
    *,
    target_order: list[str],
    group_order: list[str],
    item_order: list[str],
    item_colors: list[str],
    group_labels: dict[str, str],
    item_labels: dict[str, str],
    target_labels: dict[str, str],
    figure_title: str,
    ylabel: str,
    output_basename: str,
    output_dir: Path,
    show: bool,
    data_name: str,
) -> Path:
    """target ごとにグループ化棒グラフを描画する共通処理。"""
    _validate_grouped_scores(
        grouped_scores,
        target_order=target_order,
        group_order=group_order,
        item_order=item_order,
        data_name=data_name,
    )

    configure_matplotlib()
    fig, axes = plt.subplots(1, len(target_order), figsize=FIGURE_SIZE, sharey=True)
    if len(target_order) == 1:
        axes = [axes]

    for ax, target in zip(axes, target_order):
        heights, positions, bar_colors, group_centers, tick_labels, bar_item_keys = (
            _build_grouped_bar_data(
                grouped_scores[target],
                group_order=group_order,
                item_order=item_order,
                item_colors=item_colors,
                group_labels=group_labels,
            )
        )

        bar_container = ax.bar(
            positions,
            heights,
            width=BAR_WIDTH,
            color=bar_colors,
            alpha=BAR_ALPHA,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=0.5,
        )

        ax.set_xticks([])
        _add_bar_item_labels(ax, positions, bar_item_keys, item_labels)
        _add_model_labels(ax, group_centers, tick_labels)
        target_label = target_labels.get(target, target)
        ax.set_title(target_label, fontsize=12, fontweight="bold")
        ax.set_ylim(*YLIM)

        # 棒の上の数値ラベル。0 値でも表示する。
        _add_bar_value_labels(
            ax,
            list(bar_container.patches),
            y_offset=BAR_VALUE_LABEL_OFFSET,
            value_format=BAR_VALUE_LABEL_FORMAT,
            show_zero=True,
        )
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel(ylabel, fontsize=12)
    fig.suptitle(figure_title, fontsize=14, fontweight="bold")

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for fmt in OUTPUT_FORMATS:
        output_path = output_dir / f"{output_basename}.{fmt}"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        saved_paths.append(output_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_paths[0]


def plot_metrics_bar(
    metrics_scores: dict[str, dict[str, dict[str, float]]],
    *,
    output_dir: Path = OUTPUT_DIR,
    show: bool = False,
) -> Path:
    """全アルゴリズムを各 target サブプロット内に横並びで描画する。"""
    return _plot_grouped_bar_chart(
        metrics_scores,
        target_order=TARGET_ORDER,
        group_order=MODEL_ORDER,
        item_order=METRIC_ORDER,
        item_colors=METRIC_COLORS,
        group_labels=MODEL_LABELS,
        item_labels=METRIC_LABELS,
        target_labels=TARGET_LABELS,
        figure_title=FIGURE_TITLE,
        ylabel=YLABEL,
        output_basename=OUTPUT_BASENAME,
        output_dir=output_dir,
        show=show,
        data_name="METRICS_SCORES",
    )


def plot_feature_importance_bar(
    feature_importances: dict[str, dict[str, dict[str, float]]],
    *,
    output_dir: Path = OUTPUT_DIR,
    show: bool = False,
) -> Path:
    """特徴量重要度を各 target サブプロット内に横並びで描画する。"""
    return _plot_grouped_bar_chart(
        feature_importances,
        target_order=TARGET_ORDER,
        group_order=IMPORTANCE_MODEL_ORDER,
        item_order=FEATURE_ORDER,
        item_colors=FEATURE_COLORS,
        group_labels=IMPORTANCE_MODEL_LABELS,
        item_labels=FEATURE_LABELS,
        target_labels=TARGET_LABELS,
        figure_title=IMPORTANCE_FIGURE_TITLE,
        ylabel=IMPORTANCE_YLABEL,
        output_basename=IMPORTANCE_OUTPUT_BASENAME,
        output_dir=output_dir,
        show=show,
        data_name="FEATURE_IMPORTANCES",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3種類の目的変数について評価指標と特徴量重要度を棒グラフで描画する",
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
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="評価指標のみ描画する",
    )
    parser.add_argument(
        "--importance-only",
        action="store_true",
        help="特徴量重要度のみ描画する",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.metrics_only and args.importance_only:
        raise SystemExit("--metrics-only と --importance-only は同時に指定できません。")

    has_metrics = bool(METRICS_SCORES)
    has_importance = bool(FEATURE_IMPORTANCES)

    if args.metrics_only:
        plot_metrics = True
        plot_importance = False
    elif args.importance_only:
        plot_metrics = False
        plot_importance = True
    else:
        plot_metrics = has_metrics
        plot_importance = has_importance

    if not plot_metrics and not plot_importance:
        raise SystemExit(
            "METRICS_SCORES と FEATURE_IMPORTANCES の両方が空です。"
            " compare_models.py の出力を貼り付けてから実行してください。"
        )

    saved_paths: list[Path] = []

    if plot_metrics:
        if not has_metrics:
            raise SystemExit(
                "METRICS_SCORES が空です。"
                " compare_models.py の出力を貼り付けてから実行してください。"
            )
        saved_paths.append(
            plot_metrics_bar(
                METRICS_SCORES,
                output_dir=args.output_dir,
                show=args.show,
            )
        )

    if plot_importance:
        if not has_importance:
            raise SystemExit(
                "FEATURE_IMPORTANCES が空です。"
                " compare_models.py の出力を貼り付けてから実行してください。"
            )
        saved_paths.append(
            plot_feature_importance_bar(
                FEATURE_IMPORTANCES,
                output_dir=args.output_dir,
                show=args.show,
            )
        )

    for output_path in saved_paths:
        print(f"保存しました: {output_path}")


if __name__ == "__main__":
    main()
