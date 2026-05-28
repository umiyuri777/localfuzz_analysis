"""
speedUpItem: ランダムフォレストで不具合発見の有無を予測し、
適合率・再現率・F値・正解率を表示する。
"""

import argparse
from pathlib import Path

# プロジェクトルートをパスに追加して utils をインポート
import sys
if (project_root := Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.data_loader import load_speedup_bug_dataset
from utils.random_forest_analysis import build_random_forest_pipeline
from utils.metrics import (
    calculate_binary_metrics,
    print_binary_metrics,
    print_confusion_matrix,
)

RANDOM_STATE = 42
SPEEDUP_DIR = Path(__file__).resolve().parent
TARGET_LABELS = {
    "bug_detected_any": "5回中1回でもバグ",
    "bug_detected_all": "5回全てバグ",
    "per_run": "1回の実行でバグ",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="ランダムフォレストで不具合発見を予測")
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

    model = build_random_forest_pipeline(include_tree=False, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print_confusion_matrix(y_test, y_pred, title=f"ランダムフォレスト（目的: {target_label}）- 混同行列")
    metrics = calculate_binary_metrics(y_test, y_pred, y_pred_proba=y_pred_proba)
    print_binary_metrics(metrics, title=f"ランダムフォレスト（目的: {target_label}）- 評価指標")
    print("\n（適合率=Precision, 再現率=Recall, F値=F1-Score, 正解率=Accuracy, AUC=Area Under ROC Curve）")


if __name__ == "__main__":
    main()
