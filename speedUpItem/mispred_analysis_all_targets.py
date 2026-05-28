"""
speedUpItem: 3種類の目的変数（per_run, bug_detected_any, bug_detected_all）と
4つのモデル（ロジスティック回帰・決定木・ランダムフォレスト・勾配ブースティング）について、
誤予測したパラメータの傾向を分析するためのスクリプト。

実行すると、以下を行う:
- train_dir=tree=500, test_dir=Logs を用いて各 target ごとに4モデルを学習・評価
- 予測結果（特徴量 + 正解ラベル + 予測ラベル + 予測確率 + 正誤 + モデル名 + target名）を結合した DataFrame を作成
- 誤予測のみを抽出した DataFrame を作成
- 要約統計を標準出力に表示
- 両方の DataFrame を CSV として speedUpItem ディレクトリ直下に保存
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# プロジェクトルートをパスに追加して utils をインポート
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.data_loader import load_speedup_bug_dataset
from utils.decision_tree_analysis import build_decision_tree_pipeline
from utils.gradient_boosting_analysis import build_gradient_boosting_pipeline
from utils.logistic_regression_analysis import build_logistic_regression_pipeline
from utils.random_forest_analysis import build_random_forest_pipeline


RANDOM_STATE = 42
SPEEDUP_DIR = Path(__file__).resolve().parent

# 解析対象とする目的変数
TARGETS = ["per_run", "bug_detected_any", "bug_detected_all"]


def _build_models() -> dict[str, Pipeline]:
    """4つの分類モデルを前処理付き Pipeline で構築して返す。"""
    return {
        "logistic": build_logistic_regression_pipeline(
            include_tree=False, random_state=RANDOM_STATE, max_iter=1000,
        ),
        "tree": build_decision_tree_pipeline(
            include_tree=False, random_state=RANDOM_STATE,
        ),
        "rf": build_random_forest_pipeline(
            include_tree=False,
            random_state=RANDOM_STATE,
            n_estimators=100,
            n_jobs=-1,
        ),
        "gb": build_gradient_boosting_pipeline(
            include_tree=False, random_state=RANDOM_STATE,
        ),
    }


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """誤予測の傾向を見るための派生特徴量を追加する（集約済み cpNum / range / dir 用）。"""
    df = df.copy()

    cp_cols = [f"cpNum{i}" for i in range(1, 11)]
    range_cols = [f"cpNum{i}_range" for i in range(1, 11)]
    dir_cols = [f"cpNum{i}_dir" for i in range(1, 11)]

    # cpNum*_dir は utils.data_loader.SPEEDUP_FEATURE_COLUMNS の定義に従う
    # （cpNum10dir だけ列名がやや特殊なので、存在しない場合はスキップ）
    existing_cp_cols = [c for c in cp_cols if c in df.columns]
    existing_range_cols = [c for c in range_cols if c in df.columns]
    existing_dir_cols = [c for c in dir_cols if c in df.columns]

    if existing_cp_cols:
        df["num_cps"] = (df[existing_cp_cols] > 0).sum(axis=1)
        df["sum_cps"] = df[existing_cp_cols].sum(axis=1)

    if existing_range_cols:
        df["max_range"] = df[existing_range_cols].max(axis=1)

    if existing_dir_cols:
        dir_values = df[existing_dir_cols].to_numpy()
        # 最大値となる方角インデックス（1〜len(existing_dir_cols)）
        df["dominant_dir"] = np.argmax(dir_values, axis=1) + 1

    return df


def run_analysis() -> None:
    """全 target × 全モデルの誤予測傾向を分析し、結果を出力・保存する。"""
    train_dir = SPEEDUP_DIR / "tree=500"
    test_dir = SPEEDUP_DIR / "Logs"

    all_pred_dfs: list[pd.DataFrame] = []

    for target in TARGETS:
        print("=" * 80)
        print(f"[target={target}] データ読み込み中...")

        X_train, y_train = load_speedup_bug_dataset(
            str(train_dir),
            target=target,
            verbose=True,
            tree_value=500,
        )
        X_test, y_test = load_speedup_bug_dataset(
            str(test_dir),
            target=target,
            verbose=True,
        )

        models = _build_models()

        for model_name, model in models.items():
            print("-" * 80)
            print(f"[target={target}] モデル学習中: {model_name}")
            model.fit(X_train, y_train)

            print(f"[target={target}] 予測中: {model_name}")
            y_pred = model.predict(X_test)

            # 二値分類を想定して正例クラスの確率を取り出す
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)[:, 1]
            else:
                proba = np.full(shape=len(y_pred), fill_value=np.nan)

            df_pred = X_test.copy()
            df_pred["true_label"] = y_test.to_numpy()
            df_pred["pred_label"] = y_pred
            df_pred["pred_proba"] = proba
            df_pred["correct"] = df_pred["true_label"] == df_pred["pred_label"]
            df_pred["model_name"] = model_name
            df_pred["target_name"] = target

            all_pred_dfs.append(df_pred)

    if not all_pred_dfs:
        print("予測結果が空です。Logs ディレクトリやデータ取得処理を確認してください。")
        return

    df_pred_all = pd.concat(all_pred_dfs, ignore_index=True)
    df_pred_all = _add_derived_features(df_pred_all)

    df_mispred = df_pred_all[~df_pred_all["correct"]].copy()

    # 要約統計の表示
    print("\n" + "=" * 80)
    print("全体のレコード数:", len(df_pred_all))
    print("誤予測レコード数:", len(df_mispred))

    print("\n[target × model 別 誤予測件数]")
    mispred_counts = (
        df_mispred.groupby(["target_name", "model_name"])["correct"]
        .size()
        .unstack("model_name")
        .fillna(0)
        .astype(int)
    )
    print(mispred_counts)

    print("\n[target × model 別 正解率]")
    accuracy_table = (
        df_pred_all.groupby(["target_name", "model_name"])["correct"]
        .mean()
        .unstack("model_name")
    )
    print(accuracy_table)

    # 派生特徴の分布（誤予測 vs 全体）の簡易要約
    for col in ["num_cps", "sum_cps", "max_range", "dominant_dir"]:
        if col not in df_pred_all.columns:
            continue
        print("\n" + "-" * 80)
        print(f"[{col}] の分布要約 (target × model 別, 全体 vs 誤予測)")
        overall_desc = (
            df_pred_all.groupby(["target_name", "model_name"])[col].describe()
        )
        mispred_desc = (
            df_mispred.groupby(["target_name", "model_name"])[col].describe()
        )
        print("\n[全体]")
        print(overall_desc)
        print("\n[誤予測のみ]")
        print(mispred_desc)

    # CSV 保存
    out_all = SPEEDUP_DIR / "mispred_analysis_all_targets_pred_all.csv"
    out_mispred = SPEEDUP_DIR / "mispred_analysis_all_targets_mispred.csv"
    df_pred_all.to_csv(out_all, index=False)
    df_mispred.to_csv(out_mispred, index=False)

    print("\n" + "=" * 80)
    print("予測結果を保存しました。")
    print(f"  全レコード: {out_all}")
    print(f"  誤予測のみ: {out_mispred}")


if __name__ == "__main__":
    run_analysis()

