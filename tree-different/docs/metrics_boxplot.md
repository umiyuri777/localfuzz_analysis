# 評価指標の LaTeX 出力と箱ひげ図

task0 / task1 / task2 について、4 手法＋ベースラインの 10 分割交差検証結果を LaTeX 表として出力し、適合率・再現率・F 値の箱ひげ図を 1 枚にまとめて描画するためのスクリプト群のドキュメント。

## 概要

| スクリプト | 役割 |
|-----------|------|
| `export_latex_tables.py` | 交差検証を実行し、LaTeX 表と箱ひげ図用データを出力 |
| `plot_metrics_boxplot.py` | 箱ひげ図を生成（ラベル・文言はスクリプト内の変数で設定） |

```
Logs/ ──► export_latex_tables.py ──► LaTeX 表（論文用）
                              └──► CV_FOLD_SCORES（貼り付け用）
                                        │
                                        ▼
                              plot_metrics_boxplot.py ──► figures/metrics_boxplot_all.png
```

## 対象タスク

| task | 目的変数 | 説明 |
|------|---------|------|
| task0 | `bug_detected` | 1 回の実行でバグ発見 |
| task1 | `bug_detected_any` | 5 回中 1 回でもバグ発見 |
| task2 | `bug_detected_all` | 5 回全てバグ発見 |

## 対象モデル

| 略称 | 正式名 | 備考 |
|------|--------|------|
| BL | ベースライン | 常にバグ発見と予測（`np.ones`） |
| LR | ロジスティック回帰 | |
| DT | 決定木 | |
| RF | ランダムフォレスト | |
| GB | 勾配ブースティング | |

交差検証の設定は両スクリプト共通:

- 分割数: 10（`StratifiedKFold`）
- 乱数シード: 42
- 評価指標: 適合率（precision）・再現率（recall）・F 値（f1）

## 使い方

### 1. 評価の実行とデータ出力

```bash
# プロジェクトルートから
python tree-different/export_latex_tables.py --task all
```

ファイルにも保存する場合:

```bash
python tree-different/export_latex_tables.py --task all --output-dir tree-different/output
```

**出力内容（標準出力）**

1. 各 task ごとの LaTeX 表（評価結果・特徴量重要度・ロジスティック回帰式）
2. 末尾の **貼り付け用ブロック** `CV_FOLD_SCORES`（箱ひげ図用）

`--output-dir` 指定時は以下も保存される:

| ファイル | 内容 |
|---------|------|
| `{task_id}_metrics.tex` | 評価結果表 |
| `{task_id}_importance.tex` | 特徴量重要度表 |
| `{task_id}_logistic.tex` | ロジスティック回帰式 |
| `cv_fold_scores.py` | 箱ひげ図用データ（`CV_FOLD_SCORES` ブロック） |

### 2. 箱ひげ図用データの貼り付け

`export_latex_tables.py` の末尾に出力される `CV_FOLD_SCORES` ブロックを、`plot_metrics_boxplot.py` 先頭の同名変数にコピーする。

```python
CV_FOLD_SCORES: dict[str, dict[str, dict[str, list[float]]]] = {
    "task0": {
        "BL": { "precision": [...], "recall": [...], "f1": [...] },
        "LR": { ... },
        ...
    },
    "task1": { ... },
    "task2": { ... },
}
```

`export_latex_tables.py` 側でモデル名を短縮形（`BL`, `LR`, `DT`, `RF`, `GB`）に揃えて出力するため、ブロックごと置き換えればよい。

### 3. 箱ひげ図の生成

```bash
python tree-different/plot_metrics_boxplot.py
```

オプション:

```bash
python tree-different/plot_metrics_boxplot.py --show          # 保存後にウィンドウ表示
python tree-different/plot_metrics_boxplot.py --output-dir path/to/dir
```

**出力先（デフォルト）:** `tree-different/figures/metrics_boxplot_all.png`

---

## 箱ひげ図のレイアウト

1 枚の図に task0 / task1 / task2 の 3 サブプロットを横並びで配置する。

```
┌─────────────────────┬─────────────────────┬─────────────────────┐
│  1回の実行でバグ発見  │ 5回中1回でもバグ発見  │  5回全てバグ発見      │
│                     │                     │                     │
│ BL  LR  DT  RF  GB  │ BL  LR  DT  RF  GB  │ BL  LR  DT  RF  GB  │
│ ■■■ ■■■ ■■■ ■■■ ■■■ │ ...                 │ ...                 │
└─────────────────────┴─────────────────────┴─────────────────────┘
  凡例: ■ 適合率  ■ 再現率  ■ F値
```

- 各アルゴリズムのグループ内に 3 箱（適合率・再現率・F 値）
- 1 箱 = 10 分割交差検証の各 fold のスコア
- 指標ごとに色分け（デフォルト: 赤 / 青緑 / 青）
- Y 軸は 0〜1

---

## データ構造 `CV_FOLD_SCORES`

```text
CV_FOLD_SCORES[task_id][model_name][metric_key] = list[float]  # 長さ 10
```

| キー | 値の例 |
|------|--------|
| `task_id` | `"task0"`, `"task1"`, `"task2"` |
| `model_name` | `"BL"`, `"LR"`, `"DT"`, `"RF"`, `"GB"` |
| `metric_key` | `"precision"`, `"recall"`, `"f1"` |

---

## `export_latex_tables.py` の追加機能

従来の LaTeX 表出力に加え、箱ひげ図連携用の処理を追加している。

### ベースライン算出 `compute_baseline_fold_scores`

各 fold のテストデータに対し「常にバグあり（`y_pred = 1`）」と予測し、適合率・再現率・F 値を計算する。ノートブック（`y_pred_always_bug = np.ones(...)`）と同じ定義。

- 再現率は正例が存在する fold では常に 1.0
- 適合率・F 値は fold ごとの正例率に依存するためばらつく

### fold スコア抽出 `extract_fold_scores`

各 ML 手法の `cross_validate` 結果から、fold 別の `test_precision` / `test_recall` / `test_f1` を取り出す。

### 貼り付け用フォーマット `format_python_cv_fold_scores_block`

`plot_metrics_boxplot.py` の `CV_FOLD_SCORES` にそのまま貼れる Python 辞書リテラルを生成する。

- 型注釈・変数名・インデントをプロット側と統一
- 小数は 4 桁（`METRIC_DECIMALS = 4`）
- モデル順: `BL` → `LR` → `DT` → `RF` → `GB`

### CLI オプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--task` | `all` | `task0` / `task1` / `task2` / `all` |
| `--logs-root` | `tree-different/Logs` | 実験ログのルート |
| `--output-dir` | なし | LaTeX と `cv_fold_scores.py` の保存先 |
| `--verbose` | off | データ収集時の統計を表示 |

---

## `plot_metrics_boxplot.py` の設定変数

論文やスライドの都合で文言を変えたい場合は、スクリプト上部の変数を編集する。

### データ

| 変数 | 説明 |
|------|------|
| `CV_FOLD_SCORES` | 貼り付け用の交差検証スコア |

### ラベル・表示順

| 変数 | 説明 | デフォルト例 |
|------|------|-------------|
| `MODEL_ORDER` | アルゴリズムの並び | `["BL", "LR", "DT", "RF", "GB"]` |
| `MODEL_LABELS` | X 軸の表示名 | `"BL": "Baseline"` など |
| `TASK_ORDER` | サブプロットの並び | `["task0", "task1", "task2"]` |
| `TASK_LABELS` | サブプロットタイトル | `"task0": "1回の実行でバグ発見"` など |
| `METRIC_LABELS` | 凡例の指標名 | `"precision": "適合率"` など |
| `FIGURE_TITLE` | 図全体のタイトル | `"10分割交差検証の評価指標"` |
| `YLABEL` | Y 軸ラベル | `"スコア"` |

### 見た目

| 変数 | 説明 | デフォルト |
|------|------|-----------|
| `FIGURE_SIZE` | 図のサイズ（インチ） | `(22, 5)` |
| `YLIM` | Y 軸範囲 | `(0.0, 1.0)` |
| `METRIC_COLORS` | 指標ごとの箱の色 | 赤 / 青緑 / 青 |
| `BOX_ALPHA` | 箱の透明度 | `0.7` |
| `BOX_WIDTH` | 箱の幅 | `0.22` |
| `GROUP_GAP` | アルゴリズム間の余白 | `1.0` |
| `OUTPUT_DIR` | 保存ディレクトリ | `tree-different/figures` |
| `OUTPUT_BASENAME` | ファイル名（拡張子除く） | `metrics_boxplot_all` |
| `OUTPUT_FORMATS` | 出力形式 | `("png",)` |

> **Note:** 日本語ラベル利用時、PDF 出力はフォント設定が必要なためデフォルトは PNG のみ。

### ラベル変更の例

```python
MODEL_LABELS = {
    "BL": "ベースライン",
    "LR": "ロジスティック回帰",
    "DT": "決定木",
    "RF": "ランダムフォレスト",
    "GB": "勾配ブースティング",
}

TASK_LABELS = {
    "task0": r"\testfirst",
    "task1": r"\testsecond",
    "task2": r"\testthird",
}
```

---

## ワークフローまとめ

1. 実験ログを `tree-different/Logs/` に配置
2. `export_latex_tables.py` を実行して LaTeX 表と `CV_FOLD_SCORES` を取得
3. 必要なら `plot_metrics_boxplot.py` のラベル変数を編集
4. `CV_FOLD_SCORES` を貼り付け
5. `plot_metrics_boxplot.py` を実行して図を生成
6. 論文には `.tex` 表と `.png` 図を使用

データやモデル設定を変えた場合は、手順 2 からやり直す（箱ひげ図の数値は手計算ではなく交差検証結果を貼り付ける想定）。

---

## 関連ファイル

```
tree-different/
├── export_latex_tables.py    # 評価・LaTeX 出力・CV_FOLD_SCORES 生成
├── plot_metrics_boxplot.py   # 箱ひげ図生成
├── figures/
│   └── metrics_boxplot_all.png
├── docs/
│   └── metrics_boxplot.md    # 本ドキュメント
└── Logs/                     # 実験データ（入力）
```

共通ユーティリティ（`utils/` 配下）:

- `data_loader.py` — データ収集
- `decision_tree_analysis.py` / `random_forest_analysis.py` / `gradient_boosting_analysis.py` / `logistic_regression_analysis.py` — 各手法のパイプライン構築
- `feature_importance.py` — 特徴量重要度の LaTeX 整形
