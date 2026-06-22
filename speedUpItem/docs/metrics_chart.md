# 評価指標の LaTeX 出力と棒グラフ

per_run / bug_detected_any / bug_detected_all について、4 手法＋ベースラインの hold-out 評価結果を LaTeX 表として出力し、適合率・再現率・F 値を棒グラフで 1 枚にまとめるためのスクリプト群のドキュメント。

## 概要

| スクリプト | 役割 |
|-----------|------|
| `compare_models.py` | train/test 評価を実行し、LaTeX 表と棒グラフ用データを出力 |
| `plot_metrics_bar.py` | 棒グラフを生成（ラベル・文言はスクリプト内の変数で設定） |

```
tree=500/ ──┐
            ├──► compare_models.py ──► LaTeX 表（論文用）
Logs/    ───┘                    └──► METRICS_SCORES（貼り付け用）
                                              │
                                              ▼
                                    plot_metrics_bar.py ──► figures/metrics_bar_all.png
```

## tree-different との違い

| 項目 | tree-different | speedUpItem |
|------|----------------|-------------|
| 評価方法 | 10 分割交差検証 | hold-out（train=tree=500, test=Logs） |
| 図の種類 | 箱ひげ図（fold ごとのばらつき） | グループ化棒グラフ（単一スコア） |
| データ変数 | `CV_FOLD_SCORES`（list[float]） | `METRICS_SCORES`（float） |

交差検証を行っていないため、統計検定用の分布はなく、手法間の傾向比較用の棒グラフとする。

## 対象タスク

| target | 目的変数 | 説明 |
|--------|---------|------|
| `per_run` | `bug_detected` | 1 回の実行でバグ発見 |
| `bug_detected_any` | `bug_detected_any` | 5 回中 1 回でもバグ発見 |
| `bug_detected_all` | `bug_detected_all` | 5 回全てバグ発見 |

## 対象モデル

| 略称 | 正式名 | 備考 |
|------|--------|------|
| BL | ベースライン | 常にバグ発見と予測（`np.ones`） |
| LR | ロジスティック回帰 | |
| DT | 決定木 | |
| RF | ランダムフォレスト | |
| GB | 勾配ブースティング | |

評価設定:

- 学習データ: `speedUpItem/tree=500`
- テストデータ: `speedUpItem/Logs`
- 乱数シード: 42
- 評価指標: 適合率（precision）・再現率（recall）・F 値（f1）

## 使い方

### 1. 評価（デフォルトで全 target）

```bash
# プロジェクトルートから（引数なしで3種類すべて評価）
python speedUpItem/compare_models.py
```

単一 target のみ実行する場合:

```bash
python speedUpItem/compare_models.py --target bug_detected_any
```

**出力内容（標準出力）**

1. 各 target ごとの評価表・LaTeX 表・特徴量重要度
2. 末尾の **貼り付け用ブロック** `METRICS_SCORES`（`plot_metrics_bar.py` と同じ形式）

`speedUpItem/output/metrics_scores.py` にも自動保存される。

```python
# =============================================================================
# 貼り付け用データ（compare_models.py の出力をここにコピー）
# =============================================================================
METRICS_SCORES: dict[str, dict[str, dict[str, float]]] = {
    "per_run": {
        "BL": { "precision": 0.72, "recall": 1.00, "f1": 0.84 },
        ...
    },
    ...
}
```

コメント行から閉じ括弧 `}` までをそのまま `plot_metrics_bar.py` の同名ブロックに置き換える。

### 2. 棒グラフの生成

```bash
python speedUpItem/plot_metrics_bar.py
```

オプション:

```bash
python speedUpItem/plot_metrics_bar.py --show          # 保存後にウィンドウ表示
python speedUpItem/plot_metrics_bar.py --output-dir path/to/dir
```

**出力先（デフォルト）:** `speedUpItem/figures/metrics_bar_all.png`

---

## 棒グラフのレイアウト

1 枚の図に per_run / bug_detected_any / bug_detected_all の 3 サブプロットを横並びで配置する。

```
┌─────────────────────┬─────────────────────┬─────────────────────┐
│  1回の実行でバグ発見  │ 5回中1回でもバグ発見  │  5回全てバグ発見      │
│                     │                     │                     │
│ BL  LR  DT  RF  GB  │ BL  LR  DT  RF  GB  │ BL  LR  DT  RF  GB  │
│ ▌▌▌ ▌▌▌ ▌▌▌ ▌▌▌ ▌▌▌ │ ...                 │ ...                 │
└─────────────────────┴─────────────────────┴─────────────────────┘
  凡例: ■ 適合率  ■ 再現率  ■ F値
```

- 各アルゴリズムのグループ内に 3 本の棒（適合率・再現率・F 値）
- 指標ごとに色分け（デフォルト: 赤 / 青緑 / 青）
- Y 軸は 0〜1

---

## データ構造 `METRICS_SCORES`

```text
METRICS_SCORES[target][model_name][metric_key] = float
```

| キー | 値の例 |
|------|--------|
| `target` | `"per_run"`, `"bug_detected_any"`, `"bug_detected_all"` |
| `model_name` | `"BL"`, `"LR"`, `"DT"`, `"RF"`, `"GB"` |
| `metric_key` | `"precision"`, `"recall"`, `"f1"` |

---

## `compare_models.py` の追加機能

### `--target`（省略可）

省略時は 3 種類すべて評価し、`METRICS_SCORES` を出力する。単一のみ実行するときに指定する。

### `evaluate_models_for_target`

1 target について train/test 評価を実行し `(モデル名, 指標 dict)` のリストを返す。

### `build_plot_metrics_scores` / `format_python_metrics_scores_block`

`plot_metrics_bar.py` の `METRICS_SCORES` にそのまま貼れる Python 辞書リテラルを生成する。

- モデル順: `BL` → `LR` → `DT` → `RF` → `GB`
- 小数は 2 桁（`METRIC_DECIMALS = 2`）

### CLI オプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--target` | なし（全 target） | `per_run` / `bug_detected_any` / `bug_detected_all` |
| `--output-dir` | `speedUpItem/output` | `metrics_scores.py` の保存先 |

---

## `plot_metrics_bar.py` の設定変数

論文やスライドの都合で文言を変えたい場合は、スクリプト上部の変数を編集する。

### データ

| 変数 | 説明 |
|------|------|
| `METRICS_SCORES` | 貼り付け用の評価スコア |

### ラベル・表示順

| 変数 | 説明 | デフォルト例 |
|------|------|-------------|
| `MODEL_ORDER` | アルゴリズムの並び | `["BL", "LR", "DT", "RF", "GB"]` |
| `MODEL_LABELS` | X 軸の表示名 | `"BL": "Baseline"` など |
| `TARGET_ORDER` | サブプロットの並び | `["per_run", "bug_detected_any", "bug_detected_all"]` |
| `TARGET_LABELS` | サブプロットタイトル | `"per_run": "1回の実行でバグ発見"` など |
| `METRIC_LABELS` | 凡例の指標名 | `"precision": "適合率"` など |
| `FIGURE_TITLE` | 図全体のタイトル | `"hold-out 評価の評価指標..."` |
| `YLABEL` | Y 軸ラベル | `"スコア"` |

### 見た目

| 変数 | 説明 | デフォルト |
|------|------|-----------|
| `FIGURE_SIZE` | 図のサイズ（インチ） | `(22, 5)` |
| `YLIM` | Y 軸範囲 | `(0.0, 1.05)` |
| `METRIC_COLORS` | 指標ごとの棒の色 | 赤 / 青緑 / 青 |
| `BAR_ALPHA` | 棒の透明度 | `0.85` |
| `BAR_WIDTH` | 棒の幅 | `0.22` |
| `GROUP_GAP` | アルゴリズム間の余白 | `1.0` |
| `OUTPUT_DIR` | 保存ディレクトリ | `speedUpItem/figures` |
| `OUTPUT_BASENAME` | ファイル名（拡張子除く） | `metrics_bar_all` |
| `OUTPUT_FORMATS` | 出力形式 | `("png",)` |

---

## ワークフローまとめ

1. 実験ログを `speedUpItem/tree=500/`（学習）と `speedUpItem/Logs/`（テスト）に配置
2. `compare_models.py` を実行して `METRICS_SCORES` を取得
3. 必要なら `plot_metrics_bar.py` のラベル変数を編集
4. `METRICS_SCORES` を貼り付け
5. `plot_metrics_bar.py` を実行して図を生成
6. 論文には `.tex` 表と `.png` 図を使用

データやモデル設定を変えた場合は、手順 2 からやり直す。

---

## 関連ファイル

```
speedUpItem/
├── compare_models.py       # 評価・LaTeX 出力・METRICS_SCORES 生成
├── plot_metrics_bar.py     # 棒グラフ生成
├── figures/
│   └── metrics_bar_all.png
├── docs/
│   └── metrics_chart.md    # 本ドキュメント
├── tree=500/               # 学習データ
└── Logs/                   # テストデータ
```
