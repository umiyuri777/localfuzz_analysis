"""
実験ログ（Logs）から実験結果を読み出すモジュール。

ディレクトリ命名規則に従い、detected_bugs.csv / exe_time.csv を読み、
ロジスティック回帰用（1実行1レコード）または speedUpItem 用（flat 構造）の DataFrame を返す。
"""

import csv
from pathlib import Path
from typing import Literal, Optional, Tuple

import pandas as pd


def parse_directory_name(dir_name: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    ディレクトリ名をパースしてcpNum, cpNum_range, cpNum_dirを抽出する。

    cpNum1..10, cpNum_range1..10, cpNum_dir1..10 は同一添字位置の値が
    1つのテストケースを表す。cpNum ブロックで非ゼロが見つかった添字 i から
    cpNum_range(i+10), cpNum_dir(i+20) を読む。

    Args:
        dir_name: ディレクトリ名（例: "0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1"）

    Returns:
        (cpNum, cpNum_range, cpNum_dir) または (None, None, None)
    """
    params = dir_name.split(",")
    if len(params) < 30:
        return None, None, None

    try:
        param_index: Optional[int] = None
        for i in range(10):
            if int(params[i]) != 0:
                param_index = i
                break

        if param_index is None:
            return None, None, None

        cpnum = int(params[param_index])
        cpnum_range = int(params[10 + param_index])
        cpnum_dir = int(params[20 + param_index])
        return cpnum, cpnum_range, cpnum_dir
    except (ValueError, IndexError):
        return None, None, None


def _bug_row_to_label(bug_row: list) -> Literal["timeout", "normal", "bug"]:
    """detected_bugs.csv の1行を 'timeout' / 'normal' / 'bug' に変換する。"""
    if bug_row == ["timeout"]:
        return "timeout"
    if bug_row == ["null"]:
        return "normal"
    return "bug"


def _collect_from_param_dirs(
    param_dirs: list,
    tree_value: int,
    include_exe_time: bool,
    data_records: list,
) -> None:
    """指定したパラメータディレクトリ群から1実行1レコードを収集する。"""
    for param_dir in param_dirs:
        if not param_dir.is_dir():
            continue

        cpnum, cpnum_range, cpnum_dir = parse_directory_name(param_dir.name)
        if cpnum is None or cpnum_range is None or cpnum_dir is None:
            continue

        detected_bugs_path = param_dir / "detected_bugs.csv"
        exe_time_path = param_dir / "exe_time.csv"

        if not detected_bugs_path.exists():
            continue
        if include_exe_time and not exe_time_path.exists():
            continue

        with open(detected_bugs_path, "r", encoding="utf-8") as bug_f:
            bug_reader = csv.reader(bug_f)
            bug_rows = list(bug_reader)

        time_rows: Optional[list] = None
        if include_exe_time:
            with open(exe_time_path, "r", encoding="utf-8") as time_f:
                time_reader = csv.reader(time_f)
                time_rows = list(time_reader)

        for row_idx, bug_row in enumerate(bug_rows):
            label = _bug_row_to_label(bug_row)
            bug_detected = 1 if label == "bug" else 0
            timeout = 1 if label == "timeout" else 0

            rec = {
                "tree": tree_value,
                "cpNum": cpnum,
                "cpNum_range": cpnum_range,
                "cpNum_dir": cpnum_dir,
                "bug_detected": bug_detected,
                "timeout": timeout,
                "result_category": label,
            }
            if include_exe_time and time_rows and row_idx < len(time_rows):
                try:
                    rec["execution_time"] = int(time_rows[row_idx][0])
                except (ValueError, IndexError):
                    rec["execution_time"] = None
            data_records.append(rec)


def collect_data_per_run(
    logs_root: str = "Logs",
    include_exe_time: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    ロジスティック回帰用のデータを収集する。
    各行（各実行）ごとに1レコードを作成し、bug_detected (0/1), timeout (0/1), execution_time などを記録する。

    Args:
        logs_root: Logsディレクトリのパス（tree=* を含む親ディレクトリ）。
        include_exe_time: exe_time.csv を読み込んで execution_time 列を追加するか。
        verbose: 基本統計と分布を表示するか。

    Returns:
        収集したDataFrame。列: tree, cpNum, cpNum_range, cpNum_dir, bug_detected, timeout, (execution_time), result_category
    """
    data_records = []
    logs_path = Path(logs_root)

    for tree_dir in logs_path.glob("tree=*"):
        tree_value = int(tree_dir.name.split("=")[1])
        param_dirs = list(tree_dir.iterdir())
        _collect_from_param_dirs(param_dirs, tree_value=tree_value, include_exe_time=include_exe_time, data_records=data_records)

    df = pd.DataFrame(data_records)
    if verbose:
        print(f"データ収集完了: {len(df)}件のレコード")
        print("\nデータの基本統計:")
        print(df.describe())
        if "result_category" in df.columns:
            print("\n実行結果の分布:")
            print(df["result_category"].value_counts())
        print("\nバグ発見の有無の分布:")
        print(df["bug_detected"].value_counts())
        if "timeout" in df.columns:
            print("\nタイムアウトの有無の分布:")
            print(df["timeout"].value_counts())
    return df


# バグ予測モデル用の特徴量列名（共通化のため定数化）
BUG_PREDICTION_FEATURE_NAMES = ["tree", "cpNum", "cpNum_range", "cpNum_dir"]


def collect_data_aggregated_flat(
    logs_root: str,
    verbose: bool = True,
    tree_value: Optional[int] = None,
) -> pd.DataFrame:
    """
    tree=* なしの flat な Logs から集約データを収集する。
    各サブディレクトリ名は「cpNum1..10, cpNum_range1..10, cpNum_dir1..10」の30個のカンマ区切り。
    末尾に tree を付けた31個の形式も後方互換で受け付ける。
    tree_value を指定した場合は30個のカンマ区切り（tree なし）のディレクトリ名も受け付ける。

    Args:
        logs_root: Logs ディレクトリのパス（直下にパラメータディレクトリがある）。
        verbose: 基本統計を表示するか。
        tree_value: 指定時は dir 名を30値として扱い、レコードの tree にこの値を使う。
            None のときは30値、または末尾に tree を付けた31値を受け付ける。

    Returns:
        DataFrame。列: dir_name, tree, cpNum, cpNum_range, cpNum_dir, bug_detected_any, bug_detected_all, bug_count
    """
    data_records = []
    logs_path = Path(logs_root)

    for param_dir in logs_path.iterdir():
        if not param_dir.is_dir():
            continue

        dir_name = param_dir.name
        params = dir_name.split(",")
        if len(params) < 30:
            continue

        if tree_value is not None:
            current_tree = tree_value
        elif len(params) >= 31:
            try:
                current_tree = int(params[30])
            except (ValueError, IndexError):
                continue
        else:
            current_tree = None

        cpnum, cpnum_range, cpnum_dir = parse_directory_name(",".join(params[:30]))
        if cpnum is None or cpnum_range is None or cpnum_dir is None:
            continue

        detected_bugs_path = param_dir / "detected_bugs.csv"
        if not detected_bugs_path.exists():
            continue

        bug_results: list[str] = []
        with open(detected_bugs_path, "r", encoding="utf-8") as bug_f:
            bug_reader = csv.reader(bug_f)
            for bug_row in bug_reader:
                bug_results.append(_bug_row_to_label(bug_row))

        if len(bug_results) < 5:
            continue

        first_5 = bug_results[:5]
        bug_count = sum(1 for r in first_5 if r == "bug")
        bug_detected_any = 1 if "bug" in first_5 else 0
        bug_detected_all = 1 if all(r == "bug" for r in first_5) else 0

        data_records.append({
            "dir_name": dir_name,
            "tree": current_tree,
            "cpNum": cpnum,
            "cpNum_range": cpnum_range,
            "cpNum_dir": cpnum_dir,
            "bug_detected_any": bug_detected_any,
            "bug_detected_all": bug_detected_all,
            "bug_count": bug_count,
        })

    df = pd.DataFrame(data_records)
    if verbose and len(df) > 0:
        print(f"データ収集完了（flat）: {len(df)}件のレコード")
        print("\n1回でもバグ発見の分布:")
        print(df["bug_detected_any"].value_counts())
    return df


def collect_data_per_run_flat(
    logs_root: str,
    verbose: bool = True,
    tree_value: Optional[int] = None,
) -> pd.DataFrame:
    """
    flat な Logs から1実行1レコードでデータを収集する。
    各 detected_bugs.csv の5行を1行ずつ出力し、「1回の実行でバグ発見したか」のラベルを付与する。
    tree_value 指定時は dir 名を30値（tree なし）として扱う。

    Args:
        logs_root: Logs ディレクトリのパス（直下にパラメータディレクトリがある）。
        verbose: 基本統計を表示するか。
        tree_value: 指定時は dir 名を30値として受け付ける。
            None のときは30値、または末尾に tree を付けた31値を受け付ける。

    Returns:
        DataFrame。列: dir_name, run_index (1～5), bug_detected (0/1)
    """
    data_records = []
    logs_path = Path(logs_root)

    for param_dir in logs_path.iterdir():
        if not param_dir.is_dir():
            continue

        dir_name = param_dir.name
        params = dir_name.split(",")
        if len(params) < 30:
            continue

        if tree_value is None and len(params) >= 31:
            try:
                int(params[30])
            except (ValueError, IndexError):
                continue

        cpnum, cpnum_range, cpnum_dir = parse_directory_name(",".join(params[:30]))
        if cpnum is None or cpnum_range is None or cpnum_dir is None:
            continue

        detected_bugs_path = param_dir / "detected_bugs.csv"
        if not detected_bugs_path.exists():
            continue

        with open(detected_bugs_path, "r", encoding="utf-8") as bug_f:
            bug_reader = csv.reader(bug_f)
            bug_rows = list(bug_reader)

        for row_idx, bug_row in enumerate(bug_rows[:5]):
            label = _bug_row_to_label(bug_row)
            bug_detected = 1 if label == "bug" else 0
            data_records.append({
                "dir_name": dir_name,
                "run_index": row_idx + 1,
                "cpNum": cpnum,
                "cpNum_range": cpnum_range,
                "cpNum_dir": cpnum_dir,
                "bug_detected": bug_detected,
            })

    df = pd.DataFrame(data_records)
    if verbose and len(df) > 0:
        print(f"データ収集完了（per_run flat）: {len(df)}件のレコード")
        print("目的変数（bug_detected）の分布:")
        print(df["bug_detected"].value_counts())
    return df


# speedUpItem のバグ予測用説明変数（tree は含めない）
SPEEDUP_BUG_PREDICTION_FEATURE_NAMES = ["cpNum", "cpNum_range", "cpNum_dir"]


def load_speedup_bug_dataset(
    logs_root: str,
    target: Literal["bug_detected_any", "bug_detected_all", "per_run"] = "bug_detected_any",
    verbose: bool = True,
    tree_value: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    speedUpItem 用: Logs（flat）または単一 tree ディレクトリから (X, y) を構築する。

    dir_name は cpNum1..10, cpNum_range1..10, cpNum_dir1..10（+ 任意で tree）の構成。
    説明変数は tree-different と同様に、各ブロックの最初の非ゼロ値から
    cpNum, cpNum_range, cpNum_dir の3列に集約する（tree は含めない）。

    Args:
        logs_root: Logs ディレクトリのパス（直下にパラメータディレクトリがある flat 構造）、
                   または tree=500 のような単一 tree ディレクトリのパス。
        target: 目的変数。'bug_detected_any'（5回中1回でも）, 'bug_detected_all'（5回全て）, 'per_run'（1回の実行でバグ発見したか）。
        verbose: 件数などの表示を行うか。
        tree_value: 指定時は dir 名を30値（tree なし）として扱う。単一 tree ディレクトリ（例: tree=500）を読むときに指定。

    Returns:
        (X, y): 説明変数（cpNum, cpNum_range, cpNum_dir）の DataFrame と目的変数の Series。
    """
    if target == "per_run":
        logs_df = collect_data_per_run_flat(logs_root=logs_root, verbose=False, tree_value=tree_value)
        y = logs_df["bug_detected"]
    else:
        logs_df = collect_data_aggregated_flat(logs_root=logs_root, verbose=False, tree_value=tree_value)
        y = logs_df[target]

    if len(logs_df) == 0:
        X = pd.DataFrame(columns=SPEEDUP_BUG_PREDICTION_FEATURE_NAMES)
        y = pd.Series(dtype=int)
    else:
        X = logs_df[SPEEDUP_BUG_PREDICTION_FEATURE_NAMES].copy()

    if verbose:
        print(f"目的変数: {target}")
        print(f"データ件数: {len(X)}件")
        print("目的変数の分布:")
        print(y.value_counts())

    return X, y
