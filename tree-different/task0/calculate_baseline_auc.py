"""
task0のベースラインAUCを計算するスクリプト

ベースライン: すべてのテストサンプルに対して「バグ発見あり（1）」を予測
y_pred_always_bug = np.ones(len(y_test)) に対応
"""

import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split


def parse_directory_name(dir_name):
    """
    ディレクトリ名をパースしてcpNum, cpNum_range, cpNum_dirを抽出
    
    Args:
        dir_name: ディレクトリ名（例: "0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1"）
    
    Returns:
        tuple: (cpNum, cpNum_range, cpNum_dir) または (None, None, None)
    """
    params = dir_name.split(',')
    if len(params) < 30:
        return None, None, None
    
    try:
        cpnum = None
        cpnum_range = None
        cpnum_dir = None
        
        # cpNum (0-9番目)
        for i in range(10):
            if int(params[i]) != 0:
                cpnum = int(params[i])
                break
        
        # cpNum_range (10-19番目)
        for i in range(10, 20):
            if int(params[i]) != 0:
                cpnum_range = int(params[i])
                break
        
        # cpNum_dir (20-29番目)
        for i in range(20, 30):
            if int(params[i]) != 0:
                cpnum_dir = int(params[i])
                break
        
        return cpnum, cpnum_range, cpnum_dir
    except (ValueError, IndexError):
        return None, None, None


def collect_data(logs_root='../Logs'):
    """
    データを収集
    
    Args:
        logs_root: Logsディレクトリのパス
    
    Returns:
        pd.DataFrame: 収集したデータ
    """
    data_records = []
    logs_path = Path(logs_root)
    
    # tree=0, tree=500, tree=1000のディレクトリを処理
    for tree_dir in logs_path.glob('tree=*'):
        tree_value = int(tree_dir.name.split('=')[1])
        
        # 各パラメータディレクトリを処理
        for param_dir in tree_dir.iterdir():
            if not param_dir.is_dir():
                continue
            
            # ディレクトリ名をパース
            cpnum, cpnum_range, cpnum_dir = parse_directory_name(param_dir.name)
            if cpnum is None or cpnum_range is None or cpnum_dir is None:
                continue
            
            # ファイルパス
            detected_bugs_path = param_dir / 'detected_bugs.csv'
            
            if not os.path.exists(detected_bugs_path):
                continue
            
            # detected_bugs.csvの各行を読み込む（各実行ごとにデータポイントを作成）
            with open(detected_bugs_path, 'r') as bug_f:
                bug_reader = csv.reader(bug_f)
                
                # 各行を処理
                for bug_row in bug_reader:
                    # 実行結果を判定
                    bug_detected = 0  # バグ発見の有無（0: なし, 1: あり）
                    
                    if bug_row == ['timeout']:
                        bug_detected = 0
                    elif bug_row == ['null']:
                        bug_detected = 0
                    else:
                        bug_detected = 1
                    
                    # データポイントを作成
                    data_records.append({
                        'tree': tree_value,
                        'cpNum': cpnum,
                        'cpNum_range': cpnum_range,
                        'cpNum_dir': cpnum_dir,
                        'bug_detected': bug_detected,
                    })
    
    df = pd.DataFrame(data_records)
    print(f"データ収集完了: {len(df)}件のレコード")
    print(f"\nデータの基本統計:")
    print(df.describe())
    print(f"\nバグ発見の有無の分布:")
    print(df['bug_detected'].value_counts())
    return df


def calculate_baseline_auc(y_test):
    """
    ベースラインのAUCを計算
    
    ベースライン: すべてのテストサンプルに対して「バグ発見あり（1）」を予測
    y_pred_always_bug = np.ones(len(y_test)) に対応
    
    Args:
        y_test: テストデータの目的変数
    
    Returns:
        float: ベースラインのAUCスコア
    """
    # すべてのテストサンプルに対して「バグ発見あり（1）」を予測
    # AUC計算のため、確率予測としてすべて1.0を予測
    y_pred_proba_always_bug = np.ones(len(y_test))
    
    # AUCを計算
    auc_score = roc_auc_score(y_test, y_pred_proba_always_bug)
    
    return auc_score


def main():
    """メイン処理"""
    print("=" * 60)
    print("task0 ベースラインAUC計算スクリプト")
    print("=" * 60)
    
    # データ収集
    print("\n【ステップ1】データ収集")
    df = collect_data(logs_root="../Logs/")
    
    # 特徴量と目的変数の準備
    print("\n【ステップ2】データ分割")
    X = df[['tree', 'cpNum', 'cpNum_range', 'cpNum_dir']]
    y = df['bug_detected']
    
    # 訓練データとテストデータに分割（既存のnotebookと同じ設定）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"訓練データ数: {len(X_train)}")
    print(f"テストデータ数: {len(X_test)}")
    print(f"\n訓練データの目的変数分布:")
    print(y_train.value_counts())
    print(f"\nテストデータの目的変数分布:")
    print(y_test.value_counts())
    
    # ベースラインAUCの計算
    print("\n【ステップ3】ベースラインAUCの計算")
    print("ベースライン: すべてのテストサンプルに対して「バグ発見あり（1）」を予測")
    baseline_auc = calculate_baseline_auc(y_test)
    
    # 結果の表示
    print("\n" + "=" * 60)
    print("ベースラインAUC結果")
    print("=" * 60)
    print(f"\nベースライン（すべてバグ発見ありを予測）")
    print(f"  AUC: {baseline_auc:.4f}")
    
    print("\n" + "=" * 60)
    print("処理完了")
    print("=" * 60)


if __name__ == '__main__':
    main()
