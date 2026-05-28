import os
from pathlib import Path

# プロジェクトルートからの相対パスとして指定
LOGS_ROOT = Path("speedUpItem/tree=500")

# 削除対象のファイル名
TARGET_BASENAMES = {"positions.csv", "inputLog.csv"}


def find_targets(logs_root: Path):
    """削除対象ファイルと、そのメタファイル候補を列挙する。"""
    if not logs_root.exists():
        raise FileNotFoundError(f"Logs ディレクトリが見つかりません: {logs_root}")

    targets = []

    for dirpath, dirnames, filenames in os.walk(logs_root):
        dirpath = Path(dirpath)

        for name in filenames:
            if name in TARGET_BASENAMES:
                file_path = dirpath / name
                targets.append(file_path)

                # メタファイル（同名 + .meta）を想定
                meta_path = file_path.with_name(file_path.name + ".meta")
                if meta_path.exists():
                    targets.append(meta_path)

    # 重複除去
    targets = sorted(set(targets))
    return targets


def delete_files(files, dry_run: bool = False):
    """ファイルを削除する。dry_run=True のときは削除せずに表示だけ。"""
    if not files:
        print("削除対象のファイルは見つかりませんでした。")
        return

    print(f"対象ファイル数: {len(files)}")
    for f in files:
        print(f" - {f}")

    if dry_run:
        print("\n※ dry-run モードなので、まだファイルは削除していません。")
        print("本当に削除する場合は、スクリプト内で dry_run=False に変更してください。")
        return

    print("\n削除を実行します...")
    for f in files:
        try:
            f.unlink()
            print(f"削除: {f}")
        except FileNotFoundError:
            print(f"既に存在しないためスキップ: {f}")
        except Exception as e:
            print(f"削除失敗: {f} -> {e}")


if __name__ == "__main__":
    # 1. 削除対象を列挙
    targets = find_targets(LOGS_ROOT)

    # 2. 実際の削除処理（最初は dry_run=True で確認）
    delete_files(targets, dry_run=False)