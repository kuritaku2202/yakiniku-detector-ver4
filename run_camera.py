#!/usr/bin/env python3
"""
焼肉検出カメラアプリ - エントリーポイント

このスクリプトはカメラモードを直接起動します。
.app化用のシンプルなエントリーポイントです。
"""

import sys
import os

# バンドルされたアプリの場合、リソースパスを設定
if getattr(sys, 'frozen', False):
    # PyInstallerでバンドルされた場合
    base_path = sys._MEIPASS
    # モデルファイルのパスを更新
    os.chdir(base_path)
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

# configを動的に更新
import config
config.PROJECT_DIR = base_path
config.MODEL_DIR = os.path.join(base_path, "models")
config.BEST_MODEL = os.path.join(config.MODEL_DIR, "best.pt")

from detect_yakiniku import YakinikuDetector

def main():
    print("=" * 50)
    print("焼肉検出カメラアプリを起動します")
    print("=" * 50)
    print()
    print("操作方法:")
    print("  'q' キー: 終了")
    print("  's' キー: スクリーンショット保存")
    print()
    
    try:
        detector = YakinikuDetector()
        detector.run_camera()
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        input("Enterキーを押して終了...")
        sys.exit(1)

if __name__ == "__main__":
    main()
