#!/usr/bin/env python3
"""
焼肉検出カメラアプリ - エントリーポイント

このスクリプトはカメラモードを直接起動します。
.app化用のシンプルなエントリーポイントです。
"""

import sys
import os
import platform

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

import cv2


def detect_cameras(max_cameras=10):
    """
    接続されているカメラを検出
    
    Returns:
        list: 利用可能なカメラIDのリスト
    """
    available_cameras = []
    system = platform.system()
    
    print("カメラを検索中...")
    
    for camera_id in range(max_cameras):
        try:
            if system == "Windows":
                cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(camera_id)
            
            if cap.isOpened():
                # カメラ情報を取得
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available_cameras.append({
                    'id': camera_id,
                    'resolution': f"{width}x{height}"
                })
                cap.release()
        except:
            pass
    
    return available_cameras


def select_camera():
    """
    ユーザーにカメラを選択させる
    
    Returns:
        int: 選択されたカメラID
    """
    cameras = detect_cameras()
    
    if len(cameras) == 0:
        print("エラー: カメラが見つかりませんでした")
        input("Enterキーを押して終了...")
        sys.exit(1)
    
    if len(cameras) == 1:
        print(f"カメラが1台検出されました: カメラ {cameras[0]['id']} ({cameras[0]['resolution']})")
        return cameras[0]['id']
    
    # 複数カメラがある場合は選択させる
    print("\n" + "=" * 50)
    print("利用可能なカメラ:")
    print("=" * 50)
    
    for i, cam in enumerate(cameras):
        print(f"  [{i + 1}] カメラ {cam['id']} - 解像度: {cam['resolution']}")
    
    print()
    
    while True:
        try:
            choice = input(f"使用するカメラを選択してください (1-{len(cameras)}): ").strip()
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(cameras):
                selected = cameras[choice_num - 1]
                print(f"\nカメラ {selected['id']} を選択しました")
                return selected['id']
            else:
                print(f"1から{len(cameras)}の間で入力してください")
        except ValueError:
            print("数字を入力してください")


def main():
    print("=" * 50)
    print("焼肉検出カメラアプリを起動します")
    print("=" * 50)
    print()
    
    # カメラ選択
    camera_id = select_camera()
    
    print()
    print("操作方法:")
    print("  'q' キー: 終了")
    print("  's' キー: スクリーンショット保存")
    print()
    
    try:
        from detect_yakiniku import YakinikuDetector
        detector = YakinikuDetector()
        detector.run_camera(camera_id=camera_id)
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        input("Enterキーを押して終了...")
        sys.exit(1)


if __name__ == "__main__":
    main()
