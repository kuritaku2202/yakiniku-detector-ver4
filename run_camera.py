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


def detect_cameras(max_cameras=20):
    """
    接続されているカメラを検出
    複数のバックエンドを試行して、より多くのカメラを検出
    
    Returns:
        list: 利用可能なカメラIDのリスト
    """
    available_cameras = []
    system = platform.system()
    
    print("カメラを検索中... (しばらくお待ちください)")
    
    # Windowsの場合、複数のバックエンドを試す
    backends = []
    if system == "Windows":
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_ANY, "Auto"),
        ]
    else:
        backends = [(cv2.CAP_ANY, "Default")]
    
    found_ids = set()
    
    for backend, backend_name in backends:
        for camera_id in range(max_cameras):
            if camera_id in found_ids:
                continue
            try:
                cap = cv2.VideoCapture(camera_id, backend)
                
                if cap.isOpened():
                    # フレームを読み取って本当に動作するか確認
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # カメラ情報を取得
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                        available_cameras.append({
                            'id': camera_id,
                            'backend': backend,
                            'backend_name': backend_name,
                            'resolution': f"{width}x{height}"
                        })
                        found_ids.add(camera_id)
                        print(f"  発見: カメラID {camera_id} ({backend_name}) - {width}x{height}")
                    cap.release()
            except Exception as e:
                pass
    
    return available_cameras


def select_camera():
    """
    ユーザーにカメラを選択させる
    
    Returns:
        tuple: (カメラID, バックエンド)
    """
    cameras = detect_cameras()
    
    print()
    
    if len(cameras) == 0:
        print("自動検出でカメラが見つかりませんでした。")
        print()
        print("手動でカメラIDを入力しますか？")
        print("  [1] はい（手動入力）")
        print("  [2] いいえ（終了）")
        
        choice = input("選択 (1-2): ").strip()
        if choice == "1":
            return manual_camera_input()
        else:
            print("終了します。")
            input("Enterキーを押して終了...")
            sys.exit(0)
    
    if len(cameras) == 1:
        cam = cameras[0]
        print(f"カメラが1台検出されました: カメラ {cam['id']} ({cam['resolution']})")
        print()
        print("このカメラを使用しますか？")
        print("  [1] はい")
        print("  [2] いいえ（手動でカメラIDを入力）")
        
        choice = input("選択 (1-2): ").strip()
        if choice == "2":
            return manual_camera_input()
        return (cam['id'], cam['backend'])
    
    # 複数カメラがある場合は選択させる
    print("=" * 50)
    print("利用可能なカメラ:")
    print("=" * 50)
    
    for i, cam in enumerate(cameras):
        print(f"  [{i + 1}] カメラ {cam['id']} ({cam['backend_name']}) - 解像度: {cam['resolution']}")
    
    print(f"  [{len(cameras) + 1}] 手動でカメラIDを入力")
    print()
    
    while True:
        try:
            choice = input(f"使用するカメラを選択してください (1-{len(cameras) + 1}): ").strip()
            choice_num = int(choice)
            
            if choice_num == len(cameras) + 1:
                return manual_camera_input()
            
            if 1 <= choice_num <= len(cameras):
                selected = cameras[choice_num - 1]
                print(f"\nカメラ {selected['id']} を選択しました")
                return (selected['id'], selected['backend'])
            else:
                print(f"1から{len(cameras) + 1}の間で入力してください")
        except ValueError:
            print("数字を入力してください")


def manual_camera_input():
    """
    手動でカメラIDを入力させる
    
    Returns:
        tuple: (カメラID, バックエンド)
    """
    print()
    print("=" * 50)
    print("手動カメラID入力")
    print("=" * 50)
    print("ヒント: 通常は 0, 1, 2 などの小さい数字です")
    print("       スマホカメラアプリの場合、設定画面でIDを確認できることがあります")
    print()
    
    while True:
        try:
            camera_id = int(input("カメラIDを入力してください: ").strip())
            
            # バックエンド選択（Windowsのみ）
            backend = cv2.CAP_ANY
            if platform.system() == "Windows":
                print()
                print("バックエンドを選択してください:")
                print("  [1] DirectShow（通常のWebカメラ）")
                print("  [2] Media Foundation（新しいカメラ）")
                print("  [3] 自動")
                
                backend_choice = input("選択 (1-3) [デフォルト: 3]: ").strip()
                if backend_choice == "1":
                    backend = cv2.CAP_DSHOW
                elif backend_choice == "2":
                    backend = cv2.CAP_MSMF
                else:
                    backend = cv2.CAP_ANY
            
            # テスト接続
            print(f"\nカメラ {camera_id} に接続テスト中...")
            cap = cv2.VideoCapture(camera_id, backend)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"接続成功！ ({width}x{height})")
                    cap.release()
                    return (camera_id, backend)
                else:
                    print("カメラは開けましたが、映像を取得できませんでした。")
                    cap.release()
            else:
                print(f"カメラ {camera_id} を開けませんでした。")
            
            print("別のIDを試しますか？")
            retry = input("続行 (y/n): ").strip().lower()
            if retry != 'y':
                print("終了します。")
                input("Enterキーを押して終了...")
                sys.exit(0)
                
        except ValueError:
            print("数字を入力してください")


def main():
    print("=" * 50)
    print("焼肉検出カメラアプリを起動します")
    print("=" * 50)
    print()
    
    # カメラ選択
    camera_id, backend = select_camera()
    
    print()
    print("操作方法:")
    print("  'q' キー: 終了")
    print("  's' キー: スクリーンショット保存")
    print()
    
    try:
        from detect_yakiniku import YakinikuDetector
        detector = YakinikuDetector()
        
        # バックエンドを指定してカメラを開く
        detector.run_camera(camera_id=camera_id, backend=backend)
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        input("Enterキーを押して終了...")
        sys.exit(1)


if __name__ == "__main__":
    main()
