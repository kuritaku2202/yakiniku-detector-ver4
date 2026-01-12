"""
焼肉物体検出 - リアルタイム推論スクリプト

カメラからの映像をリアルタイムで解析し、焼肉の状態を検出します。
- 緑のバウンディングボックス: 焼けている肉
- 赤のバウンディングボックス: 焼けていない肉
"""

import cv2
import numpy as np
from ultralytics import YOLO
import config
import os
from PIL import Image, ImageDraw, ImageFont
from PIL import Image, ImageDraw, ImageFont


def put_japanese_text(img, text, position, font_size=24, color=(255, 255, 255)):
    """
    OpenCV画像に日本語テキストを描画
    
    Args:
        img: OpenCV画像（BGR）
        text: 描画するテキスト
        position: (x, y) 位置
        font_size: フォントサイズ
        color: RGB色
    """
    import platform
    
    # OpenCV画像をPIL画像に変換
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # OS別のフォントパスを試行
    font = None
    font_paths = []
    
    system = platform.system()
    if system == "Windows":
        # Windows用フォント
        font_paths = [
            "C:/Windows/Fonts/meiryo.ttc",      # メイリオ
            "C:/Windows/Fonts/msgothic.ttc",    # MSゴシック
            "C:/Windows/Fonts/YuGothM.ttc",     # 游ゴシック
            "C:/Windows/Fonts/msmincho.ttc",    # MS明朝
        ]
    elif system == "Darwin":
        # macOS用フォント
        font_paths = [
            "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
    else:
        # Linux用フォント
        font_paths = [
            "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]
    
    # フォントを順番に試行
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except:
            continue
    
    # フォントが見つからない場合はデフォルトを使用
    if font is None:
        font = ImageFont.load_default()
    
    # テキストを描画（RGBカラー）
    draw.text(position, text, font=font, fill=color)
    
    # PIL画像をOpenCV画像に戻す
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_cv


class YakinikuDetector:
    """焼肉検出器クラス"""
    
    def __init__(self, model_path=None, conf_threshold=None, iou_threshold=None):
        """
        初期化
        
        Args:
            model_path: モデルファイルのパス（Noneの場合はconfig.BEST_MODELを使用）
            conf_threshold: 信頼度閾値
            iou_threshold: IOU閾値
        """
        self.model_path = model_path or config.BEST_MODEL
        self.conf_threshold = conf_threshold or config.CONF_THRESHOLD
        self.iou_threshold = iou_threshold or config.IOU_THRESHOLD
        
        # モデルの読み込み
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                "Please train the model first by running: python judge.py"
            )
        
        print(f"Loading model from: {self.model_path}")
        self.model = YOLO(self.model_path)
        print("Model loaded successfully!")
        
        # クラス名と色の設定（Kongouデータセット用）
        self.class_names = config.CLASS_NAMES
        self.colors = {
            0: config.COLOR_COOKED,  # クラス0 = 焼けた肉 = 緑
            1: config.COLOR_ROW,     # クラス1 = 生肉 = 赤
        }
        
        # 統計情報
        self.stats = {
            "total_detections": 0,
            "row_count": 0,
            "cooked_count": 0,
        }
    
    def detect_frame(self, frame):
        """
        1フレームの画像から焼肉を検出
        
        Args:
            frame: 入力画像（numpy array）
        
        Returns:
            annotated_frame: 検出結果を描画した画像
            results: 検出結果のリスト
        """
        # YOLOで推論（agnostic_nms=Trueで異なるクラス間でもNMSを適用）
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            agnostic_nms=True,  # クラス間でもNMSを適用（二重検出を防ぐ）
            verbose=False
        )[0]
        
        # 結果を描画
        annotated_frame = frame.copy()
        detections = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.cpu().numpy()
            
            for box in boxes:
                # バウンディングボックスの座標
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # クラス名
                class_name = self.class_names.get(cls, "Unknown")
                
                # クラスに応じた描画スタイル
                if cls == 1:  # raw meat (生肉) - 角丸の四角形
                    # 白い角丸矩形
                    color = (255, 255, 255)  # 白
                    thickness = 4
                    
                    # 角丸矩形を描画
                    radius = 20
                    # 上辺
                    cv2.line(annotated_frame, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
                    # 右辺
                    cv2.line(annotated_frame, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
                    # 下辺
                    cv2.line(annotated_frame, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
                    # 左辺
                    cv2.line(annotated_frame, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
                    
                    # 四隅の角丸
                    cv2.ellipse(annotated_frame, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
                    cv2.ellipse(annotated_frame, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
                    cv2.ellipse(annotated_frame, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
                    cv2.ellipse(annotated_frame, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
                    
                else:  # cooked meat (焼けた肉) - 黄金の二重円
                    # 中心座標を計算
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    radius = max((x2 - x1), (y2 - y1)) // 2 + 15
                    
                    # 外側の黄金色の円
                    golden_color = (0, 215, 255)  # BGR: 黄金色
                    cv2.circle(annotated_frame, (center_x, center_y), radius, golden_color, 6)
                    
                    # 内側の白い円
                    white_color = (255, 255, 255)
                    cv2.circle(annotated_frame, (center_x, center_y), radius - 10, white_color, 3)
                
                # 検出結果を保存
                detections.append({
                    "class": class_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })
                
                # 統計更新
                self.stats["total_detections"] += 1
                if cls == 0:
                    self.stats["row_count"] += 1
                elif cls == 1:
                    self.stats["cooked_count"] += 1
        
        return annotated_frame, detections
    
    def run_camera(self, camera_id=None, save_video=False, output_path="output.mp4"):
        """
        カメラからリアルタイムで検出を実行
        
        Args:
            camera_id: カメラID（Noneの場合はconfig.CAMERA_IDを使用）
            save_video: 動画を保存するかどうか
            output_path: 保存する動画のパス
        """
        import platform
        camera_id = camera_id if camera_id is not None else config.CAMERA_ID
        
        # カメラをオープン（Windows対応）
        cap = None
        system = platform.system()
        
        if system == "Windows":
            # Windows: DirectShowバックエンドを試行
            print("Windows detected, trying DirectShow backend...")
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print("DirectShow failed, trying default backend...")
                cap = cv2.VideoCapture(camera_id)
        else:
            # macOS/Linux: デフォルトバックエンド
            cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            # 他のカメラIDを試行
            print(f"Failed to open camera {camera_id}, trying other cameras...")
            for try_id in range(3):
                if try_id != camera_id:
                    cap = cv2.VideoCapture(try_id, cv2.CAP_DSHOW) if system == "Windows" else cv2.VideoCapture(try_id)
                    if cap.isOpened():
                        print(f"Found camera at ID {try_id}")
                        camera_id = try_id
                        break
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera. Please check if a camera is connected.")
        
        # カメラを最高解像度に設定（4Kを要求、カメラがサポートする最大値が使用される）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)   # 4K width
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)  # 4K height
        cap.set(cv2.CAP_PROP_FPS, 30)             # 30fps
        
        # 実際に設定された解像度を取得
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        print(f"Camera opened: {width}x{height} @ {fps}fps")
        
        # カメラのウォームアップ（最初の数フレームを読み捨て）
        print("Warming up camera...")
        for _ in range(10):
            cap.read()
        print("Camera ready!")
        
        print("Press 'q' to quit, 's' to save screenshot")
        
        # フルスクリーンウィンドウの設定
        window_name = "Yakiniku Detector"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # 動画ライターの設定
        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Saving video to: {output_path}")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                # 検出を実行
                annotated_frame, detections = self.detect_frame(frame)
                
                # 凡例を画面左上に描画
                legend_y = 30
                
                # 背景（半透明の黒）
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (10, 10), (220, 120), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
                
                # 焼けた肉の凡例（黄金の円）
                legend_y += 5
                cv2.circle(annotated_frame, (35, legend_y - 5), 12, (0, 215, 255), 3)
                cv2.circle(annotated_frame, (35, legend_y - 5), 7, (255, 255, 255), 2)
                # 日本語テキスト
                annotated_frame = put_japanese_text(annotated_frame, "焼けました！", (60, legend_y - 18), 
                                                    font_size=20, color=(255, 255, 255))
                
                # 生肉の凡例（白い角丸四角）
                legend_y += 40
                # 簡略化した角丸四角
                rect_x, rect_y = 25, legend_y - 15
                rect_w, rect_h = 20, 20
                cv2.line(annotated_frame, (rect_x + 5, rect_y), (rect_x + rect_w - 5, rect_y), 
                        (255, 255, 255), 2)
                cv2.line(annotated_frame, (rect_x + rect_w, rect_y + 5), (rect_x + rect_w, rect_y + rect_h - 5), 
                        (255, 255, 255), 2)
                cv2.line(annotated_frame, (rect_x + 5, rect_y + rect_h), (rect_x + rect_w - 5, rect_y + rect_h), 
                        (255, 255, 255), 2)
                cv2.line(annotated_frame, (rect_x, rect_y + 5), (rect_x, rect_y + rect_h - 5), 
                        (255, 255, 255), 2)
                # 日本語テキスト
                annotated_frame = put_japanese_text(annotated_frame, "まだ、、", (60, legend_y - 18), 
                                                    font_size=20, color=(255, 255, 255))
                
                # 画面に表示
                cv2.imshow(window_name, annotated_frame)
                
                # 動画に書き込み
                if writer:
                    writer.write(annotated_frame)
                
                frame_count += 1
                
                # キー入力の処理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # スクリーンショットを保存
                    screenshot_path = f"screenshot_{frame_count}.jpg"
                    cv2.imwrite(screenshot_path, annotated_frame)
                    print(f"Screenshot saved: {screenshot_path}")
        
        finally:
            # リソースを解放
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            # 統計を表示
            print("\n=== Detection Statistics ===")
            print(f"Total frames processed: {frame_count}")
            print(f"Total detections: {self.stats['total_detections']}")
            print(f"Raw meat detected: {self.stats['row_count']}")
            print(f"Cooked meat detected: {self.stats['cooked_count']}")
    
    
    def run_video(self, video_path, save_video=True, output_path=None):
        """
        動画ファイルから焼肉を検出
        
        Args:
            video_path: 入力動画ファイルのパス
            save_video: 結果動画を保存するかどうか
            output_path: 出力動画のパス（Noneの場合は自動生成）
        """
        # 動画ファイルを開く
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")
        
        # 動画の情報を取得
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video opened: {width}x{height} @ {fps}fps, {total_frames} frames")
        print(f"Duration: {total_frames/fps:.1f} seconds")
        
        # 出力パスの設定
        if output_path is None:
            import os
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = f"{base_name}_detected.mp4"
        
        # 動画ライターの設定
        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Saving result to: {output_path}")
        
        frame_count = 0
        
        try:
            print("\nProcessing video...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 検出を実行
                annotated_frame, detections = self.detect_frame(frame)
                
                # 凡例を画面左上に描画
                legend_y = 30
                
                # 背景（半透明の黒）
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (10, 10), (220, 120), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
                
                # 焼けた肉の凡例（黄金の円）
                legend_y += 5
                cv2.circle(annotated_frame, (35, legend_y - 5), 12, (0, 215, 255), 3)
                cv2.circle(annotated_frame, (35, legend_y - 5), 7, (255, 255, 255), 2)
                # 日本語テキスト
                annotated_frame = put_japanese_text(annotated_frame, "焼けました！", (60, legend_y - 18), 
                                                    font_size=20, color=(255, 255, 255))
                
                # 生肉の凡例（白い角丸四角）
                legend_y += 40
                # 簡略化した角丸四角
                rect_x, rect_y = 25, legend_y - 15
                rect_w, rect_h = 20, 20
                cv2.line(annotated_frame, (rect_x + 5, rect_y), (rect_x + rect_w - 5, rect_y), 
                        (255, 255, 255), 2)
                cv2.line(annotated_frame, (rect_x + rect_w, rect_y + 5), (rect_x + rect_w, rect_y + rect_h - 5), 
                        (255, 255, 255), 2)
                cv2.line(annotated_frame, (rect_x + 5, rect_y + rect_h), (rect_x + rect_w - 5, rect_y + rect_h), 
                        (255, 255, 255), 2)
                cv2.line(annotated_frame, (rect_x, rect_y + 5), (rect_x, rect_y + rect_h - 5), 
                        (255, 255, 255), 2)
                # 日本語テキスト
                annotated_frame = put_japanese_text(annotated_frame, "まだ、、", (60, legend_y - 18), 
                                                    font_size=20, color=(255, 255, 255))
                
                # 動画に書き込み
                if writer:
                    writer.write(annotated_frame)
                
                frame_count += 1
                
                # 進捗表示（10フレームごと）
                if frame_count % 10 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"\rProgress: {frame_count}/{total_frames} frames ({progress:.1f}%)", end="")
        
        finally:
            print("\n")  # 改行
            # リソースを解放
            cap.release()
            if writer:
                writer.release()
            
            # 統計を表示
            print("\n=== Detection Statistics ===")
            print(f"Total frames processed: {frame_count}")
            print(f"Total detections: {self.stats['total_detections']}")
            print(f"Raw meat detected: {self.stats['row_count']}")
            print(f"Cooked meat detected: {self.stats['cooked_count']}")
            if save_video:
                print(f"\nResult video saved to: {output_path}")
        
        return output_path
    
    def detect_image(self, image_path, save_path=None):
        """
        画像ファイルから焼肉を検出
        
        Args:
            image_path: 入力画像のパス
            save_path: 結果を保存するパス（Noneの場合は表示のみ）
        """
        # 画像を読み込み
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # 検出を実行
        annotated_frame, detections = self.detect_frame(frame)
        
        # 結果を表示
        print(f"\nDetected {len(detections)} objects:")
        for i, det in enumerate(detections):
            print(f"  {i+1}. {det['class']}: {det['confidence']:.2f}")
        
        # 保存または表示
        if save_path:
            cv2.imwrite(save_path, annotated_frame)
            print(f"Result saved to: {save_path}")
        else:
            cv2.imshow("Detection Result", annotated_frame)
            print("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return annotated_frame, detections


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="焼肉物体検出 - リアルタイム推論")
    parser.add_argument("--mode", type=str, default="camera", 
                        choices=["camera", "image", "video"],
                        help="実行モード: camera（カメラ）、image（画像ファイル）、またはvideo（動画ファイル）")
    parser.add_argument("--image", type=str, default=None,
                        help="画像ファイルのパス（--mode image の場合）")
    parser.add_argument("--video", type=str, default=None,
                        help="動画ファイルのパス（--mode video の場合）")
    parser.add_argument("--output", type=str, default=None,
                        help="出力ファイルのパス")
    parser.add_argument("--model", type=str, default=None,
                        help="モデルファイルのパス（デフォルト: config.BEST_MODEL）")
    parser.add_argument("--camera-id", type=int, default=None,
                        help="カメラID（デフォルト: config.CAMERA_ID）")
    parser.add_argument("--conf", type=float, default=None,
                        help="信頼度閾値（デフォルト: config.CONF_THRESHOLD）")
    parser.add_argument("--save-video", action="store_true",
                        help="カメラモードで動画を保存")
    
    args = parser.parse_args()
    
    # 検出器の初期化
    detector = YakinikuDetector(
        model_path=args.model,
        conf_threshold=args.conf
    )
    
    if args.mode == "camera":
        # カメラモード
        output_path = args.output or "output.mp4"
        detector.run_camera(
            camera_id=args.camera_id,
            save_video=args.save_video,
            output_path=output_path
        )
    
    elif args.mode == "image":
        # 画像モード
        if not args.image:
            print("Error: --image argument is required for image mode")
            exit(1)
        
        detector.detect_image(args.image, args.output)
    
    elif args.mode == "video":
        # 動画モード
        if not args.video:
            print("Error: --video argument is required for video mode")
            exit(1)
        
        detector.run_video(args.video, save_video=True, output_path=args.output)
