"""
焼肉物体検出システムの設定ファイル
"""
import os

# プロジェクトディレクトリ
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# データセットのパス
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# モデル保存先
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
BEST_MODEL = os.path.join(MODEL_DIR, "best.pt")

# YOLOv8モデル設定
YOLO_MODEL = "yolov8n.pt"  # nano model (軽量版), yolov8s.pt, yolov8m.pt などもあり

# トレーニング設定
IMG_SIZE = 640  # YOLOの標準サイズ
BATCH_SIZE = 16
EPOCHS = 100
PATIENCE = 20  # Early stopping（20エポック改善なしで停止）

# クラス名（Kongouデータセットの順序）
CLASS_NAMES = {
    0: "cooked",  # 焼けた肉（Kongouでは0）
    1: "row"      # 生肉（Kongouでは1）
}

# 検出時の信頼度閾値（v2モデル用に最適化）
CONF_THRESHOLD = 0.25  # v2モデルは高精度なので0.25で十分
IOU_THRESHOLD = 0.5  # 重複検出を防ぐため少し高めに設定

# カメラ設定
CAMERA_ID = 0  # デフォルトカメラ

# 可視化の色設定（BGR）
COLOR_ROW = (0, 0, 255)      # 赤 - 生肉
COLOR_COOKED = (0, 255, 0)   # 緑 - 焼けた肉
