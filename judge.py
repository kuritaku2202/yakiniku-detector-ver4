"""
焼肉物体検出 - YOLOv8モデルのトレーニングスクリプト

このスクリプトは、YOLOv8を使用して焼肉の物体検出モデルを学習します。
- 焼けている肉 (cooked)
- 焼けていない肉 (uncooked)
の2クラスを検出し、それぞれの位置をバウンディングボックスで特定します。
"""

from ultralytics import YOLO
import os
import config

def train_yakiniku_detector():
    """
    YOLOv8モデルを使用して焼肉検出器を学習
    """
    # モデルディレクトリの作成
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # ベースモデルのロード（事前学習済みのCOCOモデル）
    print(f"Loading base model: {config.YOLO_MODEL}")
    model = YOLO(config.YOLO_MODEL)
    
    # モデル情報の表示
    print("\nModel Information:")
    print(model.info())
    
    # トレーニングの実行
    print("\nStarting training...")
    print(f"Dataset config: yakiniku.yaml")
    print(f"Image size: {config.IMG_SIZE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.EPOCHS}")
    
    results = model.train(
        data="yakiniku.yaml",           # データセット設定ファイル
        epochs=config.EPOCHS,            # エポック数
        imgsz=config.IMG_SIZE,           # 画像サイズ
        batch=config.BATCH_SIZE,         # バッチサイズ
        patience=config.PATIENCE,        # Early stopping patience
        save=True,                       # モデルを保存
        project=config.MODEL_DIR,        # 保存先ディレクトリ
        name="yakiniku_detector",        # 実験名
        exist_ok=True,                   # 既存フォルダの上書きを許可
        
        # データ拡張
        hsv_h=0.015,                     # 色相の変動
        hsv_s=0.7,                       # 彩度の変動
        hsv_v=0.4,                       # 明度の変動
        degrees=10.0,                    # 回転角度
        translate=0.1,                   # 平行移動
        scale=0.5,                       # スケール変換
        flipud=0.0,                      # 上下反転（焼肉では不要）
        fliplr=0.5,                      # 左右反転
        mosaic=1.0,                      # モザイク拡張
        mixup=0.1,                       # MixUp拡張
        
        # オプティマイザ設定
        optimizer="Adam",
        lr0=0.01,                        # 初期学習率
        lrf=0.01,                        # 最終学習率（lr0の何倍か）
        momentum=0.937,
        weight_decay=0.0005,
        
        # その他
        verbose=True,                    # 詳細出力
        seed=42,                         # 再現性のためのシード固定
        device=None,                     # 自動でGPU/CPUを選択
    )
    
    print("\nTraining completed!")
    print(f"Best model saved to: {results.save_dir}")
    
    # 最良モデルをベストモデルとしてコピー
    best_pt = os.path.join(results.save_dir, "weights", "best.pt")
    if os.path.exists(best_pt):
        import shutil
        shutil.copy(best_pt, config.BEST_MODEL)
        print(f"Best model copied to: {config.BEST_MODEL}")
    
    return model, results


def validate_model():
    """
    学習済みモデルをバリデーションセットで評価
    """
    if not os.path.exists(config.BEST_MODEL):
        print(f"Error: Model not found at {config.BEST_MODEL}")
        print("Please train the model first.")
        return
    
    print(f"\nValidating model: {config.BEST_MODEL}")
    model = YOLO(config.BEST_MODEL)
    
    # バリデーション実行
    metrics = model.val(
        data="yakiniku.yaml",
        imgsz=config.IMG_SIZE,
        batch=config.BATCH_SIZE,
        conf=config.CONF_THRESHOLD,
        iou=config.IOU_THRESHOLD,
        device=None,
    )
    
    # メトリクスの表示
    print("\n=== Validation Results ===")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    
    # クラスごとのメトリクス
    if hasattr(metrics.box, 'maps'):
        print("\nPer-class mAP50:")
        for i, class_name in config.CLASS_NAMES.items():
            if i < len(metrics.box.maps):
                print(f"  {class_name}: {metrics.box.maps[i]:.4f}")
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="焼肉物体検出モデルのトレーニング")
    parser.add_argument("--mode", type=str, default="train", 
                        choices=["train", "val", "both"],
                        help="実行モード: train（学習のみ）, val（検証のみ）, both（学習+検証）")
    
    args = parser.parse_args()
    
    if args.mode in ["train", "both"]:
        print("=" * 60)
        print("焼肉物体検出モデルのトレーニングを開始します")
        print("=" * 60)
        model, results = train_yakiniku_detector()
    
    if args.mode in ["val", "both"]:
        print("\n" + "=" * 60)
        print("モデルの検証を開始します")
        print("=" * 60)
        validate_model()
    
    print("\n処理が完了しました！")