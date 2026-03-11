#!/bin/bash
# 腳本 1 (v4)：訓練 IU X-Ray 的檢索器 (Retriever)
# (再次修正了 --img_root 路徑，指向 images 資料夾)

# --- 1. 設定絕對路徑 ---
export PROJECT_DIR="/media/md01/home/dongjhen/project/RULE"
export DATA_DIR="${PROJECT_DIR}/data/training/retriever"
export CUDA_VISIBLE_DEVICES="0" # 您的 4090 GPU

# --- 2. 設定圖片根目錄 (再次修正！) ---
# *** 修正：直接指向包含 CXR... 資料夾的 `images` 目錄 ***
export IMG_ROOT="/media/md01/public_datasets/yudong/iu_xray/images/"

# --- 3. 建立輸出資料夾 ---
mkdir -p "${PROJECT_DIR}/checkpoint/retriever_clip_iuxray/"
mkdir -p "${PROJECT_DIR}/logs/"

# --- 4. 進入程式碼所在的資料夾 ---
cd "${PROJECT_DIR}/retrieve/src" || { echo "錯誤：找不到 ${PROJECT_DIR}/retrieve/src"; exit 1; }

echo "開始訓練 IU X-Ray 檢索器 (CLIP) (v4 - 修正 img_root 指向 images)..."

# --- 5. 執行單 GPU 訓練 & 紀錄日誌 ---
python -m training.main \
    --model hf-hub:thaottn/OpenCLIP-resnet50-CC12M \
    --train-data "${DATA_DIR}/iuxray_train.json" \
    --dataset-type radiology \
    --img_root "${IMG_ROOT}" \
    --batch-size 16 \
    --precision amp \
    --workers 4 \
    --lr 0.0001 \
    --epochs 360 \
    --val-data "${DATA_DIR}/iuxray_val.json" \
    --val-frequency 10 \
    --report-to tensorboard \
    --logs "${PROJECT_DIR}/checkpoint/retriever_clip_iuxray/" \
    2>&1 | tee "${PROJECT_DIR}/logs/finetune_clip_iuxray_$(date +%Y%m%d-%H%M%S).log"

echo "IU X-Ray 檢索器訓練完成。"
