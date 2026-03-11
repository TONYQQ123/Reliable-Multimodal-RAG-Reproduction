#!/bin/bash
# 腳本 2 (v2)：訓練 Harvard 的檢索器 (Retriever)
# (根據 harvard_train.json 和實際圖片路徑修正了 --img_root)

# --- 1. 設定環境 ---
export CUDA_VISIBLE_DEVICES="0" # 您的 4090 GPU
export PROJECT_DIR="/media/md01/home/dongjhen/project/RULE"
export DATA_DIR="${PROJECT_DIR}/data/training/retriever" # JSON 檔案位置

# --- 2. 設定圖片根目錄 (再次修正！) ---
# *** 修正：精確指向包含 .jpg 檔案的那一層 ***
export IMG_ROOT="/media/md01/public_datasets/yudong/__MACOSX/Training/Training/"

# --- 3. 建立輸出資料夾 ---
mkdir -p "${PROJECT_DIR}/checkpoint/retriever_clip_harvard/"
mkdir -p "${PROJECT_DIR}/logs/"

# --- 4. 進入程式碼所在的資料夾 ---
cd "${PROJECT_DIR}/retrieve/src" || { echo "錯誤：找不到 ${PROJECT_DIR}/retrieve/src"; exit 1; }

echo "開始訓練 Harvard 檢索器 (CLIP) (v2 - 修正 img_root)..."

# --- 5. 執行單 GPU 訓練 & 紀錄日誌 ---
python -m training.main \
    --model hf-hub:thaottn/OpenCLIP-resnet50-CC12M \
    --train-data "${DATA_DIR}/harvard_train.json" \
    --dataset-type radiology \
    --img_root "${IMG_ROOT}" \
    --batch-size 16 \
    --precision amp \
    --workers 4 \
    --lr 0.0001 \
    --epochs 360 \
    --val-data "${DATA_DIR}/harvard_val.json" \
    --val-frequency 10 \
    --report-to tensorboard \
    --logs "${PROJECT_DIR}/checkpoint/retriever_clip_harvard/" \
    2>&1 | tee "${PROJECT_DIR}/logs/finetune_clip_harvard_$(date +%Y%m%d-%H%M%S).log"

echo "Harvard 檢索器訓練完成。"
