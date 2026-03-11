#!/bin/bash
# 腳本 4 (v3)：訓練 IU X-Ray DPO 模型 - 修正 train_dpo.py 路徑

# --- 1. 設定環境 ---
export CUDA_VISIBLE_DEVICES="0" # *** 使用 GPU 0 ***
export NCCL_P2P_DISABLE=1
export PROJECT_DIR="/media/md01/home/dongjhen/project/RULE"
export LLAVA_DIR="${PROJECT_DIR}/llava" # *** LLaVA 程式碼根目錄 ***

# --- 2. 基礎 SFT 模型路徑 ---
export BASE_MODEL_PATH="${PROJECT_DIR}/checkpoint/llava-med-v1.5-mistral-7b" # *** 您提供的路徑 ***

# --- 3. DPO 訓練資料 ---
export DPO_DATA_FILE="${PROJECT_DIR}/data/training/alignment/iuxray.json" # *** 您提供的路徑 ***

# --- 4. 圖片根目錄 ---
export IMG_ROOT="/media/md01/public_datasets/yudong/iu_xray/images/" # *** IU X-Ray 圖片位置 ***

# --- 5. 輸出目錄 ---
export OUTPUT_DIR="${PROJECT_DIR}/checkpoint/dpo_iuxray/" # *** DPO 模型儲存位置 ***
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PROJECT_DIR}/logs/" # 日誌目錄

# --- 6. Deepspeed 設定檔路徑 ---
export DEEPSPEED_CONFIG="${PROJECT_DIR}/scripts/zero3.json" # *** 您提供的路徑 ***

# --- 7. 進入 LLaVA 程式碼主目錄 ---
cd "${LLAVA_DIR}" || { echo "錯誤：找不到 LLaVA 程式碼目錄 ${LLAVA_DIR}"; exit 1; }

echo "開始訓練 IU X-Ray DPO 模型 (Faithfulness Scorer)..."

# --- 8. 執行單 GPU DPO 訓練 (使用 deepspeed 但只用一個 GPU) ---
# *** 修改：修正 train_dpo.py 的相對路徑 ***
deepspeed train/mytrain_dpo.py \
    --model_name_or_path ${BASE_MODEL_PATH} \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --version v1 \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --data_path ${DPO_DATA_FILE} \
    --image_folder ${IMG_ROOT} \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 1e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to tensorboard \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    2>&1 | tee "${PROJECT_DIR}/logs/train_dpo_iuxray_$(date +%Y%m%d-%H%M%S).log"

echo "IU X-Ray DPO 模型訓練完成。"
