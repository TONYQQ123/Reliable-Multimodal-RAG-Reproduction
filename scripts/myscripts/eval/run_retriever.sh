#!/bin/bash
# ------------------------------------------------------------------
# STEP 1: Run Retriever to add RAG context to the test file
# (Paths updated based on user input)
# ------------------------------------------------------------------

# --- 真實的測試集路徑 (由您提供) ---

# 1. 原始測試問題檔 (Input)
export ORIGINAL_TEST_FILE="/media/md01/home/dongjhen/project/RULE/data/test/iuxray_test.jsonl"

# 2. 測試集圖片資料夾 (Input)
export TEST_IMAGE_DIR="/media/md01/public_datasets/yudong/iu_xray/images"

# 3. 儲存 RAG 測試檔的新路徑 (Output)
export NEW_TEST_FILE="/media/md01/home/dongjhen/project/RULE/result/iuxray_test_with_reports.jsonl"
# ---

# 1. 導航到 Python 腳本所在的 "真正" 目錄
cd ~/project/RULE/retrieve/src/ || exit
export CUDA_VISIBLE_DEVICES=0 # 使用任一空閒 GPU

echo "Starting retrieval for the test file: ${ORIGINAL_TEST_FILE}..."
echo "Current directory: $(pwd)"

# 2. 執行該目錄下的 Python 腳本
python ./retrieve_clip_VQA.py \
    --img_root "${TEST_IMAGE_DIR}" \
    --train_json /media/md01/home/dongjhen/project/RULE/data/training/retriever/iuxray_train.json \
    --eval_json "${ORIGINAL_TEST_FILE}" \
    --model_name_or_path hf-hub:thaottn/OpenCLIP-resnet50-CC12M \
    --checkpoint_path /media/md01/home/dongjhen/project/RULE/checkpoint/retriever_clip_iuxray/2025_10_18-16_21_28-model_hf-hub:thaottn-OpenCLIP-resnet50-CC12M-lr_0.0001-b_16-j_4-p_amp/checkpoints/epoch_360.pt \
    --output_path "${NEW_TEST_FILE}" \
    --fixed_k 5

echo "Retrieval for test file complete. New RAG-enabled file saved to: ${NEW_TEST_FILE}"
