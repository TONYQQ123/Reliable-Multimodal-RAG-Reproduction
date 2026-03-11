#!/bin/bash
# ------------------------------------------------------------------
# STEP 2: Run RAG-enabled inference
# (修正：cd 路徑改為上三層)
# ------------------------------------------------------------------

# --- 自動切換到專案根目錄 ---
# 找到此腳本所在的目錄
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# 切換到上三層目錄 (即 /media/md01/home/dongjhen/project/RULE/)
cd "$SCRIPT_DIR/../../../"
# -----------------------------

export CUDA_VISIBLE_DEVICES=0
dataset='iuxray'

# --- 1. LLaVA-Med SFT 基礎模型 ---
export BASE_MODEL_PATH="/media/md01/home/dongjhen/project/RULE/checkpoint/llava-med-v1.5-mistral-7b"

# --- 2. 您 DPO 訓練好的 LoRA 權重路徑 ---
export LORA_PATH="/media/md01/home/dongjhen/project/RULE/checkpoint/dpo_iuxray/"

# --- 3. 測試集圖片資料夾 (必須與 Step 1 檢索時相同) ---
export TEST_IMAGE_DIR="/media/md01/public_datasets/yudong/iu_xray/images" 

# --- 4. 【關鍵】使用您在 Step 1 產生的 "RAG 檔案" ---
export RAG_TEST_FILE="/media/md01/home/dongjhen/project/RULE/result/iuxray_test_with_reports.jsonl"

# --- 5. 儲存最終答案的路徑 ---
export OUTPUT_FILE="/media/md01/home/dongjhen/project/RULE/result/iuxray_rag_answers.jsonl"
# ---

echo "Starting RAG-enabled inference..."
echo "Current directory: $(pwd)" # 確認我們在 RULE/ 根目錄
echo "Loading LLaVA model from: ${LORA_PATH}"
echo "Using RAG Question File: ${RAG_TEST_FILE}"

# 現在這個相對路徑 llava/eval/... 是相對於專案根目錄，所以是正確的
python llava/eval/model_vqa_${dataset}.py \
    --model-base "${BASE_MODEL_PATH}" \
    --model-path "${LORA_PATH}" \
    --question-file "${RAG_TEST_FILE}" \
    --image-folder "${TEST_IMAGE_DIR}" \
    --answers-file "${OUTPUT_FILE}" \
    --conv-mode vicuna_v1 \
    --temperature 0 \
    --num_beams 1

echo "Inference complete. Answers saved to: ${OUTPUT_FILE}"
