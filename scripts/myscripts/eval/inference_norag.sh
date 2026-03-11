#!/bin/bash

# --- 自動切換到專案根目錄 ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/../../"
# -----------------------------

# 設置 CUDA 設備為 0 (根據您的 nvidia-smi 輸出)
export CUDA_VISIBLE_DEVICES=0

# 設置 CUDA 函式庫路徑 (以防萬一)
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# 設置資料集名稱
dataset='iuxray'

# 現在所有相對路徑 (如 llava/eval/...) 都會從 RULE 根目錄開始計算，所以是正確的
python llava/eval/model_vqa_${dataset}.py \
    --model-base '/media/md01/home/dongjhen/project/RULE/checkpoint/llava-med-v1.5-mistral-7b' \
    --model-path '/media/md01/home/dongjhen/project/RULE/checkpoint/dpo_iuxray/' \
    --question-file '/media/md01/home/dongjhen/project/RULE/data/test/iuxray_test.jsonl' \
    --image-folder '/media/md01/public_datasets/yudong/iu_xray/images/' \
    --answers-file '/media/md01/home/dongjhen/project/RULE/result/dpo_${dataset}_eval_answers.jsonl'

echo "評估完成，答案已儲存到 /media/md01/home/dongjhen/project/RULE/result/dpo_${dataset}_eval_answers.jsonl"
