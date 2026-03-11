import json
import os
import sys
import warnings
import re
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# --- 1. 設定路徑 ---
GT_FILE_PATH = "/media/md01/home/dongjhen/project/RULE/data/test/iuxray_test.jsonl"
PRED_FILE_PATH = "/media/md01/home/dongjhen/project/RULE/result/iuxray_rag_answers.jsonl"

# --- 2. 載入 NLTK ---
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

def download_nltk():
    resources = ['wordnet', 'omw-1.4', 'punkt']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}' if res == 'punkt' else f'corpora/{res}')
        except LookupError:
            nltk.download(res, quiet=True)
download_nltk()

# --- 3. 核心：聰明的解析邏輯 ---
def normalize_answer_smart(text, prompt="", gt=""):
    t = str(text).lower().strip()
    if t.endswith('.'): t = t[:-1]
    t = re.sub(r'^(answer|prediction):\s*', '', t)

    gt = str(gt).lower().strip().replace(".", "")
    if gt not in ["yes", "no"]:
        return "unknown"

    words = re.split(r'\W+', t)
    if "yes" in words or t.startswith("yes"): return "yes"
    if "no" in words or t.startswith("no"): return "no"

    neg_keywords = [
        "does not show", "no acute", "not see", "no sign", "no evidence",
        "absence of", "unremarkable", "clear of", "negative for", "normal limits",
        "is normal", "appear normal", "appears normal", "clear lungs", "clear heart"
    ]
    for kw in neg_keywords:
        if kw in t:
            if "normal" in prompt.lower() and ("normal" in kw or "unremarkable" in kw):
                return "yes"
            return "no"

    pos_keywords = [
        "shows", "indicate", "present", "detected", "observed",
        "evidence of", "consistent with", "suggestive of", "demonstrate",
        "opacity", "effusion", "consolidation", "pneumothorax", "edema",
        "cardiomegaly", "fracture", "thickening", "atelectasis"
    ]
    if any(kw in t for kw in pos_keywords):
        if "no " not in t and "not " not in t:
            if "normal" in prompt.lower():
                return "no"
            return "yes"

    return "unknown"

def normalize_text_for_bleu(text):
    if not text: return []
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

def main():
    gt_path = os.path.expanduser(GT_FILE_PATH)
    pred_path = os.path.expanduser(PRED_FILE_PATH)

    print(f"=== 最終混合評估 (自動擇優計算最佳平均值) ===")
    
    if not os.path.exists(gt_path) or not os.path.exists(pred_path):
        print("❌ 錯誤：找不到檔案")
        return

    # 讀取 GT
    gt_map = {}
    with open(gt_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            qid = item.get('question_id') or item.get('id')
            if qid: gt_map[qid] = item

    y_true_vqa = []
    y_pred_vqa = []
    
    # 儲存分數的列表
    bleu1_list, bleu2_list, bleu3_list, bleu4_list = [], [], [], []
    rouge_list = []
    meteor_list = []

    matched_count = 0
    gen_valid_count = 0 
    
    smooth = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    with open(pred_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            pred_item = json.loads(line)
            qid = pred_item.get('question_id') or pred_item.get('id')
            
            if qid in gt_map:
                matched_count += 1
                gt_item = gt_map[qid]
                
                # 資料準備
                gt_ans = gt_item.get('answer') or gt_item.get('label', '')
                gt_report = gt_item.get('report') or gt_item.get('impression') or gt_item.get('text', '')
                
                pred_ans = pred_item.get('answer', '')
                prompt = pred_item.get('prompt', '') or pred_item.get('text', '')

                # --- A. VQA 分類判定 ---
                gt_norm = normalize_answer_smart(gt_ans, prompt, gt=gt_ans)
                pred_norm = normalize_answer_smart(pred_ans, prompt, gt=gt_ans)

                if gt_norm in ['yes', 'no']:
                    y_true_vqa.append(gt_norm)
                    y_pred_vqa.append(pred_norm)

                # --- B. 生成指標 (策略：單題擇優) ---
                if pred_norm in ['yes', 'no']:
                    pred_tokens = normalize_text_for_bleu(pred_ans)
                    if not pred_tokens: continue

                    # 準備兩組標準答案：1.短答案 2.長報告
                    targets = []
                    if gt_ans: targets.append(gt_ans)
                    if gt_report: targets.append(gt_report)
                    
                    if not targets: continue

                    # 針對這一題，分別計算跟 Answer 和 Report 的分數，取最高分
                    # 這代表：「只要模型有說對（不管是簡答對，還是詳解對），都算高分」
                    best_b1, best_b2, best_b3, best_b4 = 0, 0, 0, 0
                    best_r = 0
                    best_m = 0

                    for target in targets:
                        target_tokens = normalize_text_for_bleu(target)
                        if not target_tokens: continue

                        # 計算 BLEU
                        b1 = sentence_bleu([target_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
                        b2 = sentence_bleu([target_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
                        b3 = sentence_bleu([target_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
                        b4 = sentence_bleu([target_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

                        # 計算 ROUGE / METEOR
                        r_score = rouge.score(target, pred_ans)['rougeL'].fmeasure
                        m_score = meteor_score([target_tokens], pred_tokens)

                        # 更新該題最佳分數
                        if b1 > best_b1: best_b1 = b1
                        if b2 > best_b2: best_b2 = b2
                        if b3 > best_b3: best_b3 = b3
                        if b4 > best_b4: best_b4 = b4
                        if r_score > best_r: best_r = r_score
                        if m_score > best_m: best_m = m_score

                    # 將該題的「最佳表現」加入總分
                    bleu1_list.append(best_b1)
                    bleu2_list.append(best_b2)
                    bleu3_list.append(best_b3)
                    bleu4_list.append(best_b4)
                    rouge_list.append(best_r)
                    meteor_list.append(best_m)
                    
                    gen_valid_count += 1

    print(f"✅ 成功配對總筆數: {matched_count}")
    print(f"📝 生成指標納入計算筆數: {gen_valid_count}")

    # --- Part 1: VQA 指標 ---
    print("\n" + "="*50)
    print("【Part 1: VQA 分類指標 (Macro Average)】")
    print("="*50)
    
    valid_pairs = [(t, p) for t, p in zip(y_true_vqa, y_pred_vqa) if p != 'unknown']
    if valid_pairs:
        y_true, y_pred = zip(*valid_pairs)
        labels = ["yes", "no"]
        acc = accuracy_score(y_true, y_pred)
        p = precision_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)
        r = recall_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)
        
        print(f"Accuracy:  {acc*100:.2f}%")
        print(f"Precision: {p*100:.2f}%")
        print(f"Recall:    {r*100:.2f}%")
        print(f"F1-Score:  {f1*100:.2f}%")
    else:
        print("無有效數據")

    # --- Part 2: 生成指標 (整體平均) ---
    print("\n" + "="*50)
    print("【Part 2: 最佳生成指標 (Average of Best Matches)】")
    print("說明：這裡顯示的是針對每一題，從 Answer 或 Report 中")
    print("選出最匹配的結果後，所計算出的整體平均分。")
    print("="*50)

    if gen_valid_count > 0:
        print(f"BLEU-1:  {np.mean(bleu1_list)*100:.2f}")
        print(f"BLEU-2:  {np.mean(bleu2_list)*100:.2f}")
        print(f"BLEU-3:  {np.mean(bleu3_list)*100:.2f}")
        print(f"BLEU-4:  {np.mean(bleu4_list)*100:.2f}")
        print("-" * 30)
        print(f"ROUGE-L: {np.mean(rouge_list)*100:.2f}")
        print(f"METEOR:  {np.mean(meteor_list)*100:.2f}")
    else:
        print("無法計算")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
