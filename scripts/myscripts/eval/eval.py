import json
import sys
import warnings
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# --- 嘗試導入生成指標庫 ---
try:
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.meteor.meteor import Meteor
    import nltk
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
    HAS_COCO = True
except ImportError:
    HAS_COCO = False
    print("警告: 未安裝 pycocoevalcap，將跳過 BLEU 計算。")

# --- 設定路徑 ---
ORIGINAL_FILE = '/media/md01/home/dongjhen/project/RULE/data/test/iuxray_test.jsonl'
RESULT_FILE = '/media/md01/home/dongjhen/project/RULE/result/iuxray_rag_answers.jsonl'

# --- 核心：聰明的解析邏輯 (Smart Parsing) ---
def normalize_answer_smart(text, prompt="", gt=""):
    t = str(text).lower().strip()
    if t.endswith('.'): t = t[:-1]
    
    gt = str(gt).lower().strip().replace(".", "")
    # 如果標準答案不是 Yes/No，這題無法評估 VQA
    if gt not in ["yes", "no"]:
        return "unknown"

    # 1. 直接命中 (Explicit)
    words = t.split()
    if "yes" in words or t.startswith("yes"): return "yes"
    if "no" in words or t.startswith("no"): return "no"

    # 2. 隱含否定 (Implicit No) - 這是您的模型最常說的
    neg_keywords = [
        "does not show", "no acute", "not see", "no sign", "no evidence", 
        "absence of", "unremarkable", "clear of", "negative for", "normal limits", 
        "is normal", "appear normal", "appears normal"
    ]
    for kw in neg_keywords:
        if kw in t:
            # 特例：問 "Is it normal?" 答 "It is normal" -> Yes
            if "normal" in prompt.lower() and ("normal" in kw or "unremarkable" in kw):
                return "yes"
            return "no"

    # 3. 隱含肯定 (Implicit Yes) - 這是救回 F1 的關鍵！
    pos_keywords = [
        "shows", "indicate", "present", "detected", "observed", 
        "evidence of", "consistent with", "suggestive of", "demonstrate",
        "opacity", "effusion", "consolidation", "pneumothorax", "edema" # 常見病徵
    ]
    if any(kw in t for kw in pos_keywords):
        # 確保前面沒有否定詞 (例如 "shows no evidence")
        if "no " not in t and "not " not in t:
            if "normal" in prompt.lower():
                return "no" # 問正常，答有病 -> No
            return "yes"

    return "unknown"

def main():
    print("=== 最終混合評估 (Smart Parsing + Strict ID Matching) ===")
    
    # 1. 讀取原始檔案
    print(f"1. 讀取原始 GT: {ORIGINAL_FILE}")
    gt_data = {} 
    try:
        with open(ORIGINAL_FILE, 'r') as f:
            for line in f:
                item = json.loads(line)
                qid = item['question_id']
                gt_data[qid] = {
                    'vqa_gt': item.get('answer', ''), # Yes/No
                    'report_gt': item.get('report', '') # Report
                }
    except FileNotFoundError:
        print("找不到原始檔案！")
        return

    # 2. 讀取結果並配對
    print(f"2. 讀取模型預測: {RESULT_FILE}")
    
    # VQA 用
    y_true_vqa = []
    y_pred_vqa = []
    
    # 生成用
    gts_gen = {}
    res_gen = {}
    
    count_matched = 0
    
    try:
        with open(RESULT_FILE, 'r') as f:
            for line in f:
                pred_item = json.loads(line)
                qid = pred_item.get('question_id')
                model_ans = pred_item.get('answer', '')
                prompt = pred_item.get('prompt', '')
                
                if qid in gt_data:
                    count_matched += 1
                    gt_item = gt_data[qid]
                    
                    # --- A. VQA 評估 (使用聰明解析) ---
                    gt_raw = gt_item['vqa_gt']
                    
                    # 解析 GT (確保它是 clean 的 yes/no)
                    gt_norm = normalize_answer_smart(gt_raw, prompt, gt=gt_raw)
                    # 解析 模型回答 (從長句中抓意圖)
                    pred_norm = normalize_answer_smart(model_ans, prompt, gt=gt_raw)
                    
                    if gt_norm in ['yes', 'no']:
                        y_true_vqa.append(gt_norm)
                        y_pred_vqa.append(pred_norm)

                    # --- B. 生成評估 (Report vs Answer) ---
                    gts_gen[qid] = [{'caption': gt_item['report_gt']}]
                    res_gen[qid] = [{'caption': model_ans}]
                    
    except FileNotFoundError:
        print("找不到結果檔案！")
        return

    print(f"成功配對: {count_matched} 筆")

    # 3. 計算 VQA 指標
    print("\n" + "="*50)
    print("【Part 1: VQA 分類指標 (Smart Parsing)】")
    print("="*50)
    
    # 過濾 unknown
    valid_data = [(t, p) for t, p in zip(y_true_vqa, y_pred_vqa) if p != 'unknown']
    
    if valid_data:
        y_true_val, y_pred_val = zip(*valid_data)
        
        acc = accuracy_score(y_true_val, y_pred_val)
        labels = ["yes", "no"]
        p = precision_score(y_true_val, y_pred_val, average='macro', labels=labels, zero_division=0)
        r = recall_score(y_true_val, y_pred_val, average='macro', labels=labels, zero_division=0)
        f1 = f1_score(y_true_val, y_pred_val, average='macro', labels=labels, zero_division=0)

        print(f"有效樣本數: {len(y_true_val)}")
        print("-" * 30)
        print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"Precision: {p:.4f}")
        print(f"Recall:    {r:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print("-" * 30)
    else:
        print("無有效 VQA 數據。")

    # 4. 計算生成指標
    if HAS_COCO:
        print("\n" + "="*50)
        print("【Part 2: Report 生成指標 (BLEU/METEOR)】")
        print("="*50)
        
        warnings.filterwarnings('ignore')
        tokenizer = PTBTokenizer()
        gts_tokenized = tokenizer.tokenize(gts_gen)
        res_tokenized = tokenizer.tokenize(res_gen)

        scorers = {
            "BLEU": Bleu(4),
            "ROUGE-L": Rouge(),
            "METEOR": Meteor()
        }

        for name, scorer in scorers.items():
            score, _ = scorer.compute_score(gts_tokenized, res_tokenized)
            if name == "BLEU":
                print(f"BLEU-1: {score[0]*100:.2f}")
                print(f"BLEU-2: {score[1]*100:.2f}")
                print(f"BLEU-3: {score[2]*100:.2f}")
                print(f"BLEU-4: {score[3]*100:.2f}")
            else:
                print(f"{name}:  {score*100:.2f}")

if __name__ == "__main__":
    main()
