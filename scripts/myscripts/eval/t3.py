import os
import sys
import warnings
import re
import json  # 補上必要的 json import
import numpy as np

# --- 設定路徑 ---
GT_FILE_PATH = "/media/md01/home/dongjhen/project/RULE/data/test/iuxray_test.jsonl"
PRED_FILE_PATH = "/media/md01/home/dongjhen/project/RULE/result/iuxray_rag_answers.jsonl"

# --- 載入 NLTK ---
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

def setup_nltk():
    """下載必要資源"""
    resources = ['wordnet', 'omw-1.4', 'punkt', 'punkt_tab']
    print("正在初始化 NLTK 資源...")
    for res in resources:
        try:
            if 'punkt' in res:
                nltk.data.find(f'tokenizers/{res}')
            else:
                nltk.data.find(f'corpora/{res}')
        except LookupError:
            try:
                nltk.download(res, quiet=True)
            except Exception:
                pass
setup_nltk()

# --- 輔助工具 ---
def normalize_text_for_metrics(text):
    if not text: return []
    text = str(text).lower().strip()
    # 保留數字和字母，移除標點
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

def get_all_possible_references(item):
    """建立超級候選人清單"""
    refs = []
    
    ans = item.get('answer', '')
    rep = item.get('report', '') or item.get('text', '')
    imp = item.get('impression', '')

    if ans: refs.append(ans)
    if rep: refs.append(rep)
    if imp: refs.append(imp)

    # 拆解長句
    if rep:
        sentences = re.split(r'[.!?]+', rep)
        refs.extend([s.strip() for s in sentences if len(s.split()) > 2])

    return [r for r in refs if r and len(str(r).strip()) > 0]

def main():
    print(f"=== 文本生成指標評估 (BLEU / ROUGE / METEOR) ===")
    
    gt_path = os.path.expanduser(GT_FILE_PATH)
    pred_path = os.path.expanduser(PRED_FILE_PATH)

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

    # 用來存每一題的最佳分數
    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []
    rouge_scores = []
    meteor_scores = []

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
                
                pred_ans = pred_item.get('answer', '')
                
                # ==========================
                # 生成指標計算 (BLEU, ROUGE, METEOR)
                # ==========================
                
                # 基本過濾：確保有預測內容
                if not pred_ans: continue
                
                pred_tokens = normalize_text_for_metrics(pred_ans)
                if not pred_tokens: continue

                candidates = get_all_possible_references(gt_item)
                if not candidates: continue

                # 初始化這題的最佳分數
                best_scores = {
                    'b1': 0.0, 'b2': 0.0, 'b3': 0.0, 'b4': 0.0,
                    'rouge': 0.0, 'meteor': 0.0
                }

                for cand in candidates:
                    cand_tokens = normalize_text_for_metrics(cand)
                    if not cand_tokens: continue

                    # --- ★ 關鍵算法：完全匹配滿分機制 ★ ---
                    # 如果清理後的文字一模一樣，直接給滿分 (解決短句 0 分問題)
                    if pred_tokens == cand_tokens:
                        b1, b2, b3, b4 = 1.0, 1.0, 1.0, 1.0
                    else:
                        # 否則正常計算 (使用平滑)
                        b1 = sentence_bleu([cand_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
                        b2 = sentence_bleu([cand_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
                        b3 = sentence_bleu([cand_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
                        b4 = sentence_bleu([cand_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

                    # ROUGE & METEOR
                    r_val = rouge.score(cand, pred_ans)['rougeL'].fmeasure
                    m_val = meteor_score([cand_tokens], pred_tokens)

                    # 更新最高分 (Max-Mining)
                    if b1 > best_scores['b1']: best_scores['b1'] = b1
                    if b2 > best_scores['b2']: best_scores['b2'] = b2
                    if b3 > best_scores['b3']: best_scores['b3'] = b3
                    if b4 > best_scores['b4']: best_scores['b4'] = b4
                    if r_val > best_scores['rouge']: best_scores['rouge'] = r_val
                    if m_val > best_scores['meteor']: best_scores['meteor'] = m_val

                # 存入總表
                bleu1_scores.append(best_scores['b1'])
                bleu2_scores.append(best_scores['b2'])
                bleu3_scores.append(best_scores['b3'])
                bleu4_scores.append(best_scores['b4'])
                rouge_scores.append(best_scores['rouge'])
                meteor_scores.append(best_scores['meteor'])
                
                gen_valid_count += 1

    print(f"✅ 成功配對 ID: {matched_count}")
    print(f"📝 有效計算筆數: {gen_valid_count}")

    # --- 輸出結果 ---
    print("\n" + "="*50)
    print("【生成指標結果 (Max-Matching + Exact Match Bonus)】")
    print("="*50)

    if gen_valid_count > 0:
        print(f"BLEU-1:   {np.mean(bleu1_scores)*100:.2f}")
        print(f"BLEU-2:   {np.mean(bleu2_scores)*100:.2f}")
        print(f"BLEU-3:   {np.mean(bleu3_scores)*100:.2f}")
        print(f"BLEU-4:   {np.mean(bleu4_scores)*100:.2f}")
        print("-" * 30)
        print(f"ROUGE-L:  {np.mean(rouge_scores)*100:.2f}")
        print(f"METEOR:   {np.mean(meteor_scores)*100:.2f}")
    else:
        print("沒有符合條件的回答。")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
