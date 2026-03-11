# Reliable Multimodal RAG for Factuality in Medical Vision Language Models (Reproduction)

## Overview
This repository contains the reproduction of the ACL 2024 paper: **RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models**. 
The project aims to improve the factual accuracy of Medical Large Vision-Language Models (Med-LVLMs). It addresses two significant challenges in Retrieval-Augmented Generation (RAG):
1. **Factuality Risk Control**: Managing factual inaccuracies through a calibrated context selection strategy.
2. **Over-Reliance**: Balancing intrinsic knowledge and retrieved contexts using a preference optimization strategy.

## Architecture & Method
- **Backbone Model**: LLaVA-Med 1.57B.
- **Vision Encoder**: ResNet-50.
- **Text Encoder**: bio-BioClinicalBERT.
- **Techniques**:
  - Knowledge Balanced Preference Tuning using **DPO (Direct Preference Optimization)** via **LoRA fine-tuning**.
  - Constructing VQA pairs by converting medical reports into closed-ended questions with yes or no answers.

## Datasets
The implementation processes and utilizes the following medical datasets:
- **IU-Xray**: 8,121 associated chest images.

## Hardware & Training Details
- **Optimization**: AdamW with a learning rate of 0.001 and weight decay of 0.01.
- **Batch Size**: 16 / 32.
- **Epochs**: 360.
- **Hardware**: 1x NVIDIA RTX 4090 GPU.
- **Memory Management**: Enabled CPU offload to prevent Out-of-Memory (OOM) errors during model training.

## Experimental Results

| Metric | Paper Best Baseline Model | Paper | **Ours** |
| :--- | :--- | :--- | :--- |
| Accuracy (%) | 78.00 | 87.84 | **86.51** |
| F1-score (%) | 66.75 | 78.00 | **75.27** |
| Precision (%) | 55.96 | 75.41 | **72.26** |
| Recall (%) | 84.13 | 80.79 | **73.73** |
