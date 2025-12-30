# DistilBERT Sentiment Analysis

This repository contains an implementation of a fine-tuned **DistilBERT** model for
sentiment classification of professor reviews using transfer learning.

---

## Dataset
The dataset (`all_reviews.json`) contains student-written professor reviews along with
numeric quality ratings. Sentiment labels are derived from the quality score and mapped
to three classes:

- **Negative** (low quality ratings)
- **Neutral** (mid-range ratings)
- **Positive** (high quality ratings)

The dataset is used for supervised multi-class sentiment classification.

---

## Methodology
- Pretrained transformer model: `distilbert-base-uncased`
- Transfer learning with a fine-tuned classification head
- Text tokenization using the DistilBERT tokenizer
- Multi-class classification (negative / neutral / positive)
- Train–test split with evaluation on held-out data
- Implemented using PyTorch and Hugging Face Transformers

---

## Evaluation
- Accuracy ≈ 0.80 on the test set
- Macro F1-score ≈ 0.67
- Confusion matrix used to evaluate class-level performance

The model demonstrates strong performance across sentiment classes, including the
more challenging neutral category.

---

## Motivation
Transformer-based language models such as DistilBERT enable effective transfer learning
for text classification tasks. This project demonstrates how pretrained language models
can be adapted to domain-specific sentiment analysis with minimal feature engineering.

---

## Files
- `DistilBERT_sentiment_analysis.ipynb` — Colab notebook with full implementation
- `distilbert_sentiment_analysis.py` — Script version of the model
- `all_reviews.json` — Dataset used for training and evaluation

---
