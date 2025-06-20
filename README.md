# 📰 Fake News Detection using RNN (LSTM)

## 📌 Project Overview

Fake news is one of the major challenges in the digital age, where unverified and misleading information spreads quickly through the internet. This project aims to develop a machine learning pipeline using Recurrent Neural Networks (RNN), specifically Long Short-Term Memory (LSTM), to classify news articles as **Real** or **Fake**.

## 🧠 Problem Statement

Given a dataset of real and fake news articles, the goal is to:
- Preprocess text data using NLP techniques
- Build an RNN-based binary classifier
- Predict whether a given news article is fake or real

---

## 📁 Dataset

We use a combination of two datasets:
- `True.csv`: Contains real news articles
- `Fake.csv`: Contains fake news articles

Each CSV file includes the following columns:
- `title`: Title of the news article
- `text`: Body of the news article
- `subject`, `date` (may exist but not used directly)

A label is assigned:
- `1` for real news
- `0` for fake news

---

## ⚙️ Features

- Text preprocessing (lowercasing, stopword removal, lemmatization)
- Tokenization & Padding
- Word embedding with `Embedding` layer
- LSTM network for sequence classification
- Evaluation with classification report and confusion matrix

---

## 🛠️ Installation

Make sure you have Python 3.x and the following packages installed:

```bash
pip install pandas numpy nltk seaborn matplotlib scikit-learn tensorflow
