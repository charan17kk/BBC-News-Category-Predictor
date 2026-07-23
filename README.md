# BBC News Article Classifier

## Overview

BBC News Article Classifier is an end-to-end Natural Language Processing (NLP) application that automatically classifies news articles into predefined categories using traditional machine learning techniques.

The project demonstrates the complete NLP pipeline, including text preprocessing, feature extraction using TF-IDF, model comparison, evaluation, and deployment through an interactive Streamlit application.

---

## Business Problem

News organizations publish thousands of articles every day across multiple categories. Manually organizing these articles is time-consuming and inefficient.

This project aims to automatically classify news articles into their appropriate categories, enabling faster content organization and improving information retrieval.

---

## Dataset

- **Dataset:** BBC News Dataset
- **Categories:** 5
- **Task:** Multi-Class Text Classification

### Categories

- Business
- Entertainment
- Politics
- Sport
- Tech

---

## NLP Workflow

1. Data Loading
2. Data Understanding
3. Text Cleaning
4. Text Preprocessing
   - Lowercasing
   - Tokenization
   - Stop-word Removal
   - Text Normalization
5. TF-IDF Vectorization
6. Train-Test Split
7. Model Training
8. Model Comparison
9. Model Evaluation
10. Final Model Selection
11. Model Serialization
12. Streamlit Deployment

---

## Models Compared

The following machine learning algorithms were evaluated:

- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM

Linear SVM was selected as the final model based on its superior Precision, Recall, F1-Score, and overall classification performance.

---

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score

---

## Features

- Real-Time News Classification
- Automatic Category Prediction
- Prediction Confidence Visualization
- Interactive Streamlit Interface

---

## Tech Stack

### Programming

- Python

### NLP & Machine Learning

- Scikit-learn
- NLTK
- TF-IDF Vectorization
- Linear SVM

### Data Processing

- Pandas
- NumPy

### Deployment

- Streamlit

---

## Run Locally

```bash
git clone https://github.com/charan17kk/BBC-News-Category-Predictor.git

cd BBC-News-Category-Predictor

pip install -r requirements.txt

streamlit run app.py
```

---

## Live Demo

https://bbc-news-category-predictor-mrv8evrcrwa675s33yujjw.streamlit.app/

---

## Future Improvements

- BERT-based Text Classification
- Transformer Models
- Explainable NLP
- Multi-language News Classification
- News Recommendation System

---

## Author

**Sai Charan Kandukuri**

Aspiring Machine Learning Engineer passionate about building end-to-end machine learning solutions across classification, regression, and natural language processing.