# 🎭 Sentiment Analysis - IMDb Movie Reviews

**Level**: 🔴 Advanced  
**Type**: Natural Language Processing - Text Classification  
**Dataset**: IMDb Movie Reviews Dataset

## 📋 Project Overview

This project performs sentiment analysis on movie reviews using NLP and deep learning techniques. It classifies reviews as positive or negative, introducing text preprocessing, word embeddings, and sequence models for NLP.

## 🎯 Objectives

- Learn NLP fundamentals and text preprocessing
- Master tokenization, stemming, and lemmatization
- Implement word embeddings (Word2Vec, GloVe)
- Build LSTM/GRU models for text classification
- Apply attention mechanisms
- Compare traditional ML vs deep learning for NLP

## 📊 Dataset Information

IMDb movie reviews dataset for binary sentiment classification.

### Features
- **Review Text**: Movie review content (variable length)
- **Sentiment**: Binary labels (positive/negative)
- **Training**: 25,000 reviews
- **Testing**: 25,000 reviews

### Challenge
- **Variable Length**: Reviews have different lengths
- **Vocabulary Size**: Large vocabulary with rare words
- **Context**: Understanding sentiment requires context
- **Sarcasm/Irony**: Complex linguistic patterns

## 🔍 Key Techniques

- **Text Preprocessing**: Cleaning, tokenization, stopword removal
- **Word Embeddings**: Pre-trained GloVe, Word2Vec
- **Sequence Models**: LSTM, GRU, Bidirectional RNNs
- **Attention Mechanisms**: Focus on important words
- **Transfer Learning**: Pre-trained BERT, RoBERTa
- **Traditional ML**: TF-IDF + Logistic Regression baseline

## 📈 Expected Results

- **LSTM Model**: ~87-90% accuracy
- **BERT Model**: ~92-95% accuracy
- **Traditional ML**: ~85-88% accuracy

## 🧠 Model Architectures

### LSTM Architecture
```
Input Text → Tokenization → Embedding Layer (100d)
    ↓
LSTM Layer (128 units, bidirectional)
    ↓
Global Max Pooling
    ↓
Dense (64) + ReLU + Dropout
    ↓
Dense (1) + Sigmoid
```

### BERT Architecture
```
Input Text → BERT Tokenizer
    ↓
Pre-trained BERT Model
    ↓
Classification Head
    ↓
Binary Sentiment Output
```

## 📚 NLP Pipeline

1. **Data Loading**: Load IMDb reviews
2. **Text Cleaning**: Remove HTML, special characters
3. **Tokenization**: Split text into tokens
4. **Preprocessing**: Lowercase, remove stopwords
5. **Vectorization**: Convert text to numerical format
6. **Model Training**: Train sentiment classifier
7. **Evaluation**: Test on unseen reviews

---

**🎯 Perfect for**: Learning NLP, text classification, sequence models

**⏱️ Estimated Time**: 6-8 hours

**🎓 Difficulty**: Advanced with NLP and deep learning concepts
