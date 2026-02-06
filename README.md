# 🎬 IMDB Movie Review Sentiment Analysis (NLP)

## 📌 Project Overview

This project performs **Sentiment Analysis on IMDB movie reviews** using **Natural Language Processing (NLP)** and **Machine Learning** techniques to classify reviews as:

✅ Positive
❌ Negative

We apply preprocessing, feature extraction (BoW & TF-IDF), and multiple ML models to compare performance.

---

## 🚀 Features

* HTML tag removal
* Lowercase normalization
* Stopword removal (NLTK)
* Bag of Words (CountVectorizer)
* TF-IDF Vectorization
* GaussianNB, Random Forest, Logistic Regression
* Model performance comparison

---

## 📂 Project Structure

```
IMDB-Sentiment-Analysis/
│
├── imdb_sentiment_analysis.ipynb   # Main notebook
├── README.md
```

⚠️ Dataset is not included due to GitHub size limits.

---

## 📊 Dataset

Since the dataset size is **50MB+**, GitHub does not allow uploading it directly.

👉 Download it from Kaggle:

🔗 **IMDB Dataset (50K Reviews)**
[https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

### After downloading:

Place the file inside your project folder:

```
IMDB-Sentiment-Analysis/
│
├── IMDB Dataset.csv
├── imdb_sentiment_analysis.ipynb
```

---

## ⚙️ Workflow

### 1️⃣ Data Cleaning

* Remove HTML tags
* Convert to lowercase
* Remove stopwords
* Remove duplicates

### 2️⃣ Feature Engineering

* Bag of Words
* TF-IDF (unigram + bigram)

### 3️⃣ Model Training

* Train/Test split (80/20)
* Train multiple models
* Compare accuracy

---

## 📈 Results

| Model                        | Accuracy    |
| ---------------------------- | ----------- |
| GaussianNB                   | 63%         |
| Random Forest                | 84–85%      |
| Logistic Regression (TF-IDF) | ⭐ **88.4%** |

Best model → **TF-IDF + Logistic Regression**

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK

---

## ▶️ How to Run

### Install dependencies

```bash
pip install pandas numpy scikit-learn nltk matplotlib
```

### Download stopwords

```python
import nltk
nltk.download('stopwords')
```

### Run

Open the notebook:

```
imdb_sentiment_analysis.ipynb
```

---

## 🎯 Learning Outcomes

* NLP preprocessing pipeline
* Text vectorization techniques
* ML model comparison
* Sentiment classification
* Working with large datasets

---

## 🔮 Future Improvements

* LSTM / GRU models
* BERT / Transformers
* Hyperparameter tuning
* Deploy using Streamlit/Flask

---

## 👤 Author

**Bhupender Singh**
Data Science | Machine Learning | Analytics

GitHub: [https://github.com/bhupender5](https://github.com/bhupender5)
LinkedIn: [https://www.linkedin.com/in/bhupinder-singh-bba271187](https://www.linkedin.com/in/bhupinder-singh-bba271187)

