# Research Paper Abstract Classification using Supervised Machine Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Pipeline-orange.svg)
![NLTK](https://img.shields.io/badge/NLTK-NLP%20Preprocessing-green.svg)

> **Course Project — I (Classification) | DAC-507 / DAC-515 (2026–2027)**  
> **Reference Paper:** S. Chowdhury and M. P. Schoen, *"Research Paper Classification using Supervised Machine Learning Techniques,"* 2020 Intermountain Engineering, Technology and Computing (IETC), IEEE, pp. 1–6, doi: `10.1109/IETC47856.2020.9249211`.

---

## 1. Problem Statement & Overview
The exponential growth of peer-reviewed literature requires automated indexing systems capable of categorizing research papers accurately. This project implements a complete **Natural Language Processing (NLP) and Supervised Machine Learning Pipeline** that classifies scholarly publications into three primary fields based solely on their **abstract text**:
1. **Science** (*Physics, Biology, Chemistry, Astronomy, Genetics*)
2. **Business** (*Marketing, Finance, Entrepreneurship, Supply Chain, Accounting*)
3. **Social Science** (*Sociology, Psychology, Political Science, Anthropology, Education*)

---

## 2. Live Interactive Demo (Streamlit)
 **Live Web Application:** **[Paste Your Deployed Streamlit URL Here]**

Our interactive web application (`app.py`) provides two modes of operation:
* **Tab 1 — Single Abstract Classifier & Explainability:** Paste any research abstract to view its predicted category, calibrated **Model Confidence Scores** (`predict_proba`), and an **Influence Score Bar Chart** highlighting the exact words from the abstract that drove the model's decision.
* **Tab 2 — Batch CSV Dataset Uploader:** Upload any external `.csv` file containing an `abstract` column (e.g., `sample_upload.csv`) to automatically preprocess the text, generate predicted labels for all rows, and download the labeled dataset (`predicted_abstracts.csv`).

---

## 3. Dataset Architecture (`dataset.csv`)
* **Source:** Legally and ethically harvested via the open-access **Europe PMC RESTful API** (`https://www.ebi.ac.uk/europepmc/webservices/rest/search`).
* **Total Volume:** **1,050 peer-reviewed research abstracts**.
* **Uniform Balance:**
  * **350 abstracts per class** (`Science`: 350, `Business`: 350, `Social Science`: 350).
  * **70 abstracts per sub-class** across 15 distinct academic subtopics ($15 \times 70 = 1,050$).
* **Columns in `dataset.csv`:**
  * `subtopic`: Specific academic search domain.
  * `abstract`: Raw peer-reviewed abstract text fetched from Europe PMC.
  * `label`: Target category (`Science`, `Business`, or `Social Science`).
  * `clean_abstract`: Preprocessed, stop-word-filtered, and lemmatized text.

---

## 4. NLP Preprocessing & Feature Engineering Pipeline
Raw abstracts are transformed into numerical feature vectors through a four-stage NLP pipeline:
1. **HTML & Noise Removal:** Stripped XML/HTML tags (`<.*?>`), punctuation, URLs, and numeric digits (`[^a-zA-Z\s]`), followed by lowercase normalization.
2. **Tokenization & Academic Stop-Word Removal:** Removed standard English stop words (`nltk.corpus.stopwords`) plus high-frequency academic filler terms (`study`, `paper`, `research`, `result`, `results`, `method`, `methods`, `analysis`, `also`, `using`, `used`) and short noise tokens.
3. **WordNet Lemmatization:** Reduced inflected words to their dictionary root lemma using NLTK's `WordNetLemmatizer` to preserve human-readable keywords for filter visualizations.
4. **TF-IDF Vectorization:** Extracted the top **2,500 unigram and bigram features** (`max_features=2500, ngram_range=(1, 2)`), followed by an **80/20 stratified train-test split** (840 training papers, 210 testing papers).

---

## 5. Model Comparison & Experimental Results
All six required classifiers were trained on the 80% training split and evaluated on the 20% unseen test split using weighted Precision, Recall, F1-Score, 5-Fold Cross-Validation, and Training Time:

| Model | Hyperparameters | Accuracy | Precision | Recall | F1-Score | CV Mean | CV Std | Train Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `max_iter=1000, random_state=42` | **0.8905** | **0.8940** | **0.8905** | 0.8874 | 0.6933 | 0.0644 | 0.0676 |
| **Linear SVM** | `kernel='linear', random_state=42` | **0.8905** | 0.8936 | **0.8905** | **0.8890** | 0.6648 | 0.0716 | 0.7775 |
| **KNN** | `n_neighbors=10, metric='cosine'` | 0.8762 | 0.8752 | 0.8762 | 0.8751 | 0.6324 | 0.0620 | 0.0040 |
| **Naive Bayes** | `MultinomialNB(alpha=0.2)` | 0.8714 | 0.8773 | 0.8714 | 0.8680 | **0.7210** | 0.0605 | **0.0040** |
| **Random Forest** | `n_estimators=200, max_depth=40` | 0.8524 | 0.8608 | 0.8524 | 0.8509 | 0.6629 | 0.0478 | 1.3103 |
| **Decision Tree** | `max_depth=30, random_state=42` | 0.6857 | 0.6948 | 0.6857 | 0.6868 | 0.4676 | 0.0520 | 0.1379 |

### Key Analytical Takeaways:
* **Why Logistic Regression & Linear SVM Excelled (89.05%):** Sparse, high-dimensional TF-IDF matrices (2,500 features) are linearly separable. Hyperplane classifiers effectively weigh all domain terms simultaneously without overfitting to individual words.
* **Why Decision Tree Lagged (68.57%) vs. Random Forest (85.24%):** A single Decision Tree greedily splits on specific words and overfits sparse text vectors, whereas aggregating 200 trees in a Random Forest reduces variance and boosts accuracy by **+16.67%**.
* **Why Cosine Metric Boosted KNN (87.62%):** Measuring the angle between TF-IDF vectors (`metric='cosine'`) normalizes for abstract length and avoids Euclidean distance concentration in 2,500 dimensions.
* **Confusion Matrix & Filter Visualization Insights:** **Science** achieved the highest class-wise recall due to specialized technical vocabulary (*cell, gene, quantum, magnetic*), while minor misclassifications primarily occurred between **Business** (*market, financial, firm*) and **Social Science** (*social, student, political, education*) due to interdisciplinary overlap in socio-economic research.

---

## 6. Repository Structure & Deliverables
```text
├── Data_Collection.ipynb     # Runnable Jupyter Notebook (Scraping, NLP, Training, Plots, Export)
├── dataset.csv               # Complete 1,050-row labeled dataset (raw + cleaned abstracts)
├── app.py                    # Streamlit web application (Live tester, XAI chart, CSV uploader)
├── sample_upload.csv         # Sample unlabeled dataset for testing the batch CSV upload demo
├── best_model.pkl            # Serialized top-performing Logistic Regression model
├── tfidf_vectorizer.pkl      # Serialized 2,500-feature TF-IDF vectorizer
├── stop_words.pkl            # Serialized NLTK + academic stop-word set
├── requirements.txt          # Python dependencies for local & Streamlit Cloud execution
└── README.md                 # Project documentation and experimental summary
```

---

## 7. How to Run Locally
1. **Clone the repository and install dependencies:**
   ```bash
   git clone https://github.com/your-username/research-paper-classifier.git
   cd research-paper-classifier
   pip install -r requirements.txt
   ```
2. **Launch the interactive Streamlit demo:**
   ```bash
   streamlit run app.py
   ```
3. **Run the Jupyter Notebook:**
   Open `Data_Collection.ipynb` in VS Code, JupyterLab, or Google Colab and execute all cells sequentially.
