import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK resources quietly for Cloud server
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
lemmatizer = WordNetLemmatizer()

# Page configuration
st.set_page_config(page_title="Research Abstract Classifier", layout="centered")

st.title("Research Paper Abstract Classifier")
st.write("Classify peer-reviewed research abstracts into **Science**, **Business**, or **Social Science** using Supervised Machine Learning (TF-IDF + Logistic Regression).")

# Load saved model, vectorizer, and stop words
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    stop_words = joblib.load("stop_words.pkl")
    return model, tfidf, stop_words

model, tfidf, stop_words = load_artifacts()

# Create two tabs: Single Abstract & Batch CSV Upload
tab1, tab2 = st.tabs(["Single Abstract & Explainability", "Upload CSV Dataset"])

with tab1:
    user_abstract = st.text_area("Paste Research Abstract Here:", height=150, placeholder="Type or paste abstract text...")

    if st.button("Classify Abstract", type="primary"):
        if not user_abstract.strip():
            st.warning("Please enter some text to classify.")
        else:
            # 1. Clean text
            text = re.sub(r'<.*?>', ' ', str(user_abstract))
            text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
            words = text.split()
            clean_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
            clean_text = " ".join(clean_words)
            
            # 2. Transform using TF-IDF
            vec = tfidf.transform([clean_text])
            
            # 3. Predict class
            prediction = model.predict(vec)[0]
            probabilities = model.predict_proba(vec)[0] if hasattr(model, "predict_proba") else None
            
            st.markdown("---")
            if prediction == "Science":
                st.success(f"### Predicted Field: **Science**")
            elif prediction == "Business":
                st.info(f"### Predicted Field: **Business**")
            else:
                st.warning(f"### Predicted Field: **Social Science**")
                
            # Display Confidence Probabilities
            if probabilities is not None:
                classes = model.classes_
                prob_df = pd.DataFrame({"Category": classes, "Probability": probabilities}).set_index("Category")
                st.write("**Model Confidence Scores:**")
                st.bar_chart(prob_df)
                
            # Explainability Block
            st.markdown("### Why did the model choose this?")
            st.write("Top words from your abstract that contributed most to this prediction:")
            
            try:
                if hasattr(model, "coef_"):
                    feature_names = np.array(tfidf.get_feature_names_out())
                    class_idx = list(model.classes_).index(prediction)
                    coefs = model.coef_
                    
                    coefficients = coefs[class_idx] if coefs.ndim > 1 else coefs
                    non_zero_indices = vec.nonzero()[1]
                    
                    word_contributions = []
                    for idx in non_zero_indices:
                        if idx < len(coefficients) and idx < len(feature_names):
                            word = feature_names[idx]
                            contribution = vec[0, idx] * coefficients[idx]
                            word_contributions.append((word, contribution))
                            
                    word_contributions.sort(key=lambda x: x[1], reverse=True)
                    top_words = word_contributions[:8]
                    
                    if top_words:
                        exp_df = pd.DataFrame(top_words, columns=["Word / Phrase", "Influence Score"]).set_index("Word / Phrase")
                        st.bar_chart(exp_df)
                    else:
                        st.info(f"Classified as **{prediction}** based on overall weighted text features.")
            except Exception:
                st.info(f"The model successfully classified this abstract as **{prediction}**.")

with tab2:
    st.write("Upload any `.csv` file containing an **`abstract`** column to receive predicted labels for the entire dataset.")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    
    if uploaded_file is not None:
        new_df = pd.read_csv(uploaded_file)
        if 'abstract' not in new_df.columns:
            st.error("The uploaded CSV must contain a column named 'abstract'.")
        else:
            new_cleaned = []
            for text in new_df['abstract']:
                text = re.sub(r'<.*?>', ' ', str(text))
                text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
                words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words and len(w) > 2]
                new_cleaned.append(" ".join(words))
                
            new_X = tfidf.transform(new_cleaned)
            new_df['Predicted_Label'] = model.predict(new_X)
            
            st.success(f"Successfully classified {len(new_df)} research abstracts!")
            st.dataframe(new_df[['abstract', 'Predicted_Label']])
            
            csv_download = new_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predicted CSV", data=csv_download, file_name="predicted_abstracts.csv", mime="text/csv")