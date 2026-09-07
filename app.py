import os
import re
import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Page configuration
st.set_page_config(
    page_title="Job Category Classifier",
    page_icon="💼",
    layout="centered"
)

# Download and cache NLTK resources safely
@st.cache_resource
def get_nltk_resources():
    for resource in ["stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer

# Cache model artifacts loading
@st.cache_resource
def load_model_artifacts():
    # Use svm_model.pkl (or job_model.pkl if you renamed it)
    model_filename = "svm_model.pkl" if os.path.exists("svm_model.pkl") else "job_model.pkl"
    
    model = joblib.load(model_filename)
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, tfidf, label_encoder

stop_words, lemmatizer = get_nltk_resources()

try:
    model, tfidf, label_encoder = load_model_artifacts()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.info("Make sure `svm_model.pkl`, `tfidf_vectorizer.pkl`, and `label_encoder.pkl` are in the same folder as this script.")
    st.stop()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# UI Layout
st.title("💼 Job Category Classifier")
st.write("Enter a job description to predict its job category using NLP.")
st.divider()

job_description = st.text_area(
    "Enter Job Description",
    height=250,
    placeholder="Example: We are looking for a Data Analyst with strong SQL, Python, Excel, and Power BI skills..."
)

if st.button("🔍 Predict Job Category", use_container_width=True, type="primary"):
    if not job_description.strip():
        st.warning("Please enter a job description.")
    else:
        with st.spinner("Classifying..."):
            cleaned_text = preprocess_text(job_description)
            text_vector = tfidf.transform([cleaned_text])
            raw_prediction = model.predict(text_vector)[0]

            # Decode prediction
            if hasattr(raw_prediction, "__int__") or isinstance(raw_prediction, (int,)):
                predicted_category = label_encoder.inverse_transform([raw_prediction])[0]
            else:
                predicted_category = str(raw_prediction)

            st.success("Prediction completed successfully!")
            st.subheader("Predicted Job Category")
            st.markdown(f"### 🎯 **{predicted_category}**")

st.divider()
st.caption("NLP Job Category Classification System")