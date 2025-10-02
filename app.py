# app.py

import streamlit as st
import pandas as pd
import numpy as np
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Advanced Model Import
from sentence_transformers import SentenceTransformer
from scipy.sparse import hstack, csr_matrix

# --- Page Configuration ---
st.set_page_config(
    page_title="Bengali Hate Speech Detection",
    page_icon="🇧🇩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Model Loading (Cached for performance) ---
@st.cache_resource
def load_semantic_model():
    """Loads the Multilingual SentenceTransformer model."""
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

semantic_model = load_semantic_model()

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_data():
    """Loads the Bengali dataset and includes error handling."""
    try:
        data = pd.read_csv("bengali_hate_speech_with_explicitness.csv")
    except FileNotFoundError:
        st.error("FATAL ERROR: 'bengali_hate_speech_with_explicitness.csv' not found. Please ensure the file is in your GitHub repository's main directory.")
        st.stop()
    # Drop rows where 'text' or 'label' is missing to prevent errors
    data.dropna(subset=['text', 'label'], inplace=True)
    return data

def preprocess_bengali_text(texts):
    """Cleans and preprocesses a list of Bengali text strings."""
    stopwords = set(['এবং', 'একটি', 'এই', 'ও', 'থেকে', 'জন্য', 'জে', 'করুন', 'আপনার', 'সব', 'কে', 'সে', 'কি', 'তার', 'তিনি', 'আমি', 'আপনি'])
    
    processed_texts = []
    for text in texts:
        text = str(text)
        text = re.sub(r'http[s]?://\S+', '', text) # remove URLs
        text = re.sub(r'[^\u0980-\u09FF\s]', '', text) # Keep only Bengali characters and spaces
        tokens = text.split()
        tokens = [word for word in tokens if word not in stopwords]
        processed_texts.append(" ".join(tokens))
    return processed_texts

# --- Feature Engineering and Model Training ---
@st.cache_data
def create_hybrid_features(_data, _text_col, _label_col):
    """Generates features and encodes labels for Bengali text."""
    _data['processed_text'] = preprocess_bengali_text(_data[_text_col])
    
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=3, max_features=5000)
    X_tfidf = tfidf_vectorizer.fit_transform(_data['processed_text'])
    
    X_semantic = semantic_model.encode(_data[_text_col].tolist(), show_progress_bar=False)
    
    X_combined = hstack([X_tfidf, csr_matrix(X_semantic)]).tocsr()
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(_data[_label_col])
    
    return X_combined, y_encoded, tfidf_vectorizer, le

@st.cache_resource
def train_classifier(_X_train, _y_train):
    """Trains a Logistic Regression model."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(_X_train, _y_train)
    return model

# --- Main Application UI ---
st.title("🇧🇩 Bengali Hate Speech Detection Dashboard")
st.markdown("Analysis of the `bengali_hate_speech_with_explicitness.csv` dataset.")

data = load_data()

st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("---")
analysis_choice = st.sidebar.radio("Go to:", ("Exploratory Data Analysis", "Model Performance", "Word Cloud Visualizations"))
st.sidebar.markdown("---")


if analysis_choice == "Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis")
    st.info("This section displays the structure and distributions within your uploaded dataset.")
    
    st.subheader("Dataset Preview")
    st.dataframe(data) # Display the full loaded dataframe
    
    # Create columns for side-by-side charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution of 'label'")
        fig, ax = plt.subplots()
        label_counts = data['label'].value_counts()
        sns.barplot(x=label_counts.index, y=label_counts.values, ax=ax, palette="rocket")
        ax.set_title("Content Labels")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

    with col2:
        st.subheader("Distribution of 'target'")
        fig, ax = plt.subplots()
        target_counts = data['target'].value_counts()
        sns.barplot(x=target_counts.index, y=target_counts.values, ax=ax, palette="plasma")
        ax.set_title("Hate Speech Targets")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

    st.subheader("Distribution of 'explicitness'")
    fig, ax = plt.subplots()
    explicitness_counts = data['explicitness'].value_counts()
    sns.barplot(x=explicitness_counts.index, y=explicitness_counts.values, ax=ax, palette="viridis")
    ax.set_title("Explicitness of Content")
    ax.set_ylabel("Count")
    plt.xticks(rotation=0)
    st.pyplot(fig)


elif analysis_choice == "Model Performance":
    st.header("🚀 Model Performance Analysis")
    st.markdown("Training a model to predict the **'label'** column.")
    
    with st.spinner("Preparing features and training model... This may take a minute on the first run."):
        X_combined, y, tfidf_vectorizer, label_encoder = create_hybrid_features(data, 'text', 'label')
        class_labels = label_encoder.classes_
        X_train, X_test, y_train, y_test = train_test_split(X_combined, y, random_state=42, test_size=0.25, stratify=y)
        model = train_classifier(X_train, y_train)
    
    st.success("Model is ready!")
    st.subheader("Classifier Performance Metrics")
    y_preds = model.predict(X_test)
    
    st.text(f"Overall Accuracy: {accuracy_score(y_test, y_preds):.4f}")
    report = classification_report(y_test, y_preds, target_names=class_labels, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    
    st.sidebar.header("Try a Live Prediction")
    user_input = st.sidebar.text_area("Enter Bengali text to classify:", key="user_input")
    if st.sidebar.button("Classify"):
        if user_input:
            with st.spinner("Analyzing..."):
                processed_input = preprocess_bengali_text([user_input])
                input_tfidf = tfidf_vectorizer.transform(processed_input)
                input_semantic = semantic_model.encode([user_input])
                input_combined = hstack([input_tfidf, csr_matrix(input_semantic)]).tocsr()
                prediction_encoded = model.predict(input_combined)
                prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]
            
            st.sidebar.subheader("Prediction Result")
            st.sidebar.info(f"Predicted Class ('label'): **{prediction_label}**")
        else:
            st.sidebar.write("Please enter some text.")


elif analysis_choice == "Word Cloud Visualizations":
    st.header("☁️ Word Cloud Visualizations")
    
    font_path = 'Nikosh.ttf'
    try:
        f = open(font_path, 'rb')
        f.close()
    except FileNotFoundError:
        st.error(f"FATAL ERROR: Font file 'Nikosh.ttf' not found. Please upload it to your GitHub repository.")
        st.stop()
        
    if 'processed_text' not in data.columns:
        data['processed_text'] = preprocess_bengali_text(data['text'])
    
    st.subheader("Word Cloud for All Text")
    all_words = ' '.join([text for text in data['processed_text']])
    if all_words.strip():
        wordcloud = WordCloud(width=800, height=500, background_color='white', font_path=font_path).generate(all_words)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.warning("Not enough words to generate a word cloud.")
