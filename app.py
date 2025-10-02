import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Advanced Model Imports
from sentence_transformers import SentenceTransformer
from detoxify import Detoxify
import xgboost as xgb
from scipy.sparse import hstack, csr_matrix

# --- Page Configuration ---
st.set_page_config(
    page_title="Hate Speech Detection Dashboard",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- NLTK and Model Setup ---
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

@st.cache_resource
def load_models():
    """Loads all necessary models into memory."""
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    toxicity_model = Detoxify('original')
    return semantic_model, toxicity_model

download_nltk_data()
semantic_model, toxicity_model = load_models()


# --- Caching Functions for Performance ---
@st.cache_data
def load_data():
    """Loads and preprocesses the dataset."""
    data = pd.read_csv("HateSpeechData.csv")
    data['text length'] = data['tweet'].apply(len)
    data['processed_tweets'] = preprocess_text(data['tweet'])
    return data

def preprocess_text(tweets_series):
    """Cleans and preprocesses a series of tweets for TF-IDF."""
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    stop_words.update(["#ff", "ff", "rt"])
    processed_tweets = []
    for tweet in tweets_series:
        tweet = re.sub(r'\s+', ' ', tweet)
        tweet = re.sub(r'@[\w\-]+', '', tweet)
        tweet = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', tweet)
        tweet = re.sub("[^a-zA-Z]", " ", tweet)
        tweet = tweet.strip().lower()
        tokens = [stemmer.stem(word) for word in tweet.split() if word not in stop_words]
        processed_tweets.append(" ".join(tokens))
    return processed_tweets

# --- Advanced Feature Engineering Functions ---
@st.cache_data
def get_semantic_features(_tweets_series):
    """Generates sentence embeddings."""
    with st.spinner("Generating semantic features..."):
        embeddings = semantic_model.encode(_tweets_series.tolist(), show_progress_bar=False)
    return embeddings

@st.cache_data
def get_toxicity_features(_tweets_series):
    """Generates toxicity scores."""
    with st.spinner("Generating toxicity features..."):
        # Detoxify expects a list of strings
        toxicity_scores = toxicity_model.predict(_tweets_series.tolist())
    # Convert dictionary of lists to a DataFrame, then to a numpy array
    return pd.DataFrame(toxicity_scores).values

# --- Main Application Logic ---
class_labels = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
st.title("🗣️ Advanced Hate Speech Detection Dashboard")
st.markdown("This dashboard uses a **hybrid feature model** (TF-IDF, Semantic & Toxicity) with an **XGBoost classifier**.")

data = load_data()

# --- Sidebar ---
st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("---")
analysis_choice = st.sidebar.radio("Go to:", ("Model Performance", "Exploratory Data Analysis", "Word Cloud Visualizations"))
st.sidebar.markdown("---")


# --- Section: Model Performance (Now the main section) ---
if analysis_choice == "Model Performance":
    st.header("🚀 Advanced Model Performance")

    # 1. TF-IDF Features
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.75, min_df=5, max_features=5000)
    X_tfidf = tfidf_vectorizer.fit_transform(data['processed_tweets'])

    # 2. Semantic Features
    X_semantic = get_semantic_features(data['tweet'])

    # 3. Toxicity Features
    X_toxicity = get_toxicity_features(data['tweet'])

    # 4. Combine all features
    # Use hstack for sparse (TF-IDF) and dense matrices
    X_combined = hstack([X_tfidf, csr_matrix(X_semantic), csr_matrix(X_toxicity)]).tocsr()
    y = data['class']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, random_state=42, test_size=0.2, stratify=y)

    # Train XGBoost Model
    with st.spinner("Training XGBoost model... This is computationally intensive."):
        xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, n_estimators=100, learning_rate=0.1, max_depth=7, use_label_encoder=False, eval_metric='mlogloss')
        xgb_model.fit(X_train, y_train)

    st.subheader("XGBoost Classifier Performance")
    y_preds = xgb_model.predict(X_test)

    # Display Metrics
    st.text(f"Accuracy: {accuracy_score(y_test, y_preds):.4f}")
    st.text("Classification Report:")
    report = classification_report(y_test, y_preds, target_names=class_labels.values(), output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))

    st.text("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels.values(), yticklabels=class_labels.values(), ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    
    # --- Live Prediction ---
    st.sidebar.header("Try a Live Prediction")
    user_input = st.sidebar.text_area("Enter a text snippet to classify:")
    if st.sidebar.button("Classify"):
        if user_input:
            # Create all features for the single input
            input_tfidf = tfidf_vectorizer.transform([preprocess_text([user_input])[0]])
            input_semantic = semantic_model.encode([user_input])
            input_toxicity = pd.DataFrame(toxicity_model.predict([user_input])).values
            
            # Combine features in the same order
            input_combined = hstack([input_tfidf, csr_matrix(input_semantic), csr_matrix(input_toxicity)]).tocsr()

            # Predict
            prediction = xgb_model.predict(input_combined)
            prediction_label = class_labels[prediction[0]]
            
            st.sidebar.subheader("Prediction Result")
            if prediction_label == 'Hate Speech':
                st.sidebar.error(f"Predicted Class: **{prediction_label}**")
            elif prediction_label == 'Offensive Language':
                 st.sidebar.warning(f"Predicted Class: **{prediction_label}**")
            else:
                st.sidebar.success(f"Predicted Class: **{prediction_label}**")
        else:
            st.sidebar.write("Please enter some text to classify.")

# --- Other Sections remain the same ---
elif analysis_choice == "Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis")
    # (Code from your previous version can be pasted here without changes)
    st.subheader("Dataset Preview")
    st.dataframe(data.head())
    st.subheader("Distribution of Tweet Classes")
    fig, ax = plt.subplots()
    class_counts = data['class'].value_counts()
    sns.barplot(x=class_counts.index.map(class_labels), y=class_counts.values, ax=ax)
    st.pyplot(fig)


elif analysis_choice == "Word Cloud Visualizations":
    st.header("☁️ Word Cloud Visualizations")
    # (Code from your previous version can be pasted here without changes)
    st.subheader("All Tweets")
    all_words = ' '.join([text for text in data['processed_tweets']])
    wordcloud_all = WordCloud(width=800, height=800, background_color='white').generate(all_words)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud_all, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
