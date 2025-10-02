# # app.py

# import streamlit as st
# import pandas as pd
# import numpy as np
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# # Advanced Model Import
# from sentence_transformers import SentenceTransformer
# from scipy.sparse import hstack, csr_matrix

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="Hate Speech Detection Dashboard",
#     page_icon="🗣️",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # --- NLTK and Model Setup ---
# @st.cache_resource
# def download_nltk_data():
#     """Downloads necessary NLTK data if not already present."""
#     try:
#         nltk.data.find('corpora/stopwords')
#     except LookupError:
#         nltk.download('stopwords', quiet=True)
#     try:
#         nltk.data.find('tokenizers/punkt')
#     except LookupError:
#         nltk.download('punkt', quiet=True)

# @st.cache_resource
# def load_semantic_model():
#     """Loads the SentenceTransformer model into memory and caches it."""
#     return SentenceTransformer('all-MiniLM-L6-v2')

# # Initial setup runs once
# download_nltk_data()
# semantic_model = load_semantic_model()


# # --- Data Loading and Preprocessing Functions ---
# @st.cache_data
# def load_data():
#     """Loads the dataset from a CSV file and includes error handling."""
#     try:
#         data = pd.read_csv("HateSpeechData.csv")
#     except FileNotFoundError:
#         st.error("FATAL ERROR: 'HateSpeechData.csv' not found. Please ensure the file is in your GitHub repository's main directory and the name matches exactly.")
#         st.stop()  # Stop the app from running further
        
#     data['text length'] = data['tweet'].apply(len)
#     return data

# def preprocess_text(tweets_series):
#     """Cleans and preprocesses a series of tweets for TF-IDF."""
#     stemmer = PorterStemmer()
#     stop_words = set(stopwords.words("english"))
#     stop_words.update(["#ff", "ff", "rt"])
#     processed_tweets = []
#     for tweet in tweets_series:
#         tweet = re.sub(r'\s+', ' ', tweet)
#         tweet = re.sub(r'@[\w\-]+', '', tweet)
#         tweet = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', tweet)
#         tweet = re.sub("[^a-zA-Z]", " ", tweet)
#         tweet = tweet.strip().lower()
#         tokens = [stemmer.stem(word) for word in tweet.split() if word not in stop_words]
#         processed_tweets.append(" ".join(tokens))
#     return processed_tweets

# # --- Feature Engineering and Model Training ---
# @st.cache_data
# def create_hybrid_features(_data):
#     """Generates and combines TF-IDF and Semantic features."""
#     tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.75, min_df=5, max_features=5000)
#     # CORRECTED LINE: The typo 'tffidf_vectorizer' is now 'tfidf_vectorizer'
#     X_tfidf = tfidf_vectorizer.fit_transform(_data['processed_tweets'])
#     X_semantic = semantic_model.encode(_data['tweet'].tolist(), show_progress_bar=False)
#     X_combined = hstack([X_tfidf, csr_matrix(X_semantic)]).tocsr()
#     return X_combined, _data['class'], tfidf_vectorizer

# @st.cache_resource
# def train_classifier(_X_train, _y_train):
#     """Trains a Logistic Regression model."""
#     model = LogisticRegression(max_iter=1000, random_state=42)
#     model.fit(_X_train, _y_train)
#     return model

# # --- Main Application UI ---
# class_labels = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}

# st.title("🗣️ Advanced Hate Speech Detection Dashboard")
# st.markdown("An NLP project using a **hybrid feature model** (TF-IDF + Semantic Embeddings).")

# data = load_data()
# data['processed_tweets'] = preprocess_text(data['tweet'])

# st.sidebar.title("Dashboard Controls")
# st.sidebar.markdown("---")
# analysis_choice = st.sidebar.radio("Go to:", ("Model Performance", "Exploratory Data Analysis", "Word Cloud Visualizations"))
# st.sidebar.markdown("---")

# if analysis_choice == "Model Performance":
#     st.header("🚀 Model Performance Analysis")
#     with st.spinner("Preparing features and training model... This may take a minute on first run."):
#         X_combined, y, tfidf_vectorizer = create_hybrid_features(data)
#         X_train, X_test, y_train, y_test = train_test_split(X_combined, y, random_state=42, test_size=0.2, stratify=y)
#         model = train_classifier(X_train, y_train)
    
#     st.success("Model is ready!")
#     st.subheader("Classifier Performance Metrics")
#     y_preds = model.predict(X_test)
#     st.text(f"Overall Accuracy: {accuracy_score(y_test, y_preds):.4f}")
#     report = classification_report(y_test, y_preds, target_names=class_labels.values(), output_dict=True)
#     st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))

#     st.subheader("Confusion Matrix")
#     cm = confusion_matrix(y_test, y_preds)
#     fig, ax = plt.subplots()
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels.values(), yticklabels=class_labels.values(), ax=ax)
#     ax.set_xlabel("Predicted")
#     ax.set_ylabel("Actual")
#     st.pyplot(fig)
    
#     st.sidebar.header("Try a Live Prediction")
#     user_input = st.sidebar.text_area("Enter a text snippet to classify:", key="user_input")
#     if st.sidebar.button("Classify"):
#         if user_input:
#             with st.spinner("Analyzing..."):
#                 processed_input = preprocess_text([user_input])
#                 input_tfidf = tfidf_vectorizer.transform(processed_input)
#                 input_semantic = semantic_model.encode([user_input])
#                 input_combined = hstack([input_tfidf, csr_matrix(input_semantic)]).tocsr()
#                 prediction = model.predict(input_combined)
#                 prediction_label = class_labels[prediction[0]]
            
#             st.sidebar.subheader("Prediction Result")
#             if prediction_label == 'Hate Speech':
#                 st.sidebar.error(f"Predicted Class: **{prediction_label}**")
#             elif prediction_label == 'Offensive Language':
#                  st.sidebar.warning(f"Predicted Class: **{prediction_label}**")
#             else:
#                 st.sidebar.success(f"Predicted Class: **{prediction_label}**")
#         else:
#             st.sidebar.write("Please enter some text to classify.")

# elif analysis_choice == "Exploratory Data Analysis":
#     st.header("📊 Exploratory Data Analysis")
#     st.subheader("Dataset Preview")
#     st.dataframe(data.head())
#     st.subheader("Distribution of Tweet Classes")
#     fig, ax = plt.subplots()
#     class_counts = data['class'].value_counts()
#     sns.barplot(x=class_counts.index.map(class_labels), y=class_counts.values, ax=ax)
#     ax.set_title("Number of Tweets per Class")
#     ax.set_ylabel("Number of Tweets")
#     st.pyplot(fig)

# elif analysis_choice == "Word Cloud Visualizations":
#     st.header("☁️ Word Cloud Visualizations")
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.subheader("All Tweets")
#         all_words = ' '.join([text for text in data['processed_tweets']])
#         wordcloud_all = WordCloud(width=800, height=600, background_color='white').generate(all_words)
#         fig, ax = plt.subplots()
#         ax.imshow(wordcloud_all, interpolation='bilinear')
#         ax.axis('off')
#         st.pyplot(fig)
#     with col2:
#         st.subheader("Offensive Language")
#         offensive_words = ' '.join([text for text in data['processed_tweets'][data['class'] == 1]])
#         wordcloud_offensive = WordCloud(width=800, height=600, background_color='white').generate(offensive_words)
#         fig, ax = plt.subplots()
#         ax.imshow(wordcloud_offensive, interpolation='bilinear')
#         ax.axis('off')
#         st.pyplot(fig)
#     with col3:
#         st.subheader("Hate Speech")
#         hate_words = ' '.join([text for text in data['processed_tweets'][data['class'] == 0]])
#         wordcloud_hate = WordCloud(width=800, height=600, background_color='white').generate(hate_words)
#         fig, ax = plt.subplots()
#         ax.imshow(wordcloud_hate, interpolation='bilinear')
#         ax.axis('off')
#         st.pyplot(fig)
# app.py

import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
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

# --- Model and NLTK Setup ---
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
    return data

def preprocess_bengali_text(texts):
    """Cleans and preprocesses Bengali text."""
    # Common Bengali stopwords (this list can be expanded)
    stopwords = set(['এবং', 'একটি', 'এই', 'ও', 'থেকে', 'জন্য', 'জে', 'করুন', 'আপনার', 'সব', 'কে', 'সে', 'কি', 'তার', 'তিনি', 'আমি', 'আপনি'])
    
    processed_texts = []
    for text in texts:
        text = re.sub(r'http[s]?://\S+', '', text) # remove URLs
        # Remove non-Bengali characters and numbers, keeping spaces
        text = re.sub(r'[^\u0980-\u09FF\s]', '', text)
        tokens = text.split()
        # Remove stopwords
        tokens = [word for word in tokens if word not in stopwords]
        processed_texts.append(" ".join(tokens))
    return processed_texts

# --- Feature Engineering and Model Training ---
@st.cache_data
def create_hybrid_features(_data, _text_col, _label_col):
    """Generates TF-IDF and Semantic features for Bengali text."""
    # Preprocess text for TF-IDF
    _data['processed_text'] = preprocess_bengali_text(_data[_text_col])
    
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=3, max_features=5000)
    X_tfidf = tfidf_vectorizer.fit_transform(_data['processed_text'])
    
    X_semantic = semantic_model.encode(_data[_text_col].tolist(), show_progress_bar=False)
    
    X_combined = hstack([X_tfidf, csr_matrix(X_semantic)]).tocsr()
    
    # Encode string labels into numbers
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
st.markdown("A project using a **hybrid feature model** (TF-IDF + Multilingual Semantic Embeddings).")

data = load_data()

# Check for required columns
if 'text' not in data.columns or 'label' not in data.columns:
    st.error("CSV file must contain 'text' and 'label' columns.")
    st.stop()

st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("---")
analysis_choice = st.sidebar.radio("Go to:", ("Model Performance", "Exploratory Data Analysis", "Word Cloud Visualizations"))
st.sidebar.markdown("---")

if analysis_choice == "Model Performance":
    st.header("🚀 Model Performance Analysis")
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
            st.sidebar.info(f"Predicted Class: **{prediction_label}**")
        else:
            st.sidebar.write("Please enter some text.")

elif analysis_choice == "Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis")
    st.subheader("Dataset Preview")
    st.dataframe(data.head())
    
    st.subheader("Distribution of Labels")
    fig, ax = plt.subplots()
    class_counts = data['label'].value_counts()
    sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax)
    ax.set_title("Number of Texts per Label")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    st.pyplot(fig)

elif analysis_choice == "Word Cloud Visualizations":
    st.header("☁️ Word Cloud Visualizations")
    st.info("Displaying the most frequent words in the processed Bengali text.")
    
    # Check if font file exists
    font_path = 'Nikosh.ttf'
    try:
        # Test if the font can be loaded
        f = open(font_path, 'rb')
        f.close()
    except FileNotFoundError:
        st.error(f"FATAL ERROR: Font file 'Nikosh.ttf' not found. Please upload it to your GitHub repository.")
        st.stop()
        
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

