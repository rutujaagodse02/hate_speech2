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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- Page Configuration ---
st.set_page_config(
    page_title="Hate Speech Detection Dashboard",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- NLTK Setup ---
# This function helps cache the download process
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
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)

download_nltk_data()


# --- Caching Functions for Performance ---
@st.cache_data
def load_data():
    """Loads the dataset from a CSV file."""
    data = pd.read_csv("HateSpeechData.csv")
    # Add a text length feature for initial analysis
    data['text length'] = data['tweet'].apply(len)
    return data

@st.cache_data
def preprocess_text(tweets_series):
    """Cleans and preprocesses a series of tweets."""
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    stop_words.update(["#ff", "ff", "rt"])

    processed_tweets = []
    for tweet in tweets_series:
        # Remove extra spaces
        tweet = re.sub(r'\s+', ' ', tweet)
        # Remove @mentions
        tweet = re.sub(r'@[\w\-]+', '', tweet)
        # Remove URLs
        tweet = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', tweet)
        # Remove punctuation and numbers
        tweet = re.sub("[^a-zA-Z]", " ", tweet)
        # Remove leading/trailing whitespace and convert to lower
        tweet = tweet.strip().lower()
        # Tokenize, remove stopwords, and stem
        tokens = [stemmer.stem(word) for word in tweet.split() if word not in stop_words]
        processed_tweets.append(" ".join(tokens))
    
    return processed_tweets

@st.cache_resource
def train_models(_X_train, _y_train):
    """Trains multiple classifiers and returns them in a dictionary."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Linear SVM": LinearSVC(random_state=42, dual=True, max_iter=2000) # Added dual=True for newer sklearn versions
    }
    for model in models.values():
        model.fit(_X_train, _y_train)
    return models

# --- Main Application Logic ---

# Define class labels globally to be accessible everywhere
class_labels = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}

# Title and Introduction
st.title("🗣️ Hate Speech Detection Dashboard")
st.markdown("""
This interactive dashboard analyzes a dataset of tweets to detect hate speech and offensive language. 
You can explore the dataset, visualize key insights, and compare the performance of different machine learning models.
""")

# Load and process data
data = load_data()
data['processed_tweets'] = preprocess_text(data['tweet'])

# --- Sidebar for User Input ---
st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("---")

# Section selection
st.sidebar.header("Choose a Section")
analysis_choice = st.sidebar.radio(
    "Go to:",
    ("Exploratory Data Analysis", "Word Cloud Visualizations", "Model Performance Comparison")
)
st.sidebar.markdown("---")

# --- Section 1: Exploratory Data Analysis ---
if analysis_choice == "Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis")

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    st.subheader("Distribution of Tweet Classes")
    fig, ax = plt.subplots()
    class_counts = data['class'].value_counts()
    sns.barplot(x=class_counts.index.map(class_labels), y=class_counts.values, ax=ax)
    ax.set_title("Number of Tweets per Class")
    ax.set_ylabel("Number of Tweets")
    st.pyplot(fig)
    st.markdown("""
    The dataset is highly imbalanced, with a majority of tweets classified as **Offensive Language**. 
    This imbalance can pose a challenge for machine learning models, especially in correctly identifying the minority **Hate Speech** class.
    """)

    st.subheader("Tweet Length Analysis by Class")
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    sns.histplot(data=data, x='text length', hue='class', multiple='stack', palette='viridis', ax=ax[0])
    ax[0].set_title("Distribution of Tweet Lengths")
    sns.boxplot(data=data, x='class', y='text length', palette='viridis', ax=ax[1])
    ax[1].set_title("Box Plot of Tweet Lengths")
    ax[1].set_xticklabels(['Hate', 'Offensive', 'Neither'])
    st.pyplot(fig)
    st.markdown("""
    The box plot suggests that **Offensive Language** tweets tend to have a slightly wider range of lengths, 
    but the distributions are largely overlapping. Text length alone is not a strong differentiator.
    """)

# --- Section 2: Word Cloud Visualizations ---
elif analysis_choice == "Word Cloud Visualizations":
    st.header("☁️ Word Cloud Visualizations")
    st.markdown("Visualizing the most frequent words in different categories of tweets.")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("All Tweets")
        all_words = ' '.join([text for text in data['processed_tweets']])
        wordcloud_all = WordCloud(width=800, height=800, background_color='white').generate(all_words)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud_all, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    with col2:
        st.subheader("Offensive Language")
        offensive_words = ' '.join([text for text in data['processed_tweets'][data['class'] == 1]])
        wordcloud_offensive = WordCloud(width=800, height=800, background_color='white').generate(offensive_words)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud_offensive, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    with col3:
        st.subheader("Hate Speech")
        hate_words = ' '.join([text for text in data['processed_tweets'][data['class'] == 0]])
        wordcloud_hate = WordCloud(width=800, height=800, background_color='white').generate(hate_words)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud_hate, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

# --- Section 3: Model Performance Comparison ---
elif analysis_choice == "Model Performance Comparison":
    st.header("🤖 Model Performance Comparison")
    st.markdown("""
    Here, we train several machine learning models and evaluate their performance on the task of classifying tweets. 
    We use **TF-IDF** (Term Frequency-Inverse Document Frequency) to convert text into numerical features.
    """)

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.75, min_df=5, max_features=10000)
    X_tfidf = tfidf_vectorizer.fit_transform(data['processed_tweets'])
    y = data['class']

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, random_state=42, test_size=0.2, stratify=y)
    
    with st.spinner("Training models... This may take a moment."):
        models = train_models(X_train, y_train)

    st.sidebar.header("Select a Model for Details")
    selected_model = st.sidebar.selectbox("Choose a model to inspect:", list(models.keys()))
    
    st.subheader("Overall Accuracy Comparison")
    accuracies = {name: accuracy_score(y_test, model.predict(X_test)) for name, model in models.items()}
    accuracy_df = pd.DataFrame(accuracies.items(), columns=["Model", "Accuracy"])
    fig, ax = plt.subplots()
    sns.barplot(x="Model", y="Accuracy", data=accuracy_df, ax=ax, palette='mako')
    ax.set_title("Model Accuracy Comparison")
    ax.set_ylim(0.8, 1.0)
    st.pyplot(fig)

    st.subheader(f"Detailed Performance: {selected_model}")
    model = models[selected_model]
    y_preds = model.predict(X_test)
    
    st.text("Classification Report")
    report = classification_report(y_test, y_preds, target_names=['Hate', 'Offensive', 'Neither'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.2f}"))

    st.text("Confusion Matrix")
    cm = confusion_matrix(y_test, y_preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Hate', 'Offensive', 'Neither'], 
                yticklabels=['Hate', 'Offensive', 'Neither'], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    st.markdown("""
    **Key Observations:**
    - All models achieve high overall accuracy, largely due to their strong performance on the majority **Offensive Language** class.
    - The **Hate Speech** class (class 0) is the most challenging to predict, often being misclassified as Offensive. This is reflected in its lower precision and recall scores.
    """)

    st.sidebar.markdown("---")
    st.sidebar.header("Try a Live Prediction")
    user_input = st.sidebar.text_area("Enter a text snippet to classify:")
    if st.sidebar.button("Classify"):
        if user_input:
            processed_input = preprocess_text([user_input])
            vectorized_input = tfidf_vectorizer.transform(processed_input)
            prediction = models[selected_model].predict(vectorized_input)
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
