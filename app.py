import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Dense, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import requests
from io import BytesIO
import tempfile
import os

# Initialize session state for model, tokenizer, and encoder
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None
if 'max_len' not in st.session_state:
    st.session_state.max_len = 150

# Title for the Streamlit app
st.set_page_config(layout="wide")
st.title("Bengali Hate Speech Detection using an RNN Model")
st.markdown("This application trains a Bidirectional GRU (RNN) model to detect hate speech in Bengali text.")

# Function to load data from a local CSV file
@st.cache_data
def load_data():
    file_path = 'bengali.csv' # Updated file path
    try:
        df = pd.read_csv(file_path)
        # Using new column names 'sentence' and 'hate'
        df.dropna(subset=['sentence', 'hate'], inplace=True)
        # Map numerical hate labels to string labels for clarity
        df['hate_label'] = df['hate'].map({0: 'Not Hate', 1: 'Hate'})
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please make sure it is in the same directory as app.py.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

data = load_data()

# --- Text Preprocessing Functions ---
bengali_stopwords = [
    'এবং', 'о', 'আর', 'বা', 'কিংবা', 'হয়', 'হচ্ছে', 'হয়েছে', 'ছিল', 'আছে', 'এই',
    'সেই', 'যে', 'সে', 'কি', 'কে', 'কোন', 'কার', 'কাকে', 'আমি', 'তুমি', 'সে',
    'আমার', 'তোমার', 'তার', 'আমাদের', 'তোমাদের', 'তাদের', 'জন্য', 'থেকে', 'সঙ্গে',
    'দ্বারা', 'কাছে', 'দিকে', 'মধ্যে', 'উপরে', 'নিচে', 'পরে', 'আগে', 'এক', 'দুই'
]

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^\u0980-\u09FF\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [word for word in tokens if word not in bengali_stopwords]
    return " ".join(tokens)

if data is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Preview")
        st.write(data.head())
        st.subheader("Dataset Description")
        st.write(f"The dataset contains {data.shape[0]} rows and {data.shape[1]} columns.")
        st.write("Columns:", ", ".join(data.columns))

    with col2:
        st.subheader("Data Visualization")
        st.markdown("#### Distribution of Labels")
        fig, ax = plt.subplots()
        # Visualize the new 'hate_label' column
        sns.countplot(x='hate_label', data=data, ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("Text Preprocessing")
    # Apply preprocessing to the 'sentence' column
    data['cleaned_text'] = data['sentence'].apply(preprocess_text)
    
    st.markdown("Applied the following preprocessing steps:")
    st.markdown("- Removed URLs, mentions, and hashtags.")
    st.markdown("- Removed punctuation and numbers, keeping only Bengali characters.")
    st.markdown("- Removed common Bengali stopwords and extra whitespace.")
    st.write("#### Cleaned Text Preview:")
    st.write(data[['sentence', 'cleaned_text']].head())

    st.markdown("#### Word Cloud from Hate Speech Text")
    # Generate word cloud from sentences where hate == 1
    hate_speech_text = " ".join(text for text in data[data['hate'] == 1]['cleaned_text']).strip()
    
    if hate_speech_text:
        font_url = 'https://github.com/google/fonts/raw/main/ofl/solaimanlipi/SolaimanLipi.ttf'
        try:
            response = requests.get(font_url)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.ttf') as fp:
                fp.write(response.content)
                font_path = fp.name
            
            wordcloud = WordCloud(
                font_path=font_path, width=800, height=400,
                background_color='white', collocations=False
            ).generate(hate_speech_text)
            
            fig_wc, ax_wc = plt.subplots()
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)
            os.remove(font_path)
        except Exception as e:
            st.warning(f"Could not generate word cloud. Error: {e}")
    else:
        st.info("No words found to generate a word cloud for the hate speech category after preprocessing.")

    st.subheader("Model Training: RNN with Bidirectional GRU")

    if st.button("Start Training"):
        with st.spinner("Training in progress... This may take a few minutes."):
            texts = data['cleaned_text'].values
            # Use the new 'hate_label' as the target
            labels = data['hate_label'].values

            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels)
            num_classes = len(label_encoder.classes_)

            max_words = 10000
            tokenizer = Tokenizer(num_words=max_words)
            tokenizer.fit_on_texts(texts)
            sequences = tokenizer.texts_to_sequences(texts)
            padded_sequences = pad_sequences(sequences, maxlen=st.session_state.max_len)

            X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels)
            
            embedding_dim = 128
            gru_units = 128
            dropout_rate = 0.4
            
            input_layer = Input(shape=(st.session_state.max_len,))
            embedding_layer = Embedding(max_words, embedding_dim, input_length=st.session_state.max_len)(input_layer)
            embedding_dropout = Dropout(dropout_rate)(embedding_layer)
            gru_layer = Bidirectional(GRU(gru_units, return_sequences=False))(embedding_dropout)
            gru_dropout = Dropout(dropout_rate)(gru_layer)
            dense_layer = Dense(64, activation='relu')(gru_dropout)
            output = Dense(num_classes, activation='softmax')(dense_layer)
            
            model = Model(inputs=input_layer, outputs=output)
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            st.write("Model Summary:")
            model_summary_list = []
            model.summary(print_fn=lambda x: model_summary_list.append(x))
            st.text("\n".join(model_summary_list))

            history = model.fit(X_train, y_train, batch_size=64, epochs=5, validation_split=0.1)
            
            # Save model, tokenizer, and encoder to session state
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.label_encoder = label_encoder

            st.success("Model training completed!")

            st.subheader("Model Evaluation")
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            st.write(f"Test Accuracy: {accuracy:.4f}")
            st.write(f"Test Loss: {loss:.4f}")

            y_pred_probs = model.predict(X_test)
            y_pred = np.argmax(y_pred_probs, axis=1)

            st.text("Classification Report:")
            report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).transpose())

            st.text("Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig_cm)

            st.subheader("Training History")
            fig_hist, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.plot(history.history['accuracy'], label='Train Accuracy')
            ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
            ax1.set_title('Model Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()

            ax2.plot(history.history['loss'], label='Train Loss')
            ax2.plot(history.history['val_loss'], label='Validation Loss')
            ax2.set_title('Model Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            st.pyplot(fig_hist)

if st.session_state.model is not None:
    st.subheader("Test with Custom Input")
    custom_text = st.text_area("Enter Bengali text to classify:")
    if st.button("Classify Text"):
        if custom_text:
            cleaned_text = preprocess_text(custom_text)
            sequence = st.session_state.tokenizer.texts_to_sequences([cleaned_text])
            padded_sequence = pad_sequences(sequence, maxlen=st.session_state.max_len)
            
            prediction_probs = st.session_state.model.predict(padded_sequence)
            prediction = np.argmax(prediction_probs, axis=1)
            
            predicted_label = st.session_state.label_encoder.inverse_transform(prediction)[0]
            st.success(f"Predicted Label: **{predicted_label}**")
        else:
            st.warning("Please enter some text to classify.")

else:
    if data is None:
        st.warning("Could not load the dataset. Please ensure 'bengali.csv' is in the correct directory.")

