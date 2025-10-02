


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

# --- Page Configuration ---
st.set_page_config(layout="wide")
st.title("Bengali Hate Speech Detection")

# --- Session State Initialization ---
def init_session_state():
    for key in ['model', 'tokenizer', 'label_encoder', 'current_task']:
        if key not in st.session_state:
            st.session_state[key] = None
    if 'max_len' not in st.session_state:
        st.session_state.max_len = 150

init_session_state()

# --- Sidebar for Task Selection ---
st.sidebar.title("Select Classification Task")
task = st.sidebar.radio(
    "Choose a model to train and test:",
    ('Hate vs. Not Hate (Binary)', 'Multi-Class Hate Speech')
)
st.sidebar.info("This project is made by Rutuja Godse, Drishti and team.")


# Clear session state if task changes to ensure the correct model is used
if st.session_state.current_task != task:
    init_session_state()
    st.session_state.current_task = task

# --- Data Loading Functions ---
@st.cache_data
def load_binary_data():
    file_path = 'bengali.csv'
    try:
        df = pd.read_csv(file_path)
        df.dropna(subset=['sentence', 'hate'], inplace=True)
        df['label'] = df['hate'].map({0: 'Not Hate', 1: 'Hate'})
        df['text'] = df['sentence']
        return df[['text', 'label']]
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        st.error(f"Error loading data from {file_path}: {e}")
        return None

@st.cache_data
def load_multiclass_data():
    file_path = 'bengali2.csv'
    try:
        df = pd.read_csv(file_path)
        df.dropna(subset=['text', 'label'], inplace=True)
        return df[['text', 'label']]
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        st.error(f"Error loading data from {file_path}: {e}")
        return None

# --- Text Preprocessing ---
bengali_stopwords = [
    'এবং', 'ও', 'আর', 'বা', 'কিংবা', 'হয়', 'হচ্ছে', 'হয়েছে', 'ছিল', 'আছে', 'এই',
    'সেই', 'যে', 'সে', 'কি', 'কে', 'কোন', 'কার', 'কাকে', 'আমি', 'তুমি', 'সে',
    'আমার', 'তোমার', 'তার', 'আমাদের', 'তোমাদের', 'তাদের', 'জন্য', 'থেকে', 'সঙ্গে',
    'দ্বারা', 'কাছে', 'দিকে', 'মধ্যে', 'উপরে', 'নিচে', 'পরে', 'আগে', 'এক', 'দুই'
]

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^\u0980-\u09FF\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    return " ".join([word for word in tokens if word not in bengali_stopwords])

# --- Model Creation Function ---
def create_model(num_classes, max_len, is_multiclass=False):
    input_layer = Input(shape=(max_len,))
    embedding_layer = Embedding(10000, 128, input_length=max_len)(input_layer)
    x = Dropout(0.4)(embedding_layer)

    if is_multiclass:
        # Deeper, more powerful model for the complex multi-class task
        x = Bidirectional(GRU(128, return_sequences=True))(x)
        x = Dropout(0.4)(x)
        x = Bidirectional(GRU(64, return_sequences=False))(x)
        x = Dropout(0.4)(x)
        x = Dense(64, activation='relu')(x)
    else:
        # Standard model for the simpler binary task
        x = Bidirectional(GRU(128, return_sequences=False))(x)
        x = Dropout(0.4)(x)
        x = Dense(64, activation='relu')(x)

    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# --- Main Application Logic ---
is_multiclass_task = False
if task == 'Hate vs. Not Hate (Binary)':
    st.header("Task: Hate vs. Not Hate Classification")
    data = load_binary_data()
else: # Multi-Class Hate Speech
    st.header("Task: Multi-Class Hate Speech Classification")
    data = load_multiclass_data()
    is_multiclass_task = True


if data is not None:
    # --- UI Columns for Data Display ---
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
        sns.countplot(x='label', data=data, ax=ax, order = data['label'].value_counts().index)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    # --- Preprocessing and Word Cloud ---
    st.subheader("Text Preprocessing")
    data['cleaned_text'] = data['text'].apply(preprocess_text)
    st.write("#### Cleaned Text Preview:")
    st.write(data[['text', 'cleaned_text']].head())

    st.markdown("#### Word Cloud from Hate Speech Text")
    hate_speech_text = " ".join(text for text in data[data['label'] != 'Not Hate']['cleaned_text']).strip()
    
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

    # --- Model Training ---
    st.subheader("Model Training: RNN with Bidirectional GRU")
    if st.button("Start Training"):
        
        epochs = 10 if is_multiclass_task else 5
        spinner_message = f"Training for {epochs} epochs. This may take a while..." if is_multiclass_task else "Training in progress..."

        with st.spinner(spinner_message):
            texts = data['cleaned_text'].values
            labels = data['label'].values

            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels)
            num_classes = len(label_encoder.classes_)

            tokenizer = Tokenizer(num_words=10000)
            tokenizer.fit_on_texts(texts)
            sequences = tokenizer.texts_to_sequences(texts)
            padded_sequences = pad_sequences(sequences, maxlen=st.session_state.max_len)

            X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels)
            
            model = create_model(num_classes, st.session_state.max_len, is_multiclass=is_multiclass_task)
            
            st.write("Model Summary:")
            model_summary_list = []
            model.summary(print_fn=lambda x: model_summary_list.append(x))
            st.text("\n".join(model_summary_list))

            history = model.fit(X_train, y_train, batch_size=64, epochs=epochs, validation_split=0.1)
            
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.label_encoder = label_encoder
            st.success("Model training completed!")

            # --- Model Evaluation ---
            st.subheader("Model Evaluation")
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            st.write(f"Test Accuracy: {accuracy:.4f}")

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
            ax1.legend()
            ax2.plot(history.history['loss'], label='Train Loss')
            ax2.plot(history.history['val_loss'], label='Validation Loss')
            ax2.set_title('Model Loss')
            ax2.set_xlabel('Epoch')
            ax2.legend()
            st.pyplot(fig_hist)

# --- Custom Input Testing ---
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


# import streamlit as st
# import pandas as pd
# import numpy as np
# import re
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import classification_report, confusion_matrix
# import tensorflow as tf
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Dense, Dropout
# from tensorflow.keras.models import Model
# import matplotlib.pyplot as plt
# import seaborn as sns
# from wordcloud import WordCloud
# import requests
# from io import BytesIO
# import tempfile
# import os

# # --- Page Configuration ---
# st.set_page_config(layout="wide")
# st.title("Bengali Hate Speech Detection")

# # --- Session State Initialization ---
# def init_session_state():
#     for key in ['model', 'tokenizer', 'label_encoder', 'current_task']:
#         if key not in st.session_state:
#             st.session_state[key] = None
#     if 'max_len' not in st.session_state:
#         st.session_state.max_len = 150

# init_session_state()

# # --- Sidebar for Task Selection ---
# st.sidebar.title("Select Classification Task")
# task = st.sidebar.radio(
#     "Choose a model to train and test:",
#     ('Hate vs. Not Hate (Binary)', 'Multi-Class Hate Speech')
# )

# # Clear session state if task changes to ensure the correct model is used
# if st.session_state.current_task != task:
#     init_session_state()
#     st.session_state.current_task = task

# # --- Data Loading Functions ---
# @st.cache_data
# def load_binary_data():
#     file_path = 'bengali.csv'
#     try:
#         df = pd.read_csv(file_path)
#         df.dropna(subset=['sentence', 'hate'], inplace=True)
#         df['label'] = df['hate'].map({0: 'Not Hate', 1: 'Hate'})
#         df['text'] = df['sentence']
#         return df[['text', 'label']]
#     except FileNotFoundError:
#         st.error(f"Error: The file '{file_path}' was not found.")
#         return None
#     except Exception as e:
#         st.error(f"Error loading data from {file_path}: {e}")
#         return None

# @st.cache_data
# def load_multiclass_data():
#     file_path = 'bengali2.csv'
#     try:
#         df = pd.read_csv(file_path)
#         df.dropna(subset=['text', 'label'], inplace=True)
#         return df[['text', 'label']]
#     except FileNotFoundError:
#         st.error(f"Error: The file '{file_path}' was not found.")
#         return None
#     except Exception as e:
#         st.error(f"Error loading data from {file_path}: {e}")
#         return None

# # --- Text Preprocessing ---
# bengali_stopwords = [
#     'এবং', 'ও', 'আর', 'বা', 'কিংবা', 'হয়', 'হচ্ছে', 'হয়েছে', 'ছিল', 'আছে', 'এই',
#     'সেই', 'যে', 'সে', 'কি', 'কে', 'কোন', 'কার', 'কাকে', 'আমি', 'তুমি', 'সে',
#     'আমার', 'তোমার', 'তার', 'আমাদের', 'তোমাদের', 'তাদের', 'জন্য', 'থেকে', 'সঙ্গে',
#     'দ্বারা', 'কাছে', 'দিকে', 'মধ্যে', 'উপরে', 'নিচে', 'পরে', 'আগে', 'এক', 'দুই'
# ]

# def preprocess_text(text):
#     if not isinstance(text, str): return ""
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
#     text = re.sub(r'\@\w+|\#','', text)
#     text = re.sub(r'[^\u0980-\u09FF\s]', '', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     tokens = text.split()
#     return " ".join([word for word in tokens if word not in bengali_stopwords])

# # --- Model Creation Function ---
# def create_model(num_classes, max_len, is_multiclass=False):
#     input_layer = Input(shape=(max_len,))
#     embedding_layer = Embedding(10000, 128, input_length=max_len)(input_layer)
#     x = Dropout(0.4)(embedding_layer)

#     if is_multiclass:
#         # Deeper, more powerful model for the complex multi-class task
#         x = Bidirectional(GRU(128, return_sequences=True))(x)
#         x = Dropout(0.4)(x)
#         x = Bidirectional(GRU(64, return_sequences=False))(x)
#         x = Dropout(0.4)(x)
#         x = Dense(64, activation='relu')(x)
#     else:
#         # Standard model for the simpler binary task
#         x = Bidirectional(GRU(128, return_sequences=False))(x)
#         x = Dropout(0.4)(x)
#         x = Dense(64, activation='relu')(x)

#     output = Dense(num_classes, activation='softmax')(x)
#     model = Model(inputs=input_layer, outputs=output)
#     model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model


# # --- Main Application Logic ---
# is_multiclass_task = False
# if task == 'Hate vs. Not Hate (Binary)':
#     st.header("Task: Hate vs. Not Hate Classification")
#     data = load_binary_data()
# else: # Multi-Class Hate Speech
#     st.header("Task: Multi-Class Hate Speech Classification")
#     data = load_multiclass_data()
#     is_multiclass_task = True


# if data is not None:
#     # --- UI Columns for Data Display ---
#     col1, col2 = st.columns(2)
#     with col1:
#         st.subheader("Dataset Preview")
#         st.write(data.head())
#         st.subheader("Dataset Description")
#         st.write(f"The dataset contains {data.shape[0]} rows and {data.shape[1]} columns.")
#         st.write("Columns:", ", ".join(data.columns))

#     with col2:
#         st.subheader("Data Visualization")
#         st.markdown("#### Distribution of Labels")
#         fig, ax = plt.subplots()
#         sns.countplot(x='label', data=data, ax=ax, order = data['label'].value_counts().index)
#         plt.xticks(rotation=45, ha='right')
#         plt.tight_layout()
#         st.pyplot(fig)

#     # --- Preprocessing and Word Cloud ---
#     st.subheader("Text Preprocessing")
#     data['cleaned_text'] = data['text'].apply(preprocess_text)
#     st.write("#### Cleaned Text Preview:")
#     st.write(data[['text', 'cleaned_text']].head())

#     st.markdown("#### Word Cloud from Hate Speech Text")
#     hate_speech_text = " ".join(text for text in data[data['label'] != 'Not Hate']['cleaned_text']).strip()
    
#     if hate_speech_text:
#         font_url = 'https://github.com/google/fonts/raw/main/ofl/solaimanlipi/SolaimanLipi.ttf'
#         try:
#             response = requests.get(font_url)
#             with tempfile.NamedTemporaryFile(delete=False, suffix='.ttf') as fp:
#                 fp.write(response.content)
#                 font_path = fp.name
            
#             wordcloud = WordCloud(
#                 font_path=font_path, width=800, height=400,
#                 background_color='white', collocations=False
#             ).generate(hate_speech_text)
            
#             fig_wc, ax_wc = plt.subplots()
#             ax_wc.imshow(wordcloud, interpolation='bilinear')
#             ax_wc.axis('off')
#             st.pyplot(fig_wc)
#             os.remove(font_path)
#         except Exception as e:
#             st.warning(f"Could not generate word cloud. Error: {e}")
#     else:
#         st.info("No words found to generate a word cloud for the hate speech category after preprocessing.")

#     # --- Model Training ---
#     st.subheader("Model Training: RNN with Bidirectional GRU")
#     if st.button("Start Training"):
        
#         epochs = 10 if is_multiclass_task else 5
#         spinner_message = f"Training for {epochs} epochs. This may take a while..." if is_multiclass_task else "Training in progress..."

#         with st.spinner(spinner_message):
#             texts = data['cleaned_text'].values
#             labels = data['label'].values

#             label_encoder = LabelEncoder()
#             encoded_labels = label_encoder.fit_transform(labels)
#             num_classes = len(label_encoder.classes_)

#             tokenizer = Tokenizer(num_words=10000)
#             tokenizer.fit_on_texts(texts)
#             sequences = tokenizer.texts_to_sequences(texts)
#             padded_sequences = pad_sequences(sequences, maxlen=st.session_state.max_len)

#             X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels)
            
#             model = create_model(num_classes, st.session_state.max_len, is_multiclass=is_multiclass_task)
            
#             st.write("Model Summary:")
#             model_summary_list = []
#             model.summary(print_fn=lambda x: model_summary_list.append(x))
#             st.text("\n".join(model_summary_list))

#             history = model.fit(X_train, y_train, batch_size=64, epochs=epochs, validation_split=0.1)
            
#             st.session_state.model = model
#             st.session_state.tokenizer = tokenizer
#             st.session_state.label_encoder = label_encoder
#             st.success("Model training completed!")

#             # --- Model Evaluation ---
#             st.subheader("Model Evaluation")
#             loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
#             st.write(f"Test Accuracy: {accuracy:.4f}")

#             y_pred_probs = model.predict(X_test)
#             y_pred = np.argmax(y_pred_probs, axis=1)

#             st.text("Classification Report:")
#             report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0)
#             st.dataframe(pd.DataFrame(report).transpose())

#             st.text("Confusion Matrix:")
#             cm = confusion_matrix(y_test, y_pred)
#             fig_cm, ax_cm = plt.subplots()
#             sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
#             plt.xlabel("Predicted")
#             plt.ylabel("Actual")
#             st.pyplot(fig_cm)

#             st.subheader("Training History")
#             fig_hist, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
#             ax1.plot(history.history['accuracy'], label='Train Accuracy')
#             ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
#             ax1.set_title('Model Accuracy')
#             ax1.set_xlabel('Epoch')
#             ax1.legend()
#             ax2.plot(history.history['loss'], label='Train Loss')
#             ax2.plot(history.history['val_loss'], label='Validation Loss')
#             ax2.set_title('Model Loss')
#             ax2.set_xlabel('Epoch')
#             ax2.legend()
#             st.pyplot(fig_hist)

# # --- Custom Input Testing ---
# if st.session_state.model is not None:
#     st.subheader("Test with Custom Input")
#     custom_text = st.text_area("Enter Bengali text to classify:")
#     if st.button("Classify Text"):
#         if custom_text:
#             cleaned_text = preprocess_text(custom_text)
#             sequence = st.session_state.tokenizer.texts_to_sequences([cleaned_text])
#             padded_sequence = pad_sequences(sequence, maxlen=st.session_state.max_len)
            
#             prediction_probs = st.session_state.model.predict(padded_sequence)
#             prediction = np.argmax(prediction_probs, axis=1)
            
#             predicted_label = st.session_state.label_encoder.inverse_transform(prediction)[0]
#             st.success(f"Predicted Label: **{predicted_label}**")
#         else:
#             st.warning("Please enter some text to classify.")






