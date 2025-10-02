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
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Dense
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import requests
from io import BytesIO

# Title for the Streamlit app
st.title("Bengali Hate Speech Detection using Capsule Network")
st.markdown("This application trains a Capsule Network with GRU model to detect hate speech in Bengali text.")

# Function to load data from a raw GitHub URL
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/rezacsedu/bengali-hate-speech-with-explicitness/main/bengali_hate_speech.csv'
    try:
        df = pd.read_csv(url)
        # Drop rows with missing values in 'text' or 'label'
        df.dropna(subset=['text', 'label'], inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

data = load_data()

if data is not None:
    st.subheader("Dataset Preview")
    st.write(data.head())

    st.subheader("Dataset Description")
    st.write(f"The dataset contains {data.shape[0]} rows and {data.shape[1]} columns.")
    st.write("Columns:", ", ".join(data.columns))

    # --- Data Visualization ---
    st.subheader("Data Visualization")

    # Label Distribution
    st.markdown("#### Distribution of Labels")
    fig, ax = plt.subplots()
    sns.countplot(x='label', data=data, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # --- Text Preprocessing ---
    st.subheader("Text Preprocessing")
    
    # Define Bengali stopwords
    bengali_stopwords = [
        'এবং', 'ও', 'আর', 'বা', 'কিংবা', 'হয়', 'হচ্ছে', 'হয়েছে', 'ছিল', 'আছে', 'এই',
        'সেই', 'যে', 'সে', 'কি', 'কে', 'কোন', 'কার', 'কাকে', 'আমি', 'তুমি', 'সে',
        'আমার', 'তোমার', 'তার', 'আমাদের', 'তোমাদের', 'তাদের', 'জন্য', 'থেকে', 'সঙ্গে',
        'দ্বারা', 'কাছে', 'দিকে', 'মধ্যে', 'উপরে', 'নিচে', 'পরে', 'আগে', 'এক', 'দুই'
    ]

    def preprocess_text(text):
        # Remove URLs, mentions, and hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#','', text)
        # Remove punctuation and numbers
        text = re.sub(r'[^\u0980-\u09FF\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove stopwords
        tokens = text.split()
        tokens = [word for word in tokens if word not in bengali_stopwords]
        return " ".join(tokens)

    data['cleaned_text'] = data['text'].apply(preprocess_text)
    
    st.markdown("Applied the following preprocessing steps:")
    st.markdown("- Removed URLs, mentions, and hashtags.")
    st.markdown("- Removed punctuation and numbers, keeping only Bengali characters.")
    st.markdown("- Removed common Bengali stopwords.")
    st.markdown("- Removed extra whitespace.")

    st.write("#### Cleaned Text Preview:")
    st.write(data[['text', 'cleaned_text']].head())

    # --- Word Cloud ---
    st.markdown("#### Word Cloud from Hate Speech Text")
    hate_speech_text = " ".join(text for text in data[data['label'] != 'Not hate']['cleaned_text'])
    
    # Download a Bengali font
    font_url = 'https://github.com/google/fonts/raw/main/ofl/solaimanlipi/SolaimanLipi.ttf'
    try:
        response = requests.get(font_url)
        font_bytes = BytesIO(response.content)

        wordcloud = WordCloud(
            font_path=font_bytes,
            width=800,
            height=400,
            background_color='white',
            collocations=False
        ).generate(hate_speech_text)

        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)
    except Exception as e:
        st.warning(f"Could not generate word cloud. Error: {e}")


    # --- Model Training ---
    st.subheader("Model Training: Capsule Network with GRU")

    if st.button("Start Training"):
        with st.spinner("Training in progress... This may take a few minutes."):
            # Prepare data for model
            texts = data['cleaned_text'].values
            labels = data['label'].values

            # Encode labels
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels)
            num_classes = len(label_encoder.classes_)

            # Tokenize and pad sequences
            max_words = 10000
            max_len = 150
            tokenizer = Tokenizer(num_words=max_words)
            tokenizer.fit_on_texts(texts)
            sequences = tokenizer.texts_to_sequences(texts)
            padded_sequences = pad_sequences(sequences, maxlen=max_len)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels)
            
            # --- Capsule Network Layer Definition ---
            class CapsuleLayer(Layer):
                def __init__(self, num_capsules, dim_capsule, routings=3, **kwargs):
                    super(CapsuleLayer, self).__init__(**kwargs)
                    self.num_capsules = num_capsules
                    self.dim_capsule = dim_capsule
                    self.routings = routings

                def build(self, input_shape):
                    self.input_num_capsules = input_shape[1]
                    self.input_dim_capsule = input_shape[2]
                    self.W = self.add_weight(shape=[self.num_capsules, self.input_num_capsules, self.dim_capsule, self.input_dim_capsule],
                                             initializer='glorot_uniform',
                                             name='W')
                    self.built = True

                def call(self, inputs, training=None):
                    inputs_expand = K.expand_dims(inputs, 1)
                    inputs_tiled = K.tile(inputs_expand, [1, self.num_capsules, 1, 1])
                    inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)

                    b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsules, self.input_num_capsules])

                    for i in range(self.routings):
                        c = tf.nn.softmax(b, axis=1)
                        outputs = self.squash(K.batch_dot(c, inputs_hat, [2, 2]))
                        if i < self.routings - 1:
                            b += K.batch_dot(outputs, inputs_hat, [2, 3])
                    return outputs

                def compute_output_shape(self, input_shape):
                    return tuple([None, self.num_capsules, self.dim_capsule])

                @staticmethod
                def squash(vectors, axis=-1):
                    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
                    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
                    return scale * vectors

            # --- Build the Model ---
            embedding_dim = 128
            gru_units = 64
            
            input_layer = Input(shape=(max_len,))
            embedding_layer = Embedding(max_words, embedding_dim)(input_layer)
            gru_layer = Bidirectional(GRU(gru_units, return_sequences=True))(embedding_layer)
            
            capsule = CapsuleLayer(num_capsules=num_classes, dim_capsule=16, routings=3)(gru_layer)
            
            # Use a custom layer to compute the length of the capsule vectors
            def Length(x):
                return tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1))

            output = Dense(num_classes, activation='softmax')(tf.keras.layers.Lambda(Length)(capsule))
            
            model = Model(inputs=input_layer, outputs=output)
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            st.write("Model Summary:")
            model_summary_list = []
            model.summary(print_fn=lambda x: model_summary_list.append(x))
            st.text("\n".join(model_summary_list))

            # Train the model
            history = model.fit(X_train, y_train, batch_size=64, epochs=5, validation_split=0.1)

            st.success("Model training completed!")

            # --- Evaluation ---
            st.subheader("Model Evaluation")
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            st.write(f"Test Accuracy: {accuracy:.4f}")
            st.write(f"Test Loss: {loss:.4f}")

            # Predictions
            y_pred_probs = model.predict(X_test)
            y_pred = np.argmax(y_pred_probs, axis=1)

            # Classification Report
            st.text("Classification Report:")
            report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

            # Confusion Matrix
            st.text("Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig_cm)

            # Plot Training History
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


else:
    st.warning("Could not load the dataset. Please check the URL or your connection.")
