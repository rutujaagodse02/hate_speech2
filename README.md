Bengali Hate Speech Detection
Overview
This project is an interactive web application built with Streamlit for detecting and classifying hate speech in Bengali text. It leverages Recurrent Neural Networks (RNNs) to perform two distinct classification tasks:

Binary Classification: Identifies text as either "Hate" or "Not Hate".

Multi-Class Classification: Categorizes hate speech into more specific types such as "Personal", "Political", "Religious", and "Geopolitical".

The application provides a user-friendly interface to load data, visualize label distributions, preprocess text, train a model in real-time, evaluate its performance, and test it with custom user input.

Features
Dual-Task Functionality: Easily switch between binary and multi-class hate speech classification tasks via a simple sidebar menu.

Adaptive Model Architecture: Utilizes a standard Bidirectional GRU model for the binary task and a more powerful, stacked Bidirectional GRU model for the complex multi-class task.

Interactive Data Visualization: Displays dataset previews and visualizes the distribution of labels using bar charts.

Bengali Text Preprocessing: Includes a custom preprocessing pipeline to clean text by removing URLs, mentions, punctuation, and common Bengali stopwords.

Word Cloud Generation: Creates a word cloud from the hate speech text to highlight the most frequently used terms.

Real-Time Training & Evaluation: Train the model with a single click and view the training progress, accuracy, and loss curves.

Comprehensive Performance Metrics: Evaluates the trained model and displays the test accuracy, a detailed classification report, and a confusion matrix.

Custom Input Testing: A dedicated section to test the trained model with your own Bengali text inputs.

Datasets
This application is designed to work with two separate datasets:

bengali.csv: Used for the Binary Classification task.

sentence: The raw Bengali text.

hate: A binary label (1 for Hate, 0 for Not Hate).

bengali2.csv: Used for the Multi-Class Classification task.

text: The raw Bengali text.

label: A categorical label (e.g., "Personal", "Political", "Religious").

Note: Please ensure these CSV files are present in the root directory of the project.

Setup and Installation
To run this project on your local machine, please follow these steps:

1. Clone the Repository:

git clone [https://github.com/your-username/bengali-hate-speech-detection.git](https://github.com/your-username/bengali-hate-speech-detection.git)
cd bengali-hate-speech-detection

2. Create a Virtual Environment (Recommended):
It's a good practice to create a virtual environment to manage project dependencies.

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate

3. Install Required Libraries:
A requirements.txt file is included with all the necessary libraries.

pip install -r requirements.txt

4. Run the Streamlit Application:
Once the dependencies are installed, you can run the app with the following command:

streamlit run app.py

Your web browser should automatically open a new tab with the application running.

How to Use the Application
Select the Task: Use the sidebar on the left to choose between "Hate vs. Not Hate (Binary)" and "Multi-Class Hate Speech" classification.

Explore the Data: The app will load the corresponding dataset and display a preview, description, and a chart showing the label distribution.

Train the Model: Click the "Start Training" button. The application will preprocess the data, build the appropriate RNN model, and train it. You will see the model summary and can monitor the training progress.

Review Evaluation: After training is complete, the app will display the model's performance on the test set, including accuracy, a classification report, and a confusion matrix.

Test with Custom Input: Scroll down to the "Test with Custom Input" section, enter any Bengali text into the text area, and click "Classify Text" to get a prediction from the trained model.

Technologies Used
Backend: Python

Web Framework: Streamlit

Deep Learning: TensorFlow, Keras

Data Manipulation: Pandas, NumPy

Machine Learning: Scikit-learn

Data Visualization: Matplotlib, Seaborn, WordCloud
