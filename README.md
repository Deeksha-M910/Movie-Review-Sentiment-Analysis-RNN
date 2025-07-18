# 🎬 IMDB Movie Review Sentiment Analysis

A deep learning project to classify the sentiment of IMDB movie reviews as **Positive** or **Negative** using a **Simple RNN model**, built with **TensorFlow/Keras** and deployed via **Streamlit**.

---

## 🧩 Features
- Takes movie review text as input.
- Preprocesses input using IMDB word index mapping and padding.
- Predicts sentiment using a trained **Simple RNN model**.
- Displays sentiment with confidence probability.
- Interactive UI built with **Streamlit**.

---

## 🛠 Tech Stack
- **Python**
- **TensorFlow / Keras** — for building and training the RNN
- **Streamlit** — for deploying an interactive web app
- **NumPy, Pandas** — data manipulation
- **Matplotlib, Seaborn** — for EDA

---

## 📁 Dataset
- **Source:** IMDB Movie Reviews Dataset (via Keras Datasets)
- **Size:** 50,000 reviews labeled as positive or negative
- **Usage:** Preprocessed into sequences of integers representing word indexes.

---

## 🧠 Model Details
- **Architecture:** Simple RNN → Dense Layers → Sigmoid Output
- **Training:** Conducted on pre-tokenized IMDB data.
- **Evaluation Metric:** Accuracy on validation/test data.
- **Saved Model:** `simple_rnn_imdb.h5`

## 🗂 Project Structure

- **main.py**: Streamlit app for running the sentiment prediction interface
- **simple_rnn_imdb.h5**: Pre-trained Simple RNN model on IMDB dataset
- **requirements.txt**: List of Python libraries and dependencies
- **embedding.ipynb**: Notebook for exploring word embeddings
- **simplernn.ipynb**: Notebook for building, training, and evaluating the Simple RNN model
- **prediction.ipynb**: Notebook for testing and validating the prediction pipeline

---