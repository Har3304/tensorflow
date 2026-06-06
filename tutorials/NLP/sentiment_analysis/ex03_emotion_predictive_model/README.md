# Emotion Prediction NLP Model

A Natural Language Processing (NLP) project built with TensorFlow and BiLSTM networks to classify the emotion expressed in a text sentence.

## Features

* Text preprocessing using TensorFlow Tokenizer
* Bidirectional LSTM architecture
* Multi-class emotion classification
* Softmax probability output
* Interactive inference script
* TensorFlow/Keras model serialization

## Dataset

The model was trained using a combined emotion dataset containing labeled sentences and their corresponding emotions.

## Model Download

The trained model is hosted separately due to GitHub file size limitations.

Download the model from:

**Google Drive:**
https://drive.google.com/file/d/1fPR9kX0C-k2LgyZEbfnqbZMabze90zdd/view?usp=drive_link

After downloading, place the following files in the project root directory:

```
emotion_prediction_nlp_model.keras
emotion_tokenizer.pkl
emotion_labels.pkl
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Har3304/REPOSITORY_NAME.git
cd REPOSITORY_NAME
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Run the training script:

```bash
python train.py
```

This will generate:

```
emotion_prediction_nlp_model.keras
emotion_tokenizer.pkl
emotion_labels.pkl
```

## Inference

Run:

```bash
python inference.py
```

Example:

```
Enter text: I am feeling fantastic today

Predicted Emotion: joy
Confidence: 98.52%
```

## Model Architecture

```
Input Text
    ↓
Tokenizer
    ↓
Embedding Layer (64)
    ↓
Bidirectional LSTM (128)
    ↓
Dropout (0.3)
    ↓
Dense (64, ReLU)
    ↓
Dense (32, ReLU)
    ↓
Softmax Output
```

## Project Structure

```
.
├── train.py
├── inference.py
├── requirements.txt
├── emotion_prediction_nlp_model.keras
├── emotion_tokenizer.pkl
├── emotion_labels.pkl
├── combined_emotion.csv
└── README.md
```

## Results

The model achieves strong performance on the held-out test dataset and can predict emotions from arbitrary user text.

## License

MIT License
