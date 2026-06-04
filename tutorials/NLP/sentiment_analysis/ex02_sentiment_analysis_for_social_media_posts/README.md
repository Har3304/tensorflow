# Sentiment Analysis for Social Media Posts

A TensorFlow-based Natural Language Processing (NLP) project for multi-class sentiment classification of social media posts.

The model combines:

* Bidirectional LSTM (BiLSTM) for textual understanding
* Temporal feature engineering from dates
* Platform metadata (Twitter, Instagram, Facebook, etc.)
* Tweet timing information
* Dense neural networks for structured features

The project predicts one of three sentiment classes:

* Negative
* Neutral
* Positive

## Dataset

The dataset contains social media posts along with metadata such as:

| Feature       | Description           |
| ------------- | --------------------- |
| Year          | Year of post          |
| Month         | Month of post         |
| Day           | Day of post           |
| Time of Tweet | Time category of post |
| Platform      | Social media platform |
| Text          | Post content          |
| Sentiment     | Target label          |

## Feature Engineering

Several additional features are generated from the original date fields:

### Date Features

* Weekday
* Day of Year

### Cyclical Encoding

Weekday and yearly seasonality are encoded using sine and cosine transformations:

* sin_weekday
* cos_weekday
* sin_yearday
* cos_year

### Categorical Encoding

The following features are converted into numerical representations using one-hot encoding:

* Platform
* Time of Tweet

## Model Architecture

Text data and metadata are processed separately and merged before classification.

Text Branch:

Text
→ Tokenizer
→ Padding
→ Embedding
→ Bidirectional LSTM

Metadata Branch:

Metadata
→ Dense Layer

Fusion:

Text Features
+
Metadata Features
↓
Concatenate
↓
Dense
↓
Dropout
↓
Softmax Output

## Network Structure

Input Text
→ Embedding (64)

→ Bidirectional LSTM (32)

Metadata Input
→ Dense (32)

Concatenate

→ Dense (64)

→ Dropout (0.3)

→ Dense (3, Softmax)

## Training

The model uses:

* Optimizer: Adam
* Loss Function: Categorical Crossentropy
* Metric: Accuracy

Training Configuration:

* Epochs: 20
* Batch Size: 32
* Validation Split: Test Dataset

## Evaluation Metrics

The project reports:

* Accuracy
* Classification Report
* Precision
* Recall
* F1 Score
* Confusion Matrix
* R² Score

## Visualizations

The following visualizations are generated:

### Accuracy Curve

Shows:

* Training Accuracy
* Validation Accuracy

### Loss Curve

Shows:

* Training Loss
* Validation Loss

### Confusion Matrix

Displays class-wise prediction performance.

## Installation

Clone the repository:

```bash
git clone https://github.com/Har3304/tensorflow.git
```

Navigate to the project:

```bash
cd tutorials/NLP/sentiment_analysis/ex02_sentiment_analysis_for_social_media_posts
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

or

```bash
jupyter notebook
```

and run the notebook version.

## Project Structure

```text
ex02_sentiment_analysis_for_social_media_posts/
│
├── sentiment_analysis.csv
├── main.py
├── requirements.txt
├── README.md
│
└── outputs/
    ├── accuracy.png
    ├── loss.png
    └── confusion_matrix.png
```

## Technologies Used

* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-Learn
* Matplotlib

## Future Improvements

* Attention Mechanism
* Pretrained Word Embeddings
* BERT Integration
* Transformer Encoder Layers
* Hyperparameter Optimization
* Cross Validation
* Model Export (SavedModel / TFLite)

## License

This project is released under the MIT License.
