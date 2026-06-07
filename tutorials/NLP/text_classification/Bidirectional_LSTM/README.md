# Bidirectional LSTM Text Classification

A Natural Language Processing (NLP) project built with TensorFlow and Keras that performs text classification using a Bidirectional Long Short-Term Memory (BiLSTM) neural network.

The model processes text sequences in both forward and backward directions, allowing it to capture contextual information more effectively than a standard LSTM.

## Features

* Text preprocessing and tokenization
* Vocabulary generation
* Sequence padding
* Bidirectional LSTM architecture
* Dense classification layers
* Model evaluation with accuracy metrics
* Prediction on custom text samples
* TensorFlow/Keras implementation

## Project Structure

```
Bidirectional_LSTM/
│
├── bidirectional_lstm.py
├── dataset/
│   ├── train.csv
│   └── test.csv
│
├── saved_model/
│   └── model.keras
│
├── requirements.txt
└── README.md
```

## Model Architecture

```
Input Text
     │
Tokenizer
     │
Text Sequences
     │
Padding
     │
Embedding Layer
     │
Bidirectional LSTM
     │
Dropout
     │
Dense Layer
     │
Output Layer
```

The Bidirectional LSTM learns contextual information from both directions of a sentence, improving performance on many NLP classification tasks. TensorFlow recommends BiLSTM architectures for sequence classification problems because they capture dependencies that may appear before or after a given token.

## Installation

Clone the repository:

```bash
git clone https://github.com/Har3304/tensorflow.git
cd tensorflow/tutorials/NLP/text_classification/Bidirectional_LSTM
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The project expects a text classification dataset containing:

| Column | Description             |
| ------ | ----------------------- |
| Text   | Input sentence/document |
| Label  | Target class            |

Example:

```csv
text,label
"This movie is fantastic",positive
"This product is terrible",negative
```

## Training

Run:

```bash
python bidirectional_lstm.py
```

The script will:

1. Load the dataset
2. Preprocess text
3. Tokenize and pad sequences
4. Build the Bidirectional LSTM model
5. Train the network
6. Evaluate performance
7. Save the trained model

## Model Saving

The trained model can be saved as:

```python
model.save("model.keras")
```

Load later with:

```python
import tensorflow as tf

model = tf.keras.models.load_model("model.keras")
```

## Inference Example

```python
sample_text = [
    "This product exceeded my expectations"
]

prediction = model.predict(sample_text)
print(prediction)
```

## Evaluation Metrics

Typical metrics used:

* Accuracy
* Loss
* Validation Accuracy
* Validation Loss

Training history can be visualized using Matplotlib.

## Applications

* Sentiment Analysis
* Product Review Classification
* Social Media Analysis
* News Categorization
* Customer Feedback Analysis
* Document Classification

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Scikit-learn
* Matplotlib

## References

* TensorFlow Text Classification Tutorial: TensorFlow demonstrates Bidirectional LSTM models for NLP classification tasks and sequence learning.

## License

This project is released under the MIT License.

## Author

Harnish Gajjar

GitHub:
https://github.com/Har3304
