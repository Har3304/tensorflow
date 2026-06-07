# Inference
import joblib
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('Bideirectional_LSTM_spam_mail_model.keras')
tokenizer = joblib.load('tokenizer.pkl')
encoder = joblib.load('encoder.pkl')
max_len = model.input_shape[1]

def predict_spam(messages_list):
  sequence = tokenizer.texts_to_sequences([messages_list])
  padded = tf.keras.preprocessing.sequence.padded_sequences(sequence, padding='post', maxlen=max_len)
  raw_predictions = model.predict(padded)
  for text, prob in zip(messages_list, raw_predictions):
    class_idx = 1 if prob > 0.5 else 0
    text_label = encoder.inverse_transform(class_idx)[0]
    print('/nMessage: ', text)
    print('Prediction: ', text_label)
    print('Probability: ', prob)
