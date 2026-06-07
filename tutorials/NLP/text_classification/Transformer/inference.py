import joblib
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('transformer_spam_model.keras')
tokenizer = joblib.load('transformer_tokenizer.pkl')
encoder = joblib.load('transformer_encoder.pkl')
max_len = model.input_shape[1]

def predict_transformer_sms(messages_list):    
    sequences = tokenizer.texts_to_sequences(messages_list)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, 
        padding='post', 
        maxlen=max_len
    )
    raw_predictions = model.predict(padded_sequences)
    
    for text, prob in zip(messages_list, raw_predictions):
        prob_val = float(prob[0])
        class_idx = 1 if prob_val >= 0.5 else 0  
        text_label = encoder.inverse_transform([class_idx])[0]
        
        print(f"\nMessage: {text}")
        print(f"Prediction: {text_label} ({prob_val:.2%} confidence)")
