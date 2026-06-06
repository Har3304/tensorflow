import tensorflow as tf
import pandas as pd
import pickle
import numpy as np

model = tf.keras.models.load_model("emotion_prediction_nlp_model.keras")

with open("emotion_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

df_emotion = pd.read_csv("combined_emotion.csv")
emotion_labels = pd.get_dummies(df_emotion["emotion"]).columns.tolist()

max_len = model.input_shape[1]

while True:

    text = input("Enter text (e to exit): ")

    if text.lower() == "e":
        break

    sequence = tokenizer.texts_to_sequences([text])

    padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequence,
        maxlen=max_len,
        padding="post"
    )

    prediction = model.predict(padded, verbose=0)[0]

    predicted_index = np.argmax(prediction)
    predicted_emotion = emotion_labels[predicted_index]
    confidence = prediction[predicted_index] * 100

    print("\nPredicted Emotion:", predicted_emotion)
    print("Confidence: {:.2f}%".format(confidence))

    print("\nAll Probabilities:")
    for label, prob in zip(emotion_labels, prediction):
        print(f"{label}: {prob*100:.2f}%")

    print()
