import numpy as np
import tensorflow as tf

texts = [
    "I love this movie",
    "This film was amazing",
    "What a wonderful experience",
    "I really enjoyed it",
    "Absolutely fantastic",

    "I hate this movie",
    "This film was terrible",
    "What a bad experience",
    "I really disliked it",
    "Absolutely awful"
]

labels = [1,1,1,1,1,0,0,0,0,0]

vocab_size = 1000
max_len = 10

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

print(sequences)

padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len, padding='post')
print(padded)
X = np.array(padded)
y = np.array(labels)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(100, 128),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])

model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=50, validation_split=0.2, verbose=1)

while True:
    text = input("Enter a text (enter e to exit): ")
    if text.lower()=='e':
      break
    sequence = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_len, padding='post')
    prediction = model.predict(padded, verbose=0)[0][0]
    print(f'Confidence: {round(float(prediction)), 4}')
    
    if prediction >=0.5:
      print('Sentiment: Positive')
    else:
      print('Sentiment: Negative')
