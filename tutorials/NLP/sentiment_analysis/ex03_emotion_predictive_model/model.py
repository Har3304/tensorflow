import pandas as pd
import numpy as np
import tensorflow as tf

print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

df_emotion = pd.read_csv('combined_emotion.csv')
df_sentiment = pd.read_csv('combined_sentiment_data.csv')

pd.set_option('display.max_rows', None)
df_emotion.dropna(inplace=True)
df_sentiment.dropna(inplace=True)

print(df_emotion.head())
print(df_sentiment.head())
print(df_emotion.shape)
print(df_sentiment.shape)

# Emotion model
# X tokenization
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(df_emotion['sentence'])

sequences = tokenizer.texts_to_sequences(df_emotion['sentence'])
max_len = max(len(s) for s in sequences)

X = tf.keras.preprocessing.sequence.pad_sequences(
    sequences,
    maxlen=max_len,
    padding='post'
)

y = pd.get_dummies(df_emotion['emotion']).values.astype(np.float32)

print(X.shape, y.shape)

split = len(X) // 10 * 8
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(X_train.shape[1],), name='text'),
        tf.keras.layers.Embedding(
            input_dim=len(tokenizer.word_index) + 1,
            output_dim=64
        ),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                128,
                dropout=0.2,
                recurrent_dropout=0.0
            )
        ),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(
            y_train.shape[1],
            activation='softmax'
        ),
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

model.summary()

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=128
)

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print('\nTest Accuracy:', accuracy)
