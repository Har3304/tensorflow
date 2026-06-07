import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


# Using Bidirectional NLP

df_train = pd.read_csv('SMS_train.csv', encoding='latin1')
df_test = pd.read_csv('SMS_test.csv', encoding='latin1')

df_train.head()



# Tokanization of messages and encoding labels (train_csv)
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(df_train['Message_body'])

sequences = tokenizer.texts_to_sequences(df_train['Message_body'])

max_len = max([len(sen) for sen in sequences])
padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post', maxlen=max_len)

encoder = LabelEncoder()
y = encoder.fit_transform(df_train['Label'])

# Tokanization of messages and encoding labels (test_csv)

sequences_test = tokenizer.texts_to_sequences(df_test['Message_body'])

padded_test = tf.keras.preprocessing.sequence.pad_sequences(sequences_test, padding='post', maxlen=max_len)

y_test = encoder.transform(df_test['Label'])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)

model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_len,)),
        tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=64),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history_LSTM = model.fit(padded, y, epochs=100, callbacks=[callback], validation_data=(padded_test, y_test), verbose=1)# Visualize

tf.keras.models.save_model(model, 'Bideirectional_LSTM_spam_mail_model.keras')
joblib.dump(tokenizer, 'tokenizer.pkl')
joblib.dump(encoder, 'encoder.pkl')


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history_LSTM.history['accuracy'], label='Train')
plt.plot(history_LSTM.history['val_accuracy'], label='Test')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_LSTM.history['loss'], label='Train')
plt.plot(history_LSTM.history['val_loss'], label='Test')
plt.legend()
plt.show()

