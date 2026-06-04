import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, r2_score)

df = pd.read_csv('sentiment_analysis.csv', encoding='latin-1')

pd.set_option('display.max_rows', None)
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
df['weekday'] = df['Date'].dt.day_of_week
df['day_of_year'] = df['Date'].dt.day_of_year
df['sin_weekday'] = np.sin(2 * np.pi * df['weekday'] / 7)
df['cos_weekday'] = np.cos(2 * np.pi * df['weekday'] / 7)
df['sin_yearday'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
df['cos_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

df.dropna(inplace=True)
df.sort_values(by='Date', ascending=True, inplace=True)
df.drop('Date', axis=1, inplace=True)
df = pd.get_dummies(df, columns=['Time of Tweet', 'Platform'], dtype=int)
df_train = df.sample(frac=0.8, random_state=0)
df_test = df.drop(df_train.index)

encoder = {k:i for i,k in enumerate(sorted(np.unique(df['sentiment'])))}
decoder = {v:k for k,v in encoder.items()}

df_train['sentiment'] = (df_train['sentiment'].map(encoder))
df_test['sentiment'] = (df_test['sentiment'].map(encoder))

tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<OOV>')

tokenizer.fit_on_texts(df_train['text'])

train_sequences = (tokenizer.texts_to_sequences(df_train['text']))

max_len = __builtins__.max(len(s) for s in train_sequences)

X_text_train = (tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=max_len, padding='post'))

test_sequences = (tokenizer.texts_to_sequences(df_test['text']))

X_text_test = (tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=max_len, padding='post'))
X_meta_train = (df_train.drop(['text', 'sentiment'],axis=1).astype(np.float32).values)
X_meta_test = (df_test.drop(['text', 'sentiment'], axis=1).astype(np.float32).values)
y_train = tf.keras.utils.to_categorical(df_train['sentiment'],num_classes=3)

y_test = tf.keras.utils.to_categorical(df_test['sentiment'],num_classes=3)

vocab_size = (len(tokenizer.word_index)+ 1)

text_input = tf.keras.layers.Input(shape=(max_len,),name='text')
x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64)(text_input)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(x)
meta_input = tf.keras.layers.Input(shape=(X_meta_train.shape[1],), name='meta')

m = tf.keras.layers.Dense(32, activation='relu')(meta_input)
combined = tf.keras.layers.Concatenate()([x, m])

z = tf.keras.layers.Dense(64,activation='relu')(combined)
z = tf.keras.layers.Dropout(0.3)(z)

output = tf.keras.layers.Dense(3,activation='softmax')(z)
model = tf.keras.models.Model(inputs=[text_input, meta_input],outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit([X_text_train, X_meta_train], y_train,validation_data=([X_text_test, X_meta_test],y_test), epochs=200, batch_size=32)
loss, accuracy = model.evaluate([X_text_test,X_meta_test],y_test,verbose=0)
print('\nTest Accuracy:',accuracy)

predictions = model.predict([X_text_test, X_meta_test])
y_pred = np.argmax(predictions,axis=1)
y_true = np.argmax(y_test,axis=1)
print('\nClassification Report\n')
print(classification_report(y_true, y_pred))
print('\nConfusion Matrix\n')

cm = confusion_matrix(y_true, y_pred)
print(cm)
print('\nR2 Score:', r2_score(y_true, y_pred))

plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.figure(figsize=(6,6))
plt.imshow(cm)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0,1,2],[decoder[0], decoder[1], decoder[2]])
plt.yticks([0,1,2],[decoder[0], decoder[1], decoder[2]])

plt.xlabel('Predicted')
plt.ylabel('Actual')

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i,cm[i,j],ha='center',va='center')

plt.show()
