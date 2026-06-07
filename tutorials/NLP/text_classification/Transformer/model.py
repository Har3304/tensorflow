import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder

# Load data
df_train = pd.read_csv('SMS_train.csv', encoding='latin1')
df_test = pd.read_csv('SMS_test.csv', encoding='latin1')

# Tokenization and padding (Train)
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(df_train['Message_body'])
sequences = tokenizer.texts_to_sequences(df_train['Message_body'])
max_len = max([len(sen) for sen in sequences])
padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post', maxlen=max_len)

encoder = LabelEncoder()
y = encoder.fit_transform(df_train['Label'])

# Tokenization and padding (Test)
sequences_test = tokenizer.texts_to_sequences(df_test['Message_body'])
padded_test = tf.keras.preprocessing.sequence.pad_sequences(sequences_test, padding='post', maxlen=max_len)
y_test = encoder.transform(df_test['Label'])

# --- Custom Transformer Components ---

class TransformerEmbedding(layers.Layer):
    """Combines Word Embeddings with Position Embeddings."""
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class TransformerBlock(layers.Layer):
    """A standard Transformer Encoder Block."""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# --- Build the Transformer Model ---

vocab_size = len(tokenizer.word_index) + 1
embed_dim = 64    # Size of each word vector
num_heads = 4    # Number of attention heads
ff_dim = 64       # Hidden layer size in feed forward network inside transformer

model = tf.keras.Sequential([
    layers.Input(shape=(max_len,)),
    TransformerEmbedding(max_len, vocab_size, embed_dim),
    TransformerBlock(embed_dim, num_heads, ff_dim, rate=0.3),
    layers.GlobalAveragePooling1D(), # Condenses sequence into a single vector
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile and train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)

history_transformer = model.fit(
    padded, y, 
    epochs=100, 
    callbacks=[callback], 
    validation_data=(padded_test, y_test), 
    verbose=1
)
# Visualize

tf.keras.models.save_model(model, 'Trasnformer_spam_mail_model.keras')
joblib.dump(tokenizer, 'tokenizer.pkl')
joblib.dump(encoder, 'encoder.pkl')

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history_transformer.history['accuracy'], label='Train')
plt.plot(history_transformer.history['val_accuracy'], label='Test')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_transformer.history['loss'], label='Train')
plt.plot(history_transformer.history['val_loss'], label='Test')
plt.legend()
plt.show()

