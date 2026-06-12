# Website classification Transformer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf

df= pd.read_csv('website_classification.csv')

print(df.columns)



# Encoding category
y = pd.get_dummies(df['Category'], dtype=int).values

# Tokenizing and Padding website_url and cleaned_website_text

tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(df['cleaned_website_text'])
sequences = tokenizer.texts_to_sequences(df['cleaned_website_text'])

# Reduced max_len and embed_dim to manage memory
max_len = 500 # Reduced from 8711
padded_seq = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')



X_train, X_val, y_train, y_val = train_test_split(padded_seq, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1))


# Model

class EmbeddingAndPositionalEncoding(tf.keras.layers.Layer):
  def __init__(self, max_len, embed_dim, vocab_size):
    super().__init__()
    self.token_embed = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim = embed_dim)
    self.pos_embed = tf.keras.layers.Embedding(input_dim=max_len, output_dim=embed_dim)
  def call(self, x):
    positions = tf.range(start=0, limit=tf.shape(x)[-1])
    positions = self.pos_embed(positions)
    x = self.token_embed(x)
    return x + positions

class TransformerBlock(tf.keras.layers.Layer):
  def __init__(self, embed_dim, num_heads, ff_dim, rate=0.3):
    super().__init__()
    self.attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
    self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(ff_dim, activation='relu'), tf.keras.layers.Dense(embed_dim, activation='relu')])
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x):
    attn_output = self.attn(x, x)
    attn_output = self.dropout1(attn_output)
    out1 = self.layernorm1(x + attn_output)
    ffn_output = self.ffn(out1)
    ffn_output = self.dropout2(ffn_output)
    return self.layernorm2(out1+ffn_output)

# compilation
# max_len is now 512 (defined above)

vocabsize = len(tokenizer.word_index) + 1
embed_dim = 64 # Reduced from 64
num_heads= 2
ff_dim = 64

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

inputs=tf.keras.layers.Input(shape=(max_len,))
embedding_layer = EmbeddingAndPositionalEncoding(max_len=max_len, vocab_size=vocabsize, embed_dim=embed_dim)
x = embedding_layer(inputs)

transformer_block = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)
x = transformer_block(x)

x = tf.keras.layers.GlobalAveragePooling1D()(x)
# x2 = tf.keras.layers.GlobalMaxPooling1D()(x)
# x = tf.keras.layers.Concatenate()([x1, x2])
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(y.shape[1], activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

# Added batch_size to model.fit to control memory usage
model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=1, epochs=200, batch_size=32, callbacks=[callback])

model = tf.keras.Model(inputs=inputs, outputs=outputs)
