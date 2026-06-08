import tensorflow as tf

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
      super().__init__()
      self.token_embed = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
      self.pos_embed = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim) # Corrected max_len to maxlen
    def call(self, x):
      maxlen = tf.shape(x)[-1]
      positions = tf.range(start=0, limit=maxlen, delta=1)
      positions = self.pos_embed(positions)
      x = self.token_embed(x)
      return x + positions

class TransformerBlock(tf.keras.layers.Layer): # Corrected TrasnformerBlock to TransformerBlock
  def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
    super().__init__()
    self.attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
    self.ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(ff_dim, activation='relu'),
        tf.keras.layers.Dense(embed_dim)
    ])
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon = 1e-6) # Corrected LayerNormalizaion to LayerNormalization
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
  def call(self, x, training=False): # Added training argument and used x instead of inputs
    attn_output = self.attn(x, x)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)
    ffn_output = self.ffn(out1)
    ffn_output = self.dropout2(ffn_output, training = training)
    return self.layernorm2(out1 + ffn_output)


vocab_size = 20000
maxlen = 200
embed_dim = 32
num_heads = 2
ff_dim = 32

(x_train, y_train), (x_val, y_val) = tf.keras.datasets.imdb.load_data(num_words=vocab_size) # Corrected imbd to imdb
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = tf.keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen) # Corrected x_test to x_val

inputs = tf.keras.layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)

transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)

x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(20, activation='relu')(x)
x = tf.keras.layers.Dropout(0.1)(x)
output = tf.keras.layers.Dense(2, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=output)

model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val))
