import pickle
import pandas as pd
import numpy as np
import tensorflow as tf

df = pd.read_csv("IT Support Ticket Data.csv")
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
print("GPU Details: ", gpus)


@tf.keras.utils.register_keras_serializable()
class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len, embed_dim, vocab_size):
        super().__init__()
        self.token_embed = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_embed = tf.keras.layers.Embedding(input_dim=max_len, output_dim=embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[-1], delta=1)
        positions = self.pos_embed(positions)
        x = self.token_embed(x)
        return x + positions

@tf.keras.utils.register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim)])
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
        return self.layernorm2(out1 + ffn_output)

df["Output"] = ("<start> "+ "Department: "+ df["Department"].astype(str)+ " | Priority: "+ df["Priority"].astype(str)+ " | Tags: "+ df["Tags"].str.strip("[]").str.replace("'", "", regex=False)+ " <end>")

input_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="")
input_tokenizer.fit_on_texts(df["Body"])
encoder_sequences = (input_tokenizer.texts_to_sequences(df["Body"]))
encoder_max_len = max(len(seq) for seq in encoder_sequences)
encoder_input = (tf.keras.preprocessing.sequence.pad_sequences(encoder_sequences, maxlen=encoder_max_len, padding="post"))

target_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="")
target_tokenizer.fit_on_texts(df["Output"])
decoder_sequences = (target_tokenizer.texts_to_sequences(df["Output"]))
decoder_max_len = max(len(seq) for seq in decoder_sequences)
decoder_sequences = (tf.keras.preprocessing.sequence.pad_sequences(decoder_sequences, maxlen=decoder_max_len, padding="post"))

decoder_input = decoder_sequences[:, :-1]
decoder_output = decoder_sequences[:, 1:]

encoder_vocab_size = (len(input_tokenizer.word_index) + 1)

decoder_vocab_size = (len(target_tokenizer.word_index) + 1)

embed_dim = 128
num_heads = 4
ff_dim = 256

encoder_inputs = tf.keras.layers.Input(shape=(encoder_max_len,))
encoder_embedding = (TokenAndPositionEmbedding(encoder_max_len, embed_dim, encoder_vocab_size)(encoder_inputs))
encoder_transformer = (TransformerBlock(embed_dim, num_heads, ff_dim)(encoder_embedding))
encoder_context = (tf.keras.layers.GlobalAveragePooling1D()(encoder_transformer))
encoder_context = (tf.keras.layers.Dense(embed_dim, activation="relu")(encoder_context))

decoder_inputs = tf.keras.layers.Input(shape=(decoder_max_len - 1,))
decoder_embedding = (TokenAndPositionEmbedding(decoder_max_len, embed_dim, decoder_vocab_size)(decoder_inputs))
context = tf.keras.layers.RepeatVector(decoder_max_len - 1)(encoder_context)
decoder_concat = tf.keras.layers.Concatenate()([decoder_embedding, context])
decoder_transformer = (TransformerBlock(embed_dim * 2, num_heads,ff_dim)(decoder_concat))
outputs = tf.keras.layers.Dense(decoder_vocab_size,activation="softmax")(decoder_transformer)

model = tf.keras.Model([encoder_inputs, decoder_inputs], outputs)

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.summary()

model.fit([encoder_input, decoder_input], decoder_output, epochs=20, batch_size=32, validation_split=0.2)

model.save("ticket_transformer.keras")

metadata = {
    "input_tokenizer": input_tokenizer,
    "target_tokenizer": target_tokenizer,
    "encoder_max_len": encoder_max_len,
    "decoder_max_len": decoder_max_len
}

with open("ticket_transformer_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("Saved Successfully")
