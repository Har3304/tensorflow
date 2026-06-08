import pickle
import numpy as np
import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len, embed_dim, vocab_size):
        super().__init__()
        self.token_embed = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.pos_embed = tf.keras.layers.Embedding(max_len, embed_dim)

    def call(self, x):
        positions = tf.range(0, tf.shape(x)[-1], 1)
        return self.token_embed(x) + self.pos_embed(positions)

@tf.keras.utils.register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim)
        ])
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(rate)
        self.drop2 = tf.keras.layers.Dropout(rate)

    def call(self, x):
        attn = self.drop1(self.attn(x, x))
        out1 = self.norm1(x + attn)
        ffn = self.drop2(self.ffn(out1))
        return self.norm2(out1 + ffn)

with open("ticket_transformer_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

input_tokenizer = metadata["input_tokenizer"]
target_tokenizer = metadata["target_tokenizer"]
encoder_max_len = metadata["encoder_max_len"]
decoder_max_len = metadata["decoder_max_len"]

model = tf.keras.models.load_model("ticket_transformer.keras", custom_objects={"TokenAndPositionEmbedding": TokenAndPositionEmbedding,
        "TransformerBlock": TransformerBlock})

start_token = target_tokenizer.word_index["start"]
end_token = target_tokenizer.word_index["end"]

def predict_ticket(text, max_new_tokens=50):
    seq = input_tokenizer.texts_to_sequences([text])
    encoder_input = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=encoder_max_len, padding="post")
    generated = [start_token]
    for _ in range(max_new_tokens):
        decoder_input = tf.keras.preprocessing.sequence.pad_sequences(
            [generated], maxlen=decoder_max_len-1, padding="post")
        pred = model.predict([encoder_input, decoder_input],verbose=0)
        next_token = np.argmax(pred[0, len(generated)-1])
        if next_token == end_token:
            break
        generated.append(next_token)
    words = [target_tokenizer.index_word[token] for token in generated[1:] if token in target_tokenizer.index_word]
    return " ".join(words)

while True:
    text = input("\nTicket (e to exit): ")
    if text.lower() == "e":
        break
    prediction = predict_ticket(text)
    print("\nPrediction:")
    print(prediction)
