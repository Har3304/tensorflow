import tensorflow as tf
import numpy as np

tf.random.set_seed(42)

X = tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.float32)
y = tf.multiply(X, 3)

X = tf.reshape(X, shape=(-1,1))
y = tf.reshape(y, shape=(-1,1))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(1,)) 
])

model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), metrics=['mae'])
model.fit(X, y, epochs=200, verbose=0)
test_input = tf.constant([[10.0]], dtype=tf.float32)
prediction = model.predict(test_input)

print(f"Prediction for X=10: {prediction[0][0]:.4f}")
