import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

data = np.sin(np.linspace(0, 100, 100)) + np.random.normal(0, 0.2, 100)

input_width = 30
label_width = 1
shift = 1

total_window_size = input_width + shift

dataset = tf.keras.utils.timeseries_dataset_from_array(
    data = data[:-shift],
    targets = data[input_width:],
    sequence_length = input_width,
    batch_size = 32,
    shuffle = True
)

# Model Architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_width, 1)),
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(label_width)
])

model.summary()

# Model compilation
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
    loss='mse',
    metrics=['mae']
)

history = model.fit(dataset, epochs=750)

predictions=[]
for d in range(len(data)-input_width):
  sample_window = data[d:d+input_width].reshape(1, input_width, 1)
  prediction = model.predict(sample_window)
  print(f"Predicted next value: {prediction[0][0]}")
  predictions.append(prediction[0][0])

# Visualize actual vs predcition
plt.plot(data, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()
