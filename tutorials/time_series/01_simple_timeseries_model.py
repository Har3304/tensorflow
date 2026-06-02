import numpy as np
import tensorflow as tf

X = np.arange(1, 100, 1)

input_width = 30
label_width = 1
shift = 1

dataset = tf.keras.utils.timeseries_dataset_from_array(
            data = X[:-shift],
            targets = X[shift + input_width - 1:], 
            sequence_length = input_width,
            batch_size = 32,
            shuffle = True
)
model = tf.keras.Sequential(
    [tf.keras.layers.Input(shape=(input_width, 1)),
     tf.keras.layers.LSTM(64, return_sequences=False), 
     tf.keras.layers.Dense(32, activation='relu'),
     tf.keras.layers.Dense(label_width)]
)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
    loss='mse',
    metrics=['mae'])

history = model.fit(dataset, epochs=250)

sample_window = np.arange(1, 31, 1).reshape(1, input_width, 1)
print(sample_window)

prediction = model.predict(sample_window)
print(f'Next value in the time-series: {prediction[0][0]}')
