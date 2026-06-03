# Using Bidirectional LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('date_count.csv')
df.head()

df['Date'] = pd.to_datetime(df['Date'])
df= df.sort_values("Date")
df.dropna(inplace=True)                  

df['month'] = df['Date'].dt.month
df['weekday'] = df['Date'].dt.weekday
df['day_of_year'] = df['Date'].dt.dayofyear

df['sin_day'] = np.sin(2*np.pi*df['day_of_year']/365.25)
df['cos_day'] = np.cos(2*np.pi*df['day_of_year']/365.25)

for lag in range(1, 7):
  df[lag] = df['count'].shift(lag)

# Rolling mean
df['mean_7'] = df['count'].rolling(7).mean()
df['mean_14'] = df['count'].rolling(14).mean()
df['mean_30'] = df['count'].rolling(30).mean()

df['std_7'] = df['count'].rolling(7).std()

# Drop NaN values AFTER all feature engineering is done
df.dropna(inplace=True)
print("NaN values after preprocessing:")
print(df.isnull().sum())

df.drop('Date', axis=1, inplace=True)

X = df.drop('count', axis=1)
y = df['count']

X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_scaled = X_scaler.fit_transform(X.values)
y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))

window_size = 60

X_seq= []
y_seq= []

for i in range(window_size, len(X_scaled)):
  X_seq.append(X_scaled[i-window_size:i])
  y_seq.append(y_scaled[i])
  
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

split = int(len(X_seq)*0.8)

X_train = X_seq[:split]
y_train = y_seq[:split]
X_test = X_seq[split:]
y_test = y_seq[split:]

model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=False, input_shape=(window_size, X_seq.shape[2]))), # Changed return_sequences to False
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
])

model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['mae'])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), batch_size=32, callbacks=[early_stop])

pred = model.predict(X_test)

pred_inverse = y_scaler.inverse_transform(pred)
y_test_inverse = y_scaler.inverse_transform(y_test)

# Metrics
print(f'R2 Score: {r2_score(y_test_inverse, pred_inverse)}')
print(f'MAPE: {mean_absolute_percentage_error(y_test_inverse, pred_inverse)}')
print(f'Accuracy: {100 - mean_absolute_percentage_error(y_test_inverse, pred_inverse) * 100:.2f}%')

# Visualization
plt.plot(y_test_inverse, label='Actual')
plt.plot(pred_inverse, label='Predicted')
plt.legend()
plt.show()
