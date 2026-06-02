import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('house_price_regression_dataset.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print(df.columns)
print(df.head())


print('\n')
print('#'*20)
print('Important features')
print('#'*20)
print('\n')

features = df.corr().loc['House_Price', :].sort_values(ascending=False)[1:3]
print(features)


# Data Preprocessing
X = df[features.index]
y = df['House_Price']

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))

X_scaled = tf.constant(X_scaled, dtype=tf.float32)
y_scaled = tf.constant(y_scaled, dtype=tf.float32)

print(X_scaled[:5])
print(y_scaled[:5])

X_train, X_test, y_train, y_test = train_test_split(X_scaled.numpy(), y_scaled.numpy(), test_size=0.2, random_state=42)

# Model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(128, activation='linear'),
    tf.keras.layers.Dense(1, activation='linear'),])

model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['mae'])
history = model.fit(X_train, y_train, epochs=100)


# Loss Visualization and test vs prediction
fig, ax = plt.subplots(1, 2 ,figsize=(10, 5))
ax[0].plot(history.history['loss'])
ax[0].set_title('Model Loss')
ax[0].set_ylabel('Loss')
ax[0].set_xlabel('Epoch')

pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)


ax[1].scatter(X_pca.flatten()[:10], y[:10], c='b', label='Actual data')
ax[1].scatter(X_pca.flatten()[:10], y_scaler.inverse_transform(model.predict(tf.constant(X_scaled))).flatten()[:10], c='g', label='predicted data')
ax[1].set_title('Evaluation')
ax[1].set_ylabel('Housing Price')
ax[1].set_xlabel('PCA-1 of X')


ax[1].legend()
plt.show()

# Inference Evaluation
while True:
  test_dict={}
  should_quit = False
  for k in X.columns:
      value = input(f'Enter {k} or press q to quit: ')
      if value.lower() == 'q':
          should_quit = True
          break
      test_dict[k] = [float(value)]
  
  if should_quit:
      print("Exiting evaluation.")
      break
  if not test_dict:
      print("No data entered for evaluation. Exiting.")
      break

  print(test_dict)
  df = pd.DataFrame(test_dict)
  transformed_df = X_scaler.transform(df)
  
  print("Scaled input:")
  print(transformed_df)
  
  model_prediction_scaled = model.predict(tf.constant(transformed_df, dtype=tf.float32))
  
  print("Model prediction (scaled):")
  print(model_prediction_scaled)
  
  final_prediction = y_scaler.inverse_transform(model_prediction_scaled)
  
  print("Predicted House Price:")
  print(final_prediction[0][0])    
