import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error

class BidirectionalLSTM():
  def __init__(self, csv, w=30, model_path=None, feature_scaler_path=None, target_scaler_path=None):
    self.window_size = w
    self.df = pd.read_csv(csv)
    self.df["date"] = pd.to_datetime(self.df["date"])
    self.df = self.df.sort_values("date").reset_index(drop=True)
    self.target_cols = [
        "meantemp",
        "humidity",
        "wind_speed",
        "meanpressure"
    ]
    self.model_path = model_path
    self.feature_scaler_path = feature_scaler_path
    self.target_scaler_path = target_scaler_path
    self.model = tf.keras.models.load_model(self.model_path) if model_path else None
    for col in self.target_cols:
      for m in [7, 14, 30]:
          self.df[f"{col}_mean_{m}"] = self.df[col].rolling(m).mean()

      self.df[f"{col}_std_7"] = self.df[col].rolling(7).std()

      for lag in [1, 4, 7, 30]:
          self.df[f"{col}_lag_{lag}"] = self.df[col].shift(lag)

    self.df["day_of_year"] = self.df["date"].dt.dayofyear
    self.df["day_of_week"] = self.df["date"].dt.dayofweek
    self.df["month"] = self.df["date"].dt.month

    self.df["sin_day"] = np.sin(
        2 * np.pi * self.df["day_of_year"] / 365.25
    )

    self.df["cos_day"] = np.cos(
        2 * np.pi * self.df["day_of_year"] / 365.25
    )

    self.df["sin_month"] = np.sin(
        2 * np.pi * self.df["month"] / 12
    )

    self.df["cos_month"] = np.cos(
        2 * np.pi * self.df["month"] / 12
    )

    self.df["sin_weekday"] = np.sin(
        2 * np.pi * self.df["day_of_week"] / 7
    )

    self.df["cos_weekday"] = np.cos(
        2 * np.pi * self.df["day_of_week"] / 7
    )

    self.df = self.df.dropna().reset_index(drop=True)

    feature_cols = [
        c for c in self.df.columns
        if c not in ["date"] + self.target_cols
    ]

    X = self.df[feature_cols].values
    y = self.df[self.target_cols].values
    if feature_scaler_path and target_scaler_path:
      self.feature_scaler = joblib.load(self.feature_scaler_path)
      self.target_scaler = joblib.load(self.target_scaler_path)
      
      self.feature_scaler = self.feature_scaler.partial_fit(X)
      self.target_scaler = self.target_scaler.partial_fit(y)

      X = self.feature_scaler.transform(X)
      y = self.target_scaler.transform(y)
    else:
      self.feature_scaler = MinMaxScaler()
      self.target_scaler = MinMaxScaler()

      X = self.feature_scaler.fit_transform(X)
      y = self.target_scaler.fit_transform(y)
    X_seq = []
    y_seq = []

    for i in range(self.window_size, len(X)):
        X_seq.append(X[i - self.window_size:i])
        y_seq.append(y[i])

    self.X_seq = np.array(X_seq)
    self.y_seq = np.array(y_seq)

  def train(self, epochs=100, batch_size=32):
    if self.model_path:
      self.model = tf.keras.models.load_model(self.model_path)
      history = self.model.partial_fit(self.X_seq, self.y_seq, epochs=100, batch_size=32, ) # callbacks=[early_stop]  
    else:
      self.model = tf.keras.Sequential([tf.keras.layers.Input(shape=(self.window_size, self.X_seq.shape[2])),
                                  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),tf.keras.layers.Dropout(0.3),
                                  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                                  tf.keras.layers.Dropout(0.3),
                                  tf.keras.layers.Dense(64, activation="relu"),
                                  tf.keras.layers.Dense(32, activation="relu"),
                                  tf.keras.layers.Dense(len(self.target_cols))])

      self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.Huber(),metrics=["mae"])

      # early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

      history = self.model.fit(self.X_seq, self.y_seq, epochs=100, batch_size=32, ) # callbacks=[early_stop]
    joblib.dump(self.feature_scaler, 'feature_scaler.pkl')
    joblib.dump(self.target_scaler, 'target_scaler.pkl')
    self.model.save('bidirectional_weather_model.keras')
    return self.model, history
  
  def predict(self):
    if self.model and self.feature_scaler and self.target_scaler:
      pass
    else:
      return 'No model found! Please enter correct model_path.'
    self.model = tf.keras.models.load_model(self.model_path)
    self.feature_scaler = joblib.load(self.feature_scaler_path)
    self.target_scaler = joblib.load(self.target_scaler_path)
    pred = self.model.predict(self.X_seq)
    pred = self.target_scaler.inverse_transform(pred)
    actual = self.target_scaler.inverse_transform(self.y_seq)
    r2 = r2_score(actual, pred, multioutput="uniform_average")
    mape = mean_absolute_percentage_error(actual, pred)
    print(f"R² Score: {r2:.4f}")
    print(f"MAPE: {mape:.4f}")
    print(f"Accuracy: {100 - (mape*100):.4f}")
    for i, col in enumerate(self.target_cols):
        plt.figure(figsize=(12, 5))
        plt.plot(actual[:, i], label=f"Actual {col}")
        plt.plot(pred[:, i], label=f"Predicted {col}")
        plt.title(col)
        plt.legend()
        plt.show()
