# Weather Parameters Predictor using Bidirectional LSTM

A TensorFlow-based deep learning project for multivariate weather forecasting using a Bidirectional Long Short-Term Memory (BiLSTM) network.

This project predicts multiple weather parameters simultaneously from historical weather observations and engineered temporal features.

## Features

* Predicts multiple weather variables:

  * Mean Temperature (`meantemp`)
  * Humidity (`humidity`)
  * Wind Speed (`wind_speed`)
  * Mean Pressure (`meanpressure`)
* Bidirectional LSTM architecture
* Automatic feature engineering
* Rolling statistics generation
* Lag feature generation
* Seasonal cyclic encoding using sine and cosine transformations
* Feature and target scaling using MinMaxScaler
* Model persistence and reloading
* Interactive prediction visualization
* Real-time simulation of forecasts
* Multi-output regression

---

## Dataset

The model is designed for weather datasets containing the following columns:

| Column       | Description          |
| ------------ | -------------------- |
| date         | Observation date     |
| meantemp     | Mean temperature     |
| humidity     | Humidity percentage  |
| wind_speed   | Wind speed           |
| meanpressure | Atmospheric pressure |

Example:

```csv
date,meantemp,humidity,wind_speed,meanpressure
2013-01-01,10.0,84.5,0.0,1015.0
2013-01-02,7.4,92.0,2.98,1018.0
```

---

## Feature Engineering

The model automatically generates:

### Rolling Means

For each target variable:

* 7-day moving average
* 14-day moving average
* 30-day moving average

### Rolling Standard Deviation

* 7-day rolling standard deviation

### Lag Features

For each target variable:

* Lag 1
* Lag 4
* Lag 7
* Lag 30

### Cyclic Time Features

The following cyclical encodings are generated:

* Day of year
* Day of week
* Month

Using:

```python
sin(x)
cos(x)
```

to preserve temporal seasonality.

---

## Model Architecture

```text
Input Sequence
        │
        ▼
Bidirectional LSTM (128 units)
        │
     Dropout
        │
        ▼
Bidirectional LSTM (64 units)
        │
     Dropout
        │
        ▼
Dense (64, ReLU)
        │
        ▼
Dense (32, ReLU)
        │
        ▼
Dense (4 Outputs)
```

Outputs:

```text
[
  meantemp,
  humidity,
  wind_speed,
  meanpressure
]
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Har3304/tensorflow.git
cd tensorflow/tutorials/time_series/lstm/weather_parameters_predictor
```

Install dependencies:

```bash
pip install tensorflow pandas numpy matplotlib plotly scikit-learn joblib
```

---

## Training

```python
model = BidirectionalLSTM(
    csv="DailyDelhiClimateTrain.csv",
    w=30
)

model.train(
    epochs=100,
    batch_size=32
)
```

Generated files:

```text
feature_scaler.pkl
target_scaler.pkl
bidirectional_weather_model.keras
```

---

## Prediction

```python
predictor = BidirectionalLSTM(
    csv="DailyDelhiClimateTest.csv",
    model_path="bidirectional_weather_model.keras",
    feature_scaler_path="feature_scaler.pkl",
    target_scaler_path="target_scaler.pkl"
)

predictor.predict()
```

Metrics reported:

* R² Score
* Mean Absolute Percentage Error (MAPE)

Prediction graphs are generated for:

* Mean Temperature
* Humidity
* Wind Speed
* Mean Pressure

---

## Real-Time Forecast Simulation

The project includes a simulation mode that visualizes actual versus predicted values over time.

```python
predictor.simulate()
```

Visualization includes:

* Live prediction updates
* Actual observations
* Forecasted observations
* Multi-variable tracking

---

## Project Structure

```text
weather_parameters_predictor/
│
├── DailyDelhiClimateTrain.csv
├── DailyDelhiClimateTest.csv
│
├── feature_scaler.pkl
├── target_scaler.pkl
├── bidirectional_weather_model.keras
│
├── weather_predictor.py
└── README.md
```

---

## Technologies Used

* Python
* TensorFlow / Keras
* Pandas
* NumPy
* Scikit-Learn
* Plotly
* Matplotlib
* Joblib

---

## Future Improvements

* Attention-based forecasting
* GRU and Transformer models
* Hyperparameter optimization
* Multi-step forecasting
* Recursive forecasting
* Weather anomaly detection
* Model explainability with SHAP
* Deployment using FastAPI

---

## References

* TensorFlow Time Series Forecasting Tutorial
* Bidirectional LSTM Networks
* Deep Learning for Time Series Forecasting Literature Survey

---

## License

This project is released under the MIT License.

Feel free to use, modify, and distribute the code for educational and research purposes.
