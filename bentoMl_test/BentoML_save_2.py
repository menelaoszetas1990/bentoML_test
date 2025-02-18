import pandas as pd
import numpy as np
import bentoml
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class EnergyConsumptionXGB:
    """
    XGBoost Model Wrapper to handle scaling inside the model.
    """

    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler  # Store scaler internally

    def predict(self, input_series):
        """Receives raw input and applies scaling inside the model."""
        return self.__call__(input_series)

    def __call__(self, input_series):
        """Make the model callable so BentoML can use it."""
        input_series = np.array(input_series, dtype=np.float32)

        # Ensure input size is exactly 12
        if input_series.shape[0] != 12:
            raise ValueError(f"Expected input series of length 12, but got {len(input_series)}")

        # Normalize input inside the model
        input_series_scaled = self.scaler.transform(input_series.reshape(-1, 1)).flatten()
        input_series_scaled = input_series_scaled.reshape(1, -1)  # (1, 12)

        # Predict scaled value
        prediction_scaled = self.model.predict(input_series_scaled)

        # Inverse transform to get original scale
        prediction = self.scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]

        return prediction


# Load dataset
hyper_dataset = pd.read_csv('../data/data_consumption.csv', usecols=['Time', 'consumption'])

# Use only the consumption column for training
df = hyper_dataset[['consumption']].values

# Normalize data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)


# Convert to sequences (last 12 values -> next value)
def to_sequences(data, sequence_size=12):
    X, y = [], []
    for i in range(len(data) - sequence_size):
        X.append(data[i:i+sequence_size].flatten())  # Flatten sequence
        y.append(data[i+sequence_size][0])  # Next value as target
    return np.array(X), np.array(y)


sequence_size = 12
X, y = to_sequences(df_scaled, sequence_size)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, shuffle=False)

# Train XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, learning_rate=0.05)
xgb_model.fit(X_train, y_train)

# Wrap model with scaling embedded inside
model_wrapper = EnergyConsumptionXGB(xgb_model, scaler)

# Save model with BentoML
bentoml.picklable_model.save_model(
    "energy_consumption_forecast_xgb",
    model_wrapper,
)

print("XGBoost model saved successfully.")
