# Importing the libraries
import numpy as np
import pandas as pd
import bentoml


class BentoMl:
    def __init__(self):
        pass

    @staticmethod
    def run_model():
        # Training the model on the Training set
        def to_sequences(dataset_x, _sequence_size=1):
            x = []

            for i in range(len(dataset_x) - _sequence_size + 1):
                window = dataset_x[i:(i + _sequence_size), 0:X_test.shape[1]]
                x.append(window)

            return np.array(x)

        model = bentoml.keras.load_model('energy_consumption:latest')
        bentoml_model = bentoml.keras.get('energy_consumption:latest')
        scaler_X = bentoml_model.custom_objects['scaler_X']
        scaler_Y = bentoml_model.custom_objects['scaler_Y']

        X_test = pd.read_csv('../data/data_consumption.csv', usecols=['consumption']).head(12).values

        X_test_scaled = scaler_X.transform(X_test[:, :])
        test_X = to_sequences(X_test_scaled, 12)

        test_predict = model.predict(test_X)
        test_predict = scaler_Y.inverse_transform(test_predict)

        print(test_predict)


if __name__ == '__main__':
    BentoMl.run_model()
    print('END')
