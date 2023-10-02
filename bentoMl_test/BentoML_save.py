# Importing the libraries
import numpy as np
import pandas as pd
import bentoml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM
from keras import Sequential
from keras.callbacks import LearningRateScheduler


class BentoMl:
    def __init__(self):
        pass

    @staticmethod
    def save_model(_max_epochs=10, _learning_rate=0.001, _sequence_size=10, _batch_size=16, _hidden_layers=1):
        # Training the model on the Training set
        def to_sequences(dataset_x, dataset_y, _sequence_size=1):
            x, y = [], []

            for i in range(len(dataset_x) - _sequence_size - 1):
                window = dataset_x[i:(i + _sequence_size), 0:df_X.shape[1]]
                x.append(window)
                y.append(dataset_y[i + _sequence_size])

            return np.array(x), np.array(y)

        hyper_dataset = pd.read_csv('../data/cape1.csv', usecols=['trim', 'sog', 'stw', 'wspeedbf', 'wdir', 'me_power'])

        df_X = hyper_dataset[['trim', 'sog', 'stw', 'wspeedbf', 'wdir']].values
        df_y = hyper_dataset['me_power'].values.reshape(-1, 1)

        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=.1, random_state=1, shuffle=False)

        # Feature Scaling
        sc1 = MinMaxScaler()
        X_train_scaled = sc1.fit_transform(X_train[:, :])
        X_test_scaled = sc1.transform(X_test[:, :])
        sc2 = MinMaxScaler()
        y_train_scaled = sc2.fit_transform(y_train[:, :])
        y_test_scaled = sc2.transform(y_test[:, :])

        train_X, train_y = to_sequences(X_train_scaled, y_train_scaled, _sequence_size)
        test_X, test_y = to_sequences(X_test_scaled, y_test_scaled, _sequence_size)

        model = Sequential()
        lr = LearningRateScheduler(lambda _: _learning_rate)

        if _hidden_layers >= 3:
            model.add(LSTM(128, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2]),
                           return_sequences=True))
            model.add(LSTM(64, activation='relu', return_sequences=True))
            model.add(LSTM(32, activation='relu', return_sequences=True))
            model.add(LSTM(16, activation='relu', return_sequences=False))
        elif _hidden_layers >= 2:
            model.add(LSTM(128, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2]),
                           return_sequences=True))
            model.add(LSTM(64, activation='relu', return_sequences=True))
            model.add(LSTM(32, activation='relu', return_sequences=False))
        elif _hidden_layers >= 1:
            model.add(LSTM(128, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2]),
                           return_sequences=True))
            model.add(LSTM(64, activation='relu', return_sequences=False))
        else:
            model.add(LSTM(128, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2]),
                           return_sequences=False))

        model.add(Dense(y_train.shape[1]))

        model.compile(optimizer='adam', loss='mse')
        model.summary()

        # fit the model
        model.fit(train_X, train_y, epochs=_max_epochs, batch_size=_batch_size,
                  validation_data=(test_X, test_y), verbose=0, callbacks=[lr])

        print('BendoML start saving')
        bentoml.keras.save_model('cape1_LSTM', model, custom_objects={"scaler_X": sc1, "scaler_Y": sc2})
        print('BendoML end saving')


if __name__ == '__main__':
    BentoMl.save_model()
    print('saved at C:\\Users\\Da-Wi\\bentoml\\models')
    print('END')
