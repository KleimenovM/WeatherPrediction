import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout

from sklearn.model_selection import train_test_split

from reformat_data import split_data, encryption, get_prediction_data, get_data_from_five_numbers


def establish_a_model(method):
    input_shape = (56, 5)

    # OLD MODEL
    # model = keras.Sequential([
    #     Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
    #     MaxPooling2D((2, 2), strides=1, padding='valid'),
    #     Conv2D(64, (3, 3), padding='same', activation='relu'),
    #     MaxPooling2D((2, 2), strides=2, padding='valid'),
    #     Flatten(),
    #     Dense(128, activation='relu'),
    #     Dense(5)
    # ])

    # NEW MODEL
    model = keras.Sequential([
        Conv1D(32, 3, padding='same', activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2, strides=1, padding="valid"),
        Conv1D(64, 3, padding='same', activation='relu'),
        MaxPooling1D(pool_size=2, strides=2, padding="valid"),
        # Dropout(.4),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(5)
    ])

    # print(model.summary())
    return model


def training_procedure(method=1):
    X, y, decipher = split_data('weather2014-2022.csv', n=5000, method=method)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # X_train = np.expand_dims(X_train, axis=3)
    # X_test = np.expand_dims(X_test, axis=3)

    model = establish_a_model(method)
    model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

    # res1 = []
    # for i in range(5):
    #     res1.append(model.predict([X_test[0:1]])
    #     print(res1)

    prediction = model.predict(X_test[0:2])
    real = y_test[0:2]
    print(f'PREDICTED: {encryption(prediction, decipher)}')
    print(f'REAL: {encryption(real, decipher)}')
    return model, decipher


def predict_weather(filename, model, decipher, method):
    x = get_prediction_data(filename, method=method)
    print(x.shape)
    prediction = model.predict(np.array([x]))
    print(f'PREDICTED: {encryption(prediction, decipher)}')
    return encryption(prediction, decipher)


if __name__ == '__main__':
    Method = 1
    Model, Decipher = training_procedure(method=Method)
    get_data_from_five_numbers(predict_weather('mod20decPrediction.csv', Model, Decipher, Method)[0])
