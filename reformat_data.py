import numpy as np
import pandas as pd


def normalize(column):
    """
    Normalize data to get appropriate results from a neural network
    Activation function 'softmax' returns values in [0, 1]
    Thus, to be able to convert data back inputs must belong to the same interval [0, 1]

    :param column: data column (pd.Series)
    :return: normalized data column (pd.Series) with values in [0, 1]
    """
    col_min, col_max = np.min(column), np.max(column)
    col_diam = col_max - col_min

    def encryption_function(val):
        return val * col_diam + col_min

    return (column - col_min) / col_diam, encryption_function


def encryption(data, decipher):
    # ToDo: DESCRIPTION!
    """

    :param data:
    :param decipher:
    :return:
    """
    round_value = [1, 1, 0, 5, 5]
    results = []
    for line in data:
        res_l = []
        for i in range(len(line)):
            encoded = np.round(decipher[i](line[i]), round_value[i])
            res_l.append(encoded)
        results.append(res_l)
    return results


def appropriate_data_format(file):
    """
    Reads the .csv file and returns data in an appropriate form
    :param file: string, 'filename.csv'
    :return: data as a pandas Dataframe and a encryption functions list
    """
    # read data
    data = pd.read_csv(file, dtype=float)

    # data normalization, translation to [0, 1]
    columns = data.columns
    decipher = []
    for i, c in enumerate(columns[2:]):
        d_c, f_c = normalize(data[c])
        data[c] = d_c
        decipher.append(f_c)

    return data, decipher


def split_data(file, n: int = 10, method=2):
    """
    Form a list of data appropriate for machine learning
    the first day is chosen in a random way

    :param method: X-data arrangement
    :param file: data filename (.csv)
    :param n: number of samples
    :return: three objects
    X - tensor 56x5 (seven consequent days weather features)
    Y - array 1x5 (last day - 15:00 - features)
    decipher - encryption functions list
    """

    table, decipher = appropriate_data_format(file)

    s = table.index.size
    d = s // n  # size of train-test split

    # MESSAGE
    print('-----')
    print('Reading and formatting done')

    counter: int = 0  # counts the number of data included
    if method == 1:
        X = np.zeros([n, 56, 5])
    else:
        X = np.zeros([n, 56, 7])
    Y = np.zeros([n, 5])

    dates = table.date
    unique_dates = np.unique(dates)

    while counter < n:
        # generate a random number to start an 8-days row
        k = np.random.randint(0, unique_dates.size - 9)
        date_k = unique_dates[k]

        # check if there exists a continuous date row
        if unique_dates[k + 7] > date_k + 7:
            continue

        # gather all the parameters in X and Y

        # THERE EXIST TWO WAYS OF X-arrangement
        # ONE: exclude columns 0 and 1 as there are date and daytime stored
        if method == 1:
            x = table.loc[dates.isin(np.arange(date_k, date_k + 7, 1))][table.columns[2:]].to_numpy()
        # TWO: include data and normalize these lines after data array formation
        else:
            x = table.loc[dates.isin(np.arange(date_k, date_k + 7, 1))].to_numpy()

        y = table.loc[dates == date_k + 7][table.columns[2:]].to_numpy()[5]
        X[counter] = x
        Y[counter] = y

        counter += 1

    # MESSAGE
    print('-----')
    print('Train and test data organized!')

    X = X.reshape([n, 56, 5])

    return X, Y, decipher


def get_prediction_data(file, method):
    table, decipher = appropriate_data_format(file)

    X = table[table.columns[2:]].to_numpy()[:56]

    return X


def get_data_from_five_numbers(line):
    print(f'Температура: {line[0]} °С')
    print(f'Давление: {line[1]} мм. рт. ст.')
    print(f'Влажность: {line[2]} %')
    speed = (line[3]**2 + line[4]**2)**0.5
    angle = np.arccos(line[3] / speed) / np.pi * 180
    print(f'Направление ветра: {np.round(angle)}° к направлению на восток')
    print(f'Скорость ветра: {np.round(speed, 1)} м/с')
    return


if __name__ == '__main__':
    # ENCRYPTION TRY
    x, y, des = split_data('weather2014-2022.csv', n=100)
    a = [[0.61848235, 0.46895117, 0.30913854, 0.65802354, 0.27043423]]
    print(encryption(a, des))

    # PREPARATION TRY
    x = get_prediction_data('mod20decPrediction.csv', method=1)
    print(x)
    print(x.shape)


