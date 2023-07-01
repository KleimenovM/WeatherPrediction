import numpy as np
import pandas as pd
from datetime import date


# Wind direction dictionary
Dict = {'Штиль, безветрие': 0,
        'Ветер, дующий с востока': 1, 'Ветер, дующий с востоко-северо-востока': 2,
        'Ветер, дующий с северо-востока': 3, 'Ветер, дующий с северо-северо-востока': 4,
        'Ветер, дующий с севера':5, 'Ветер, дующий с северо-северо-запада': 6,
        'Ветер, дующий с северо-запада': 7,'Ветер, дующий с западо-северо-запада': 8,
        'Ветер, дующий с запада': 9, 'Ветер, дующий с западо-юго-запада': 10,
        'Ветер, дующий с юго-запада': 11, 'Ветер, дующий с юго-юго-запада': 12,
        'Ветер, дующий с юга': 13, 'Ветер, дующий с юго-юго-востока': 14,
        'Ветер, дующий с юго-востока': 15, 'Ветер, дующий с востоко-юго-востока': 16}


def process_one_line(line):
    """
    Shift from strings to floats
    :param line: one string line
    :return: list of line parts
    """
    line = line.replace('\"', '').strip()
    return line.split(';')


def read_file(filename):
    """
    data file reading function
    :param filename: '_.txt'
    :return: list of lists with parts of each line
    """
    with open(filename, 'r', encoding='utf-8') as to_read:
        lines = to_read.readlines()

    data = []
    for line in lines:
        res = process_one_line(line)
        data.append(res)

    return data[6:]


def map_unique(column):
    """
    ToDo: DESCRIPTION!
    :param column:
    :return:
    """
    definitions = np.unique(column)
    n = definitions.size
    dictionary = {definitions[i]: i for i in range(n)}
    return dictionary


def convert_date_to_integer(value):
    """
    ToDo: DESCRIPTION!
    :param value:
    :return:
    """
    parts = [int(v) for v in value.split('.')]
    d = date(parts[2], parts[1], parts[0])
    d.toordinal()
    return d.toordinal()


def work_with_wind(direction, speed):
    angles = (direction.map(Dict) - 1) / 16 * 2 * np.pi  # wind to angle
    speed = speed.astype('float')

    print(speed)

    cord1, cord2 = speed * np.cos(angles), speed * np.sin(angles)

    return cord1, cord2


def arrange_data(data):
    """
    ToDo: DESCRIPTION!
    :param data:
    :return:
    """
    n, m = len(data[0]), len(data)

    res = []
    for i in range(n):
        res.append([])
        for j in range(1, m):
            res[i].append(data[j][i])

    # make a dataframe with correct subscriptions
    table = pd.DataFrame(data[1:])
    h = table.columns.size
    table = table.drop(columns=np.arange(29, h))
    columns = data[0]
    table.columns = columns
    tmp = table.index
    table = table[::-1]
    table.index = tmp

    # expand time data column
    time = table[columns[0]].str.split(' ', expand=True)
    table[columns[0]] = time[time.columns[0]]
    table = table.rename(columns={columns[0]: 'date'})
    table.insert(1, 'time', time[time.columns[1]])

    # column 'DD' contains strings data
    # need to change it for integer numbers by mapping each unique string with a number
    d, f = work_with_wind(table.DD, table.Ff)
    table['WindX'] = d
    table['WindY'] = f
    print(table)

    useful = ['date', 'time', 'T', 'Po', 'U', 'WindX', 'WindY']
    table = table[useful]
    table = table.replace('', np.nan)
    table = table.dropna(axis='rows', how='any')

    # exclude data that are not in the map below
    t_map = {'00:00': 0, '03:00': 1, '06:00': 2, '09:00': 3, '12:00': 4, '15:00': 5, '18:00': 6, '21:00': 7}
    table.time = table.time.map(t_map)
    table = table.loc[table.time.isin(np.arange(0, 8))]
    # print(table)

    table.date = table.date.map(convert_date_to_integer)

    return table, d


def remove_reduced_data(table, if_check=False):
    """
    Removes dates where not all the possible daytime are registered
    If there's less than 8 daytime lines, replace date with NaN and remove the row from the table

    :param if_check: controlling procedure
    :param table: pandas DataFrame
    :return: cleaned pandas DataFrame
    """

    # find all possible dates
    date_data = table.date
    n = date_data.size
    dates = np.unique(date_data)

    for d in dates:
        counter = n - np.count_nonzero(date_data - d)
        if counter != 8:
            table = table.replace(d, np.nan)

    table = table.dropna()

    # now check whether it worked in a correct way: compare number of daytime values
    if if_check:
        time_data = np.unique(table.time)
        m = table.time.size
        for t in time_data:
            print(m - np.count_nonzero(table.time - t))

    return table


def full_file_processing(filename):
    """
    This function processes the whole file from reading to data withdrawal
    :param filename:
    :return:
    """
    print(filename)
    data = read_file(filename)

    # pd.dataframe with readable data and a dictionary for 'DD' translation
    data, d = arrange_data(data)
    data = pd.DataFrame(data, dtype=float)  # make float

    # leave only dates with full feature set
    data = remove_reduced_data(data)

    return data, d


def prepare_training_data():
    files = [str(year) + 'weather.csv' for year in range(2014, 2023)]
    dis = []
    frames = []
    for file in files:
        data, d = full_file_processing(file)
        data.to_csv('mod' + file, index=False)
        frames.append(data)
        dis.append(d)

    with open('dict.txt', 'w') as out_file:
        print(dis[0], file=out_file)

    final = pd.concat(frames)
    final.to_csv('weather2014-2022.csv', index=False)
    return


def prepare_prediction_data():
    file = '20decPrediction.csv'
    data, d = full_file_processing(file)
    data.to_csv('mod' + file, index=False)
    return


if __name__ == '__main__':
    prepare_training_data()
    # prepare_prediction_data()
