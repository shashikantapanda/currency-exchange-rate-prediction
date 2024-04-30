import warnings

from pandas import read_csv
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot

train_ratio = 0.83
test_ratio = 1 - train_ratio
prev_data_points_numbers = 3
data_set_len = 0
set_train_len = 0
set_test_len = 0


def load_data():
    data_set_frame = read_csv(r"C:\Users\User\Desktop\ME\MY PROJECTS\currency_exchange_rate_prediction-main\usdinr_d.csv", header=0)
    data_set_frame_series = data_set_frame.squeeze()
    column_headers = data_set_frame_series.columns.values.tolist()
    col_idx = column_headers.index('Close')

    data_file = read_csv(r"C:\Users\User\Desktop\ME\MY PROJECTS\currency_exchange_rate_prediction-main\usdinr_d.csv", usecols=[col_idx], engine='python')
    load_data = []
    for data_point in data_file.values.tolist():
        load_data.append(data_point[0])
    global data_set_len
    data_set_len = len(load_data)
    return load_data


def split_train_test(load_data, train_ratio, test_ratio):
    global set_train_len, set_test_len
    set_train_len = int(data_set_len * train_ratio)
    set_test_len = data_set_len - set_train_len
    training_set, testing_set = load_data[0:set_train_len], load_data[set_train_len:data_set_len]
    return training_set, testing_set


def modify_data_set_rnn(training_set, testing_set):
    train_actual = []
    train_predict = []
    for interval in range(len(training_set) - prev_data_points_numbers - 1):
        train_actual.append(training_set[interval: interval + prev_data_points_numbers])
        train_predict.append(training_set[interval + prev_data_points_numbers])

    test_actual = []
    test_predict = []

    for interval in range(len(testing_set) - prev_data_points_numbers - 1):
        test_actual.append(testing_set[interval: interval + prev_data_points_numbers])
        test_predict.append(testing_set[interval + prev_data_points_numbers])

    return train_actual, train_predict, test_actual, test_predict


def build_rnn(train_actual, train_predict):
    recurrent_neural_network = Sequential()

    recurrent_neural_network.add(Dense(12, input_dim=prev_data_points_numbers, activation="relu"))
    recurrent_neural_network.add(Dense(8, activation="relu"))
    recurrent_neural_network.add(Dense(1))

    recurrent_neural_network.compile(loss='mean_squared_error', optimizer='adam')
    recurrent_neural_network.fit(train_actual, train_predict, epochs=50, batch_size=2, verbose=2)

    return recurrent_neural_network


def predict_rnn(recurrent_neural_network, train_actual, test_actual):
    training_predict, testing_predict = recurrent_neural_network.predict(
        train_actual), recurrent_neural_network.predict(test_actual)

    print('\t The prediction for the next day:', testing_predict[-1])
    return training_predict, testing_predict


def evaluate_performance_rnn(recurrent_neural_network, test_actual, test_predict):
    # mse_training = recurrent_neural_network.evaluate(train_actual, train_predict, verbose=0)
    mse_testing = recurrent_neural_network.evaluate(test_actual, test_predict, verbose=0)
    print('\t Performance Evaluation: Testing Mean Square Error:', mse_testing)


def rnn_graph_plot(load_data, training_predict, testing_predict, file_name):
    training_data_trend = [None] * data_set_len
    testing_data_trend = [None] * data_set_len

    training_data_trend[prev_data_points_numbers:len(training_predict) + prev_data_points_numbers] = list(training_predict[:, 0])
    testing_data_trend[prev_data_points_numbers - 1:len(training_predict) + prev_data_points_numbers] = list(testing_predict[:, 0])

    actual = pyplot.plot(load_data[int(train_ratio * data_set_len):], label="Actual data points", color="blue")
    testing = pyplot.plot(testing_data_trend, label="Testing prediction", color="red")

    pyplot.ylabel('Currency Values for 1 USD')
    pyplot.xlabel('Number of Days')
    pyplot.title("USD/INR' : Actual vs Predicted using RNN")

    pyplot.legend()

    pyplot.savefig(file_name)
    pyplot.clf()


def rnn_model():

    data = load_data()

    training_set, testing_set = split_train_test(data, train_ratio, test_ratio)
    train_actual, train_predict, test_actual, test_predict = modify_data_set_rnn(training_set, testing_set)

    rnn = build_rnn(train_actual, train_predict)

    training_predict, testing_predict = predict_rnn(rnn, train_actual, test_actual)

    evaluate_performance_rnn(rnn, test_actual, test_predict)

    print('RNN Graph has been generated in PNG format.')
    rnn_graph_plot(data, training_predict, testing_predict, "rnn_tp.png")

    return training_predict, testing_predict


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    rnn_model()  # setting the entry point
