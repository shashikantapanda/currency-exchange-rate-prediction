import warnings
from algo_1_arima import *
from matplotlib import pyplot
from pandas import read_csv
from algo_2_rnn import *

train_ratio = 0.83
test_ratio = 1 - train_ratio
prev_data_points_number = 3
data_set_len = 0
set_train_len = 0
set_test_len = 0

def plot(load_data, pred_train, pred_test, pred_test_arima, filename):
    global data_set_len
    data_set_len = len(load_data)
    training_data_trend = [None] * data_set_len
    testing_data_trend = [None] * data_set_len

    training_data_trend[prev_data_points_number:len(pred_train) + prev_data_points_number] = list(pred_train[:, 0])
    testing_data_trend[prev_data_points_number - 1:len(pred_train) + prev_data_points_number] = list(pred_test[:, 0])

    actual = pyplot.plot(load_data[int(train_ratio * data_set_len):], label="Actual data points", color="blue")
    testing_rnn = pyplot.plot(testing_data_trend, label="Testing prediction RNN", color="red")
    testing_arima = pyplot.plot(pred_test_arima, label="Testing prediction ARIMA", color="green")

    pyplot.ylabel('INR Value for 1 USD')
    pyplot.xlabel('Number of Days')
    pyplot.title("USD/INR' : Actual vs Predicted")

    pyplot.legend()
    pyplot.savefig(filename)
    pyplot.clf()

def main():
    load_data, pred_test_arima = arima_model()
    pred_train, pred_test = rnn_model()
    print('Plotting combined graph of both the models...')
    plot(load_data, pred_train, pred_test, pred_test_arima, "main.png")

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()  # setting the entry point
