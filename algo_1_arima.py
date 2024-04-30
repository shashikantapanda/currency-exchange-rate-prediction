# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 19:07:32 2024

@author: User
"""


import warnings
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

train_ratio = 0.83

def load_data():
    data_frame = pd.read_csv(r"C:\Users\User\Desktop\ME\MY PROJECTS\currency_exchange_rate_prediction-main\usdinr_d.csv", header=0)
    data_frame_series = data_frame.squeeze()
    close_idx = data_frame_series.columns.get_loc('Close')

    data_file = pd.read_csv(r"C:\Users\User\Desktop\ME\MY PROJECTS\currency_exchange_rate_prediction-main\usdinr_d.csv", usecols=[close_idx], engine='python')
    load_data = data_file.values.flatten().tolist()
    return load_data

def split_train_test(load_data, train_ratio):
    data_set_len = len(load_data)
    set_train_len = int(data_set_len * train_ratio)
    set_train, set_test = load_data[0:set_train_len], load_data[set_train_len:data_set_len]
    return set_train, set_test

def fit_arima_model(set_train, set_test):
    pred_test = []
    pred_train = list(set_train)
    for set_test_idx in range(len(set_test)):
        arima = ARIMA(pred_train, order=(5, 1, 0))
        arima_model = arima.fit()
        forecasting = arima_model.forecast()[0]
        pred_test.append(forecasting)
        pred_train.append(set_test[set_test_idx])

    return pred_test

def arima_graph_plot(actual_test, pred_test, file_name):
    plt.plot(actual_test, label="Actual data points", color="blue")
    plt.plot(pred_test, label="Testing prediction", color="green")

    plt.ylabel('Currency Values for 1 USD')
    plt.xlabel('Number of Days')
    plt.title("'USD/INR' : Actual vs Predicted using ARIMA")

    plt.legend()
    plt.savefig(file_name)
    plt.clf()

def evaluate_performance_arima(actual_test, pred_test):
    return mean_squared_error(actual_test, pred_test)

def arima_model():
    data = load_data()
    training_actual_arima, actual_test_arima = split_train_test(data, train_ratio)
    pred_test_arima = fit_arima_model(training_actual_arima, actual_test_arima)
    mse_arima = evaluate_performance_arima(actual_test_arima, pred_test_arima)
    print('\t Performance Evaluation: Testing Mean Square Error:', mse_arima)
    print('ARIMA Graph has been generated in PNG format.')
    arima_graph_plot(actual_test_arima, pred_test_arima, "arima_tp.png")
    return data, pred_test_arima

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    arima_model()
