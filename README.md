# USD/INR Currency Exchange Rate Prediction

This project aims to predict the currency exchange rate between the US Dollar and the Indian Rupee (USD/INR) using two different machine learning algorithms: ARIMA and RNN.

The project consists of three Python files:

algo_1_arima.py: Implements the ARIMA model for predicting USD/INR exchange rates.

algo_2_rnn.py: Implements the RNN model for predicting USD/INR exchange rates.

main.py: Runs both ARIMA and RNN models and compares their performance.

## Dataset
The dataset used in this project is the historical daily data of USD/INR exchange rate from Jan 2020 to Dec 2023. The dataset can be found in the file 'usdinr_d.csv'.

## Usage
To run the project, simply run main.py. The script will train both ARIMA and RNN models on the provided dataset and compare their performance. The final output of the script will be the MSE (Mean Squared Error) of each model on the test data.

## Dependencies
This project requires the following Python libraries:

##### python 3.0 or later(3.8 preferable)
##### pandas
##### matplotlib
##### scikit-learn
##### statsmodels
##### keras
##### warnings

## Installation
Clone the repository: git clone https://github.com/shashikantapanda/currency-exchange-rate-prediction.git

Install the required packages: 
pandas
matplotlib
warnings
scikit-learn
keras
statsmodels

## File Descriptions
algo_1_arima.py: Contains the implementation of the ARIMA model for predicting USD/INR exchange rates.

algo_2_rnn.py: Contains the implementation of the RNN model for predicting USD/INR exchange rates.

main.py: Contains the driver code that runs both ARIMA and RNN models and compares their performance.

## Results
Upon running **main.py**, the script trains both ARIMA and RNN models and prints the Mean Squared Error (MSE) for each model on the test data. The script also generates a graph comparing the actual and predicted exchange rates for both models.

Based on the provided dataset, we found that the ARIMA model outperformed the RNN model in terms of MSE. However, the performance of each model may vary based on the dataset used.

## Acknowledgements
The dataset used in this project was obtained from Stooq (https://stooq.com/q/d/?s=usdinr&c=0&d1=20200101&d2=20221231)). We also thank the developers of the required libraries used in this project.
