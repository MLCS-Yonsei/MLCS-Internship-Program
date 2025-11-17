# Neural Networks

## Recurrent Neural Networks (RNNs) & Regression Models
RNN is a network that can deal with sequential data such as time series data. However, classic (or "vanilla") RNNs can suffer from the vanishing gradient problem, which leads to the loss of former information. LSTM(Long short-term memory) can handle this problem by providing a short-term memory for RNN that can last thousands of timesteps, thus 'long short-term memory'.

In this course, we are going to train an LSTM regression model which predicts future stock prices based on historical data on stock prices. We will use the financial data from yFinance, which can be downloaded with a python library 'yfinance'. To deal with such tabular data, we can use a dataframe with 'pandas' python library.

Please refer to the example code (003_lstm.py) and the material ('RNN_LSTM.pdf') uploaded above. You may change the structure of the LSTM and tune the hyperparameters in the example code, to increase the performance of the model. The sample code uses only one feature, the stock price, and makes a prediction of one future time step. You may make a training dataset consisting of multiple features, or make predictions of multiple time steps(which will make the LSTM structure 'many-to-many').

## Author
Bogyeong Suh
