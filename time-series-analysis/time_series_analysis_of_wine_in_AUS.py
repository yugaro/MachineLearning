import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# Define functions to create input data correct labels
def create_dataset(dataset, look_back):
    data_X, data_Y = [], []
    for i in range(look_back, len(dataset)):
        data_X.append(dataset[i-look_back:i, 0])
        data_Y.append(dataset[i, 0])
    return np.array(data_X), np.array(data_Y)

# Reading the dataset
dataframe = pd.read_csv("../data/monthly-australian-wine-sales-th-sparkling.csv",usecols=[1],engine='python',skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
print(dataset)
print(type(dataset))

# Split the dataset into training data, test data
train_size = int(len(dataset) * 0.67)
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# Data scaling (normalization)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler_train = scaler.fit(train)
train = scaler_train.transform(train)
test = scaler_train.transform(test)

# Create input data and correct answer labels
look_back = 3
train_X, train_Y = create_dataset(train, look_back)
test_X, test_Y = create_dataset(test, look_back)

# Data Formatting
train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)

# Creation and training of the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(look_back, 1), return_sequences=True))
model.add(LSTM(32))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_X, train_Y, epochs=15, batch_size=1, verbose=2)

# Creation of predictive data
train_predict = model.predict(train_X)
test_predict = model.predict(test_X)

# Restore the scaled data
train_predict = scaler_train.inverse_transform(train_predict)
train_Y = scaler_train.inverse_transform([train_Y])
test_predict = scaler_train.inverse_transform(test_predict)
test_Y = scaler_train.inverse_transform([test_Y])

# Prediction accuracy calculation
train_score = math.sqrt(mean_squared_error(train_Y[0], train_predict[:, 0]))
print('Train Score: %.2f RMSE' % (train_score))
test_score = math.sqrt(mean_squared_error(test_Y[0], test_predict[:, 0]))
print('Test  Score: %.2f RMSE' % (test_score))

# Data Formatting for Plotting
train_predict_plot = np.empty_like(dataset)
train_predict_plot[:, :] = np.nan
train_predict_plot[look_back:len(train_predict)+look_back, :] = train_predict
test_predict_plot = np.empty_like(dataset)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict)+(look_back*2):len(dataset), :] = test_predict

# plot data
plt.title("monthly-champagne-sales-in-1000s")
plt.xlabel("time(month)")
plt.ylabel("Earnings")
plt.plot(dataset, label='dataset')
plt.plot(train_predict_plot, label='train_predict')
plt.plot(test_predict_plot, label='test_predict')
plt.legend(loc='lower right')
plt.show()
