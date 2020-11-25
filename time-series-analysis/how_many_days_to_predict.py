import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from sklearn.preprocessing import StandardScaler
plt.style.use('ggplot')

# # Make it a two-dimensional array so that each array is the length of window_size
def apply_window(data, window_size):
    sequence_length = window_size
    window_data = []
    for index in range(len(data) - window_size):
        window = data[index: index + sequence_length]
        window_data.append(window)
    return np.array(window_data)

# Separate training data and test data
def split_train_test(data, train_rate=0.7):
    row = round(train_rate * data.shape[0])
    train = data[:row]
    test = data[row:]
    return train, test

def data_load():
    df_uniqlo_stock = pd.read_csv("../data/UNIQLO_data_2015-2020.csv")
     # Recognize the Data column as date data
    df_uniqlo_stock['Date'] = pd.to_datetime(df_uniqlo_stock['Date'])
    # Sorted by date
    df_uniqlo_stock.sort_values(by='Date', inplace=True)
     # Extract the end value
    close_ts = df_uniqlo_stock['Close']
    close_ts = np.expand_dims(close_ts, 1)
    return close_ts

def train_model(X_train, y_train, units=15):
     # Get the format of the input data
    input_size = X_train.shape[1:]
    # Define layers
    model = Sequential()
    model.add(LSTM(
            input_shape=input_size,
            units=units,
            dropout = 0.1,
            return_sequences=False,))
    model.add(Dense(units=1))

    model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
    model.fit(X_train, y_train,
                        epochs=10, validation_split=0.3, verbose=2, shuffle=False)
    return model

def predict_ten_days(data, model,day):
    # How many days to predict
    prediction_len = day
    prediction_seqs = []
    curr_frame = data.copy()
    data_len = len(curr_frame.flatten())
    predicted = []

    for j in range(prediction_len):
        # data input and prediction
        pred = model.predict(curr_frame.flatten().reshape(1,-1,1),verbose=0)
        # Get the result of the prediction
        prediction_seqs.append(pred[0,0])
        # Scrape one input data and use the result of the prediction for the next input data
        curr_frame = curr_frame[1:, :]
        curr_frame = np.insert(curr_frame, data_len - 1, pred[0,0], axis=0)
    return prediction_seqs

# Data length to be entered into the model
window_size = 15

# Get UNIQLO stock price and closing price
close_ts = data_load()

# Split data for training and learning
train, test = split_train_test(close_ts)

# Normalize the data
scaler = StandardScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)

# window_size+1 length, displaced one by one horizontally to make the training data
train = apply_window(train, window_size+1)
# Make the test data by shifting the length of window_size+1, one by one across
test = apply_window(test, window_size+1)

X_train = train[:, :-1]
y_train = train[:,  -1]
X_test = test[:, :-1]
y_test = test[:,  -1]

# Get a learning model
model = train_model(X_train, y_train,units=15)

# Specify from which position the test data will be used for prediction data
start_point = 90
day=5
prediction_seqs = predict_ten_days(X_test[start_point], model,day)

# The resulting output
plt.figure(figsize=(15, 8))
plt.plot(range(start_point-window_size, start_point + day), scaler.inverse_transform(y_test[start_point-window_size:start_point + day]), label='True Data')
plt.plot(range(start_point - 1, start_point + day), scaler.inverse_transform(np.insert(prediction_seqs, 0, y_test[start_point-1])), label='Prediction')
plt.legend()
plt.show()
