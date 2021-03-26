# 回帰分析を行うLSTMモデルです
# モジュールを読み込む
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from sklearn.preprocessing import StandardScaler
import json
plt.style.use('ggplot')

# window_sizeに分けて時系列データのデータセットを作成
def apply_window(data, window_size):
    sequence_length = window_size
    window_data = []
    for index in range(len(data) - window_size):
        window = data[index: index + sequence_length]
        window_data.append(window)
    return np.array(window_data)

# 訓練データとテストデータに分ける
def split_train_test(data, train_rate=0.9):
    # データの7割（デフォルト値0.7）を訓練用データとし、残りをテスト用データとする
    row = round(train_rate * data.shape[0])
    train = data[:row]
    test = data[row:]
    return train, test

def data_load():
    # ファイル読み込み
    df_uniqlo_stock = pd.read_csv("./UNIQLO_data_2015-2020.csv")
    # Data列を日付データとして認識
    df_uniqlo_stock['Date'] = pd.to_datetime(df_uniqlo_stock['Date'])
    # 日付順に並び替え
    df_uniqlo_stock.sort_values(by='Date', inplace=True)
    # 終値を抽出
    close_ts = df_uniqlo_stock['Close']
    close_ts = np.expand_dims(close_ts, 1)
    return close_ts

def train_model(X_train, y_train, units):
    # 入力データの形式を取得
    input_size = X_train.shape[1:]

    # レイヤーを定義
    model = Sequential()
    model.add(LSTM(input_shape=input_size,units=units,dropout = 0.1,return_sequences=True,))
    model.add(LSTM(32,return_sequences=False))
    model.add(Dense(units=1))
    model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
    
    model.fit(X_train, y_train,epochs=10, verbose=2, validation_split=0.3 ,shuffle=False)
    return model

def predict_ten_days(data, model,day):
    # 何日後を予測するか
    prediction_len = day
    prediction_seqs = []

    curr_frame = data.copy()
    data_len = len(curr_frame.flatten())
    predicted = []

    for j in range(prediction_len):
        # データを入力して予測
        pred = model.predict(curr_frame.flatten().reshape(1,-1,1),verbose=0)
        # 予測結果を取得
        prediction_seqs.append(pred[0,0])
        #print(prediction_seqs)
        # 入力データを１つ削り、予測結果を次の入力データに利用
        curr_frame = curr_frame[1:, :]
        #print(curr_frame)
        curr_frame = np.insert(curr_frame, data_len - 1, pred[0,0], axis=0)
        #print(curr_frame)
    return prediction_seqs

# モデルに入力するデータ長
window_size = 15


##############################
#with open("./covid19-master/data/data.json", "r") as f:
#    train = json.load(f)

#data=train["transition"]["carriers"]
#dataset=[row[3] for row in data]
#dataset=np.array(dataset)
#close_ts=dataset.astype('float32')
#close_ts = np.expand_dims(close_ts, 1)

##############################
# ユニクロの株価・終値を取得
close_ts = data_load()
#print(len(close_ts))
#print(close_ts)
#print(type(close_ts))
# データを訓練用・学習用に分割
train, test = split_train_test(close_ts)
train_len=len(train)
#print(train_len)
#print(len(test))
# データを正規化
scaler = StandardScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)

# window_size+1の長さで、横に1個ずつずらしてトレーニングデータを作る
train = apply_window(train, window_size+1)
#print(train)
# window_size+1の長さで、横に1個ずつずらしてテストデータを作る
test = apply_window(test, window_size+1)

# 訓練用の入力データ
X_train = train[:, :-1]
#print(X_train.shape)
# 訓練用の正解ラベル
y_train = train[:,  -1]

# テスト用の入力データ
#print(test)
X_test = test[:, :-1]
#print(X_test)
# テスト用の正解ラベル
y_test = test[:,  -1]

# 学習モデルを取得
model = train_model(X_train, y_train,units=64)

# テストデータをどの位置から予測データに使うかを指定する
start_point = 111
day=20
#print(X_test.shape)
#print("_____________________")
#print(X_test[start_point])
#print("_____________________")
prediction_seqs = predict_ten_days(X_test[start_point], model,day)

print(y_test[start_point-1])
print(prediction_seqs)

# 結果の出力
plt.figure(figsize=(15, 8))
#plt.plot(range(0,len(close_ts)),close_ts,label='full Data',color='red')
#plt.plot(range(start_point-window_size+train_len, start_point+train_len), scaler.inverse_transform(y_test[start_point-window_size:start_point]), label='True Data',color='green')
plt.plot(range(start_point - 1+train_len, start_point + day+train_len), scaler.inverse_transform(np.insert(prediction_seqs, 0, y_test[start_point-1])), label='Prediction',color="blue")
#plt.plot(range(start_point - 1+train_len, start_point + day+train_len), scaler.inverse_transform(np.array(prediction_seqs)), label='Prediction',color="blue")
plt.legend()
plt.show()


#a = np.arange(4)
#print(a)
# [0 1 2 3]
#print(np.insert(a, 2, 100))
# [  0   1 100   2   3]

