# 回帰分析を行うLSTMモデルです
# モジュールを読み込む
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.layers.recurrent import LSTM,SimpleRNN
from sklearn.preprocessing import StandardScaler
import json
plt.style.use('ggplot')

# window_sizeに分けて時系列データのデータセットを作成
def apply_window(data, window_size):
    window_data = []
    if(len(data)==window_size):
        window_data.append(data)
        return np.array(window_data)
    for index in range(len(data) - window_size):
        window = data[index: index + window_size]
        window_data.append(window)
    return np.array(window_data)

# 訓練データとテストデータに分ける
def split_train_test(data, window_size):
    # データの7割（デフォルト値0.7）を訓練用データとし、残りをテスト用データとする
    data_len=len(data)
    train = data[:data_len-1]
    test = data[data_len-window_size-1:]
    return train, test

def data_load():
    data = pd.read_csv("./total_cases.csv")
    data['date'] = pd.to_datetime(data['date'])
    data.sort_values(by='date', inplace=True)
    data_US = data['United States']
    data_Ja = data['Japan']
    data_US = np.expand_dims(data_US, 1)
    data_Ja = np.expand_dims(data_Ja, 1)
    return data_Ja,data_US


def train_model(X_train, y_train):
    # 入力データの形式を取得
    # print(y_train.shape)
    input_size = X_train.shape[1:]
    #print(input_size)
    # レイヤーを定義
    model = Sequential()
    model.add(LSTM(input_shape=input_size,units=30,return_sequences=False,batch_size=1))
    model.add(Dense(units=2))
    model.add(Activation("linear"))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x=X_train,y=y_train,epochs=10, verbose=2,shuffle=False,batch_size=1)

    return model

def predict_ten_days(data, model,day):
    # 何日後を予測するか
    prediction_len = day
    #prediction_seqs = []

    curr_frame = data.copy()
    #data_len = len(curr_frame.flatten())

    for j in range(prediction_len):
        # データを入力して予測
        #print(curr_frame.flatten().reshape(1,-1,2))
        #print(curr_frame.flatten().reshape(1,-1,2).shape)
        pred = model.predict(curr_frame.flatten().reshape(1,-1,2),verbose=0)
        #print(pred)
        #print(pred[0])

        # 予測結果を取得
        if j==0:
            prediction_seqs=[pred[0]]
        else :
            prediction_seqs=np.vstack((prediction_seqs, pred[0]))

        # 入力データを１つ削り、予測結果を次の入力データに利用
        #print(curr_frame[1:, :])
        curr_frame = curr_frame[1:, :]
        curr_frame = np.insert(curr_frame, len(curr_frame)-1, pred[0], axis=0)

    #print(prediction_seqs)
    return prediction_seqs

# モデルに入力するデータ長
window_size = 15

#ユニクロの株価・終値を取得
data_Ja,data_US = data_load()

#print(data_US.shape)
#print(data_Ja.shape)

data=np.concatenate([data_Ja,data_US],axis=1)
#print(data.shape)
# データを訓練用・学習用に分割
train, test = split_train_test(data,window_size)
print(train.shape)
print(test.shape)


# データを正規化
scaler = StandardScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)

# window_size+1の長さで、横に1個ずつずらしてトレーニングデータを作る
train = apply_window(train, window_size+1)
#print(train.shape)
# window_size+1の長さで、横に1個ずつずらしてテストデータを作る
test = apply_window(test, window_size+1)
#print(test.shape)
# 訓練用の入力データ
X_train = train[:, :-1]
y_train = train[:,  -1]

#テスト用の入力データ
X_test = test[:, :-1]
#print(X_test.shape)
# テスト用の正解ラベル
y_test = test[:,  -1]
# 学習モデルを取得
model = train_model(X_train, y_train)

# テストデータをどの位置から予測データに使うかを指定する
start_point = 0
day=30
#print(X_test[start_point])
#print(X_test[start_point].shape)
prediction_seqs = predict_ten_days(X_test[start_point], model,day)
print("__________________________________")
print(prediction_seqs)
print("__________________________________")

predict=np.insert(prediction_seqs, 0, y_test[0],axis=0)
#print(predict)
predict_data=scaler.inverse_transform(predict)
p_data_Ja=predict_data[:,0]
#print(p_data_Ja)
p_data_US=predict_data[:,1]
#print(predict)
# 結果の出力
plt.figure(figsize=(15, 8))
plt.plot(range(0,len(data_Ja)),data_Ja,label='full Data',color='red')
plt.plot(range(len(data_Ja)-1, len(data_Ja) + day), p_data_Ja, label='Prediction',color="blue")
plt.legend()
plt.show()


#a = np.arange(4)
#print(a)
# [0 1 2 3]
#print(np.insert(a, 2, 100))
# [  0   1 100   2   3]


