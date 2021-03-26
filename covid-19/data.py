import json
import numpy as np
import pandas as pd

data = pd.read_csv("./new_cases.csv")


# Data列を日付データとして認識
data['date'] = pd.to_datetime(data['date'])
# 日付順に並び替え
data.sort_values(by='date', inplace=True)
# 終値を抽出
close_ts = data['United States']
close_ts = np.expand_dims(close_ts, 1)
print(close_ts)
