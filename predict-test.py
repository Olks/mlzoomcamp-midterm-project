#!/usr/bin/env python
# coding: utf-8

import requests
import numpy as np
import pandas as pd


url = 'http://localhost:9696/predict'

test_data_file = 'data/test_data.csv'

df = pd.read_csv(test_data_file)
df_json = df.to_dict()

response = requests.post(url, json=df_json).json()

df = pd.DataFrame(response).transpose()
df["timestamp"] = pd.to_datetime(df["timestamp"].astype(np.int64), origin='unix', unit='ms')
print(f"Sleep events for the patient:")
print(df)
