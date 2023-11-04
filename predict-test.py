#!/usr/bin/env python
# coding: utf-8

import requests
import pandas as pd

from io import StringIO


url = 'http://localhost:9696/predict'

test_data_file = 'data/test_data.csv'

df = pd.read_csv(test_data_file)
df_json = df.to_json()

response = requests.post(url, json=df_json).json()

df = pd.DataFrame(response).transpose()
df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), origin='unix', unit='ms')
print(f"Sleep events for the patient:")
print(df)

