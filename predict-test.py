#!/usr/bin/env python
# coding: utf-8

import json
import requests
import numpy as np
import pandas as pd


url = 'http://localhost:9696/predict'

test_data_file = 'data/test_data.json'

 
# Opening JSON file
with open(test_data_file) as json_file:
    data = json.load(json_file)
    
response = requests.post(url, json=data).json()

df = pd.DataFrame(response).transpose()
df["timestamp"] = pd.to_datetime(df["timestamp"].astype(np.int64), origin='unix', unit='ms')
print(f"Sleep events for the patient:")
print(df)
