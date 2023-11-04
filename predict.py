import pickle
import numpy as np
import polars as pl
import pandas as pd
import xgboost as xgb

from io import StringIO

from flask import Flask
from flask import request
from flask import jsonify

from helpers import create_rolling_features, create_sleeping_time_vars, clean_predictions, get_events


model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
	model = pickle.load(f_in)

app = Flask('sleep_detection')

@app.route('/predict', methods=['POST'])
def predict():
	sleep_series = request.get_json()
	df = pd.read_json(StringIO(sleep_series))
	df_pl = pl.from_pandas(df)

	# Create features
	df_pl = df_pl.with_columns(
		pl.col('dt_minute').str.to_datetime(),
		pl.col('dt_minute').str.to_datetime().dt.hour().alias('hour')
	)

	df_pl = create_sleeping_time_vars(df_pl)

	df_pl = create_rolling_features(df_pl)

	with open('data/features_to_exclude.txt') as f:
		vars_to_exclude = [line.rstrip() for line in f]
	df_pl = df_pl.drop(vars_to_exclude)

	features = [f for f in df_pl.columns if f not in ["dt_minute","step","series_id"]]
	df_pd = df_pl.to_pandas()
	X = df_pd[features]
	
	dX = xgb.DMatrix(X, feature_names=features)
	y_proba_multi = model.predict(dX)
	y_proba = y_proba_multi[:,1]
	y_pred = np.argmax(y_proba_multi, axis=1)
	df_pd["probability"] = y_proba
	df_pd["prediction"] = y_pred
	
	# Clean predictions
	df_pd = clean_predictions(df_pd)
	
	# Get onsets and wakeups
	df_events = get_events(pl.from_pandas(df_pd))
	
	return df_events.to_pandas().to_json(orient="index")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
