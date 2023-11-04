#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import datetime
import pickle
import copy
import matplotlib.pyplot as plt

import polars as pl
pl.Config.set_fmt_str_lengths(100)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

import xgboost as xgb
from xgboost import plot_importance

from helpers import create_rolling_features, create_sleeping_time_vars, calculate_metrics


# Parameters
K_FOLD_SPLITS = 5
MAX_DEPTH = 6
MIN_CHILD_WEIGHT = 250
MODEL_FILE_PATH = 'model.bin'
LABELS = "target"

# Load the data
train_data_file = 'data/sleep_data_120series_7days.csv'
df = pl.read_csv(train_data_file, dtypes={
    "dt_minute":pl.Datetime
}).sort(by=["series_id","dt_minute"])


# Load the list of correlated features to exclude
with open('data/features_to_exclude.txt') as f:
    vars_to_exclude = [line.rstrip() for line in f]
    

# Prepare the data
# Add hour and sleep time variables
df = df.with_columns(
    pl.col('dt_minute').dt.hour().alias('hour')
)
df = create_sleeping_time_vars(df)


# Create rolling features
df = create_rolling_features(df)
df = df.drop(vars_to_exclude)

# Split for train_full and test set
# Determine sizes
series_ids_shuffle = df["series_id"].unique().sort().shuffle(seed=4)
test_set_size = int(series_ids_shuffle.shape[0] * 0.2)
train_set_size = series_ids_shuffle.shape[0] - test_set_size

# Select sets
df_train_night_ids = series_ids_shuffle[:train_set_size]
df_test_night_ids = series_ids_shuffle[train_set_size:]

df_train_full = df.filter(pl.col('series_id').is_in(df_train_night_ids))
df_test = df.filter(pl.col('series_id').is_in(df_test_night_ids))


def k_fold_split(df, k):

	# Shuffle series_id's
	series_ids_shuffle = df["series_id"].unique().sort().shuffle(seed=4)
	val_set_size = int(series_ids_shuffle.shape[0] * (1/k))

	# Save each of k parts as an element of a list to later use k-fold in validation
	val_sets = []
	for n in range(k):
		val_set_ids = series_ids_shuffle[n*val_set_size:(n+1)*val_set_size]
		val_set = df.filter(pl.col('series_id').is_in(val_set_ids))
		val_set_pd = val_set.to_pandas()
		val_sets.append(val_set_pd)
	return val_sets
    

def train(dtrain, dval, max_depth, min_child_weight):
    
	watchlist = [(dtrain, 'train'), (dval, 'val')]

	xgb_params = {
		'eta': 0.3, 
		'max_depth': max_depth,
		'min_child_weight': min_child_weight,
		'objective': 'multi:softprob', 
		'num_class': 3,
		'eval_metric': 'auc',
		'nthread': 8,
		'seed': 1,
		'verbosity': 1
	}

	model = xgb.train(
		xgb_params, 
		dtrain,
		evals=watchlist,
		verbose_eval=10, 
		num_boost_round=30
	)
	return model

def predict(model, dtest):
	y_test_proba = model.predict(dtest)
	y_test_pred = np.argmax(y_test_proba, axis=1)
	return y_test_pred

def get_f1_score(y_test_pred,y_test):
	_, _, sleep_f1_score = calculate_metrics(y_test_pred,y_test,target=1)
	return sleep_f1_score


# K-Fold Validation
val_sets = k_fold_split(df_train_full, K_FOLD_SPLITS)

for split in range(K_FOLD_SPLITS):
	print(f"Running k-fold validation for splin number: {split}.")
	val_sets_copy = copy.deepcopy(val_sets)
	
	df_val = val_sets_copy.pop(split)
	df_train = pd.concat(val_sets_copy)

	y_val = df_val[LABELS].values
	y_train = df_train[LABELS].values
	del df_val[LABELS] 
	del df_train[LABELS] 
	features = [f for f in df_train.columns if f not in ["dt_minute","step","series_id"]]

	X_train = df_train[features]
	X_val = df_val[features]
	
	dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
	dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)
	model = train(dtrain, dval, max_depth=MAX_DEPTH, min_child_weight=MIN_CHILD_WEIGHT)
	y_val_pred = predict(model, dval)
	sleep_f1_score = get_f1_score(y_val_pred, y_val)
	print(f'Sleeping F1 score: {sleep_f1_score}')


# Training the final model
print('Training the final model')
df_train_full_pd = df_train_full.to_pandas()
df_test_pd = df_test.to_pandas()
y_train_full = df_train_full_pd[LABELS].values
y_test = df_test_pd[LABELS].values
del df_train_full_pd[LABELS] 
del df_test_pd[LABELS]

features = [f for f in df_train_full_pd.columns if f not in ["dt_minute","step","series_id"]]
features
X_train_full = df_train_full_pd[features]
X_test = df_test_pd[features]

dtrain_full = xgb.DMatrix(X_train_full, label=y_train_full, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)

model = train(dtrain_full, dtest, max_depth=MAX_DEPTH, min_child_weight=MIN_CHILD_WEIGHT)
y_test_pred = predict(model, dtest)
sleep_f1_score = get_f1_score(y_test_pred,y_test)
print(f'Sleeping F1 score: {sleep_f1_score}')


# Save model to pickel file
pickle.dump(model, open(MODEL_FILE_PATH, 'wb'))
print(f'The model is saved to {MODEL_FILE_PATH}')

