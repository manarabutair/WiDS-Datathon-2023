import catboost as cb
import lightgbm as lgb
import pandas as pd
import json
import os
import re

path_to_json = './'

json_text = json.load(open(os.path.join(path_to_json, 'Settings.json')))

print("Reading Data...")
train_1 = pd.read_csv(json_text['CLEAN_PATH'] + "cleaned_train1.csv")
train_2 = pd.read_csv(json_text['CLEAN_PATH'] + "cleaned_train2.csv")

train_1 = train_1.rename(columns=lambda x: re.sub('[^A-Za-z0-9_-]+', '', x))
train_2 = train_2.rename(columns=lambda x: re.sub('[^A-Za-z0-9_-]+', '', x))

y1 = train_1['contest-tmp2m-14d__tmp2m']
train_1.drop('contest-tmp2m-14d__tmp2m', axis=1, inplace=True)

y2 = train_2['contest-tmp2m-14d__tmp2m']
train_2.drop('contest-tmp2m-14d__tmp2m', axis=1, inplace=True)

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'max_depth': 8,
    'learning_rate': 0.03,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 100,
    'subsample_for_bin': 200000,
    'n_estimators': 15000,
    'device_type': 'gpu'
}

print("Training Lightgbm")
lgb1 = lgb.LGBMRegressor(**params)

lgb2 = lgb.LGBMRegressor(**params)

print("Training first lightgbm model...")

if os.path.exists(json_text["MODELS_DIR"] + 'lgbm1.txt'):
    os.remove(json_text["MODELS_DIR"] + 'lgbm1.txt')

lgb1.fit(train_1, y1)
lgb1.booster_.save_model(json_text["MODELS_DIR"] + 'lgbm1.txt')

print("First lightgbm model trained")

print("Training second lightgbm model...")

if os.path.exists(json_text["MODELS_DIR"] + 'lgbm2.txt'):
    os.remove(json_text["MODELS_DIR"] + 'lgbm2.txt')

lgb2.fit(train_2, y2)

lgb2.booster_.save_model(json_text["MODELS_DIR"] + 'lgbm2.txt')
print("Second lightgbm model trained")

params = {
    'iterations': 25000,
    'verbose': 1000,
    'learning_rate': 0.0980689972639084,
    'l2_leaf_reg': 2.3722386345448316,
    'max_depth': int(6.599144674342465),
    'loss_function': 'RMSE',
    'model_size_reg': 0.4833187897595954,
    'task_type': "GPU"
}

print("Training Catboost")
catboost1 = cb.CatBoostRegressor(**params)
catboost2 = cb.CatBoostRegressor(**params)

print("Training first catboost model...")

catboost1.fit(train_1, y1)

if os.path.exists(json_text["MODELS_DIR"] + 'cb1.cbm'):
    os.remove(json_text["MODELS_DIR"] + 'cb1.cbm')

catboost1.save_model(json_text["MODELS_DIR"] + 'cb1.cbm')

print("First catboost model trained")

print("Training second catboost model...")

catboost2.fit(train_2, y2)

if os.path.exists(json_text["MODELS_DIR"] + 'cb2.cbm'):
    os.remove(json_text["MODELS_DIR"] + 'cb2.cbm')

catboost2.save_model(json_text["MODELS_DIR"] + 'cb2.cbm')

print("Second catboost model trained")

print("Saving Models!")







