import catboost as cb
import lightgbm as lgb
import pandas as pd
import json
import os
import re

path_to_json = './'

json_text = json.load(open(os.path.join(path_to_json, 'Settings.json')))

print("Reading Data...")
test = pd.read_csv(json_text["RAW_DATA_DIR"] + "test_data.csv")
test_1 = pd.read_csv(json_text['CLEAN_PATH'] + "cleaned_test1.csv")
test_2 = pd.read_csv(json_text['CLEAN_PATH'] + "cleaned_test2.csv")

test_1 = test_1.rename(columns=lambda x: re.sub('[^A-Za-z0-9_-]+', '', x))
test_2 = test_2.rename(columns=lambda x: re.sub('[^A-Za-z0-9_-]+', '', x))

print("Loading models...")
lgb1 = lgb.Booster(model_file=json_text["MODELS_DIR"] + 'lgbm1.txt')
lgb2 = lgb.Booster(model_file=json_text["MODELS_DIR"] + 'lgbm2.txt')

cb1 = cb.CatBoostRegressor()
cb2 = cb.CatBoostRegressor()
cb1 = cb1.load_model(json_text["MODELS_DIR"] + 'cb1.cbm')
cb2 = cb2.load_model(json_text["MODELS_DIR"] + 'cb2.cbm')

print("Predicting data...")
preds_cb1 = cb1.predict(test_1)
preds_cb2 = cb2.predict(test_2)

preds_lgb1 = lgb1.predict(test_1)
preds_lgb2 = lgb2.predict(test_2)

print("Predictions done!")

preds1 = preds_cb1 * 0.30 + preds_lgb1 * 0.70
preds2 = preds_cb2 * 0.70 + preds_lgb2 * 0.30

predictions = preds1 * 0.90 + preds2*0.10

print("Saving submission...")
submission = pd.DataFrame()
submission['index'] = test['index']
submission['contest-tmp2m-14d__tmp2m'] = predictions

if os.path.exists(json_text["SUBMISSION_DIR"] + 'sub.csv'):
    os.remove(json_text["SUBMISSION_DIR"] + 'sub.csv')

submission.set_index("index").to_csv(json_text["SUBMISSION_DIR"] + 'sub.csv')

print("Done!")
