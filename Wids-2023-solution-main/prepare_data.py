import json
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

path_to_json = './'

json_text = json.load(open(os.path.join(path_to_json, 'Settings.json')))


def parse_date(df):
    df['year'] = df['startdate'].dt.year
    df['month'] = df['startdate'].dt.month
    df['day'] = df['startdate'].dt.day
    df['dayofyear'] = df.startdate.dt.day_of_year


print("Reading Data...")
train = pd.read_csv(json_text['RAW_DATA_DIR'] + "train_data.csv", parse_dates=['startdate'])
test = pd.read_csv(json_text['RAW_DATA_DIR'] + "test_data.csv", parse_dates=['startdate'])

print("Psuedo Labelling...")
target = 'contest-tmp2m-14d__tmp2m'
train_pseudo = test.copy()
y_test_pred = pd.read_csv(json_text['PSUEDO_SUB_FILE_PATH'])['contest-tmp2m-14d__tmp2m']
train_pseudo[target] = y_test_pred

X_train_pseudo = train.copy()
X_train_pseudo[target] = train['contest-tmp2m-14d__tmp2m']

train_mod = pd.concat([X_train_pseudo.copy(), train_pseudo], axis=0).reset_index(drop=True)
features = [c for c in test.columns if (c != 'id')]
features.append(target)

train = train_mod[features]

print("Psuedo Labeling Done!")

scale = 14

train.loc[:, 'lat'] = round(train.lat, scale)
train.loc[:, 'lon'] = round(train.lon, scale)

test.loc[:, 'lat'] = round(test.lat, scale)
test.loc[:, 'lon'] = round(test.lon, scale)

train['loc_group'] = train.groupby(['lat', 'lon']).ngroup()
test['loc_group'] = test.groupby(['lat', 'lon']).ngroup()

train = train.sort_values(by=['loc_group', 'startdate']).ffill()

parse_date(train)
parse_date(test)
print("Main preprocessing done!")

train1 = train.copy()
train2 = train.copy()

test1 = test.copy()
test2 = test.copy()

print("Preparing first data")
le = LabelEncoder()
train1['climateregions__climateregion'] = le.fit_transform(train1['climateregions__climateregion'])
test1['climateregions__climateregion'] = le.transform(test1['climateregions__climateregion'])

irrelevant_cols = ['index', 'startdate']
features1 = [col for col in test.columns if col not in irrelevant_cols]
features1_train = features1 + ['contest-tmp2m-14d__tmp2m']

train1 = train1[features1_train]
test1 = test1[features1]
print("First data prepared!")

# Second Model

print("Preparing second data")
features_to_keep = ['lon',
                    'contest-rhum-sig995-14d__rhum',
                    'nmme0-prate-34w__ccsm30',
                    'contest-slp-14d__slp',
                    'contest-wind-vwnd-925-14d__wind-vwnd-925',
                    'nmme-prate-56w__ccsm3',
                    'nmme-prate-56w__nasa',
                    'nmme-prate-56w__nmmemean',
                    'contest-wind-uwnd-250-14d__wind-uwnd-250',
                    'contest-prwtr-eatm-14d__prwtr',
                    'contest-wind-vwnd-250-14d__wind-vwnd-250',
                    'contest-precip-14d__precip',
                    'contest-wind-h850-14d__wind-hgt-850',
                    'contest-wind-uwnd-925-14d__wind-uwnd-925',
                    'contest-wind-h500-14d__wind-hgt-500',
                    'elevation__elevation',
                    'wind-vwnd-250-2010-2',
                    'wind-vwnd-250-2010-3',
                    'wind-vwnd-250-2010-4',
                    'wind-vwnd-250-2010-5',
                    'wind-vwnd-250-2010-6',
                    'wind-vwnd-250-2010-7',
                    'wind-vwnd-250-2010-8',
                    'wind-vwnd-250-2010-9',
                    'wind-vwnd-250-2010-10',
                    'wind-vwnd-250-2010-11',
                    'wind-vwnd-250-2010-12',
                    'wind-vwnd-250-2010-13',
                    'wind-vwnd-250-2010-14',
                    'wind-vwnd-250-2010-15',
                    'wind-vwnd-250-2010-16',
                    'wind-vwnd-250-2010-17',
                    'wind-vwnd-250-2010-18',
                    'wind-vwnd-250-2010-19',
                    'wind-vwnd-250-2010-20',
                    'wind-uwnd-250-2010-3',
                    'wind-uwnd-250-2010-4',
                    'wind-uwnd-250-2010-5',
                    'wind-uwnd-250-2010-6',
                    'wind-uwnd-250-2010-7',
                    'wind-uwnd-250-2010-8',
                    'wind-uwnd-250-2010-9',
                    'wind-uwnd-250-2010-10',
                    'wind-uwnd-250-2010-11',
                    'wind-uwnd-250-2010-12',
                    'wind-uwnd-250-2010-13',
                    'wind-uwnd-250-2010-14',
                    'wind-uwnd-250-2010-15',
                    'wind-uwnd-250-2010-16',
                    'wind-uwnd-250-2010-17',
                    'wind-uwnd-250-2010-18',
                    'wind-uwnd-250-2010-19',
                    'wind-uwnd-250-2010-20',
                    'mjo1d__phase',
                    'mjo1d__amplitude',
                    'mei__nip',
                    'wind-hgt-850-2010-2',
                    'wind-hgt-850-2010-4',
                    'wind-hgt-850-2010-6',
                    'wind-hgt-850-2010-7',
                    'wind-hgt-850-2010-9',
                    'wind-hgt-850-2010-10',
                    'sst-2010-2',
                    'sst-2010-3',
                    'sst-2010-4',
                    'sst-2010-5',
                    'sst-2010-6',
                    'sst-2010-7',
                    'sst-2010-8',
                    'sst-2010-9',
                    'sst-2010-10',
                    'wind-hgt-500-2010-2',
                    'wind-hgt-500-2010-4',
                    'wind-hgt-500-2010-5',
                    'wind-hgt-500-2010-6',
                    'wind-hgt-500-2010-7',
                    'wind-hgt-500-2010-8',
                    'wind-hgt-500-2010-9',
                    'wind-hgt-500-2010-10',
                    'icec-2010-2',
                    'icec-2010-3',
                    'icec-2010-4',
                    'icec-2010-6',
                    'icec-2010-7',
                    'icec-2010-8',
                    'icec-2010-9',
                    'icec-2010-10',
                    'wind-uwnd-925-2010-2',
                    'wind-uwnd-925-2010-3',
                    'wind-uwnd-925-2010-4',
                    'wind-uwnd-925-2010-5',
                    'wind-uwnd-925-2010-6',
                    'wind-uwnd-925-2010-7',
                    'wind-uwnd-925-2010-8',
                    'wind-uwnd-925-2010-9',
                    'wind-uwnd-925-2010-10',
                    'wind-uwnd-925-2010-11',
                    'wind-uwnd-925-2010-12',
                    'wind-uwnd-925-2010-13',
                    'wind-uwnd-925-2010-14',
                    'wind-uwnd-925-2010-15',
                    'wind-uwnd-925-2010-16',
                    'wind-uwnd-925-2010-17',
                    'wind-uwnd-925-2010-18',
                    'wind-uwnd-925-2010-19',
                    'wind-uwnd-925-2010-20',
                    'wind-hgt-10-2010-3',
                    'wind-hgt-10-2010-4',
                    'wind-hgt-10-2010-5',
                    'wind-hgt-10-2010-6',
                    'wind-hgt-10-2010-7',
                    'wind-hgt-10-2010-8',
                    'wind-hgt-10-2010-9',
                    'wind-hgt-10-2010-10',
                    'wind-hgt-100-2010-2',
                    'wind-hgt-100-2010-3',
                    'wind-hgt-100-2010-4',
                    'wind-hgt-100-2010-5',
                    'wind-hgt-100-2010-6',
                    'wind-hgt-100-2010-7',
                    'wind-hgt-100-2010-8',
                    'wind-hgt-100-2010-9',
                    'wind-hgt-100-2010-10',
                    'wind-vwnd-925-2010-1',
                    'wind-vwnd-925-2010-2',
                    'wind-vwnd-925-2010-3',
                    'wind-vwnd-925-2010-4',
                    'wind-vwnd-925-2010-5',
                    'wind-vwnd-925-2010-6',
                    'wind-vwnd-925-2010-7',
                    'wind-vwnd-925-2010-8',
                    'wind-vwnd-925-2010-9',
                    'wind-vwnd-925-2010-10',
                    'wind-vwnd-925-2010-11',
                    'wind-vwnd-925-2010-12',
                    'wind-vwnd-925-2010-13',
                    'wind-vwnd-925-2010-14',
                    'wind-vwnd-925-2010-15',
                    'wind-vwnd-925-2010-16',
                    'wind-vwnd-925-2010-17',
                    'wind-vwnd-925-2010-18',
                    'wind-vwnd-925-2010-19',
                    'wind-vwnd-925-2010-20',
                    'loc_group',
                    'climateregions__climateregion',
                    'year',
                    'day',
                    'dayofyear']

features_to_keep_train = features_to_keep + ['contest-tmp2m-14d__tmp2m']

train2 = train2[features_to_keep_train]
test2 = test2[features_to_keep]

train2 = pd.get_dummies(data=train2, columns=['climateregions__climateregion'])
test2 = pd.get_dummies(data=test2, columns=['climateregions__climateregion'])
print("Second data prepared!")

print("Saving cleaned datasets")

if os.path.exists(json_text["CLEAN_PATH"] + "cleaned_train1.csv"):
    os.remove(json_text["CLEAN_PATH"] + "cleaned_train1.csv")

if os.path.exists(json_text["CLEAN_PATH"] + "cleaned_train2.csv"):
    os.remove(json_text["CLEAN_PATH"] + "cleaned_train2.csv")

if os.path.exists(json_text["CLEAN_PATH"] + "cleaned_test1.csv"):
    os.remove(json_text["CLEAN_PATH"] + "cleaned_test1.csv")

if os.path.exists(json_text["CLEAN_PATH"] + "cleaned_test2.csv"):
    os.remove(json_text["CLEAN_PATH"] + "cleaned_test2.csv")

train1.to_csv(json_text["CLEAN_PATH"] + "cleaned_train1.csv")
train2.to_csv(json_text["CLEAN_PATH"] + "cleaned_train2.csv")
test1.to_csv(json_text["CLEAN_PATH"] + "cleaned_test1.csv")
test2.to_csv(json_text["CLEAN_PATH"] + "cleaned_test2.csv")

print("Done!")
