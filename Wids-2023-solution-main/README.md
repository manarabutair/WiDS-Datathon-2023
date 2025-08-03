# Hello!

Below you can find a outline of how to reproduce my solution for the WiDS Datathon 2023 competition.
If you run into any trouble with the setup/code or have any questions feel free to contact me.

## CONTENTS
prepare_data.py     :  This code run does preprocessing methods applied to the data\
train.py            :  This code trains the data on 4 models\
predict.py          :  This code predicts the test data

## HARDWARE
Processor  : Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz, 2496 Mhz, 4 Core(s), 8 Logical Processor(s)\
RAM        : 16.0 GB 2667MHz\
GPU        : NVIDIA GeForce GTX 1650

## SOFTWARE (python packages are detailed separately in `requirements.txt`):
Python 3.10.10

## DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)
mkdir -p data/\
mkdir -p data/raw/\
mkdir -p data/clean/\
mkdir -p submission/\
mkdir -p models/\
cd data/raw/\
kaggle competitions download -c widsdatathon2023 -f train_data.csv\
kaggle competitions download -c widsdatathon2023 -f test_data.csv

## DATA PROCESSING
Expected to run for 5 minutes\
Uses psuedo labeling from a previously trained model\
Includes few features extraction and feature selections\
Saves 4 files, 2 training files and 2 test files each test and train contains a certain set of features\
To run the code for preprocessing: python ./prepare_data.py

## MODEL BUILD
Model building is expected to run for 1 hour\
Contains 2 lightgbm models and 2 catboost models, each of these 2 models trained on the cleaned trained data\
The 4 models are saved in models folder\
To run the code for model building: python ./train.py

## MODEL PREDICTIONS
Model predictions should come out instantly\
The code reads the cleaned test files, and the built models for predictions\
Each model predicts the result and then the predictions are ensembled together\
The latest predictions are then saved in the submission folder\
To run the code for model predictions: python ./predict.py
