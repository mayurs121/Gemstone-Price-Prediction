import os
import sys

import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(xtrain,ytrain,xtest,ytest,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(xtrain,ytrain)

            # Predict Training data
            y_train_pred = model.predict(xtrain)

            # Predict Testing data
            y_test_pred =model.predict(xtest)

            # Get R2 scores for train and test data
            train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(ytest,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured in evaluate_models (utils)')
        raise CustomException(e,sys)
    
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function')
        raise CustomException(e,sys)