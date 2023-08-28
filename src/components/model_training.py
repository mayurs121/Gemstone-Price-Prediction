# Basic Import
import numpy as np
import pandas as pd
from dataclasses import dataclass
import sys
import os

# Modelling
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models





@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def model_trainer_start(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            xtrain, ytrain, xtest, ytest = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "GradientBoosting Regressor":GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report:dict = evaluate_models(xtrain,ytrain,xtest,ytest,models)

            
            logging.info(f'Model Report : {model_report}')
            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6 :
                logging.info('Best model has r2 Score less than 60%')
                raise CustomException('No Best Model Found')
            
            
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            logging.info('Hyperparameter tuning started for catboost')

            # Hyperparameter tuning on Catboost
            # Initializing catboost
            cbr = CatBoostRegressor(verbose=False)

            # Creating the hyperparameter grid
            param_dist = {'depth'          : [4,5,6,7,8,9, 10],
                          'learning_rate' : [0.01,0.02,0.03,0.04],
                          'iterations'    : [300,400,500,600]}

            #Instantiate RandomSearchCV object
            rscv = RandomizedSearchCV(cbr , param_dist, scoring='r2', cv =5, n_jobs=-1)

            # Fit the model
            rscv.fit(xtrain, ytrain)

            best_cbr = rscv.best_estimator_

            logging.info('Hyperparameter tuning complete for Catboost')

            logging.info('Hyperparameter tuning started for KNN')

            # Initialize knn
            knn = KNeighborsRegressor()

            # parameters
            k_range = list(range(2, 31))
            param_grid = dict(n_neighbors=k_range)

            # Fitting the cvmodel
            grid = GridSearchCV(knn, param_grid, cv=5, scoring='r2',n_jobs=-1)
            grid.fit(xtrain, ytrain)

            best_knn = grid.best_estimator_

            logging.info('Hyperparameter tuning Complete for KNN')

            logging.info('Voting Regressor model training started')

            # Creating final Voting regressor
            vr = VotingRegressor([('cbr',best_cbr),('xgb',XGBRegressor()),('knn',best_knn)], weights=[3,2,1])
            vr.fit(xtrain, ytrain)
           
            logging.info('Voting Regressor Training Completed')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = vr
            )
            logging.info('Model pickle file saved')
            # Evaluating Ensemble Regressor (Voting Classifier on test data)
            predicted = vr.predict(xtest)


            r2_Scr = r2_score(ytest, predicted)
            
            return r2_Scr
        
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)