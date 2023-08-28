import sys 
import os 
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler 
from src.utils import save_object


@dataclass
class DataTransformConfig:
    preprocessor_data_path = os.path.join("artifacts",'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transform_config = DataTransformConfig()

    def data_transformation_object(self):
        '''
        this function returns transormation pipeline as preprocessor
        '''
        try:
            
            # Define categorical & numerical features
            cat_features = ['cut', 'color','clarity']
            num_features = ['carat', 'depth','table', 'x', 'y', 'z']

            #define the categorical variables with their values for ordinal encoding 
            cut_feature = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_feature = ['F','J', 'G', 'E', 'D', 'H', 'I']
            clarity_feature = ['VS2', 'SI2', 'VS1', 'SI1', 'IF', 'VVS2', 'VVS1', 'I1'] 

            #creating pipeline 

            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )  

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encoder',OrdinalEncoder(categories=[cut_feature,color_feature,clarity_feature])),
                    ('scaler',StandardScaler())
                ]
            ) 

            #column transformation  

            from sklearn.compose import ColumnTransformer 

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,num_features),
                    ('cat_pipeline',cat_pipeline,cat_features)
                ]
            )            
           
            return preprocessor
        
        except Exception as e:
            logging.info('Exception occured in Data Transformation Phase')
            raise CustomException(e,sys)
    
    def Data_Transformation_Start(self,train_path,test_path):
        try:
            logging.info("Read train and test data")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.data_transformation_object()

            target_col = 'price' 

            input_col_train_df=train_df.drop(columns=[target_col],axis=1)
            target_col_train_df=train_df[target_col]

            input_col_test_df=test_df.drop(columns=[target_col],axis=1)
            target_col_test_df=test_df[target_col] 

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            ) 
           
            input_col_train_arr = preprocessor_obj.fit_transform(input_col_train_df)
            input_col_test_arr = preprocessor_obj.transform(input_col_test_df)

            #concating array 

            train_arr = np.c_[input_col_train_arr, np.array(target_col_train_df)]
            test_arr = np.c_[input_col_test_arr, np.array(target_col_test_df)]
 
            save_object(

                file_path=self.data_transform_config.preprocessor_data_path,
                obj=preprocessor_obj 
                )
            
            return(
                train_arr,
                test_arr,
                self.data_transform_config.preprocessor_data_path
            )

        except Exception as e:
            logging.info('Exception occured in Data_Transformation_Start function')
            raise CustomException(e,sys)
