import os
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from exception import CustomException
from logger import logging
from .feature_engineering import FeatureEngineering
from imblearn.combine import SMOTETomek

from utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:  
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 
                                  'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
            
            # derived_feature = ['is_good']  

            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            

            logging.info(f"Numerical columns: {numerical_features}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",numerical_pipeline,numerical_features)
                ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys) from e
        


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            feature_engineer = FeatureEngineering()
            train_df = feature_engineer.transform(train_df)
            test_df = feature_engineer.transform(test_df)

            logging.info("Feature engineering completed (created 'is_good' column)")
            
            logging.info(f"train df: {train_df.head(5)}")
            logging.info(f"value count is_good: {train_df['is_good'].value_counts()}")
            logging.info(f"test df: {test_df.head(5)}")
            logging.info(f"value count is_good: {test_df['is_good'].value_counts()}")

            preprocessing_obj = self.get_data_transformer_object()


            target_column = "is_good"
            numerical_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 
                                  'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

            input_feature_train_df = train_df.drop(columns=['type','quality'],axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=['type','quality'],axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info(f"train df features: {input_feature_train_df.columns}")
            logging.info(f"test df features: {input_feature_test_df.columns}")
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Apply Sampling technique(SMOTETomek) to the training data
            # smote_tomek = SMOTETomek()
            # input_feature_train_arr, target_feature_train_df = smote_tomek.fit_resample(input_feature_train_arr, target_feature_train_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("Preprocessor pickle file saved")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error("Error occurred during data transformation")
            raise CustomException(e, sys) from e