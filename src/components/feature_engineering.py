import os
import sys
 
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin

from exception import CustomException
from logger import logging

class FeatureEngineering(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, df:pd.DataFrame):
        try:
            if 'quality' in df.columns:   
                # New target variable for binary classification instead of regression/multi class classification  
                df['is_good'] = df['quality'].apply(lambda x: 1 if x >=7 else 0) 
            return df
        except Exception as e:
            logging.error("Error occurred during feature engineering")
            raise CustomException(e, sys) from e
        
