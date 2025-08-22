import os
import sys
from exception import CustomException
from logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from components.data_transformation import DataTransformationConfig
from components.data_transformation import DataTransformation

@dataclass
class DataIngestinConfig:   # defining class DataIgestionConfig 
    train_data_path:str = os.path.join('artifacts',"train.csv") # variable to store file path for training data
    test_data_path:str = os.path.join('artifacts',"test.csv")# variable to store file path for test data
    raw_data_path:str = os.path.join('artifacts',"data.csv")# variable to store file path fro original data



class DataIgestion: # defining class DataIngestion
    def __init__(self): # constructor
        self.ingestion_config = DataIngestinConfig() # creating an pbject of class DataIngestionConfig

    def initiate_data_ingestion(self): # function for initaiting data ingestion
        logging.info("Entered the data ingestion component")
        try:
            df = pd.read_csv("notebook/data/winequality.csv")
            logging.info("Load the dataset as dataframe.")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Iniating train test split")
            
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=56)
            
            logging.info("train test split initiated")

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)# converting training set into csv and saving it in path defined in dataingestionconfig

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)# conveerting test set into csv and saving it in path defined in dataingestionconfig 

            logging.info("Data ingestion completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        
        except Exception as e:
            raise CustomException (e,sys)
        

if __name__ == "__main__":
    obj = DataIgestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)