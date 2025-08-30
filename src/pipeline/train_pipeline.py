import sys
import os

from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainer
from components.model_evaluation import ModelEvaluation

from exception import CustomException
from logger import logging


class TrainingPipeline:
    def run(self):
        try:
            logging.info("Starting training pipeline")

            # Step 1: Data Ingestion
            data_ingestion = DataIngestion()
            train_data, test_data = data_ingestion.initiate_data_ingestion()

            # Step 2: Data Transformation
            data_transformation = DataTransformation()
            train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)

            # Step 3: Model Training
            model_trainer = ModelTrainer()
            model = model_trainer.initiate_model_trainer(train_arr,test_arr)

            # Step 4: Model Evaluation
            model_evaluation = ModelEvaluation()
            model_evaluation.initiate_model_evaluation(train_arr, test_arr, model)

            logging.info("Training pipeline completed successfully")

        except Exception as e:
            logging.error(f"Error occurred in training pipeline: {e}")
            raise CustomException(e, sys)




if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()