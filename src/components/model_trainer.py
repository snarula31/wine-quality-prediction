import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score,f1_score


from exception import CustomException
from logger import logging

from utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model1.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Logistic Regression":LogisticRegression(),
                "Random Forest":RandomForestClassifier(),
                "Decision Tree":DecisionTreeClassifier(),
                "LightGBM":LGBMClassifier(),
                "XGBoost":XGBClassifier(),
                "KNN":KNeighborsClassifier(),
                "SVC":SVC()
            }
            
            params = {
                "Logistic Regression": {
                    # 'class_weight': ['balanced', None],

                },
                "Random Forest": {
                    "n_estimators": [100, 150, 200, 250],
                    "max_depth": [10, 15, 20, 25],
                    # "class_weight": ['balanced', None]
                },
                "Decision Tree": {
                    "splitter": ["best", "random"],
                    "max_depth": [10, 15, 20, 25],
                    # "class_weight": ['balanced', None]
                },
                "LightGBM": {
                    "n_estimators": [100, 150, 200, 250],
                    "max_depth": [10, 15, 20, 25],
                    # "class_weight": ['balanced', None]
                },
                "XGBoost": {
                    "n_estimators": [100, 150, 200, 250],
                    "max_depth": [10, 15, 20, 25],
                    # "class_weight": ['balanced', None]
                },
                "KNN": {
                    "n_neighbors": [3, 5, 7]
                },
                "SVC": {
                    "C": [0.1, 1, 10],
                    "gamma": [0.1, 1, 10],
                    # "class_weight": ['balanced', None]
                }
            }

            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train,X_test=X_test,
                                                y_test=y_test, models=models, params=params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model1 found on both training and testing dataset: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            logging.info("Model saved successfully.")

            predicted = best_model.predict(X_test)

            f1_scr = f1_score(y_test, predicted)

            return f1_scr
            
        except Exception as e:
            logging.error("Error occurred during model training")
            raise CustomException(e, sys) from e