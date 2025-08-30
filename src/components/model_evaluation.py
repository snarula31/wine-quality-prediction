import  os
import sys
import pandas as pd
import numpy as np

from sklearn.metrics import (f1_score, precision_score, recall_score, 
                             roc_auc_score, precision_recall_curve, auc)

from exception import CustomException
from logger import logging

from utils import load_object


class ModelEvaluation:
    def eval_metrics(self,actual,pred_prob):
        pred = (pred_prob > 0.5).astype(int)

        f1 = f1_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual,pred)

        # Calculate AUPRC
        pr_curve_precision, pr_curve_recall, _ = precision_recall_curve(actual, pred_prob)
        auprc = auc(pr_curve_recall, pr_curve_precision)

        return f1, precision, recall, auprc

    def initiate_model_evaluation(self,train_array,test_array,model_path):
        try:
            logging.info("Loading model for evaluation")

            model = load_object(file_path=model_path)

            logging.info("Model loaded successfully")

            logging.info("Starting model evaluation")
            
            X_test, y_test = (test_array[:,:-1],
                              test_array[:,-1])
            
            predict_prob = model.predict_proba(X_test)[:,1]

            f1, precision, recall, auprc = self.eval_metrics(y_test, predict_prob)

            logging.info("="*20 + " MODEL EVALUATION METRICS " + "="*20)
            logging.info(f"F1-Score: {f1:.4f}")
            logging.info(f"Precision: {precision:.4f}")
            logging.info(f"Recall: {recall:.4f}")
            logging.info(f"Area Under PR Curve (AUPRC): {auprc:.4f}")
            logging.info("="*20)

            return f1, precision, recall, auprc
            

        except Exception as e:
            raise CustomException(e, sys)
