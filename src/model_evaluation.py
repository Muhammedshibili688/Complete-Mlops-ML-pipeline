import pandas as pd
import os
import numpy as np
import logging
import json
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score 
import yaml
from dvclive import Live

# Ensure the log directory exists
log_dir = 'log'
os.makedirs(log_dir, exist_ok = True)

logger = logging.getLogger('model_evaluation')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_yaml(file_path: str) -> dict:
    '''Load yaml file for model evaluation'''
    try:
        with open(file_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug("yaml file for model evaluation loaded succesfully..")
        return params
    
    except FileNotFoundError as e:
        logger.debug("File not found in the given path %s", file_path)
        raise

    except yaml.YAMLError as e:
        logger.debug("yaml Error occured %s", e)
        raise

    except Exception as e:
        logger.exception("Unexpected Error happened %s", e)
        raise

def load_model(file_path: str):
    '''
    Try to fetch model from given file path
    '''
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Fetched model successfully from %s', file_path)
        return model
    
    except FileNotFoundError as e:
        logger.debug("File not found on given path : %s", e)
        raise
    except Exception as e:
        logger.exception("Unexpected error occured while loading the model %s", e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    ''''
    Loading test file from file path
    '''
    try:
        test_df = pd.read_csv(file_path)
        logger.debug("Gathered test data from %s successfully", file_path)
        return pd.DataFrame(test_df)
    
    except FileNotFoundError as e:
        logger.debug("File not found while loading test at %s", e)
        raise
    except Exception as e:
        logger.exception("Unexpected error occured in loading test data %s", e)
        raise


def evaluate_model(clf, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    '''
    Evaluating model-clf by passing parameters:
    x_test and y_test
    '''
    try :
        y_pred = clf.predict(x_test)
        y_pred_proba = clf.predict_proba(x_test)[:,1]

        # logger.debug("y_test type: %s", type(y_test))
        # logger.debug("y_test shape: %s", y_test.shape)
        # logger.debug("y_test unique values: %s", y_test.unique()[:10])


        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        model_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logger.debug("Model evaluation completed and model metrics calculated")
        return model_metrics, y_pred
    
    except Exception as e:
        logger.exception("Unexpected error occured while model evaluating and calculating model metrics %s", e)
        raise

def save_metrics(file_path: str, metrics: dict) -> None:
    '''
    Save the model metrics to given file path
    '''
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok = True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent = 4)
        logger.debug("Metrics saved to %s", file_path)

    except Exception as e:
        logger.debug("Unexpected error occured while saving metrics to file location %s", e)
        raise

def main():
    try:
        params = load_yaml(file_path = 'params.yaml')
        # try to add "./" before path string    
        clf = load_model('./models/model.pkl')
        test_df =load_data('./data/processed/test_tfidf.csv')

        x_test = test_df.iloc[:,:-1]
        y_test = test_df.iloc[:,-1]

        metrics, y_pred = evaluate_model(clf, x_test, y_test)

        # Experiment tracking using dvclive
        with Live(save_dvc_exp = True) as live:
            live.log_metric("accuracy", accuracy_score(y_test, y_pred))
            live.log_metric("precision", precision_score(y_test, y_pred))
            live.log_metric("recall", recall_score(y_test, y_pred))

            live.log_params(params)

        save_metrics('reports/metrics.json', metrics)

    except Exception as e:
        logger.exception("Unexpected error occured while model evaluation ..")


if __name__ == "__main__":
    main()