import pandas as pd
import os
import logging
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import yaml

# Esure the log directory 
log_dir = 'log'
os.makedirs(log_dir, exist_ok = True)

logger = logging.getLogger('model_training')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_log_path = os.path.join(log_dir, "model_training.log")
file_handler = logging.FileHandler(file_log_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Load yaml file

def load_params(file_path: str) -> dict:
    'load yaml for model_trainig'
    try:
        with open(file_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug("yaml file for model traing loaded succesfully ..")
        return params
    except FileNotFoundError as e:
        logger.debug("File not found on given file path %s", file_path)
        raise
    except yaml.YAMLError as e:
        logger.debug("yaml error occured %s", e)
        raise
    except Exception as e:
        logger.exception(f"Unexpected error occured {e}")
        raise

def load_data(file_path : str) -> pd.DataFrame:
    '''
    Load csv data from file path for model training

    param : file_path to csv needed to be giiven
    returns Dataframe
    '''
    try:
        df = pd.read_csv(file_path)
        logger.debug("csv file loaded from %s with shape %s", file_path, df.shape)
        return df
     
    except pd.errors.ParserError as e:
        logger.debug('Error while parsing the csv file')
        raise

    except FileNotFoundError as e:
        logger.debug("File not found at %s", file_path)
        raise

    except Exception as e:
        logger.exception("Unexpected error occured !")
        raise



def train_model(x_train: np.ndarray, y_train: np.ndarray, param: dict) -> RandomForestClassifier:
    '''
    Trainig the model
    
    : param x_train
    : param y_train
    : param param dictionary of hyper-parameters, leaf_node etc
    : return Trained RandomForestClassifier
    '''
    try:
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("Number of rows in x_train and y_train are diffreent x_train = %s  y_train = %s", 
                             x_train.shape[0], y_train.shape[0])
        
        logger.debug("Initialising Random Forest Classifier with parameters %s", param)
        clf = RandomForestClassifier(n_estimators = param['n_estimators'], random_state = param['random_state'])

        logger.debug('Model trainig started with %d samples', x_train.shape[0])
        clf.fit(x_train, y_train)
        logger.debug("Model traing completed ...")

        return clf
    
    except ValueError as e:
        logger.debug("value Error during traing, moslty because of x_train and y_train rows mismatch. %s", e)
        raise

    except Exception as e:
        logger.debug("Unexpected Error while training model %s", e)
        raise

def save_model(file_path: str, model) -> None:
    '''
    : param file path where the model is saved
    : model the classifier model returned from train_model function need to be passed here.
    '''

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok = True)

        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug("model saved to : %s", file_path)

    except FileNotFoundError as e:
        logger.debug("File not found at the given path %s", file_path)
        raise
    except Exception as e:
        logger.debug("Unexpected Error found while saving the model at %s", file_path)
        raise

def main():
    try:
        #params = {'n_estimators' : 25, 'random_state' : 2}
        params = load_params(file_path = 'params.yaml')['model_training']
        train_data = load_data('./data/processed/train_tfidf.csv')
        x_train = train_data.iloc[:,:-1]
        y_train  =train_data.iloc[:,-1]

        clf = train_model(x_train, y_train, params)

        model_save_path  = 'models/model.pkl'
        save_model(model_save_path, clf)

    except Exception as e:
        logger.debug('Failed to complete model building process')
        print(f"Error : {e}")

if __name__ == "__main__":
    main()
