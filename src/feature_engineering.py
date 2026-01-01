import pandas as pd
import numpy as np
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml 

# Ensure the log directory
log_dir = "log"
os.makedirs(log_dir, exist_ok = True)

logger = logging.getLogger("feature_engineering")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_log_path = os.path.join(log_dir, "feature_engineering.log")
file_handler = logging.FileHandler(file_log_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter) 
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Load yaml file

def load_params(file_path: str)-> dict:
    '''Load yaml file for feature engineering'''
    try:
        with open(file_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug("yaml file for feature engineering loaded succesfully..")
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
        


def load_data(file_path:str) -> pd.DataFrame:
    '''Load data from csv file'''
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace = True)
        logger.debug("Data loaded and Nans filled from %s", file_path)
        return df
    
    except pd.errors.ParserError as e:
        logger.debug("Failed to Parse csv file %s", e)
        raise
    except Exception as e:

        logger.debug("Unexpected erro occured while loading the csv file %s", e)
        raise

def apply_tfidf(train_data : pd.DataFrame, test_data : pd.DataFrame, max_features: int) -> tuple:
    "apply tfidf to data"
    try:
        vectoriser = TfidfVectorizer(max_features = max_features)

        x_train = train_data['text'].values
        x_test = test_data['text'].values
        y_train = train_data['target'].values
        y_test = test_data['target'].values

        x_train_bow = vectoriser.fit_transform(x_train)
        x_test_bow = vectoriser.transform(x_test)

        train_df = pd.DataFrame(x_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(x_test_bow.toarray())
        test_df['label'] = y_test

        logger.debug('Tfidf applied and data transformed')
        return train_df, test_df
    
    except Exception as e:
        logger.debug("Error during bag of words transformation %s", e)
        raise

def save_data(df : pd.DataFrame, file_path: str) -> None:
    ''''Save the dataframe to a csv file'''
    try :
        os.makedirs(os.path.dirname(file_path), exist_ok = True)
        df.to_csv(file_path, index = False)
        logger.debug("Data save to %s", file_path)

    except Exception as e:
        logger.debug("Unexpected error occured while saving the data %s", e)
        raise

def main():
    try:
        params = load_params(file_path = 'params.yaml')
        max_features = params['feature_engineering']['max_features']
        train_data = load_data('./data/interim/processed_train.csv')
        test_data = load_data('./data/interim/processed_test.csv')

        train_df, test_df = apply_tfidf(train_data, test_data, max_features)
        
        save_data(train_df, os.path.join("./data", "processed", "train_tfidf.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_tfidf.csv"))

    except Exception as e:
        logger.debug("Failed to complete feature engineering process %s", e)
        print(f"Error  : {e}")

if __name__ == '__main__':
    main()
