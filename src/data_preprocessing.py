import os
import pandas as pd
import numpy as np
import logging
import nltk
import string
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download("stopwords")
nltk.download("punkt")

# Ensure log directory existes
log_dir = "log"
os.makedirs(log_dir, exist_ok = True)

logger = logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "data_preprocessing.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    '''Transform text by removing lowercase, tokenisation, punctuation, stemming, and removing stopwords'''
    ps = PorterStemmer()
    # conver to lower case
    text = text.lower()
    # tokenization
    text = nltk.word_tokenize(text)
    # removing non alphanumeric characters
    text = [word for word in text if word.isalnum()]
    # Remove stop words and punctuation
    text = [word for word in text if word not in stopwords.words("english") and word not in string.punctuation]
    # Stemming
    text = [ps.stem(word) for word in text]
    return " ".join(text)

def preprocess_df(df: pd.DataFrame, text_column = "text", target_column = "target") -> pd.DataFrame:
    '''Preprocess the dataframe by transforming text and encoding target labels'''
    try:
        logger.debug("Strating Preprocessing of dataframe")
        # Encode the target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug("Encoded target column %s", target_column)

        # Remove duplicate rows
        df.drop_duplicates(inplace = True,  keep = "first")
        logger.debug("Duplicte rows removed")

        # Apply the text transformation
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug("Text column transformed %s", text_column)
        return df
    
    except KeyError as e:
        logger.error("Column not found %s", e)
        raise

    except Exception as e:
        logger.error("Unexpected error during datframe processing %s", e)
        raise

def main(text_column = "text", target_column = "target"):
    ''''Main function to preprocess data'''
    try:
        # Fetch the data from dataframe
        train_data = pd.read_csv('./data/raw/train.csv') 
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug("Loaded train and test data properly")

        # Transform the data
        processed_train_data = preprocess_df(train_data, text_column, target_column)
        processed_test_data = preprocess_df(test_data, text_column, target_column)

        # Stoe the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok = True)

        processed_train_data.to_csv(os.path.join(data_path, "processed_train.csv"), index = False)
        processed_test_data.to_csv(os.path.join(data_path, "processed_test.csv"), index = False)

        logger.debug("Processed data saved to %s", data_path)

    except FileNotFoundError as e:
        logger.error("File not found %s", e)
        raise

    except pd.errors.EmptyDataError as e:
        logger.error("No data: %s", e)
        raise

    except Exception as e:
        logger.error("Failed to complete data preprocessing pipeline %s", e)
        print(f"Error : {e}")
        raise

if __name__ == "__main__":
    main()