import os
import pandas as pd
import numpy as np
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml
from google.oauth2 import service_account

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.file_name = self.config["bucket_file_name"]
        self.train_test_ratio = self.config["train_ratio"]

        os.makedirs(RAW_DIR, exist_ok=True)

        logger.info(f"Data Ingestion started with {self.bucket_name} and the file name {self.file_name}")

    
    def download_csv_from_gcp(self):
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)

            blob.download_to_filename(RAW_FILE_PATH)

            logger.info(f"CSV file is successfully downloaded to {RAW_FILE_PATH}")

        except Exception as e:
            logger.error(f"Error while downloading the csv file: {e}")
            raise CustomException(f"Failed to download csv file: {e} ", e)

        
    def split_data(self):
        try:
            logger.info("Splitting the data into train and test sets")
            data = pd.read_csv(RAW_FILE_PATH) # Read the csv file into a pandas dataframe

            train_data, test_data = train_test_split(data, test_size=1-self.train_test_ratio, random_state=42)

            train_data.to_csv(TRAIN_FILE_PATH)
            test_data.to_csv(TEST_FILE_PATH)

            logger.info(f"Train data saved to {TRAIN_FILE_PATH}")
            logger.info(f"Test data saved to {TEST_FILE_PATH}")

        except Exception as e:
            logger.error("Error occurred while splitting the data into train and test sets")
            raise CustomException("Failed to split the data into train and test sets", e)



    def run(self):
        try:
            logger.info("Starting the data ingestion process")

            self.download_csv_from_gcp() # Download the csv file from GCP bucket
            self.split_data() # Split the data into train and test sets
            logger.info("Data ingestion process completed successfully")

        except CustomException as ce:
            logger.error(f"Custom Exception: {str(ce)}")

        finally:
            logger.info("Data ingestion Completed")


if __name__ == "__main__":

    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()
