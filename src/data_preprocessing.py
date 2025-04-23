import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


logger = get_logger(__name__)

class DataProcessor:

    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)


        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir, exist_ok=True)

        logger.info("Data Processor initialized")

    def preprocess_data(self, df):
        try:
            logger.info(" Starting the Preprocessing data")

            # Drop unnecessary columns
            logger.info("Dropping unnecessary columns")
            df.drop(columns=["Booking_ID"], inplace=True)
            df.drop_duplicates(inplace=True)

            cat_cols = self.config['data_processing']['categorical_columns']
            num_cols = self.config['data_processing']['numerical_columns']

            logger.info("Applying Label Encoding")
            label_encoder = LabelEncoder()
            mappings = {}

            for column in cat_cols:
                df[column] = label_encoder.fit_transform(df[column])
                # Store the mapping of original labels to encoded labels
                mappings[column] = {label: code for label, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}

            logger.info("Label Mapping are: ")

            for col, mapping in mappings.items():
                logger.info(f"{col}: {mapping}")

            logger.info("Skewness handling")

            df['no_of_previous_cancellations'] = np.log1p(df['no_of_previous_cancellations'])
            df['no_of_previous_bookings_not_canceled'] = np.log1p(df['no_of_previous_bookings_not_canceled'])

            skew_threshold = self.config['data_processing']['skewness_threshold']
            skewness = df[num_cols].apply(lambda x: x.skew()).sort_values(ascending=False)

            for column in skewness[skewness > skew_threshold].index:
                df[column] = np.log1p(df[column])
            
            return df
        
        except Exception as e:
            logger.error(f"Error occurred during data preprocessing: {e}")
            raise CustomException("Failed to preprocess the data", e)
                

    def balanced_data(self, df):
        try:
            logger.info("Balancing the data using SMOTE")
            X = df.drop(columns=["booking_status"])
            y = df["booking_status"]
            
            smote = SMOTE()
            X_resampled, y_resampled = smote.fit_resample(X, y)

            balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_df["booking_status"] = y_resampled
            
            logger.info("Data balanced successfully")
            return balanced_df
            
        except Exception as e:
            logger.error(f"Error occurred during data balancing: {e}")
            raise CustomException("Failed to balance the data", e)
                    
    def select_feature(self, df):
        try:
            logger.info("Feature selection using Random Forest")
            X = df.drop(columns=["booking_status"])
            y = df["booking_status"]

            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)

            feature_importance =model.feature_importances_
            feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

            num_features_to_select = self.config['data_processing']['no_of_features']

            top_10_features = feature_importance_df['Feature'].head(num_features_to_select).values

            top_10_df = df[top_10_features.tolist() + ['booking_status']]

            logger.info("Feature selection completed")
            logger.info(f"Top {num_features_to_select} features selected: {top_10_features}")

            return top_10_df
        
        except Exception as e:
            logger.error(f"Error occurred during feature selection: {e}")
            raise CustomException("Failed to select features", e)
        
    
    def save_data(self, df, file_path):
        try:
            logger.info(f"Saving processed data to {file_path}")
            df.to_csv(file_path, index=False)
            logger.info("Data saved successfully")
        except Exception as e:
            logger.error(f"Error occurred while saving the data: {e}")
            raise CustomException("Failed to save the processed data", e)
        
    def process(self):
        try:
            logger.info("Loading data from RAW directory")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            train_df = self.balanced_data(train_df)
            test_df = self.balanced_data(test_df)

            train_df = self.select_feature(train_df)
            test_df = train_df[train_df.columns]

            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)

            logger.info("Data processing completed successfully")

        except Exception as e:
            logger.error(f"Error occurred during data processing: {e}")
            raise CustomException("Failed to process the data", e)
        

if __name__ == "__main__":
    try:
        processor = DataProcessor(
            train_path=TRAIN_FILE_PATH,
            test_path=TEST_FILE_PATH,
            processed_dir=PROCESSED_DIR,
            config_path=CONFIG_PATH
        )
        processor.process()
    except CustomException as ce:
        logger.error(f"Custom Exception: {str(ce)}")
         
