import os
import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

# Configuration class to store file paths
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")


# Main Data Ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")

        try:
            # Step 1: Read the dataset
            df = pd.read_csv(r"notebooks\data\dirty_house_prices.csv")  # raw string to avoid escape warning
            logging.info("Dataset read successfully")

            # Step 2: Create 'artifacts' directory if not exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            logging.info("Artifacts folder created (if not already)")

            # Step 3: Save the raw dataset as raw.csv
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved at %s", self.ingestion_config.raw_data_path)

            # Step 4: Split dataset into train and test sets
            train_set, test_set = train_test_split(df, train_size=0.25, random_state=42)
            logging.info("Train-Test split completed: Train size = %d, Test size = %d", len(train_set), len(test_set))

            # Step 5: Save train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train & Test data saved successfully")

            logging.info("Data ingestion process completed successfully")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("Error during data ingestion: %s", e)
            raise CustomException(e, sys)


# Run the script directly (main entry point)
if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    print(f"Data ingestion completed.\nTrain data: {train_path}\nTest data: {test_path}")
