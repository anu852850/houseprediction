import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transform_object(self):
        try:
            numerical_columns = ["bedrooms", "bathrooms", "area_sqft", "year_built"]
            categorical_columns = ["has_garage"]

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numerical_pipeline, numerical_columns),
                    ("cat", categorical_pipeline, categorical_columns)
                ]
            )

            logging.info("Data transformation pipeline created successfully")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            # Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Training and testing data loaded successfully")

            # ----- BEGIN: data cleaning / sanitization -----
            missing_tokens = ["unknown", "n/a", "na", "none", "missing", "unk", "null"]
            for df in (train_df, test_df):
                # Replace common tokens with NaN
                df.replace({tok: np.nan for tok in missing_tokens}, inplace=True)

                # Strip + lower-case text columns for consistency
                for col in df.select_dtypes(include=["object"]).columns:
                    df[col] = df[col].astype(str).str.strip().replace({'nan': np.nan})
                    try:
                        df[col] = df[col].str.lower()
                    except Exception:
                        pass

            # Convert known numeric columns to numeric (coerce invalid -> NaN)
            numeric_cols = ["area_sqft", "bedrooms", "bathrooms", "year_built"]
            for c in numeric_cols:
                if c in train_df.columns:
                    train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
                if c in test_df.columns:
                    test_df[c] = pd.to_numeric(test_df[c], errors="coerce")

            logging.info("Data cleaning completed")
            # ----- END: data cleaning / sanitization -----

            # Split input and target
            target_column = "price"
            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("Separated input and target features")

            # Transform data
            preprocessor = self.get_data_transform_object()
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # Combine transformed features with target column
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            logging.info(f"Preprocessor object saved at {self.data_transformation_config.preprocessor_obj_file_path}")
            logging.info("Data transformation completed successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataTransformation()
    train_arr, test_arr, path = obj.initiate_data_transformation(
        train_path="artifacts/train.csv",
        test_path="artifacts/test.csv"
    )
    print("Transformation complete")
    print("Preprocessor saved at:", path)
