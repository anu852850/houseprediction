import os
import sys
import dill
import pandas as pd

from src.exception import CustomException

class PredictPipeline:
    def __init__(self):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            # Load preprocessor
            with open(preprocessor_path, "rb") as f:
                self.preprocessor = dill.load(f)

            # Load model
            with open(model_path, "rb") as f:
                self.model = dill.load(f)

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame):
        """
        features: DataFrame with raw input
        """
        try:
            # Transform raw input into expanded features (e.g., 206 features)
            X_processed = self.preprocessor.transform(features)

            # Predict using trained model
            preds = self.model.predict(X_processed)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, area_sqft, bedrooms, bathrooms, location, year_built, has_garage):
        self.area_sqft = area_sqft
        self.bedrooms = bedrooms
        self.bathrooms = bathrooms
        self.location = location
        self.year_built = year_built
        self.has_garage = has_garage

    def get_data_as_df(self):
        try:
            custom_data_dict = {
                "area_sqft": [float(self.area_sqft)],
                "bedrooms": [float(self.bedrooms)],
                "bathrooms": [float(self.bathrooms)],
                "location": [self.location],
                "year_built": [int(self.year_built)],
                "has_garage": [self.has_garage],
            }
            return pd.DataFrame(custom_data_dict)

        except Exception as e:
            raise CustomException(e, sys)
