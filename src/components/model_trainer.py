import numpy as np
import pandas as pd
import os
import sys
from dataclasses import dataclass 

# Linear models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, SGDRegressor

# Tree-based models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor

# Support Vector Machines
from sklearn.svm import SVR

# Neighbors
from sklearn.neighbors import KNeighborsRegressor

# Gaussian Process
from sklearn.gaussian_process import GaussianProcessRegressor

# Neural Networks
from sklearn.neural_network import MLPRegressor

# Metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

# Dataclass to store the best model
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Splitting the train and test input data")
            x_train, y_train = train_array[:, :-1], train_array[:, -1]
            x_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Dictionary of models
            models = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "ElasticNet": ElasticNet(),
                "BayesianRidge": BayesianRidge(),
                "SGDRegressor": SGDRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "RandomForest": RandomForestRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "ExtraTrees": ExtraTreesRegressor(),
                "SVR": SVR(),
                "KNeighbors": KNeighborsRegressor(),
                "GaussianProcess": GaussianProcessRegressor(),
                "MLP": MLPRegressor()
            }

            # Evaluate all models
            model_report: dict = evaluate_models(x_train, y_train, x_test, y_test, models)

            # Select best model
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            # Check model quality
            if best_model_score < 0.6:
                raise CustomException("No good model found", sys)

            # Save best model
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            return best_model_name, best_model_score

        except Exception as e:
            raise CustomException(e, sys)


# ===============================
# Run directly (for testing)
# ===============================
if __name__ == "__main__":
    # Example: dummy arrays (replace with actual pipeline outputs)
    train_array = np.random.rand(100, 6)   # 100 rows, 5 features + 1 target
    test_array = np.random.rand(20, 6)     # 20 rows, 5 features + 1 target
    preprocessor_path = "artifacts/preprocessor.pkl"

    trainer = ModelTrainer()
    best_model_name, best_model_score = trainer.initiate_model_trainer(
        train_array, test_array, preprocessor_path
    )
    print(f"Best Model: {best_model_name}")
    print(f"R2 Score: {best_model_score}")
