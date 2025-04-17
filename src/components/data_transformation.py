import os
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.uitils import save_object, evaluate_models

@dataclass
class DataTransformationConfig:
    preprocessed_object_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        # Define numerical and categorical columns as class attributes
        self.numerical_cols = [
            'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 
            'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
            'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 
            'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 
            'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 
            'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 
            'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'
        ]
        self.categorical_cols = [
            'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 
            'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 
            'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 
            'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
            'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 
            'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 
            'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 
            'SaleType', 'SaleCondition'
        ]

    def get_data_transformer_object(self):
        try:
            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f"Numerical columns: {self.numerical_cols}")
            logging.info(f"Categorical columns: {self.categorical_cols}")
            
            preprocessor = ColumnTransformer(
                [
                    ('numerical_pipeline', numerical_pipeline, self.numerical_cols),
                    ('categorical_pipeline', categorical_pipeline, self.categorical_cols)
                ]
            )

            return preprocessor
            
        except Exception as e:
            logging.info("Error in get_data_transformer_object")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Log shapes of loaded DataFrames
            logging.info(f"Train data shape: {train_df.shape}")
            logging.info(f"Test data shape: {test_df.shape}")

            # Verify train data has enough rows
            if train_df.shape[0] == 0:
                raise CustomException("Train data is empty", sys)
            if train_df.shape[0] == 1:
                raise CustomException("Train data has only one row", sys)

            preprocessor = self.get_data_transformer_object()

            target_column_name = 'SalePrice'

            # Ensure target column exists in train data
            if target_column_name not in train_df.columns:
                raise CustomException(f"Target column '{target_column_name}' not found in train data", sys)

            # Split features and target for train data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Log shapes after splitting
            logging.info(f"Input feature train shape: {input_feature_train_df.shape}")
            logging.info(f"Target feature train shape: {target_feature_train_df.shape}")

            # Test data may not have target column
            input_feature_test_df = test_df  # Do not attempt to drop target_column_name
            target_feature_test_df = None  # Set to None if test data has no target

            # Ensure test data has all required columns
            missing_cols = [col for col in self.numerical_cols + self.categorical_cols if col not in input_feature_test_df.columns]
            if missing_cols:
                raise CustomException(f"Test data missing columns: {missing_cols}", sys)

            # Log shape of test features
            logging.info(f"Input feature test shape: {input_feature_test_df.shape}")

            # Apply preprocessing
            logging.info("Applying preprocessing object on training and testing dataframes")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            
            # Log shape of preprocessed training array
            logging.info(f"Preprocessed train array shape: {input_feature_train_arr.shape}")

            # Verify row count after preprocessing
            if input_feature_train_arr.shape[0] != input_feature_train_df.shape[0]:
                raise CustomException(
                    f"Preprocessing reduced train rows from {input_feature_train_df.shape[0]} to {input_feature_train_arr.shape[0]}",
                    sys
                )

            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # Log shape of preprocessed test array
            logging.info(f"Preprocessed test array shape: {input_feature_test_arr.shape}")

            # Combine features and target for train data
            logging.info("Combining preprocessed features and target for train data")
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]

            # For test data, only return features since target may not exist
            test_arr = input_feature_test_arr

            # Log final array shapes
            logging.info(f"Final train array shape: {train_arr.shape}")
            logging.info(f"Final test array shape: {test_arr.shape}")

            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessed_object_file_path,
                obj=preprocessor
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessed_object_file_path
            )
            
        except Exception as e:
            logging.info("Error in initiate_data_transformation")
            raise CustomException(e, sys)