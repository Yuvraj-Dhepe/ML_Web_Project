# Doing all the ETL steps in this file.
# Doing all types of data transformation like feature engineering, feature selection, feature scaling, data cleaning, handling null values etc.

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import sys
import os
from dataclasses import dataclass
from src.utils import save_object


# Defining the paths for the data ingestion
# di_obj = DataIngestion.DataIngestionConfig() # Not required as we already are doing the Doing the addition of paths in data_ingestion.py itself.
# di_obj.train_data_path = "data/train_data.csv"
# di_obj.test_data_path = "data/test_data.csv"

@dataclass #This is a decorator which is used to create a dataclass variables.
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    
    def __init__(self,transformation_config: DataTransformationConfig = DataTransformationConfig()):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ["writing_score","reading_score"]
            categorical_columns = ["gender",
                                   "race_ethnicity",
                                   "parental_level_of_education",
                                   "lunch",
                                   "test_preparation_course"]
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy = 'median')),
                    ("scaler",StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy = 'most_frequent')),
                    ("one_hot_encoder",OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f"Numerical columns:{numerical_columns}")
            logging.info(f"Categorical columns:{categorical_columns}")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )
            
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
            
        try:
            train_df = pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)
            
            
            logging.info("Read train and test data completed") 
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "math_score"
            numerical_columns = ["writing_score","reading_score"]
            
            input_feature_train_df = train_df.drop(columns = [target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns = [target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
                ]
            
            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
                ]
            
            logging.info(f"Saved Preprocessing object.")
            # Saving the pkl file
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
                
        except Exception as e:
            raise CustomException(e,sys)
            
        
# class MyDataTransformation:
#     '''
#     Data Transformation class 
#     '''
#     def __init__(self,ingestion_config: DI.DataIngestionConfig = DI.DataIngestionConfig()):
#         self.ingestion_config = ingestion_config
#         self.ingestion_obj = DI.DataIngestion(self.ingestion_config)
        
#     def initiate_data_transformation(self):
#         '''
#         Apply simple transformation on the categorical and numerical data
#         '''
        
#         try:
#             logging.info("Initiating data transformation")
            
#             logging.info("Initiating data ingestion via data ingestion class")
#             df_train = pd.read_csv(self.ingestion_config.train_data_path)
#             df_test = pd.read_csv(self.ingestion_config.test_data_path)
#             logging.info("Data ingestion completed")
            
#             logging.info("Data copy creation")
#             # Creating a copy of the data
#             self.df_train_copy = df_train.copy()
#             self.df_test_copy = df_test.copy()
#             logging.info("Data copy creation completed")
            
#             logging.info("Data transformation via column transformers started")
#             # Create column transformer with 3 types of transformers
#             num_features = self.df_train_copy.select_dtypes(exclude='object').columns
#             cat_features = self.df_train_copy.select_dtypes(include='object').columns
            
#             # Creating a column transformer
#             numeric_transformer = StandardScaler()
#             oh_transformer = OneHotEncoder(handle_unknown='ignore')
            
#             preprocessor = ColumnTransformer(
#                 [
#                     ('num',numeric_transformer,num_features),
#                     ('cat',oh_transformer,cat_features),
#                 ]
#             )
#             logging.info("Data transformation via column transformers completed")
            
#             self.df_train_copy = preprocessor.fit_transform(self.df_train_copy)
#             self.df_test_copy = preprocessor.fit_transform(self.df_test_copy)        
            
#             logging.info("Data transformation completed")
#             return (
#                 self.df_train_copy, 
#                 self.df_test_copy
#             )
#         except Exception as e:
#             logging.error("Error occured in data transformation")
#             raise CustomException(e,sys)

# if __name__ == "__main__":
#     dt_obj = MyDataTransformation()
#     df_train_copy, df_test_copy = dt_obj.initiate_data_transformation()
#     logging.info(df_train_copy)