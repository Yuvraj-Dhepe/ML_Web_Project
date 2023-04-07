# Used to ingest data from a warehouse, database or any other source into the project, usually a job of big data team.
# Data ingestion plays a very important role in the project, as it is the first step in the data pipeline, cause a data is the most important asset of the project.
# There a seperate big data team which ensures to get data from various sources and store it in different places like Hadoop, MongoDB etc.
# Data ingestion is the process of loading data from a source system into a data warehouse, data lake or data mart.
# As Data Scientists we need to know how to get data from various sources like hadoop, mongodb, mysql, oracle etc and store it in such a way to make it available for analysis.

import os
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    '''
    Used for defining the configuration for data ingestion.
    '''
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv') 
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    '''
    Used for ingesting data by making use of the configuration defined in DataIngestionConfig.
    '''
    def __init__(self,ingestion_config: DataIngestionConfig):
        self.ingestion_config = ingestion_config
    
    def initiate_data_ingestion(self):
        try:
            # Reading data here.
            logging.info("Initiating data ingestion")
            data = pd.read_csv('data/StudentsPerformance.csv')
                        
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            logging.info("Data ingestion completed")
            
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(data,test_size = 0.2, random_state = 18)

            train_set.to_csv(self.ingestion_config.train_data_path,index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path,index = False, header = True)
            logging.info("Train test split ingestion completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error("Error occured in data ingestion")
            raise CustomException(e,sys)
    

if __name__ == '__main__':
    di_config = DataIngestionConfig()
    obj = DataIngestion(di_config)
    obj.initiate_data_ingestion()