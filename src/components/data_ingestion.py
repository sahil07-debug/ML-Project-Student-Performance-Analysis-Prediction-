import os
import sys
from src.exception import custom_exception
from src.logger import logging
import pandas as pd


from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','data.csv')


class DataIngestion:
    def __init__(self):
        self.ingention_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv(r'notebook\StudentsPerformance.csv')
            logging.info('read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingention_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingention_config.raw_data_path,index=False,header=True)

            logging.info('train_test_split initiated')
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingention_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingention_config.test_data_path,index=False,header=True)

            logging.info('ingestion of the data completed')

            return(
                self.ingention_config.train_data_path,
                self.ingention_config.test_data_path
            )
        except Exception as e:
            raise custom_exception(e,sys)



if __name__=='__main__':
    obj=DataIngestion()
    obj.initiate_data_ingestion()
