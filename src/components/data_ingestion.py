import os
import sys
from src.exception import custom_exception
from src.logger import logging
import pandas as pd


from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import datatransformation,datatransformationConfig
from src.components.model_trainer import Model_trainer,ModelTrainer_comfig

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
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=datatransformation()
    train_arr,test_arr,_= data_transformation.initiate_data_transformation(train_data,test_data)

    model=Model_trainer()
    print(model.initiate_model_trainer(train_arr,test_arr,))

