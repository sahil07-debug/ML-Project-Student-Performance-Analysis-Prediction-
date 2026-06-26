import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import custom_exception
from src.logger import logging
from src.utils import save_object

@dataclass
class datatransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts',"preprocessor.pkl")

    
class datatransformation:
    def __init__(self):
        self.data_transformation_comfig=datatransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_column=["reading_score","writing_score"]
            categorical_column=["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]
            
            num_pipeline=Pipeline(
                steps=[
                      ("imputer", SimpleImputer(strategy="median")),
                      ("scalar", StandardScaler())
                      ]
                              )

            categorical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("oneHotencoder",OneHotEncoder()),
                    ("scalar",StandardScaler(with_mean=False)) 
                ]
            )

            logging.info("numerical columns standard scaling complected")
            logging.info("categorical columns encoding complected")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_column),
                    ("cat_pipeline",categorical_pipeline,categorical_column)
                ]
            )

            return preprocessor
        except Exception as e:
            raise custom_exception(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read train and test data completed")
            logging.info("obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            num_column_name=["reading_score","writing_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name])
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name])
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"applying preprocessor object on training dataframe and testing dataframe"
            )
            input_feature_train_arr= preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            logging.info(f"saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_comfig.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_comfig.preprocessor_obj_file_path
            )
        except Exception as e:
            raise custom_exception(e,sys)
        





        
        

            

        


 
            
