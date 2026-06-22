import sys
import os
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import custom_exception
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainer_comfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class Model_trainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainer_comfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("splittiog training and test input data")
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models={
    "Random Forest":RandomForestRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Logestic regression": LinearRegression(),
    "KNeighborsRegressor" : KNeighborsRegressor(),
    "Adaboost" : AdaBoostRegressor(),
    "Gradient boost": GradientBoostingRegressor(),
    "Lasso":Lasso(),
    "Ridge":Ridge(),
    "XGBRegressor":XGBRegressor(),
    "CatBoostRegressor":CatBoostRegressor(verbose=False),
    "LinearRegression":LinearRegression()
}
            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
        
            best_model_score=max(sorted(model_report.values()))

            best_mosel_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_mosel_name]
            if best_model_score<0.6:
                raise custom_exception("No best model found")
            
            logging.info(f'best found model on both training and test dataset')
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)
            r2_squar=r2_score(y_test,predicted)
            return r2_squar

        except Exception as e:
            raise custom_exception(e,sys)
