import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,
                 age: int,
                 sex: bool,
                 sick: bool,             
                 pregnant: bool,      
                 thyroid: bool,      
                 surgery: bool,         
                 I131: bool,               
                 treatment: bool,          
                 lithium: bool,            
                 goitre: bool,        
                 tumor: bool, 
                 TSH: int,
                 T3: int,
                 TT4: int,
                 T4U: int,
                 FTI: int):
        
        self.age = age
        self.sex = sex
        self.sick = sick
        self.pregnant = pregnant
        self.thyroid = thyroid
        self.surgery = surgery
        self.I131 = I131
        self.treatment = treatment
        self.lithium = lithium
        self.goitre = goitre
        self.tumor = tumor
        self.TSH = TSH
        self.T3 = T3
        self.TT4 = TT4
        self.T4U = T4U
        self.FTI = FTI

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "sex": [self.sex],
                "sick": [self.sick],
                "pregnant": [self.pregnant],
                "thyroid surgery": [self.thyroid],
                "I131 treatment": [self.I131],
                "lithium": [self.lithium],
                "goitre": [self.goitre],
                "tumor": [self.tumor],
                "TSH": [self.TSH],
                "T3": [self.T3],
                "TT4": [self.TT4],
                "T4U": [self.T4U],
                "FTI": [self.FTI],
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
