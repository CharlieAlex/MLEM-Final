import os
import pandas as pd
from datetime import date
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')
from etl_func.etl_data import read_stop_words

rawdata_path:str = '/Users/alexlo/Desktop/Project/MLEM_Final/rawdata'
workdata_path:str = '/Users/alexlo/Desktop/Project/MLEM_Final/workdata'

os.chdir(workdata_path)
company:str = '聯電'
stock_code:str = '2303'
day_arg:int = 2
cutoff_arg:int  = 0.03
features_num:int  = 2000
data_time:tuple = tuple([date(2019,1,1), date(2021,12,31)])
word_df:pd.DataFrame = pd.read_parquet(company + '_word_df.parquet')
stock_df:pd.DataFrame = pd.read_parquet(stock_code + '_stock_df.parquet')
stop_words:list[str] = read_stop_words()
classifier_dict:dict = {
    'kNN': KNeighborsClassifier,
    'Ridge': RidgeClassifier,
    'Desision Tree': DecisionTreeClassifier,
    'Random Forest': RandomForestClassifier,
    'Gradient Boosting': GradientBoostingClassifier,
    'MLP': MLPClassifier,
}

args_dict:dict = {
    'rawdata_path': rawdata_path,
    'workdata_path': workdata_path,
    'data_time': data_time,
    'day_arg': day_arg,
    'cutoff_arg': cutoff_arg,
    'company': company,
    'stock_code': stock_code,
    'features_num': features_num,
    'classifier_dict': classifier_dict,
    'word_df': word_df,
    'stock_df': stock_df,
    'stop_words': stop_words,
}

class args_class:
    def __init__(self, args:dict):
        for arg in args:
            setattr(self, arg, args[arg])