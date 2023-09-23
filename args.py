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

rawdata_path = '/Users/alexlo/Desktop/Project/MLEM_Final/rawdata'
workdata_path = '/Users/alexlo/Desktop/Project/MLEM_Final/workdata'

data_time = tuple([date(2019,1,1), date(2021,12,31)])
day_arg = 2
cutoff_arg = 0.03
company = '聯電'
stock_code = '2303'
features_num = 2000

classifier_dict = {
    'kNN': KNeighborsClassifier,
    'Ridge': RidgeClassifier,
    'Desision Tree': DecisionTreeClassifier,
    'Random Forest': RandomForestClassifier,
    'Gradient Boosting': GradientBoostingClassifier,
    'MLP': MLPClassifier,
}

os.chdir(workdata_path)
word_df = pd.read_parquet(company + '_word_df.parquet')
stock_df = pd.read_parquet(stock_code + '_stock_df.parquet')
stop_words = read_stop_words()