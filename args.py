import os
import pandas as pd
from datetime import date
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from etl_func.etl_data import read_stop_words
import warnings
warnings.filterwarnings('ignore')

# 可變參數
dir_path:str = '/Users/alexlo/Desktop/Project/MLEM_Final/'         # 檔案路徑
company:str = '台積電'          # 公司名稱
stock_code:str = '2330'      # 股票代碼
kw_list = [company, ]        # 關鍵字搜尋清單（通常一定會放公司名稱）
kw_title_num = 1             # 關鍵字搜尋標題數量
kw_content_num = 3           # 關鍵字搜尋內文數量
day_arg:int = 2              # 跟幾天前的股價相比
cutoff_arg:int  = 0.03       # 跌漲幅度門檻(e.g. 0.03 代表跌漲幅度超過 3% 才算跌或漲)
features_num:int  = 2000     # 選取幾個關鍵詞作為預測

##########################################################################################
# 不變參數
rawdata_path:str = dir_path + 'rawdata'
workdata_path:str = dir_path + 'workdata'
datafile_name:dict[str, str] = { # 原始檔案名稱
    'bbs': 'bda2022_mid_bbs_2019-2021.csv',
    'news2019': 'bda2022_mid_news_2019.csv',
    'news2020': 'bda2022_mid_news_2020.csv',
    'news2021': 'bda2022_mid_news_2021.csv',
    'forum2019': 'bda2022_mid_forum_2019.csv',
    'forum2020': 'bda2022_mid_forum_2020.csv',
    'forum2021': 'bda2022_mid_forum_2021.csv',
}
classifier_dict:dict = { # 演算法
    'kNN': KNeighborsClassifier,
    'Ridge': RidgeClassifier,
    'Desision Tree': DecisionTreeClassifier,
    'Random Forest': RandomForestClassifier,
    'Gradient Boosting': GradientBoostingClassifier,
    'MLP': MLPClassifier,
}
data_time:tuple = tuple([date(2019,1,1), date(2021,12,31)])
os.chdir(rawdata_path)
stop_words:list[str] = read_stop_words()
try:
    os.chdir(workdata_path)
    word_df:pd.DataFrame = pd.read_parquet(company + '_word_df.parquet')
    stock_df:pd.DataFrame = pd.read_parquet(stock_code + '_stock_df.parquet')
except:
    print('No word_df or stock_df')

##########################################################################################
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
    'stop_words': stop_words,
}
try:
    args_dict.update({
        'word_df': word_df,
        'stock_df': stock_df,
    })
except:
    print('No word_df or stock_df')

class args_class:
    def __init__(self, args:dict):
        for arg in args:
            setattr(self, arg, args[arg])