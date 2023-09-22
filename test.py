import os
from datetime import date
from etl_func.etl_data import *
from df_func.make_XY import *

rawdata_path = '/Users/alexlo/Desktop/Project/MLEM_Final/rawdata'
os.chdir(rawdata_path)
keywords = ['聯電']
keywords_times_titles = 1
keywords_times_content = 2
article_df = pd.read_csv('bda2022_mid_bbs_2019-2021.csv')
article_df = transform_article_df(article_df, keywords, keywords_times_titles, keywords_times_content)
with open('stopwords_zh.txt', 'r') as file:
    stop_words = file.read().splitlines()
print('finish loading article')

workdata_path = '/Users/alexlo/Desktop/Project/MLEM_Final/workdata'
os.chdir(workdata_path)
all_words = np.load('聯電.npy').tolist()
stock_df = pd.read_parquet('聯電.parquet').astype({'Date':'datetime64[ns]'})
stock_df = transform_stock_df(stock_df, D=2, cutoff=3)
print('finish loading stock')

data_time = tuple([date(2019,1,1), date(2021,12,31)])
words_matrix = Words_Matrix(all_words, stop_words, article_df, stock_df, data_time)
print(words_matrix.main_df)