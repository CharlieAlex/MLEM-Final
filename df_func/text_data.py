import pandas as pd
import numpy as np
from datetime import datetime
from typing import Self
from sklearn.feature_extraction.text import TfidfVectorizer  #把所有關鍵字變成文字向量模型的套件
from sklearn.feature_selection import SelectKBest            #用來挑出最好幾個的關鍵字的套件
from sklearn.feature_selection import chi2                   #SelectKBest 要輸入的 score function

class Words_Matrix:
    def __init__(
            self:Self,
            all_words:list,
            stop_words:list,
            article_df:pd.DataFrame,
            stock_df:pd.DataFrame,
            data_time:tuple[datetime, datetime],
        ) -> None:
        self.all_words = all_words
        self.stop_words = stop_words
        self.article_df = article_df
        self.stock_df = stock_df
        self.start_date, self.end_date = data_time
        self.tfidf_matrix = pd.DataFrame()

    def get_main_df(self:Self)->pd.DataFrame:
        '''
        Create main dataset by merging stock and article dataset.
        '''
        main_df = pd.merge(article_df, stock_df, left_on='Post_Time', right_on='Date_L') \
                    [['Post_Time', 'Title', 'Content', 'Label']]
        self.main_timerange = main_df['Post_Time'].between(self.start_date, self.end_date)
        self.article_timerange = article_df['Post_Time'].between(self.start_date, self.end_date)

    def get_tfidf_matrix(self:Self)->pd.DataFrame:
        '''
        Turn the words list into tfidf matrix.

        This func. is for get_X().
        '''
        vectorizer = TfidfVectorizer(stop_words=self.stop_words)                      
        tfidf_matrix = vectorizer.fit_transform(self.all_words)       
        tfidf_matrix = pd.DataFrame(
            data=tfidf_matrix.toarray(),
            columns=vectorizer.get_feature_names_out(),
        )
        self.tfidf_matrix = tfidf_matrix

    def get_X(self:Self)->pd.DataFrame:
        '''
        create X.
        '''
        X = self.get_tfidf_matrix(all_words, stop_words)
        X['Date'] = article_df[self.article_timerange]['Post_Time']
        X_temp = pd.DataFrame({'Date':self.main_df[self.main_timerange]['Post_Time']})
        X = pd.merge(X, X_temp, how='inner', on='Date')
        X = X.drop(columns=['Date']).reset_index(drop=True)
        return X
    
    def get_y(self:Self)->pd.DataFrame:
        return self.main_df[self.main_timerange]['Label'].reset_index(drop=True)

    def feature_better_X(X, y, k, method = chi2):
        ### 進一步縮小 X 矩陣內詞的數量，挑出更好的關鍵詞
        selector = SelectKBest(method, k = k).fit(X, y)           
        k_features = X.columns[selector.get_support()]  
        X = X[k_features]
        return X, k_features

if __name__ == '__main__':
    import os
    rawdata_path = '/Users/alexlo/Desktop/Project/Project_MLEM/rawdata'
    workdata_path = '/Users/alexlo/Desktop/Project/Project_MLEM/workdata'
    os.chdir(workdata_path)

    code = '2330'
    all_words = np.load(code + '_words.npy').tolist()
    article_df = pd.read_csv('bda2022_mid_bbs_2019-2021.csv')
    stock_df = pd.read_parquet('聯電.parquet').astype({'Date':'datetime64'})
    
    with open('stopwords_zh.txt', 'r') as file:
        stop_words = file.read().splitlines() 
    
    words_matrix = Words_Matrix(all_words, stop_words, article_df, stock_df)