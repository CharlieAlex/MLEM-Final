import pandas as pd
import numpy as np
from datetime import datetime
from typing import Self
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from functools import cached_property

class Words_Matrix:
    def __init__(
            self:Self,
            all_words:pd.DataFrame,
            stop_words:list[str],
            article_df:pd.DataFrame,
            stock_df:pd.DataFrame,
            data_time:tuple[datetime, datetime],
        ) -> None:
        self.all_words = all_words
        self.stop_words = stop_words
        self.article_df = article_df
        self.stock_df = stock_df
        self.start_date, self.end_date = data_time


    @cached_property
    def tfidf_matrix(self:Self)->pd.DataFrame:
        '''
        Turn the words list into tfidf matrix.

        This func. is for get_X().
        '''
        vectorizer = TfidfVectorizer(stop_words=self.stop_words)
        matrix = pd.DataFrame(
            data=vectorizer.fit_transform(self.all_words['content']).toarray(),
            columns=vectorizer.get_feature_names_out(),
            )
        matrix['Date'] = pd.to_datetime(self.all_words['post_time'], format='%Y-%m-%d').dt.date
        return matrix

    @cached_property
    def XY_matrix(self:Self)->pd.DataFrame:
        date_filter:pd.Series[bool] = self.tfidf_matrix['Date'].between(self.start_date, self.end_date)
        X = self.tfidf_matrix[date_filter]
        return pd.merge(
            X, self.stock_df[['Date_L', 'Label']],
            left_on='Date', right_on='Date_L'
            ).reset_index(drop=True)

    @cached_property
    def X_matrix(self:Self)->pd.DataFrame:
        return self.XY_matrix.drop(columns=['Date', 'Date_L', 'Label'])

    @cached_property
    def Y_matrix(self:Self)->pd.DataFrame:
        return self.XY_matrix['Label']

    def feature_X_byChi2(self, k):
        '''Find the k best features by Chi2 test.

        Args:
            k (int): The number of features to select.

        Output:
            filtered X matrix
        '''
        selected_label:np.ndarray[bool] = (
            SelectKBest(chi2, k=k)
            .fit(self.X_matrix, self.Y_matrix)
            .get_support()
            )
        k_features:list[str] = self.X_matrix.columns[selected_label]
        return self.X_matrix[k_features]

if __name__ == '__main__':
    pass