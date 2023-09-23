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
            word_df:pd.DataFrame,
            stock_df:pd.DataFrame,
            data_time:tuple[datetime, datetime],
            stop_words:list[str],
        ) -> None:
        self.word_df = word_df
        self.stock_df = stock_df
        self.start_date, self.end_date = data_time
        self.stop_words = stop_words

    @cached_property
    def tfidf_matrix(self:Self)->pd.DataFrame:
        '''
        Turn the words list into tfidf matrix.

        This func. is for get_X().
        '''
        vectorizer = TfidfVectorizer(stop_words=self.stop_words)
        matrix = pd.DataFrame(
            data=vectorizer.fit_transform(self.word_df['content']).toarray(),
            columns=vectorizer.get_feature_names_out(),
            )
        matrix['Date'] = pd.to_datetime(self.word_df['post_time'], format='%Y-%m-%d').dt.date
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

def feature_X_byChi2(X:pd.DataFrame, Y:pd.DataFrame, k):
    '''Find the k best features by Chi2 test.

    Args:
        k (int): The number of features to select.

    Output:
        filtered X matrix
    '''
    selected_label:np.ndarray[bool] = (
        SelectKBest(chi2, k=k)
        .fit(X, Y)
        .get_support()
        )
    k_features:list[str] = X.columns[selected_label]
    return X[k_features]

if __name__ == '__main__':
    pass