import pandas as pd
from df_func.make_XY import Words_Matrix, feature_X_byChi2
from etl_func.etl_data import transform_stock_df
from args import (
    stock_df, word_df, data_time, stop_words,
    features_num, classifier_dict,
    day_arg, cutoff_arg
)
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

class Predict_Machine:
    def __init__(self,
        train_words_matrix:Words_Matrix,
        test_words_matrix:Words_Matrix,
        features_num:int,
        classifier,
        ) -> None:
        self.train_matrix, self.test_matrix  = train_words_matrix, test_words_matrix
        self.train_X, self.train_Y = self.train_matrix.X_matrix, self.train_matrix.Y_matrix
        self.test_X, self.test_Y = self.test_matrix.X_matrix, self.test_matrix.Y_matrix
        self.classifier = classifier
        self.k_feature = feature_X_byChi2(self.train_X, self.train_Y, k=features_num)

    @property
    def trained_classifier(self):
        self.classifier.fit(self.train_X[self.k_feature], self.train_Y)
        return self.classifier

    @property
    def XY(self)->pd.DataFrame:
        XY = self.test_matrix.XY_matrix
        time_filter = XY['Date'].between(self.test_matrix.start_date, self.test_matrix.end_date)
        notna_filter = XY['Label'].notna()
        return XY[time_filter & notna_filter]

    @property
    def XY_hat(self)->pd.DataFrame:
        XY_hat = self.XY
        XY_hat['Label_hat'] = self.trained_classifier.predict(self.test_X[self.k_feature])
        return XY_hat

    @property
    def YY_hat(self)->pd.DataFrame:
        ''' Predict the label for each day.
        Compare the number of up label and down label to determine the final label.
        '''
        Y_hat = (
            self.XY_hat
            .groupby(['Date'])['Label_hat']
            .value_counts()
            .reset_index()
            .sort_values(by=['Date', 'count'], ascending=False)
            .drop_duplicates(subset=['Date'], keep='first')
            .astype({'Date': 'datetime64[ns]'})
        )
        return (
            pd.merge(Y_hat, self.test_matrix.stock_df, on='Date', how='left')
            .dropna(subset=['Label'])
            .astype({'Label': 'bool'})
            [['Date', 'Label', 'Label_hat']]
        )

    def show_accuracy(self)->float:
        return accuracy_score(self.YY_hat['Label'], self.YY_hat['Label_hat'])

    def show_confusion(self):
        return confusion_matrix(self.YY_hat['Label'], self.YY_hat['Label_hat'])

if '__name__' == '__main__':
    pass

# def compute_mon_day(i):
#     i = i + 1
#     if i < 9:
#         test_mon = i + 3
#     elif i == 9:
#         test_mon = 12
#     else:
#         test_mon = i - 9

#     train_mon_start = i
#     if train_mon_start > 12:
#         train_mon_start = train_mon_start - 12
#     train_mon_end = train_mon_start + 2
#     if train_mon_end > 12:
#          train_mon_end = train_mon_end - 12

#     train_day = compute_day(train_mon_end)
#     test_day = compute_day(test_mon)
#     return train_mon_start, train_mon_end, test_mon, train_day, test_day


# ### 計算日期函數 3：我知道這三個寫很爛，反省中...
# def compute_year(i):
#     i = i + 1
#     train_year_start = 2020
#     train_year_end = 2020
#     test_year_start = 2020
#     test_year_end = 2020

#     if i >= 10:
#         test_year_start = 2021
#         test_year_end = 2021

#     if i in [11, 12]:
#         train_year_start = 2020
#         train_year_end = 2021
#     elif i >= 13:
#         train_year_start = 2021
#         train_year_end = 2021
#     return train_year_start, train_year_end, test_year_start, test_year_end