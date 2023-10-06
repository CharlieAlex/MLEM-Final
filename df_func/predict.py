import pandas as pd
import datetime
from df_func.make_XY import Words_Matrix, feature_X_byChi2
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
        return pd.DataFrame(
            confusion_matrix(self.YY_hat['Label'], self.YY_hat['Label_hat']),
            index=['Positive', 'Negative'],
            columns=['True', 'False'],
            )

class Date_Machine:
    def __init__(self, train_duration, test_duration, data_time) -> None:
        self.train_duration:int = train_duration
        self.test_duration:int = test_duration
        self.start_date:datetime.date = data_time[0]
        self.end_date:datetime.date = data_time[1]
        self.index = 0

    @property
    def date_df(self)->pd.DataFrame:
        df = pd.DataFrame()
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='MS')
        df['Start_Date'] = date_range
        df['End_Date'] = date_range + pd.offsets.MonthEnd()
        df['Index'] = df.index + 1
        return df

    def get_start_date(self, index:int):
        return self.date_df['Start_Date'].iloc[index].date()

    def get_end_date(self, index:int):
        return self.date_df['End_Date'].iloc[index].date()

    @property
    def train_date(self):
        return tuple([
            self.get_start_date(self.index),
            self.get_end_date(self.index+self.train_duration-1),
        ])

    @property
    def test_date(self):
        return tuple([
            self.get_start_date(self.index+self.train_duration),
            self.get_end_date(self.index+self.train_duration+self.test_duration-1),
        ])

if '__name__' == '__main__':
    pass