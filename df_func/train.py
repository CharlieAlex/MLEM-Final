from tqdm import tqdm
from df_func.make_XY import Words_Matrix, feature_X_byChi2
from etl_func.etl_data import transform_stock_df
from sklearn.model_selection import cross_val_score
import pandas as pd
from typing import Callable

train_function = Callable[[list], dict]

def train(
        classifier_dict_:dict,
        X_:pd.DataFrame,
        Y_:pd.DataFrame,
        cv_:int)->dict():

    scores_dict = dict()
    for classifier in classifier_dict_:
        try:
            clf = classifier_dict_[classifier]()
            scores = cross_val_score(estimator=clf, X=X_ , y=Y_, cv=cv_)
            print(classifier, ':', round(scores.mean(), 3))
            print(scores)
            scores_dict[classifier] = scores
        except:
            print('there is an error when traning', classifier)
    return scores_dict

def create_train_function(arg_name)->train_function:
    def train_function(arg_list:list)->dict():
        from args import args_dict, args_class
        all_results = dict()
        for value_ in (arg_list):
            args_dict[arg_name] = value_
            args = args_class(args_dict)
            words_matrix = Words_Matrix(
                word_df=args.word_df,
                stock_df=transform_stock_df(
                    args.stock_df, D=args.day_arg, cutoff=args.cutoff_arg
                    ),
                data_time=args.data_time,
                stop_words=args.stop_words
                )
            X, Y = words_matrix.X_matrix, words_matrix.Y_matrix
            X = feature_X_byChi2(X, Y, k=args.features_num)

            print("資料中漲跌的比例: \n", Y.value_counts())

            all_results[f'{value_}'] = (
                train(classifier_dict_=args.classifier_dict, X_=X, Y_=Y, cv_=5)
                )
        return all_results
    return train_function

def train_lag_cutoff(lag_list:list[int], cut_list:list[float])->dict():
    '''用幾天前的股價預測今天的股價&用幾%作為漲跌幅
    1. lag: lag=2 表示用當天和2天前的股價做比較，計算漲跌比率
    2. cut: cut=0.01 表示如果漲跌比率>1%才作為漲，否則視為跌
    '''
    from args import (
        stock_df, word_df, data_time, stop_words,
        features_num, classifier_dict
    )

    all_results = dict()
    for lag in (lag_list):
        for cut in tqdm(cut_list):
            words_matrix = Words_Matrix(
                word_df=word_df,
                stock_df=transform_stock_df(stock_df, D=lag, cutoff=cut),
                data_time=data_time,
                stop_words=stop_words
                )
            X, Y = words_matrix.X_matrix, words_matrix.Y_matrix
            X = feature_X_byChi2(X, Y, k=features_num)

            print("資料中漲跌的比例: \n", Y.value_counts())

            all_results[f'lag{lag}cut{cut}'] = (
                train(classifier_dict_=classifier_dict, X_=X, Y_=Y, cv_=5)
                )
    return all_results

if __name__ == '__main__':
    pass