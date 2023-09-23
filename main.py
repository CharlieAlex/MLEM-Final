def step1_get_stock():
    from etl_func.etl_data import extract_stock_df
    import os
    rawdata_path = '/Users/alexlo/Desktop/Project/MLEM_Final/rawdata'
    workdata_path = '/Users/alexlo/Desktop/Project/MLEM_Final/workdata'
    os.chdir(rawdata_path)
    stock_code = '2303'
    stock_df = extract_stock_df(stock_code)
    
    os.chdir(workdata_path)
    stock_df.to_parquet(stock_code + '_stock_df.parquet', index=False)

def step2_get_data():
    from etl_func.cut_text import Cut_Machine
    import pandas as pd
    import datetime
    import os
    rawdata_path = '/Users/alexlo/Desktop/Project/MLEM_Final/rawdata'
    workdata_path = '/Users/alexlo/Desktop/Project/MLEM_Final/workdata'
    data_time = (datetime.date(2019,1,1), datetime.date(2021,12,31))
    company = '聯電'
    keywords_list = [company]
    kw_title_num = 1
    kw_content_num = 2
    source_dict = {
        'bbs': 'bda2022_mid_bbs_2019-2021.csv',
        'news2019': 'bda2022_mid_news_2019.csv',
        'news2020': 'bda2022_mid_news_2020.csv',
        'news2021': 'bda2022_mid_news_2021.csv',
    }

    final_df = pd.DataFrame()
    for source in source_dict:
        os.chdir(rawdata_path)
        cut_machine = Cut_Machine(articles_source=source_dict[source], data_time=data_time)
        cut_machine.filter_article(
            keywords=keywords_list,
            title_times=kw_title_num,
            content_times=kw_content_num
            )
        word_df = cut_machine.Pool_sep_all_articles()
        print('finish cut:' + source)
        final_df = pd.concat([final_df, word_df], ignore_index=True)

    os.chdir(workdata_path)
    final_df.to_parquet(company + '_word_df.parquet', index=False)

def step3_get_XY():
    from df_func.make_XY import Words_Matrix
    from etl_func.etl_data import transform_stock_df, read_stop_words
    import pandas as pd
    import os
    from datetime import date
    rawdata_path = '/Users/alexlo/Desktop/Project/MLEM_Final/rawdata'
    workdata_path = '/Users/alexlo/Desktop/Project/MLEM_Final/workdata'
    data_time = tuple([date(2019,1,1), date(2021,12,31)])
    day_arg = 2 
    cutoff_arg = 0.03
    company = '聯電'
    stock_code = '2303'

    os.chdir(workdata_path)
    stop_words = read_stop_words(workdata_path)
    word_df = pd.read_parquet(company + '_word_df.parquet')
    stock_df = pd.read_parquet(stock_code + '_stock_df.parquet')
    stock_df = transform_stock_df(stock_df, D=day_arg, cutoff=cutoff_arg)

    words_matrix = Words_Matrix(word_df, stock_df, data_time, stop_words)
    X = words_matrix.feature_X_byChi2(k=1000)
    Y = words_matrix.Y_matrix
    return X, Y

def step4_train_test(X, Y):
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import RidgeClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    import warnings
    warnings.filterwarnings('ignore')

    classifier_list = {
        'kNN': KNeighborsClassifier,
        'Ridge': RidgeClassifier,
        'Desision Tree': DecisionTreeClassifier,
        'Random Forest': RandomForestClassifier,
        'Gradient Boosting': GradientBoostingClassifier,
        'MLP': MLPClassifier,
    }

    for classifier in classifier_list:
        clf = classifier_list[classifier]()
        scores = cross_val_score(clf, X , Y, cv = 5)
        print(classifier, ':', round(scores.mean(), 3))
        print(scores)

if __name__ == '__main__':
    # step1_get_stock()
    # step2_get_data()
    # X, Y = step3_get_XY()
    # step4_train_test(X, Y)

    from df_func.train import train_lag_cutoff
    import pandas as pd
    results = train_lag_cutoff(lag_list=[2,3], cut_list=[0, 0.01])
    df = pd.DataFrame(results)
    df.to_csv('測試.csv', index=False)