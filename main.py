
def step1_get_data():
    from etl_func.cut_text import Cut_Machine
    import numpy as np
    import datetime
    import os
    # setting
    os.chdir('/Users/alexlo/Desktop/Project/MLEM_Final/rawdata')
    bbs = 'bda2022_mid_bbs_2019-2021.csv'
    # news19 = 'bda2022_mid_news_2019.csv'
    # news20 = 'bda2022_mid_news_2020.csv'
    # news21 = 'bda2022_mid_news_2021.csv'
    data_time = (datetime.date(2019,1,1), datetime.date(2021,12,31))
    company = '聯電'
    keywords_list = [company]

    # get data
    cut_machine = Cut_Machine(articles_source=bbs, data_time=data_time)
    cut_machine.article_df = cut_machine.filter_article(keywords=keywords_list, title_times=1, content_times=3)
    final_df = cut_machine.Pool_sep_all_articles()

    # save data
    os.chdir('/Users/alexlo/Desktop/Project/MLEM_Final/workdata')
    final_df.to_csv(company + '.csv', index=False)
    # np.save(company, np.array(final_df))

def step2_get_XY():
    from df_func.make_XY import Words_Matrix
    from etl_func.etl_data import transform_article_df, transform_stock_df
    import pandas as pd
    import os
    from datetime import date

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
    all_words = pd.read_csv('/Users/alexlo/Desktop/Project/MLEM_Final/workdata/聯電.csv')
    stock_df = pd.read_parquet('聯電.parquet').astype({'Date':'datetime64[ns]'})
    stock_df = transform_stock_df(stock_df, D=2, cutoff=3)
    print('finish loading stock')

    data_time = tuple([date(2019,1,1), date(2021,12,31)])
    words_matrix = Words_Matrix(all_words, stop_words, article_df, stock_df, data_time)
    X = words_matrix.feature_X_byChi2(k=1000)
    Y = words_matrix.Y_matrix

if __name__ == '__main__':
    pass