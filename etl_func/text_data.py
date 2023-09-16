from typing import Self
from datetime import datetime
from cut_text import get_words_list
import numpy as np
import pandas as pd
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")

class Words_Dataset:
    def __init__(self:Self, data_source:str, data_time:tuple[datetime, datetime])->None:
        self.article_df:pd.DataFrame = pd.read_csv(data_source)
        self.article_df['post_time'] = pd.to_datetime(self.article_df['post_time']).dt.date
        self.start_date:datetime = data_time[0]
        self.end_date:datetime = data_time[1]

    def filter_article(self:Self, keywords:list[str], title_times:int, content_times:int)->None:
        '''
        Select articles that contain keywords in title or content at least X times.
        '''
        title_filter = (self.article_df['title'].str.count('|'.join(keywords)) >= title_times)
        content_filter = (self.article_df['content'].str.count('|'.join(keywords)) >= content_times)
        self.article_df = (
            self
            .article_df[ title_filter | content_filter]
            .reset_index(drop = True)
        )

    def get_indexlist_for_multi(self, num:int=50)->None:
        num_rows = self.article_df.shape[0]
        start_index_list = [i*num for i in range(int(num_rows / num)+1)]
        end_index_list = [(i+1)*num for i in range(int(num_rows / num))]
        end_index_list.append(num_rows)
        self.index_list:list[list[int]] = [
            [start_index_list[i], end_index_list[i]]
            for i in range(len(start_index_list))
        ]

    def get_targetDF(self:Self, index:list[int])->pd.DataFrame:
        return self.article_df.iloc[index[0]:index[1]].reset_index(drop=True)

    def get_words_list_aux(self, index_list):
        df = self.get_targetDF(index_list)
        return get_words_list(df)

    def Pool_get_words_list(self)->list[str]:
        final_list:list[str] = []
        # method: 1
        # pool = ThreadPool(8)
        # pool.map(self.get_words_list_aux, index_list)
        # pool.close()
        # pool.join()

        # method: 2
        with Pool(processes=8) as pool:
            print()
            for index, result in enumerate(pool.imap(self.get_words_list_aux, self.index_list)):
                final_list += result
                print('finish', index)
        return final_list

if __name__ == '__main__':
    import datetime
    import os
    # setting
    os.chdir('/Users/alexlo/Desktop/Project/MLEM_Final/rawdata')
    bbs = 'bda2022_mid_bbs_2019-2021.csv'
    news19 = 'bda2022_mid_news_2019.csv'
    news20 = 'bda2022_mid_news_2020.csv'
    news21 = 'bda2022_mid_news_2021.csv'
    data_time = (datetime.date(2019,1,1), datetime.date(2021,12,31))
    company = '聯電'
    keywords_list = [company]

    # get data
    words_dataset = Words_Dataset(data_source=news21, data_time=data_time)
    words_dataset.filter_article(keywords=keywords_list, title_times=1, content_times=3)
    words_dataset.get_indexlist_for_multi()
    final_list_ = words_dataset.Pool_get_words_list()
    final_list_ = np.array(final_list_)

    # save data
    os.chdir('/Users/alexlo/Desktop/Project/MLEM_Final/workdata')
    np.save(company, final_list_ )