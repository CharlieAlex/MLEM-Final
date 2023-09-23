import pandas as pd
from multiprocessing import Pool
from typing import Self
from datetime import datetime
from functools import cached_property
import warnings
warnings.filterwarnings("ignore")
from .cut_text_aux import sep_all_articles, cut_df_by_num

class Cut_Machine:
    def __init__(self:Self, articles_source:str, data_time:tuple[datetime, datetime])->None:
        self.article_df:pd.DataFrame = pd.read_csv(articles_source)
        self.article_df['post_time'] = pd.to_datetime(self.article_df['post_time']).dt.date
        self.start_date, self.end_date = data_time

    def create_filter(self:Self, col_name:str, filter_times:int, keywords:list[str]):
        '''
        Filter for the function: filter_article().
        '''
        return self.article_df[col_name].str.count('|'.join(keywords)) >= filter_times

    def filter_article(self:Self, keywords:list[str], title_times:int, content_times:int)->None:
        '''
        Select articles that contain keywords in title or content at least X times.
        '''
        title_filter = self.create_filter('title', title_times, keywords)
        content_filter = self.create_filter('content', content_times, keywords)
        self.article_df = self.article_df[title_filter | content_filter].reset_index(drop = True)

    @cached_property
    def index_list(self:Self)->list[list[int]]:
        '''
        Create the index list for multiprocessing.
        '''
        return cut_df_by_num(num=50, num_rows=self.article_df.shape[0])

    def get_targetDF(self:Self, index:list[int])->pd.DataFrame:
        '''
        Accroding to the index list, get the filtered dataframe.
        '''
        return self.article_df.iloc[index[0]:index[1]].reset_index(drop=True)

    def sep_all_articles_aux(self:Self, index:list[int]):
        df = self.get_targetDF(index)
        return sep_all_articles(df)

    def Pool_sep_all_articles(self:Self)->pd.DataFrame:
        word_df = pd.DataFrame()
        with Pool(processes=8) as pool:
            for index, result_df in enumerate(pool.imap(self.sep_all_articles_aux, self.index_list)):
                word_df = pd.concat([word_df, result_df], ignore_index=True)
        word_df['post_time'] = word_df['post_time'].astype('string') #for saving to parquet
        return word_df

if __name__ == '__main__':
    print('This is cut_text.py')