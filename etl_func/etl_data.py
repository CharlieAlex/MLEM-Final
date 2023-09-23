import pandas as pd
import datetime

def extract_stock_df(stock_code:str)->pd.DataFrame:
    '''
    Extract selected stock data from the raw dataset stock_data_2019-2021.xlsx.

    Return a dataframe with columns: Code, Date, Price
    '''
    stock_df = pd.DataFrame()
    for sheets in ['上市2019', '上市2020', '上市2021']:
        stock_1y_data = pd.read_excel('stock_data_2019-2021.xlsx', sheet_name=sheets)
        stock_filter:pd.Series.bool = stock_1y_data['證券代碼'].str.startswith(stock_code)
        stock_1y_data:pd.DataFrame = (
            stock_1y_data[stock_filter][['證券代碼', '年月日', '收盤價(元)']]
            .astype({'年月日':'datetime64[ns]', '收盤價(元)':'float32'})
            .sort_values(by='年月日')
        )
        stock_df = pd.concat([stock_df, stock_1y_data])
    return (
        stock_df
        .reset_index(drop = True)
        .rename(columns={'證券代碼':'Code', '年月日':'Date', '收盤價(元)':'Price'})
    )

def transform_stock_df(stock_df: pd.DataFrame, D: int, cutoff: float) -> pd.DataFrame:
    stock_df= stock_df.astype({'Date':'datetime64[ns]'})
    stock_df['Change'] = (
        stock_df['Price']
        .rolling(window = D)
        .apply(lambda df: (df[D-1] - df[0]) / df[0], raw = True)
    )
    stock_df['Label'] = stock_df['Change'] > cutoff
    stock_df['Diff_Days'] = datetime.timedelta(days=D-1)
    stock_df['Date_L'] = (stock_df['Date'] - stock_df['Diff_Days']).dt.date
    return stock_df

def read_stop_words()->list[str]:
    with open('stopwords_zh.txt', 'r') as file:
        stop_words = file.read().splitlines()
    return stop_words

# def transform_article_df(
#         article_df:pd.DataFrame,
#         kw:list[str],
#         kw_title_num:int,
#         kw_content_num:int) -> pd.DataFrame:
#     criteria1 = (article_df['title'].str.count('|'.join(kw)) >= kw_title_num)
#     criteria2 = (article_df['content'].str.count('|'.join(kw)) >= kw_content_num)
#     article_df = (
#         article_df[(criteria1) | (criteria2)]
#         .rename(columns={'post_time':'Post_Time', 'title':'Title', 'content':'Content'})
#         .reset_index(drop = True)
#     )
#     article_df['Post_Time'] = pd.to_datetime(article_df['Post_Time']).dt.date
#     return article_df[['Title', 'Content', 'Post_Time']]

if __name__ == '__main__':
    pass
    import os

    workdata_path = '/Users/alexlo/Desktop/Project/MLEM_Final/workdata'
    os.chdir(workdata_path)
    stock_df = pd.read_parquet('聯電.parquet').astype({'Date':'datetime64[ns]'})
    stock_df = transform_stock_df(stock_df, D=2, cutoff=3)
    print(stock_df.head())

    # rawdata_path = '/Users/alexlo/Desktop/Project/MLEM_Final/rawdata'
    # os.chdir(rawdata_path)
    # article_df = pd.read_csv('bda2022_mid_bbs_2019-2021.csv')
    # keywords = ['聯電']
    # keywords_times_titles = 1
    # keywords_times_content = 2
    # article_df = transform_article_df(article_df, keywords, keywords_times_titles, keywords_times_content)
    # print(article_df.head())