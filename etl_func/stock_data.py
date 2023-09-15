import pandas as pd

def get_StockDF(stock_code:str)->pd.DataFrame:
    '''
    Extract selected stock data from the dataset.

    Return a dataframe with columns: Code, Date, Price
    '''
    stock_df = pd.DataFrame()
    for sheets in ['上市2019', '上市2020', '上市2021']:
        stock_data = pd.read_excel('stock_data_2019-2021.xlsx', sheet_name=sheets)
        target = stock_data['證券代碼'].str.startswith(stock_code)
        target = (
            target[['證券代碼', '年月日', '收盤價(元)']]
            .astype({'年月日':'datetime64', '收盤價(元)':'float32'})
            .sort_values(by='年月日')
        )
        stock_df = pd.concat([stock_df, target])
    stock_df = (
        stock_df
        .reset_index(drop = True)
        .rename(columns={'證券代碼':'Code', '年月日':'Date', '收盤價(元)':'Price'})
    )
    return stock_df

if __name__ == '__main__':
    df = get_StockDF('2303')
    print(df.head())