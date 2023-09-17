import pandas as pd
import datetime

def transform_stock_df(stock_df: pd.DataFrame, D: int, cutoff: float) -> pd.DataFrame:
    stock_df['Change'] = (
        stock_df['Price']
        .rolling(window = D)
        .apply(lambda df: (df[D-1] - df[0]) / df[0], raw = True)
    )
    stock_df['Label'] = stock_df['Change'] > cutoff
    stock_df['Diff_Days'] = datetime.timedelta(days=D-1) 
    stock_df['Date_L'] = (stock_df['Date'] - stock_df['Diff_Days']).dt.date
    return stock_df
