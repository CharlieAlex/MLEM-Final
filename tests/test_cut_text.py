from etl_func.cut_text import Cut_Machine
import pytest
import os
import datetime
import pandas as pd

@pytest.fixture
def Test_Cut_Machine():
    os.chdir('/Users/alexlo/Desktop/Project/MLEM_Final/rawdata')
    bbs = 'bda2022_mid_bbs_2019-2021.csv'
    data_time = (datetime.date(2019,1,1), datetime.date(2021,12,31))

    instance = Cut_Machine(articles_source=bbs, data_time=data_time)
    instance.article_df = instance.article_df.head(100)
    yield instance

def test_filter_article(Test_Cut_Machine: Cut_Machine)->None:
    kw = '台積電'
    title_num = 2
    content_num = 3

    Test_Cut_Machine.filter_article([kw], title_num, content_times=3)
    df = Test_Cut_Machine.article_df[['title', 'content']].loc[1]
    title_criteria = df['title'].count('|'.join([kw])) >= title_num
    content_criteria = df['content'].count('|'.join([kw])) >= content_num
    assert (title_criteria | content_criteria) == True

def test_get_indexlist_for_multi(Test_Cut_Machine: Cut_Machine)->None:
    Test_Cut_Machine.article_df = pd.DataFrame({
        'content':[
            '我正在寫程式測試。我正在寫程式測試。',
            '我正在寫程式測試。我正在寫程式測試。',
            '我正在寫程式測試。我正在寫程式測試。',
            '我正在寫程式測試。我正在寫程式測試。',
            '我正在寫程式測試。我正在寫程式測試。',
            '我正在寫程式測試。我正在寫程式測試。',
            '我正在寫程式測試。我正在寫程式測試。',
        ]
    })
    Test_Cut_Machine.get_indexlist_for_multi(3)
    assert Test_Cut_Machine.index_list == [[0, 3], [3, 6], [6, 7]]

def test_get_targetDF(Test_Cut_Machine: Cut_Machine)->None:
    assert Test_Cut_Machine.get_targetDF([0, 3]).shape[0] == 3

def test_Pool_get_words_list(Test_Cut_Machine: Cut_Machine)->None:
    pass
    Test_Cut_Machine.article_df = pd.DataFrame({
        'content':[
            '我正在寫程式測試。我正在寫程式測試。',
            '我正在寫程式測試。我正在寫程式測試。',
            '我正在寫程式測試。我正在寫程式測試。',
            '我正在寫程式測試。我正在寫程式測試。',
            '我正在寫程式測試。我正在寫程式測試。',
            '我正在寫程式測試。我正在寫程式測試。',
            '我正在寫程式測試。我正在寫程式測試。',
        ]
    })
    Test_Cut_Machine.index_list = [[0, 3], [3, 6], [6, 7]]
    target = [
            '我 正在 寫 程式 測試 我 正在 寫 程式 測試',
            '我 正在 寫 程式 測試 我 正在 寫 程式 測試',
            '我 正在 寫 程式 測試 我 正在 寫 程式 測試',
            '我 正在 寫 程式 測試 我 正在 寫 程式 測試',
            '我 正在 寫 程式 測試 我 正在 寫 程式 測試',
            '我 正在 寫 程式 測試 我 正在 寫 程式 測試',
            '我 正在 寫 程式 測試 我 正在 寫 程式 測試',
        ]
    assert target == Test_Cut_Machine.Pool_get_words_list()


if __name__ == "__main__":
    print('This is test_cut_text.py')