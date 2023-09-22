from etl_func.cut_text_aux import *

def test_remove_nonChinese():
    assert remove_nonChinese("我正在寫Python程式測試") == "我正在寫程式測試"

def test_remove_numberFirst():
    word_list_ = ['哈哈', '一個', '三個']
    assert remove_numberFirst(word_list_) == ['哈哈']

def test_sep_sentence():
    str = '我正在寫程式測試'
    assert sep_sentence(str, '') == '我 正在 寫 程式 測試'

def test_sep_article():
    article = '我正在寫程式測試。我正在寫程式測試。'
    article_list = []
    assert sep_article(article, article_list) == ['我 正在 寫 程式 測試 我 正在 寫 程式 測試']

def test_get_words_list():
    df = pd.DataFrame({'content':['我正在寫程式測試。我正在寫程式測試。', '我正在寫程式測試。我正在寫程式測試。']})
    assert get_words_list(df) == [
        '我 正在 寫 程式 測試 我 正在 寫 程式 測試',
        '我 正在 寫 程式 測試 我 正在 寫 程式 測試'
        ]

def test_cut_df_by_num():
    assert cut_df_by_num(50, 120) == [[0, 50], [50, 100], [100, 120]]

if __name__ == "__main__":
    print('This is test_cut_text_aux.py')