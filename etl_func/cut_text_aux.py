# This file is for the class 'Words_Dataset', which contains functions that cut the text into words.

import re
import pandas as pd
import contextlib # for silencing the monpa hello message
with contextlib.redirect_stdout(None):
    import monpa
    from monpa import utils

def remove_nonChinese(sentence:str)->str:
    return re.sub(r'[^\u4e00-\u9fa5]+', '', sentence)

def remove_numberFirst(word_list:list[str]) -> list[str]:
    remove_list:list[str] = []
    for word in word_list:
        if word.startswith(('一', '二','三','四','五','六','七','八','九','十')):
            remove_list.append(word)
    return [word for word in word_list if word not in remove_list]

def sep_sentence(sentence:str, string_:str)->list[str]:
    '''
    Separate a sentence into words and return a list of words.

    E.G. 'fdsfsafdf' -> 'fds fsa fdf'
    '''
    sentence = remove_nonChinese(sentence)
    word_list = monpa.cut_batch(sentence)[0]
    if word_list is not None:
        word_list = remove_numberFirst(word_list)
        string_ += ' '.join(word_list) #將切好的字串list用空白隔開變成一整個字串
    return string_

def sep_article(article:str, article_list:list[str])->list[str]:
    sentence_list = utils.short_sentence(article) #先把一篇文章切成很多句
    article_in_words = ''
    for sentence in sentence_list: #再針對每一句切成很多個字
        article_in_words = sep_sentence(sentence, article_in_words)
    article_list.append(article_in_words)
    return article_list

def sep_all_articles(df:pd.DataFrame)->pd.DataFrame:
    '''
    Turn the articles dataframe (only content) into an artilces list with a date list.

    E.G. [['sfdsdfs'], ['fsfafasf']] -> [['sfd sdfs'], ['fs fa fa sf']]

    E.G. [5/6, 6/7]
    '''
    article_list:list[str] = []
    date_list:list[str] = []
    for i in range(0, df.shape[0]): #幾篇文章就要跑幾次
        try:
            article = df['content'][i]
            article_list = sep_article(article, article_list)
        except:
            article_list.append('')
        date_list.append(df['post_time'][i])
    return pd.DataFrame({'post_time':date_list, 'content':article_list})

def cut_df_by_num(num:int=50, num_rows:int=120)-> list[list[int]]:
    '''
    Cut dataframe rows into several pieces.
    E.G. num = 50, num_rows = 120
    return [[0, 50], [50, 100], [100, 120]]
    '''
    start_index_list = [i*num for i in range(int(num_rows / num)+1)]
    end_index_list = [(i+1)*num for i in range(int(num_rows / num))]
    end_index_list.append(num_rows)
    return [ [start_index_list[i], end_index_list[i]] for i in range(len(start_index_list)) ]