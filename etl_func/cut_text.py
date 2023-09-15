import monpa
from monpa import utils
import re
import numpy as np
import pandas as pd

def remove_nonChinese(sentence:str)->str:
    return re.sub(r'[^\u4e00-\u9fa5]+', '', sentence)

def remove_numberFirst(word_list:list[str]) -> list[str]:
    for word in word_list:
        if word.startswith(('一', '二','三','四','五','六','七','八','九','十')):
            word_list.remove(word)
    return word_list

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

def get_words_list(df:pd.DataFrame,)->list[str]:
    '''
    Turn the articles dataframe (only content) into an artilces list, and then

    Turn the articles list into another list in which each article is a string of words separated by ' '.

    E.G. [['sfdsdfs'], ['fsfafasf']] -> [['sfd sdfs'], ['fs fa fa sf']]
    '''
    article_list:list[str] = []
    error_times:int = 0
    for i in range(0, df.shape[0]): #幾篇文章就要跑幾次
        try:
            article = df['content'][i]
            article_list = sep_article(article, article_list)
        except:
            error_times += 1
            article_list.append('')
    return article_list

# def save_article_list(self, article_list:np.ndarray[str], index:list[int])->None:
#     save_name = (self.data_source.split('_')[2] + self.data_source.split('_')[3])[:-4] + '_' + str(index[0])
#     np.save(save_name, article_list)