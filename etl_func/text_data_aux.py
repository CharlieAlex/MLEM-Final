def combine_np_files(stock_code):
    '''
    Combine all the np files startwith 'stock_code'
    '''
    os.chdir(workdata_path)
    file_list = [file for file in os.listdir() if file.startswith(stock_code)]
    words_array = np.array([])
    for file in file_list:
        words_array = np.append(words_array, np.load(file))
        words_array = np.array(words_array)
        os.remove(file)
        np.save(stock_code + '_words', words_array)
    return words_array

def drop_bda2022(stock_code):
    '''
    os 刪除所有bda2022開頭的檔案，測試使用
    '''
    os.chdir(workdata_path)
    for file in os.listdir():
        if file.startswith(stock_code):
            os.remove(file)

