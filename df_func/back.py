from df_func import Words_Matrix, feature_X_byChi2
from etl_func.etl_data import transform_stock_df
from args import (
    stock_df, word_df, data_time, stop_words,
    features_num, classifier_dict,
    day_arg, cutoff_arg
)

train_words_matrix = Words_Matrix(
    word_df=word_df,
    stock_df=transform_stock_df(stock_df, D=day_arg, cutoff=cutoff_arg),
    data_time=data_time,
    stop_words=stop_words
    )
train_X, train_Y = train_words_matrix.X_matrix, train_words_matrix.Y_matrix
train_X = feature_X_byChi2(train_X, train_Y, k=features_num)

MLPclf = MLPClassifier(solver='lbfgs',hidden_layer_sizes=[150, 150])
MLPclf.fit(X_train, y_train)
test_data = get_all_prediction(MLPclf, stock_text_data, test_startDate, test_endDate, X_test)
final_test_data = get_final_prediction2(test_data)
MLPscore, test_stock_data, MLPconfusion = get_accuracy_score(final_test_data, stock_data,test_startDate, test_endDate)
MLPscore_all.append(MLPscore)


## raw code
test_data = get_all_prediction(MLPclf, stock_text_data, test_startDate, test_endDate, X_test)
final_test_data = get_final_prediction2(test_data)
### 每一天預測屬於漲、跌的分別有幾篇文章
def get_all_prediction(ML_model, stock_text_data, test_startDate, test_endDate, X_test):
    test_data = stock_text_data[stock_text_data['post_time'].between(test_startDate, test_endDate)]
    test_data = test_data[test_data['label'].notna()].reset_index(drop = True)
    test_data['predict_label'] = ML_model.predict(X_test)
    test_data = test_data[['post_time', 'predict_label', 'label']]        #label 純粹是要用來計算pred-label多少，沒意義
    return test_data

def get_final_prediction2(test_data):
    final_test_data = test_data.groupby(['post_time']).predict_label.apply(eval_prediction).reset_index()
    final_test_data = final_test_data[final_test_data["predict_label"] != 0].reset_index()
    return final_test_data

for i in range(18):
    print(i+1)
    ### Setting Date
    mon1, mon2, mon3, train_day, test_day, mon4 = compute_mon_day(i)
    year1, year2, year3, year4 = compute_year(i)
    train_startDate = datetime.date(year1,mon1,1)
    train_endDate = datetime.date(year2,mon2,train_day)
    test_startDate = datetime.date(year3,mon3,1)
    test_endDate = datetime.date(year4,mon4,test_day)

    ### Get x, y training data
    train_allWords = get_allWords(stock_text_data, train_startDate, train_endDate) #要跑超久要小心
    X_train, y_train = get_X_y(train_allWords, train_startDate, train_endDate, stock_text_data, stopwords)
    X_train, k_features = featuring_X(X_train, y_train, k= 500, method = chi2)

    ### Get x, y testing data
    test_allWords = get_allWords(stock_text_data, test_startDate, test_endDate)  #要跑超久要小心
    X_test, y_test = get_X_y(test_allWords, test_startDate, test_endDate, stock_text_data, stopwords)
    X_test = X_test.reindex(k_features, axis=1, fill_value=0)

    ### MLP 演算法
    MLPclf = MLPClassifier(solver='lbfgs',hidden_layer_sizes=[150, 150])
    MLPclf.fit(X_train, y_train)
    test_data = get_all_prediction(MLPclf, stock_text_data, test_startDate, test_endDate, X_test)
    final_test_data = get_final_prediction2(test_data)
    MLPscore, test_stock_data, MLPconfusion = get_accuracy_score(final_test_data, stock_data,test_startDate, test_endDate)
    MLPscore_all.append(MLPscore)
    print(MLPscore)
    try:
        MLPmatrix_all = MLPmatrix_all + MLPconfusion
        MLPconfu_matrix2 = pretty_confusion(MLPconfusion)
        print(MLPconfu_matrix2)
    except ValueError:
        pass

    ### Ridge
    RDclf = RidgeClassifier(alpha = 0.05)
    RDclf.fit(X_train, y_train)
    test_data = get_all_prediction(RDclf, stock_text_data, test_startDate, test_endDate, X_test)
    final_test_data = get_final_prediction2(test_data)
    RDscore, test_stock_data, RDconfusion = get_accuracy_score(final_test_data, stock_data,test_startDate, test_endDate)
    RDscore_all.append(RDscore)
    print(RDscore)
    try:
        RDmatrix_all = RDmatrix_all + RDconfusion
        RDconfu_matrix2 = pretty_confusion(RDconfusion)
        print(RDconfu_matrix2)
    except ValueError:
        pass


MLPconfu_matrix = pretty_confusion(MLPmatrix_all)
RDconfu_matrix = pretty_confusion(RDmatrix_all)






def compute_mon_day(i):
    i = i + 1
    if i < 9:
        test_mon = i + 3
    elif i == 9:
        test_mon = 12
    else:
        test_mon = i - 9

    train_mon_start = i
    if train_mon_start > 12:
        train_mon_start = train_mon_start - 12
    train_mon_end = train_mon_start + 2
    if train_mon_end > 12:
         train_mon_end = train_mon_end - 12

    train_day = compute_day(train_mon_end)
    test_day = compute_day(test_mon)
    return train_mon_start, train_mon_end, test_mon, train_day, test_day


### 計算日期函數 3：我知道這三個寫很爛，反省中...
def compute_year(i):
    i = i + 1
    train_year_start = 2020
    train_year_end = 2020
    test_year_start = 2020
    test_year_end = 2020

    if i >= 10:
        test_year_start = 2021
        test_year_end = 2021

    if i in [11, 12]:
        train_year_start = 2020
        train_year_end = 2021
    elif i >= 13:
        train_year_start = 2021
        train_year_end = 2021
    return train_year_start, train_year_end, test_year_start, test_year_end
