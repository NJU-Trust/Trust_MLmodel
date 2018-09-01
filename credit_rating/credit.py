import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split
from hep_ml.nnet import MLPRegressor
from sklearn import metrics
from sklearn.externals import joblib
import seaborn as sns
import copy
import matplotlib.pyplot as plt

def mono_bin(Y, X, n=10):
    r = 0
    good=Y.sum()
    bad=Y.count()-good
    while np.abs(r) < 1:
        d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, n)})
        d2 = d1.groupby('Bucket', as_index = True)
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
        n = n - 1
    d3 = pd.DataFrame(d2.X.min(), columns = ['min'])
    d3['min']=d2.min().X
    d3['max'] = d2.max().X
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    d3['woe']=np.log((d3['rate']/good)/((1-d3['rate'])/bad))
    d3['goodattribute']=d3['sum']/good
    d3['badattribute']=(d3['total']-d3['sum'])/bad
    iv=((d3['goodattribute']-d3['badattribute'])*d3['woe']).sum()
    d4 = (d3.sort_index(by = 'min')).reset_index(drop=True)
    woe=list(d4['woe'].round(3))
    cut=[]
    cut.append(float('-inf'))
    for i in range(1,n+1):
        qua=X.quantile(i/(n+1))
        cut.append(round(qua,4))
    cut.append(float('inf'))
    return d4,iv,cut,woe

def woe_value(d1):
    d2 = d1.groupby('Bucket', as_index = True)
    good=train_y.sum()
    bad=train_y.count()-good
    d3 = pd.DataFrame(d2.X.min(), columns = ['min'])
    d3['min']=d2.min().X
    d3['max'] = d2.max().X
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    d3['woe'] = np.log((d3['rate']/good)/((1-d3['rate'])/bad))
    d3['goodattribute']=d3['sum']/good
    d3['badattribute']=(d3['total']-d3['sum'])/bad
    iv=((d3['goodattribute']-d3['badattribute'])*d3['woe']).sum()
    d4 = (d3.sort_index(by = 'min')).reset_index(drop=True)
    woe=list(d4['woe'].round(3))
    return d4,iv,woe

def get_score(coe,woe,factor):
    scores=[]
    for w in woe:
        score=round(coe*w*factor,0)
        scores.append(score)
    return scores

def trans_woe(var,var_name,x_woe,x_cut):
    woe_name = var_name + '_woe'
    for i in range(len(x_woe)):
        if i == 0:
            var.loc[(var[var_name]<=x_cut[i+1]),woe_name] = x_woe[i]
        elif (i>0) and (i<= len(x_woe)-2):
            var.loc[((var[var_name]>x_cut[i])&(var[var_name]<=x_cut[i+1])),woe_name] = x_woe[i]
        else:
            var.loc[(var[var_name]>x_cut[len(x_woe)-1]),woe_name] = x_woe[len(x_woe)-1]
    return var


if __name__=='__main__':
    train_data = pd.read_csv('cs-training.csv')
    train_data = train_data.iloc[:, 1:]
    mData = train_data.iloc[:, [5, 0, 1, 2, 3, 4, 6, 7, 8, 9]]
    train_known = mData[mData.MonthlyIncome.notnull()].as_matrix()
    train_unknown = mData[mData.MonthlyIncome.isnull()].as_matrix()
    train_X = train_known[:, 1:]
    train_y = train_known[:, 0]
    rfr = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=3, n_jobs=-1)
    rfr.fit(train_X, train_y)
    predicted_y = rfr.predict(train_unknown[:, 1:]).round(0)
    train_data.loc[train_data.MonthlyIncome.isnull(), 'MonthlyIncome'] = predicted_y
    train_data = train_data.dropna()
    train_data = train_data.drop_duplicates()
    train_data = train_data[train_data['NumberOfTime30-59DaysPastDueNotWorse'] < 90]
    train_data = train_data[train_data.age > 0]
    train_data['SeriousDlqin2yrs'] = 1 - train_data['SeriousDlqin2yrs']
    y = train_data.iloc[:, 0]
    X = train_data.iloc[:, 1:]
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=0)
    x1_d, x1_iv, x1_cut, x1_woe = mono_bin(train_y, train_X.RevolvingUtilizationOfUnsecuredLines)
    x2_d, x2_iv, x2_cut, x2_woe = mono_bin(train_y, train_X.age)
    x4_d, x4_iv, x4_cut, x4_woe = mono_bin(train_y, train_X.DebtRatio)
    x5_d, x5_iv, x5_cut, x5_woe = mono_bin(train_y, train_X.MonthlyIncome)

    d1 = pd.DataFrame({"X": train_X['NumberOfTime30-59DaysPastDueNotWorse'], "Y": train_y})
    d1['Bucket'] = d1['X']
    d1_x1 = d1.loc[(d1['Bucket'] <= 0)]
    d1_x1.loc[:, 'Bucket'] = "(-inf,0]"
    d1_x2 = d1.loc[(d1['Bucket'] > 0) & (d1['Bucket'] <= 1)]
    d1_x2.loc[:, 'Bucket'] = "(0,1]"
    d1_x3 = d1.loc[(d1['Bucket'] > 1) & (d1['Bucket'] <= 3)]
    d1_x3.loc[:, 'Bucket'] = "(1,3]"
    d1_x4 = d1.loc[(d1['Bucket'] > 3) & (d1['Bucket'] <= 5)]
    d1_x4.loc[:, 'Bucket'] = "(3,5]"
    d1_x5 = d1.loc[(d1['Bucket'] > 5)]
    d1_x5.loc[:, 'Bucket'] = "(5,+inf)"
    d1 = pd.concat([d1_x1, d1_x2, d1_x3, d1_x4, d1_x5])
    x3_d, x3_iv, x3_woe = woe_value(d1)
    x3_cut = [float('-inf'), 0, 1, 3, 5, float('+inf')]

    d1 = pd.DataFrame({"X": train_X['NumberOfOpenCreditLinesAndLoans'], "Y": train_y})
    d1['Bucket'] = d1['X']
    d1_x1 = d1.loc[(d1['Bucket'] <= 1)]
    d1_x1.loc[:, 'Bucket'] = "(-inf,1]"
    d1_x2 = d1.loc[(d1['Bucket'] > 1) & (d1['Bucket'] <= 2)]
    d1_x2.loc[:, 'Bucket'] = "(1,2]"
    d1_x3 = d1.loc[(d1['Bucket'] > 2) & (d1['Bucket'] <= 3)]
    d1_x3.loc[:, 'Bucket'] = "(2,3]"
    d1_x4 = d1.loc[(d1['Bucket'] > 3) & (d1['Bucket'] <= 5)]
    d1_x4.loc[:, 'Bucket'] = "(3,5]"
    d1_x5 = d1.loc[(d1['Bucket'] > 5)]
    d1_x5.loc[:, 'Bucket'] = "(5,+inf)"
    d1 = pd.concat([d1_x1, d1_x2, d1_x3, d1_x4, d1_x5])
    x6_d, x6_iv, x6_woe = woe_value(d1)
    x6_cut = [float('-inf'), 1, 2, 3, 5, float('+inf')]

    d1 = pd.DataFrame({"X": train_X['NumberOfTimes90DaysLate'], "Y": train_y})
    d1['Bucket'] = d1['X']
    d1_x1 = d1.loc[(d1['Bucket'] <= 0)]
    d1_x1.loc[:, 'Bucket'] = "(-inf,0]"
    d1_x2 = d1.loc[(d1['Bucket'] > 0) & (d1['Bucket'] <= 1)]
    d1_x2.loc[:, 'Bucket'] = "(0,1]"
    d1_x3 = d1.loc[(d1['Bucket'] > 1) & (d1['Bucket'] <= 3)]
    d1_x3.loc[:, 'Bucket'] = "(1,3]"
    d1_x4 = d1.loc[(d1['Bucket'] > 3) & (d1['Bucket'] <= 5)]
    d1_x4.loc[:, 'Bucket'] = "(3,5]"
    d1_x5 = d1.loc[(d1['Bucket'] > 5)]
    d1_x5.loc[:, 'Bucket'] = "(5,+inf)"
    d1 = pd.concat([d1_x1, d1_x2, d1_x3, d1_x4, d1_x5])
    x7_d, x7_iv, x7_woe = woe_value(d1)
    x7_cut = [float('-inf'), 0, 1, 3, 5, float('+inf')]

    d1 = pd.DataFrame({"X": train_X['NumberRealEstateLoansOrLines'], "Y": train_y})
    d1['Bucket'] = d1['X']
    d1_x1 = d1.loc[(d1['Bucket'] <= 0)]
    d1_x1.loc[:, 'Bucket'] = "(-inf,0]"
    d1_x2 = d1.loc[(d1['Bucket'] > 0) & (d1['Bucket'] <= 1)]
    d1_x2.loc[:, 'Bucket'] = "(0,1]"
    d1_x3 = d1.loc[(d1['Bucket'] > 1) & (d1['Bucket'] <= 2)]
    d1_x3.loc[:, 'Bucket'] = "(1,2]"
    d1_x4 = d1.loc[(d1['Bucket'] > 2) & (d1['Bucket'] <= 3)]
    d1_x4.loc[:, 'Bucket'] = "(2,3]"
    d1_x5 = d1.loc[(d1['Bucket'] > 3)]
    d1_x5.loc[:, 'Bucket'] = "(3,+inf)"
    d1 = pd.concat([d1_x1, d1_x2, d1_x3, d1_x4, d1_x5])
    x8_d, x8_iv, x8_woe = woe_value(d1)
    x8_cut = [float('-inf'), 0, 1, 2, 3, float('+inf')]

    d1 = pd.DataFrame({"X": train_X['NumberRealEstateLoansOrLines'], "Y": train_y})
    d1['Bucket'] = d1['X']
    d1_x1 = d1.loc[(d1['Bucket'] <= 0)]
    d1_x1.loc[:, 'Bucket'] = "(-inf,0]"
    d1_x2 = d1.loc[(d1['Bucket'] > 0) & (d1['Bucket'] <= 1)]
    d1_x2.loc[:, 'Bucket'] = "(0,1]"
    d1_x3 = d1.loc[(d1['Bucket'] > 1) & (d1['Bucket'] <= 3)]
    d1_x3.loc[:, 'Bucket'] = "(1,3]"
    d1_x4 = d1.loc[(d1['Bucket'] > 3)]
    d1_x4.loc[:, 'Bucket'] = "(3,+inf]"
    d1 = pd.concat([d1_x1, d1_x2, d1_x3, d1_x4])
    x9_d, x9_iv, x9_woe = woe_value(d1)
    x9_cut = [float('-inf'), 0, 1, 3, float('+inf')]

    d1 = pd.DataFrame({"X": train_X['NumberOfDependents'], "Y": train_y})
    d1['Bucket'] = d1['X']
    d1_x1 = d1.loc[(d1['Bucket'] <= 0)]
    d1_x1.loc[:, 'Bucket'] = "(-inf,0]"
    d1_x2 = d1.loc[(d1['Bucket'] > 0) & (d1['Bucket'] <= 1)]
    d1_x2.loc[:, 'Bucket'] = "(0,1]"
    d1_x3 = d1.loc[(d1['Bucket'] > 1) & (d1['Bucket'] <= 2)]
    d1_x3.loc[:, 'Bucket'] = "(1,2]"
    d1_x4 = d1.loc[(d1['Bucket'] > 2) & (d1['Bucket'] <= 3)]
    d1_x4.loc[:, 'Bucket'] = "(2,3]"
    d1_x5 = d1.loc[(d1['Bucket'] > 3) & (d1['Bucket'] <= 5)]
    d1_x5.loc[:, 'Bucket'] = "(3,5]"
    d1_x6 = d1.loc[(d1['Bucket'] > 5)]
    d1_x6.loc[:, 'Bucket'] = "(5,+inf]"
    d1 = pd.concat([d1_x1, d1_x2, d1_x3, d1_x4, d1_x5, d1_x6])
    x10_d, x10_iv, x10_woe = woe_value(d1)
    x10_cut = [float('-inf'), 0, 1, 2, 3, 5, float('+inf')]

    # corr = train_data.corr()
    # xticks = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
    # yticks = list(corr.index)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 1, 1)
    # sns.heatmap(corr, annot=True, cmap='rainbow', ax=ax1, annot_kws={'size': 5, 'color': 'blue'})
    # ax1.set_xticklabels(xticks, rotation=0, fontsize=10)
    # ax1.set_yticklabels(yticks, rotation=0, fontsize=10)
    # plt.show()

    # informationValue = []
    # informationValue.append(x1_iv)
    # informationValue.append(x2_iv)
    # informationValue.append(x3_iv)
    # informationValue.append(x4_iv)
    # informationValue.append(x5_iv)
    # informationValue.append(x6_iv)
    # informationValue.append(x7_iv)
    # informationValue.append(x8_iv)
    # informationValue.append(x9_iv)
    # informationValue.append(x10_iv)
    # index = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
    # index_num = range(len(index))
    # ax = plt.bar(index_num, informationValue, tick_label=index)
    # plt.show()

    x1_name = 'RevolvingUtilizationOfUnsecuredLines'
    x2_name = 'age'
    x3_name = 'NumberOfTime30-59DaysPastDueNotWorse'
    x7_name = 'NumberOfTimes90DaysLate'
    x9_name = 'NumberOfTime60-89DaysPastDueNotWorse'

    train_X = trans_woe(train_X, x1_name, x1_woe, x1_cut)
    train_X = trans_woe(train_X, x2_name, x2_woe, x2_cut)
    train_X = trans_woe(train_X, x3_name, x3_woe, x3_cut)
    train_X = trans_woe(train_X, x7_name, x7_woe, x7_cut)
    train_X = trans_woe(train_X, x9_name, x9_woe, x9_cut)
    train_X = train_X.iloc[:, -5:]

    mlp = MLPRegressor(layers=[6, 11, 7], trainer='irprop-', scaler='iron', epochs=100)
    abc = AdaBoostRegressor(n_estimators=100, base_estimator=mlp, learning_rate=0.8)
    abc.fit(train_X, train_y)
    test_X = trans_woe(test_X, x1_name, x1_woe, x1_cut)
    test_X = trans_woe(test_X, x2_name, x2_woe, x2_cut)
    test_X = trans_woe(test_X, x3_name, x3_woe, x3_cut)
    test_X = trans_woe(test_X, x7_name, x7_woe, x7_cut)
    test_X = trans_woe(test_X, x9_name, x9_woe, x9_cut)
    test_X = test_X.iloc[:, -5:]

    pred_y = abc.predict(test_X)
    fpr, tpr, threshold = metrics.roc_curve(test_y, pred_y)
    rocauc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % rocauc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()
    p = 10 / np.log(2)
    q = 50 - 10 * np.log(10) / np.log(2)
    x_coe = [2.6084, 0.6327, 0.5151, 0.5520, 0.5747, 0.4074]
    baseScore = round(q + p * x_coe[0], 0)
    x1_score = get_score(x_coe[1], x1_woe, p)
    x2_score = get_score(x_coe[2], x2_woe, p)
    x3_score = get_score(x_coe[3], x3_woe, p)
    x7_score = get_score(x_coe[4], x7_woe, p)
    x9_score = get_score(x_coe[5], x9_woe, p)
    print(x1_score)
    print(x2_score)
    print(x3_score)
    print(x7_score)
    print(x9_score)
