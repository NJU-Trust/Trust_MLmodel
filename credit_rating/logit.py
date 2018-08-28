import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split
import patsy
from hep_ml.nnet import MLPRegressor
from sklearn import metrics

def load_data():
    diabetes = datasets.load_boston()
    X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.20, random_state=0)
    return X_train, X_test, y_train, y_test
def test_LogisticRegression_multiomaial(X_train, X_test, y_train, y_test):
    cls = LogisticRegression(multi_class='ovr',solver='lbfgs')
    cls.fit(X_train, y_train)
    denom = (2.0 * (1.0 + np.cosh(cls.decision_function(X_train))))
    F_ij = np.dot((X_train / denom[:, None]).T, X_train)
    Cramer_Rao = np.linalg.inv(F_ij)
    sigma_estimates = np.array(
        [np.sqrt(Cramer_Rao[i, i]) for i in range(Cramer_Rao.shape[0])])
    z_scores = cls.coef_[0] / sigma_estimates
    p_values = [stats.norm.sf(abs(x)) * 2 for x in z_scores]
    print("p_value:%s"%(p_values))

if __name__=='__main__':
    X_train,X_test,y_train,y_test=load_data()
    # test_LogisticRegression_multiomaial(X_train,X_test,y_train,y_test)
    mlp = MLPRegressor(layers=[11, 7], trainer='irprop-', scaler='iron', epochs=1000)
    abc = AdaBoostRegressor(n_estimators=50, base_estimator=mlp, learning_rate=1)
    abc.fit(X_train,y_train)
    y_pred = abc.predict(X_test)
    print "MSE:", metrics.mean_squared_error(y_test, y_pred)
    # rfr = RandomForestRegressor(n_estimators=1000, criterion='mse', max_features='sqrt', max_depth=None, min_samples_split=2, bootstrap=True, n_jobs=1, random_state=1)
    # rfr.fit(X_train,y_train)
    # y_pred = rfr.predict(X_test)
    # print "MSE:", metrics.mean_squared_error(y_test, y_pred)

