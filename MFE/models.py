from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn import linear_model

def svr_model(X, y, X_test):
    svr = SVR(kernel='linear')
    svr.fit(X, y)
    return svr.predict(X_test)

def brr_model(X, y, X_test):
    brr = linear_model.BayesianRidge()
    brr.fit(X.toarray(), y)
    return brr.predict(X_test)

def xgb_model(X, y, X_test):
    xgb = XGBRegressor(n_estimators=800, seed=42, learning_rate = 0.015, max_depth=5, eval_metric='rmse')
    xgb.fit(X, y)
    return xgb.predict(X_test)