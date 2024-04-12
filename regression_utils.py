import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.linear_model import RidgeCV, LinearRegression, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor


def comp_pearson_corpca(X, y, cnty_sel, method='lin', corparm=[], pcaparm=[]):
    # method can be 'lin', 'rf': random forest
    kf = KFold(n_splits=10, random_state=47, shuffle=True)

    predictions = []
    result = {'Pearsonr': [], 'Pearsonr_fold': [], 'MSE': [], 'MSE_fold': []}

    # if X is not a list, then convert it to a list
    if type(X) is not list:
        XAll = [X]
    else:
        XAll = X

    # start kfold
    for train_index, test_index in kf.split(XAll[0]):

        # split y
        y_train, y_test = y[train_index], y[test_index]

        # split X
        X_trains = []
        X_tests = []
        for i, X in enumerate(XAll):

            # feature selection
            X_train = X[train_index]
            X_test = X[test_index]
            if corparm[i] is not None:
                fwe = SelectKBest(f_regression, k=corparm[i])
                X_train = fwe.fit_transform(X_train, y_train)
                X_test = fwe.transform(X_test)
            if pcaparm[i] is not None:
                pca = PCA(n_components=pcaparm[i])
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

            X_trains.append(X_train)
            X_tests.append(X_test)

        if len(X_trains) > 1:

            X_train = np.concatenate(X_trains, 1)
            X_test = np.concatenate(X_tests, 1)
        else:
            X_train = X_trains[0]
            X_test = X_tests[0]

        if method == 'lin':
            model = LinearRegression()
        elif method == 'rf':
            model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=0)
        elif method == 'ridgecv':
            model = RidgeCV(alphas=np.array([1.00000e+03, 1.00000e-01, 1.00000e+00,
                                             1.00000e+01, 1.00000e+02, 1.00000e+04, 1.00000e+05]))
        elif method == 'lassocv':
            model = LassoCV()
        elif method == 'nn':
            model = MLPRegressor(hidden_layer_sizes=(128, 64))
        else:
            raise ("Not implemented method")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions.append(dict(zip(test_index, y_pred)))

        result['Pearsonr_fold'].append(pearsonr(y_pred, y_test)[0])
        result['MSE_fold'].append(mean_squared_error(y_test , y_pred))

    predictions_dict = {}
    for d in predictions:
        predictions_dict.update(d)

    predictions_lst = [predictions_dict[key] for key in sorted(predictions_dict.keys(), reverse=False)]
    predictions_np = np.array(predictions_lst)
    predictions_dict = dict(zip(cnty_sel, predictions_lst))

    pr = pearsonr(np.array(predictions_lst), y)[0]
    mse = mean_squared_error(y, np.array(predictions_lst))

    return pr, mse, np.mean(result['Pearsonr_fold']), np.mean(result['MSE_fold']), predictions_np, predictions_dict


def build_corpca_config():
    corpcas = []
    for i in range(1,5):
        for j in range(1,5):
            cor = 50 * i
            pca = 10 * j
            if (.6*cor > pca):
                corpcas.append((cor, pca))

    for i in range(1, 5):
        corpcas.append((None, i*10))
        corpcas.append((i*10, None))
    return corpcas

def build_corpca_config_small():
    corpcas = []
    for i in range(2,4):
        for j in range(2,4):
            cor = 50 * i
            pca = 10 * j
            if (.6*cor > pca):
                corpcas.append((cor, pca))

    for i in range(2, 4):
        corpcas.append((None, i*10))
        corpcas.append((i*10, None))
    return corpcas



def read_county_list():
    # read county list  and convert to str
    county_list = pd.read_csv('./data/county_list.csv', header=None)
    county_list = county_list.iloc[:, 0].tolist()
    county_list_str = []
    for c in county_list:
        if c < 10000:
            county_list_str.append('0' + str(c))
        else:
            county_list_str.append(str(c))
    county_list = county_list_str

    return county_list



if __name__ == '__main__':
    corpcas = build_corpca_config()