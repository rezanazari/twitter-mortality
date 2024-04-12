"""
Here, we train the regressions based on Twitter language model on mor1 and then test it in a different mortality mor2.
This experiment illustrates how the Twitter language models can generalize to other mortality categories.
"""
import argparse
import os
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.linear_model import RidgeCV, LinearRegression, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from regression_utils import build_corpca_config, read_county_list

warnings.filterwarnings("ignore")

corpcas = build_corpca_config()

mor_cats = ['a_all', 'a_accident', 'a_alzheimer', 'a_cancer', 'a_cereb', 'a_diabetes',
            'a_heart', 'a_influenza', 'a_nephritis', 'a_respiratory', 'a_suicide']

county_list = read_county_list()
year_list = [str(i) for i in list(np.arange(2009, 2019))]
methods_list = ['ridgecv']

# create result folder
folder_name = 'transfer_kfold'
if not os.path.exists('./results_02272024/' + folder_name):
    os.makedirs('./results_02272024/' + folder_name)


def comp_pearson_corpca_transfer(X1, y1, county1, X2, y2, county2, method='lin', corparm=[], pcaparm=[]):
    # method can be 'lin', 'rf': random forest
    kf = KFold(n_splits=10, random_state=47, shuffle=True)

    predictions = []
    result = {'Pearsonr': [], 'Pearsonr_fold': [], 'MSE': [], 'MSE_fold': []}

    # if X1 is not a list, then convert it to a list
    if type(X1) is not list:
        XAll1 = [X1]
    else:
        XAll1 = X1

    # if X2 is not a list, then convert it to a list
    if type(X2) is not list:
        XAll2 = [X2]
    else:
        XAll2 = X2

    # start kfold
    for train_index, _ in kf.split(XAll1[0]):

        # counties selected in the mor1
        county_train = list(np.array(county1)[train_index])

        # split y1
        y_train = y1[train_index]

        # counties selected in mor1 train that are in mor2
        county_test = list(set(county2).difference(set(county_train)))
        test_index = []
        for i in range(len(county2)):
            if county2[i] in county_test:
                test_index.append(i)
        test_index = np.array(test_index)
        y_test = y2[test_index]


        # split X
        X_trains = []
        X_tests = []
        for i in range(len(XAll1)):
            # feature selection
            X_train = XAll1[i][train_index]
            X_test = XAll2[i][test_index]
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

    # one county can appear in more than one test dataset, so we average them
    df = pd.DataFrame(predictions)
    predictions_dict = dict(df.mean())

    predictions_lst = [predictions_dict[key] for key in sorted(predictions_dict.keys(), reverse=False)]
    predictions_np = np.array(predictions_lst)
    predictions_dict = dict(zip(county2, predictions_lst))

    pr = pearsonr(np.array(predictions_lst), y2)[0]
    mse = mean_squared_error(y2, np.array(predictions_lst))

    return pr, mse


def standard_scalar_tw(X, enabled=True):
    if type(X) is not list:
        raise ("X should be a list")
    # standardize
    X_std = []
    if enabled:
        for x in X:
            scaler = StandardScaler()
            x = scaler.fit_transform(x)
            X_std.append(x)
    else:
        X_std = X

    return X_std


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--min_freq", type=int, default=10000)
    parser.add_argument("--outtype", type=str, default='norm')
    parser.add_argument("--mor1", type=str, default='all')
    parser.add_argument("--mor2", type=str, default='heart')
    args, _ = parser.parse_known_args()

    ################
    mor1 = "a_" + args.mor1
    mor2 = "a_" + args.mor2
    outtype = args.outtype
    results = []

    outcome = pd.read_csv('features/outControls_normalized.csv')

    for year in np.arange(2009, 2019):
        outcome_y = outcome[outcome['year'] == year]
        # select counties
        outcome_sel = outcome_y[outcome_y['userwordtotal'] >= args.min_freq]

        outcome_sel1 = outcome_sel[~outcome_sel['{}_{}'.format(mor1, outtype)].isna()]
        cnty_sel1 = list(outcome_sel1['cnty'])
        print("{} train counties selected in {} in {}".format(len(cnty_sel1), mor1, year))

        outcome_sel2 = outcome_sel[~outcome_sel['{}_{}'.format(mor2, outtype)].isna()]
        cnty_sel2 = list(outcome_sel2['cnty'])
        print("{} test counties selected in {} in {}".format(len(cnty_sel2), mor2, year))

        # get outcome
        y1 = outcome_sel1['{}_{}'.format(mor1, outtype)].values
        num_rows1 = len(y1)
        y2 = outcome_sel2['{}_{}'.format(mor2, outtype)].values
        num_rows2 = len(y2)

        # create indices
        ind_sel1 = []
        for i, cnty in enumerate(county_list):
            if int(cnty) in cnty_sel1:
                ind_sel1 += [i]
        ind_sel2 = []
        for i, cnty in enumerate(county_list):
            if int(cnty) in cnty_sel2:
                ind_sel2 += [i]

        # load twitter features
        X_twt = sparse.load_npz('./features/twitter_by_year/countytopic/topic_{}.npz'.format(year)).toarray()
        X_twd = sparse.load_npz('./features/twitter_by_year/countydictionary/dictionary_{}.npz'.format(year)).toarray()
        X_twg = sparse.load_npz('./features/twitter_by_year/county13gram/13gram_{}.npz'.format(year)).toarray()
        X_tws = [X_twt, X_twd, X_twg]
        X_tws_std = standard_scalar_tw(X_tws)

        X1 = []
        X2 = []
        for i in range(len(X_tws_std)):
            X1.append(X_tws_std[i][ind_sel1, :])
            X2.append(X_tws_std[i][ind_sel2, :])

        ###############
        # Regular regression
        ## dimentionality reduction
        for corpca in corpcas:
            cor_parm = corpca[0]
            pca_parm = corpca[1]

            # if X is not a list, then convert it to a list
            if type(X1) is not list:
                X1 = [X1]
            if type(X2) is not list:
                X2 = [X2]

            # create Xs for twitter and all variables
            for method in methods_list:
                pr, mse = comp_pearson_corpca_transfer(X1, y1, cnty_sel1, X2, y2,cnty_sel2,
                                                           method=method,
                                                           corparm=[cor_parm] * 3,
                                                           pcaparm=[pca_parm] * 3)
                res = [mor1, mor2, year, 'tw', method, cor_parm, pca_parm, pr, mse, num_rows1, num_rows2]
                print(res)
                results.append(res)

        ##########
        # save
        result_df = pd.DataFrame(results,
                                 columns=['mor1', 'mor2', 'year', 'type', 'method', 'kbest',
                                          'pca', 'pr', 'mse', 'rows1', 'rows2'],
                                 index=None)
        result_df.to_csv('./results_02272024/{}/{}_{}_{}_{}.csv'.format(folder_name, mor1, mor2, outtype, args.min_freq),
                         index=False)
