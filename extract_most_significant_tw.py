"""
This python file reads Twitter language features and also mortality categories and extracts 10 most significant
language features that has the highest pearson-r correlation with each mortality category.
"""
import argparse
import warnings
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from regression_utils import read_county_list

warnings.filterwarnings("ignore")

county_list = read_county_list()
year_list = [str(i) for i in list(np.arange(2009, 2010))]

mor_cats = ['a_accident', 'a_alzheimer', 'a_cancer', 'a_cereb', 'a_diabetes',
            'a_heart', 'a_influenza', 'a_nephritis', 'a_respiratory', 'a_suicide', 'a_all']
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--min_freq", type=int, default=10000)
    parser.add_argument("--outtype", type=str, default='norm')
    args, _ = parser.parse_known_args()
    standardize = True

    ################
    # read all cause morality and merge with the controls
    ################
    outtype = args.outtype
    results = []

    ################
    # extract most signficant Tritter language features for each mortality and year
    ################
    data_all = []
    outcome_all = pd.read_csv('features/outControls_normalized.csv')
    for mor in mor_cats:
        print('Category', mor)
        X_tws_sel_all = []
        cnty_sel_all = []
        year_sel_all = []
        y_sel_all = []

        for year in year_list:
            outcome = outcome_all[outcome_all['year'] == int(year)]

            # select counties
            outcome_sel = outcome[outcome['userwordtotal'] >= args.min_freq]
            outcome_sel = outcome_sel[~outcome_sel[mor].isna()]
            cnty_sel = list(outcome_sel['cnty'])

            # get outcome
            y = outcome_sel['{}_{}'.format(mor, outtype)].values
            # create indices
            ind_sel = []
            for i, cnty in enumerate(county_list):
                if int(cnty) in cnty_sel:
                    ind_sel += [i]

            # load twitter features
            X_twt = sparse.load_npz('./features/twitter_by_year/countytopic/topic_{}.npz'.format(year))
            X_twt = X_twt[ind_sel, :]
            X_twt = X_twt.toarray()

            X_twd = sparse.load_npz('./features/twitter_by_year/countydictionary/dictionary_{}.npz'.format(year))
            X_twd = X_twd[ind_sel, :]
            X_twd = X_twd.toarray()

            X_twg = sparse.load_npz('./features/twitter_by_year/county13gram/13gram_{}.npz'.format(year))
            X_twg = X_twg[ind_sel, :]
            X_twg = X_twg.toarray()

            X_tws = [X_twt, X_twd, X_twg]
            X_tws = np.concatenate(X_tws, 1)
            # standardize tweets
            if standardize:
                scaler = StandardScaler()
                X_tws = scaler.fit_transform(X_tws)

            # gather
            X_tws_sel_all.append(X_tws)
            cnty_sel_all.append(cnty_sel)
            y_sel_all.append(y)
            year_sel_all.append([year] * len(y))

        X_tws_merge = np.concatenate(X_tws_sel_all, 0)
        cnty_merge = np.concatenate(cnty_sel_all, 0)
        y_sel_merge = np.concatenate(y_sel_all, 0)
        year_merge = np.concatenate(year_sel_all, 0)

        # remove columns to make the x nonsingular
        sel = VarianceThreshold(threshold=.1)
        X_tws_red = sel.fit_transform(X_tws_merge)

        # select highly correlated cols
        corr = []
        for k in range(X_tws_red.shape[1]):
            corr.append(pearsonr(X_tws_red[:, k], y_sel_merge)[0])

        corr_idx_sorted = np.argsort(corr)[::-1][:10]
        X_tws_sel = X_tws_red[:, corr_idx_sorted]

        tw_cols = ["tw_{}_{}".format(mor, i) for i in range(X_tws_sel.shape[1])]
        data = pd.DataFrame(X_tws_sel, columns=tw_cols)
        data['cnty'] = cnty_merge
        data['year'] = year_merge
        cols = data.columns.tolist()
        cols = cols[-2:] + cols[:-2]
        data = data[cols]
        data_all.append(data)

    data_all = reduce(lambda x, y: pd.merge(x, y, on=['cnty', 'year'], how='outer'), data_all)
    data_all.to_csv("./features/most_significant_twt_features_pr_10.csv", index=None)
