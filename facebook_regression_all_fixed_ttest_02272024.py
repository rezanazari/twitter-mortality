"""
The main taining for the regression models.
Using this entrypoint, we can train models for different mortality categories per each year.
We have three settings:
 1) train only based on traditional county-level controls
 2) train using Twitter language feature only
 3) train using both traditional and Twitter language features
For each of these settings, we find the best performing configuration setting, and then use these configurations
in bootstrapping.
"""
import argparse
import glob
import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from regression_utils import comp_pearson_corpca, build_corpca_config, read_county_list

warnings.filterwarnings("ignore")

corpcas = build_corpca_config()

ses = ['n_medinc', 'n_white', 'n_insur', 'n_unemp', 'n_black', 'n_public_income', 'n_divorced_fe', "n_pov", 'n_ssi',
       'n_homevlu', 'n_hsdplm', 'n_lesshs', 'n_colledge', 'n_deeppov', 'n_deeppovkds', 'n_deeppov65p',
       'n_house_phone', "n_snap", "n_mcr_pen"]
lang = ['n_hispanic', 'n_foreign_born', 'n_nonenglish']
behavior = ['n_inactivity', 'n_obesity', 'n_alcohol', 'n_cigar', 'n_crime', 'n_opioid']
policy = ['n_beds', 'n_nurse_home', 'n_hospt', 'n_md', "n_rural_clinic", "n_psych", "n_nsf", "n_outpatient"]
environment = ['n_park', 'n_urban', 'n_rent', "n_pop_mile", 'n_agric', 'n_construct', 'n_teach', 'n_manufact']

control_cols = ses + lang + behavior + policy + environment
controls_dict = {
    'ses': ses,
    'lang': lang,
    'behavior': behavior,
    'policy': policy,
    'environment': environment,
    'controls': control_cols
}

county_list = read_county_list()
methods_list = ['lin', 'ridgecv']

# create result folder
folder_name = 'ttest'
if not os.path.exists('./results/' + folder_name):
    os.makedirs('./results/' + folder_name)
if not os.path.exists('./results/ypreds_' + folder_name):
    os.makedirs('./results/ypreds_' + folder_name)


# type can be "tw" or "all_res_ridge"
def find_best_param(mor, year, type):
    """
    This function finds the best configuration parameters for each regression.
    params: 
        mor: The mortality name
        year: year of the regression
        type: type can be "tw" (when only having the twitter language features) or 
               "all_res_ridge" (when having the two-staged residualized regression.
    """
    files = glob.glob('./results/config_selection/{}_{}_norm_*'.format(mor, year))
    assert (len(files) == 1)
    data = pd.read_csv(files[0])
    data = data[data['type'] == type]
    data = data[data['method'] == 'ridgecv']
    data = data.loc[data['pr'].idxmax()]
    assert (len(data) == 13)
    if np.isnan(data['kbest']):
        return (None, int(data['pca']))
    elif np.isnan(data['pca']):
        return (int(data['kbest']), None)
    else:
        return (int(data['kbest']), int(data['pca']))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--min_freq", type=int, default=10000)
    parser.add_argument("--outtype", type=str, default='norm')
    parser.add_argument("--mor", type=str, default='suicide')
    parser.add_argument("--year", type=int, default=2009)
    parser.add_argument("--bt", type=int, default=1)
    args, _ = parser.parse_known_args()
    start_time = time.time()
    ################
    rnd = np.random.RandomState(47)
    mor = "a_" + args.mor

    outtype = args.outtype

    outcome = pd.read_csv('features/outControls_normalized.csv')

    year = args.year
    # for year in np.arange(2009, 2019):
    results = []

    outcome_y = outcome[outcome['year'] == year]
    # select counties
    outcome_sel = outcome_y[outcome_y['userwordtotal'] >= args.min_freq]
    outcome_sel = outcome_sel[~outcome_sel['{}_{}'.format(mor, outtype)].isna()]
    cnty_sel = list(outcome_sel['cnty'])
    print("{} counties selected in {} in {}".format(len(cnty_sel), mor, year))

    # get outcome
    y = outcome_sel['{}_{}'.format(mor, outtype)].values

    # bootstrap loop
    for bt in range(args.bt):
        rnd_ind = np.arange(y.shape[0])
        rnd.shuffle(rnd_ind)
        y_t = y[rnd_ind]
        cnty_sel_t = list(np.array(cnty_sel)[rnd_ind])

        # other controls
        yres = {}
        ypreds = {}
        for con, vars in controls_dict.items():
            for method in methods_list:
                X_oth = outcome_sel[vars].values
                num_rows = X_oth.shape[0]
                X_oth_t = X_oth[rnd_ind]

                pr, mse, pr_fold, mse_fold, ypred_t, ypred_dict_t = comp_pearson_corpca(X_oth_t, y_t, cnty_sel_t,
                                                                                        method=method,
                                                                                        corparm=[None],
                                                                                        pcaparm=[None])
                res = [mor, year, bt, con, method, None, None, None, pr, mse, pr_fold, mse_fold, num_rows]
                print(res)
                results.append(res)

                if con == 'controls':
                    ypreds[method] = ypred_t
                    yres[method] = y_t - ypred_t
                    X_oth_controls_t = X_oth_t

        # create indices
        ind_sel = []
        for i, cnty in enumerate(county_list):
            if int(cnty) in cnty_sel:
                ind_sel += [i]

        # load twitter features
        X_twt = sparse.load_npz('./features/twitter_by_year/countytopic/topic_{}.npz'.format(year))
        X_twt = X_twt[ind_sel, :]
        X_twt = X_twt.toarray()
        X_twt_t = X_twt[rnd_ind]

        X_twd = sparse.load_npz('./features/twitter_by_year/countydictionary/dictionary_{}.npz'.format(year))
        X_twd = X_twd[ind_sel, :]
        X_twd = X_twd.toarray()
        X_twd_t = X_twd[rnd_ind]

        X_twg = sparse.load_npz('./features/twitter_by_year/county13gram/13gram_{}.npz'.format(year))
        X_twg = X_twg[ind_sel, :]
        X_twg = X_twg.toarray()
        X_twg_t = X_twg[rnd_ind]

        X_tws_t = [X_twt_t, X_twd_t, X_twg_t]

        for std in [True]:

            # standardize
            X_tws_std_t = []
            if std:
                for xtw in X_tws_t:
                    scaler = StandardScaler()
                    xtw = scaler.fit_transform(xtw)
                    X_tws_std_t.append(xtw)
            else:
                X_tws_std_t = X_tws_t

            ###############
            # Regular regression
            ## dimentionality reduction
            corpcas = [find_best_param(mor, year, "tw")]
            for corpca in corpcas:
                cor_parm = corpca[0]
                pca_parm = corpca[1]

                # create Xs for twitter and all variables
                method = 'ridgecv'

                pr, mse, pr_fold, mse_fold, _, ypred_dict = comp_pearson_corpca(X_tws_std_t, y_t, cnty_sel_t,
                                                                                method=method,
                                                                                corparm=[cor_parm] * 3,
                                                                                pcaparm=[pca_parm] * 3)
                res = [mor, year, bt, 'tw', method, std, cor_parm, pca_parm, pr, mse, pr_fold, mse_fold, num_rows]
                print(res)
                results.append(res)

            #################
            # residual regression with first step coming from ridgecv
            corpcas = [find_best_param(mor, year, "all_res_ridge")]
            for corpca in corpcas:
                cor_parm = corpca[0]
                pca_parm = corpca[1]

                method = 'ridgecv'
                _, _, _, _, yres_pred, _ = comp_pearson_corpca(X_tws_std_t, yres['ridgecv'], cnty_sel_t,
                                                               method=method,
                                                               corparm=[cor_parm] * 3,
                                                               pcaparm=[pca_parm] * 3)

                ypred_t = ypreds['ridgecv'] + yres_pred
                pr = pearsonr(ypred_t, y_t)[0]
                mse = mean_squared_error(y_t, ypred_t)

                res = [mor, year, bt, 'all_res_ridge', method, std, cor_parm, pca_parm, pr, mse, None, None,
                       num_rows]
                print(res)
                results.append(res)

    ##########
    # save
    result_df = pd.DataFrame(results,
                             columns=['mor', 'year', 'bt', 'type', 'method', 'standardize', 'kbest',
                                      'pca', 'pr', 'mse', 'pr_fold', 'mse_fold', 'rows'],
                             index=None)
    result_df.to_csv('./results/{}/{}_{}_{}_{}.csv'.format(folder_name, mor, year, outtype, args.min_freq), index=False)
