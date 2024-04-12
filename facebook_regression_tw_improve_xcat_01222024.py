"""
This experiment evaluates how Twitter language features boost the prediction power of each subcategory of traditional
controls.
"""
import argparse
import os
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from regression_utils import comp_pearson_corpca, build_corpca_config_small, read_county_list
warnings.filterwarnings("ignore")

corpcas = build_corpca_config_small()

ses = ['n_medinc', 'n_white', 'n_insur', 'n_unemp', 'n_black', 'n_public_income', 'n_divorced_fe', "n_pov", 'n_ssi',
       'n_homevlu', 'n_hsdplm', 'n_lesshs', 'n_colledge', 'n_deeppov', 'n_deeppovkds', 'n_deeppov65p',
       'n_house_phone', "n_snap", "n_mcr_pen"]
lang = ['n_hispanic', 'n_foreign_born', 'n_nonenglish']
behavior = ['n_inactivity', 'n_obesity', 'n_alcohol', 'n_cigar', 'n_crime', 'n_opioid']
policy = ['n_beds', 'n_nurse_home', 'n_hospt', 'n_md', "n_rural_clinic", "n_psych", "n_nsf", "n_outpatient"]
environment = ['n_park', 'n_urban', 'n_rent', "n_pop_mile", 'n_agric', 'n_construct', 'n_teach', 'n_manufact']

control_cols = ses + lang + behavior + policy + environment
controls_dict = {'ses': ses,
                 'lang': lang,
                 'behavior': behavior,
                 'policy': policy,
                 'environment': environment,
                 'controls': control_cols
                 }

county_list = read_county_list()
year_list = np.arange(2009, 2019)
methods_list = ['lin', 'ridgecv']

# create result folder
folder_name = 'improve_subx_2024'
if not os.path.exists('./results/' + folder_name):
    os.makedirs('./results/' + folder_name)
if not os.path.exists('./results/ypreds_' + folder_name):
    os.makedirs('./results/ypreds_' + folder_name)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--min_freq", type=int, default=10000)
    parser.add_argument("--outtype", type=str, default='norm')
    parser.add_argument("--mor", type=str, default='heart')
    args, _ = parser.parse_known_args()

    ################
    mor = "a_" + args.mor
    outtype = args.outtype
    results = []

    outcome = pd.read_csv('features/outControls_normalized.csv')

    for year in year_list:
        outcome_y = outcome[outcome['year'] == year]
        # select counties
        outcome_sel = outcome_y[outcome_y['userwordtotal'] >= args.min_freq]
        outcome_sel = outcome_sel[~outcome_sel['{}_{}'.format(mor, outtype)].isna()]
        cnty_sel = list(outcome_sel['cnty'])
        print("{} counties selected in {} in {}".format(len(cnty_sel), mor, year))

        # get outcome
        y = outcome_sel['{}_{}'.format(mor, outtype)].values

        # other controls
        yres = {}
        ypreds = {}
        for con, vars in controls_dict.items():
            yres[con] = {}
            ypreds[con] = {}
            for method in methods_list:
                X_oth = outcome_sel[vars].values
                num_rows = X_oth.shape[0]
                pr, mse, pr_fold, mse_fold, ypred, ypred_dict = comp_pearson_corpca(X_oth, y, cnty_sel, method=method,
                                                                                    corparm=[None],
                                                                                    pcaparm=[None])
                res = [mor, year, con, method, None, None, None, pr, mse, pr_fold, mse_fold, num_rows]
                print(res)
                results.append(res)

                ypreds[con][method] = ypred
                yres[con][method] = y - ypred

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

        for std in [True]:

            # standardize
            X_tws_std = []
            if std:
                for xtw in X_tws:
                    scaler = StandardScaler()
                    xtw = scaler.fit_transform(xtw)
                    X_tws_std.append(xtw)
            else:
                X_tws_std = X_tws

            ###############
            # Regular regression
            ## dimentionality reduction
            for corpca in corpcas:
                cor_parm = corpca[0]
                pca_parm = corpca[1]

                # create Xs for twitter and all variables
                for method in methods_list:
                    pr, mse, pr_fold, mse_fold, _, ypred_dict = comp_pearson_corpca(X_tws_std, y, cnty_sel,
                                                                                    method=method,
                                                                                    corparm=[cor_parm] * 3,
                                                                                    pcaparm=[pca_parm] * 3)
                    res = [mor, year, 'tw', method, std, cor_parm, pca_parm, pr, mse, pr_fold, mse_fold, num_rows]
                    print(res)
                    results.append(res)

                # for con, vars in controls_dict.items():
                #     X_oth = outcome_sel[vars].values
                #     for method in methods_list:
                #         pr, mse, pr_fold, mse_fold, _, ypred_dict = comp_pearson_corpca([X_oth] + X_tws_std,
                #                                                                         y,
                #                                                                         cnty_sel,
                #                                                                         method=method,
                #                                                                         corparm=[None] + [cor_parm] * 3,
                #                                                                         pcaparm=[None] + [pca_parm] * 3)
                #
                #         res = [mor, year, 'all_'+con, method, std, cor_parm, pca_parm, pr, mse, pr_fold, mse_fold,
                #                num_rows]
                #         print(res)
                #         results.append(res)

            #################

            # residual regression
            for corpca in corpcas:

                cor_parm = corpca[0]
                pca_parm = corpca[1]

                for con, vars in controls_dict.items():
                    for method in methods_list:
                        _, _, _, _, yres_pred, _ = comp_pearson_corpca(X_tws_std, yres[con]['lin'], cnty_sel,
                                                                       method=method,
                                                                       corparm=[cor_parm] * 3,
                                                                       pcaparm=[pca_parm] * 3)

                        ypred = ypreds[con]['lin'] + yres_pred
                        pr = pearsonr(ypred, y)[0]
                        mse = mean_squared_error(y, ypred)

                        res = [mor, year, 'allreslin_'+con, method, std, cor_parm, pca_parm, pr, mse, None, None,
                               num_rows]
                        print(res)
                        results.append(res)

            # residual regression
            for corpca in corpcas:

                cor_parm = corpca[0]
                pca_parm = corpca[1]

                for con, vars in controls_dict.items():
                    for method in methods_list:
                        _, _, _, _, yres_pred, _ = comp_pearson_corpca(X_tws_std, yres[con]['ridgecv'], cnty_sel,
                                                                       method=method,
                                                                       corparm=[cor_parm] * 3,
                                                                       pcaparm=[pca_parm] * 3)

                        ypred = ypreds[con]['ridgecv'] + yres_pred
                        pr = pearsonr(ypred, y)[0]
                        mse = mean_squared_error(y, ypred)

                        res = [mor, year, 'allresridge_'+con, method, std, cor_parm, pca_parm, pr, mse, None, None,
                               num_rows]
                        print(res)
                        results.append(res)

        ##########
        # save
        result_df = pd.DataFrame(results,
                                 columns=['mor', 'year', 'type', 'method', 'standardize', 'kbest',
                                          'pca', 'pr', 'mse', 'pr_fold', 'mse_fold', 'rows'],
                                 index=None)
        result_df.to_csv('./results/{}/{}_{}_{}.csv'.format(folder_name, mor, outtype, args.min_freq), index=False)
