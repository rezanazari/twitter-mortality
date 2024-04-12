"""
This python file merges the age adjusted mortality rates with traditional controls. We apply some transformations to
the traditional controls to make them follow a close-to-normal distribution or their ranges scaled to small numbers.
"""

from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from regression_utils import read_county_list

mor_cats = ["all", "accident", "alzheimer", "cancer", "cereb", "diabetes",
            "heart", "influenza", "nephritis", "respiratory", "suicide"]
year_list = np.arange(2009, 2019)
county_list = read_county_list()
county_list = [int(c) for c in county_list]
if __name__ == "__main__":
    """
    1) This file prepares normalized controls and outcomes. 
    """

    # load controls
    dfWordCount = pd.read_csv('./data/userwordtotal.csv', index_col=0)
    controls = pd.read_csv("data/traditional_controls.csv")
    controls = controls.rename(columns={'county': 'cnty'})
    # exclude records that are not in countylist
    controls = controls[controls.cnty.isin(county_list)]
    del controls['good_air']
    del controls['air']

    # merge userwordtotal
    controls = controls.merge(dfWordCount, on=['cnty', 'year'])

    # load crime and merge with the original controls
    crime = pd.read_csv(
        "./data/crime_data_w_population_and_crime_rate.csv")
    crime = crime.rename(columns={"fips": "cnty"})
    controls = controls.merge(crime, on="cnty", how="outer")

    # drop county-years with missing 'pop'
    controls = controls[~controls['pop'].isna()]

    # drop controls if no opioid
    controls['opioid_miss'] = controls['opioid'].isna().astype(int)
    controls['opioid'] = controls['opioid'].fillna(0)

    # drop if less than 1000 userwordtotal
    controls = controls[controls['userwordtotal'] >= 1000].reset_index(drop=True)

    # print missing values from columns
    controls['pop_mile']
    a = controls.isna().sum()
    for i, j in a.iteritems():
        print(i, j)

    # drop all data with missing value (4 rows in cnty 51515)
    controls = controls[~controls.isnull().any(axis=1)]

    # log transform columns
    controls['rent'] = np.log(1 + controls['rent'])
    controls['medinc'] = np.log(1 + controls['medinc'])
    controls['homevlu'] = np.log(1 + controls['homevlu'])
    controls['white'] = np.log(1 + (100 - controls['white']))
    controls['black'] = np.log(1 + controls['black'])
    controls['hispanic'] = np.log(1 + controls['hispanic'])
    controls['foreign_born'] = np.log(1 + controls['foreign_born'])
    controls['nonenglish'] = np.log(1 + controls['nonenglish'])
    controls['beds'] = np.log(1 + controls['beds'] / controls['pop'] * 1000)
    controls['nurse_home'] = np.log(1 + controls['nurse_home'] / controls['pop'] * 10000)
    controls['hospt'] = np.log(1 + controls['hospt'] / controls['pop'] * 10000)
    controls['rehab'] = np.log(1 + controls['rehab'] / controls['pop'] * 10000)
    controls['fqhc'] = np.log(1 + controls['fqhc'] / controls['pop'] * 10000)
    controls['rural_clinic'] = np.log(1 + controls['rural_clinic'] / controls['pop'] * 10000)
    controls['md'] = np.log(1 + controls['md'] / controls['pop'] * 10000)
    controls['park'] = np.log(1 + controls['park'] / controls['pop'] * 10000)
    controls['opioid'] = np.log(1 + controls['opioid'])
    controls['psych'] = np.log(1 + (controls['psych_lt'] + controls['psych_short']) * 10)
    controls['snap'] = np.log(1 + controls['snap'] * 1000)
    controls['outpatient'] = np.log(1 + controls['outpatient'] * 10)
    controls['nsf'] = np.log(1 + controls['nsf'] * 10)
    controls['pop_mile'] = np.log(1 + controls['pop_mile'] * 10)
    controls['ssi'] = np.log(1 + controls['ssi'] / controls['pop'] * 10000)
    controls['hsdplm'] = np.log(1 + controls['hsdplm'] / controls['pop'] * 10000)
    controls['lesshs'] = np.log(1 + controls['lesshs'] / controls['pop'] * 10000)
    controls['house_phone'] = np.log(1 + (100 - controls['house_phone']))

    # normalize columns
    norm_cols = ["white", "black", "hispanic", "medinc", "unemp", "insur", "foreign_born", "public_income",
                 "divorced_fe", "rent", "nonenglish", "beds", "nurse_home", "hospt", "fqhc",
                 "rural_clinic", "md", "opioid", "inactivity", "obesity", "park", "urban", "marijuana", "cocaine",
                 "alcohol", "cigar", "crime", "psych", "snap", "mcr_pen", "outpatient", "nsf",
                 "pop_mile", "pov", 'ssi', 'homevlu', 'hsdplm', 'lesshs',
                 'colledge', 'deeppov', 'deeppovkds', 'deeppov65p', 'house_phone', 'agric', 'construct', 'teach',
                 'manufact']


    # normalize columns by year
    controls_lst = []
    for year in year_list:
        controls_y = controls[controls['year'] == year].copy()
        for k in norm_cols:
            controls_y["n_" + k] = (controls_y[k] - np.mean(controls_y[k])) / np.std(controls_y[k])
            controls_y["n_" + k] = controls_y.groupby('cnty')["n_" + k].apply(
                lambda group: group.interpolate(method="index"))

        controls_lst.append(controls_y)

    controls = pd.concat(controls_lst)
    controls = controls.sort_values(by=['cnty', 'year'], ascending=True)

    for n in norm_cols[0:10]:
        plt.clf()
        plt.hist(controls["n_" + n])
        plt.title("n_" + n)
        plt.show()


    ##########################################################
    # read age adjusted outcomes
    ##########################################################
    dfList = []
    for mor in mor_cats:
        f_pd_ys = []
        for year in year_list:
            f = "./data/age_adjusted_mortality/{}/{}_{}.txt".format(
                mor, mor, year)

            f_pd = pd.read_csv(f, delimiter="\t")
            f_pd = f_pd[~f_pd["Notes"].notna()]
            f_pd = f_pd[~(f_pd["Age Adjusted Rate"] == "Unreliable")]
            f_pd = f_pd[~(f_pd["Age Adjusted Rate"] == "Missing")]

            f_pd["a_" + mor] = f_pd["Age Adjusted Rate"].astype(float)
            f_pd["year"] = int(year)
            f_pd["cnty"] = f_pd["County Code"].astype(int)
            f_pd = f_pd[["cnty", "year", "a_" + mor]]
            f_pd_ys.append(f_pd)
        f_pd_ys = pd.concat(f_pd_ys)
        dfList.append(f_pd_ys)
    df_adj = reduce(lambda x, y: pd.merge(x, y, on=["cnty", "year"], how="outer"), dfList)

    # read not adjusted mortality
    dfList = []
    for mor in mor_cats:
        f_pd_ys = []
        for year in year_list:
            f = "./data/mortality_data/{}/{}_{}.csv".format(mor, mor, year)

            f_pd = pd.read_csv(f, delimiter=",")
            f_pd = f_pd[~(f_pd["Crude Rate"] == "Unreliable")]
            f_pd = f_pd[~(f_pd["Crude Rate"] == "Missing")]

            f_pd["r_" + mor] = f_pd["Crude Rate"].astype(float)
            f_pd["year"] = int(year)
            f_pd["cnty"] = f_pd["fips"].astype(int)
            f_pd = f_pd[["cnty", "year", "r_" + mor]]
            f_pd_ys.append(f_pd)
        f_pd_ys = pd.concat(f_pd_ys)
        dfList.append(f_pd_ys)
    df_notadj = reduce(lambda x, y: pd.merge(x, y, on=["cnty", "year"], how="outer"), dfList)

    outcomes = reduce(lambda x, y: pd.merge(x, y, on=["cnty", "year"], how="outer"), [df_notadj, df_adj])
    outcomes = outcomes[~outcomes['r_all'].isna()]
    outcomes = outcomes.sort_values(by=['cnty', 'year'], ascending=True)

    # merge with outcome variables
    data = pd.merge(outcomes, controls, on=['cnty', 'year'], how='inner')

    # generate norm of output variables
    for mor in mor_cats:
        data["r_" + mor + "_norm"] = (data["r_" + mor] - np.mean(data["r_" + mor])) / \
                                     np.std(data["r_" + mor])
        data["a_" + mor + "_norm"] = (data["a_" + mor] - np.mean(data["a_" + mor])) / \
                                     np.std(data["a_" + mor])

    data.to_csv('./features/outControls_normalized.csv', index=False)
