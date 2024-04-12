"""
This file reads the outcomes of the regressions and runs a ttest to
verify if using Twitter features improves the predictions or not.
"""

import numpy as np
import pandas as pd
import glob
import scipy

col_order = ['a_all', 'a_heart', 'a_cancer', 'a_accident', 'a_respiratory', 'a_cereb',
             'a_alzheimer', 'a_diabetes', 'a_influenza', 'a_nephritis', 'a_suicide']
x_label = ['All Causes', 'Heart', 'Cancer', 'Accidents', 'Respiratory', 'Stroke',
           'Alzheimer', 'Diabetes', 'Influenza', 'Nephritis', 'Suicide']

MethodName = {'lin': 'Linear',
              'ridgecv': 'RidgeCV',
              'lassocv': 'LassoCV',
              'rf': 'Random Forest'}
Categories = ['Twitter', 'Traditional Predictors', 'Traditional Predictors and Twitter']
Cols = ['tw', 'controls', 'all_res_ridge']

if __name__ == '__main__':
    files = glob.glob('results/ttest/*_norm_*')
    method = 'ridgecv'
    std = True

    # read data
    data = []
    for f in files:
        data.append(pd.read_csv(f))
    data = pd.concat(data)
    data = data[data['method'] == method]

    data["Variables"] = data["type"].map(dict(zip(Cols, Categories)))
    data = data[~data['Variables'].isna()]
    data = data.sort_values(by=['mor', 'year', 'bt'])

    data = data[data['mor'] == "a_all"]

    data1 = data[data["type"] == "controls"]
    data2 = data[data["type"] == "all_res_ridge"]

    x1 = np.array(data1['pr'])
    x2 = np.array(data2['pr'])

    res = scipy.stats.ttest_ind(x1, x2, equal_var=False, alternative='less')
    print(res)
