import pandas as pd
from sklearn.feature_selection import RFE
import copy
import numpy as np
import warnings
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import yaml


def add_binary_where_the_nan_was(table, column):
    """takes the table and next to each columnputs information whether there was a nan"""
    table[f'bool_nan_{column}'] = table[column].isna()
    return table


def get_rid_of_outliers(table, column):
    up, low = np.percentile(table[column], [1, 99])
    y = np.clip(table[column], up, low)
    # pd.Series(y).hist(bins=30)
    table = table.drop(columns=[column])
    table[f'no_outliers_{column}'] = pd.Series(y)
    return table


def WoE_for_categorical_values(table, column, ret_woe=False):
    """
    takes a table, with a categorical value in columns and returns WOE and IV for that variable
    """

    different_values = table[column].unique().shape[0]
    list_of_bads = [0] * different_values  # in each element of a list it contains woe for a corresponding interval
    list_of_goods = [0] * different_values
    for i in range(len(table[str(column)])):  # iterate over every sample
        # print(i)
        for j in range(different_values):  # how many separate values are there to deal with
            if table[column][i] == table[column].unique()[j] and table['target'][i] == 1:  # default is an event here
                list_of_bads[j] += 1
            elif table[column][i] == table[column].unique()[j] and table['target'][i] == 0:
                list_of_goods[j] += 1

    total_bads = table.target.sum()
    total_goods = len(table.target) - total_bads
    distr_goods = []
    distr_bads = []
    WoE = []

    for i in range(len(list_of_goods)):
        distr_goods.append(list_of_goods[i] / total_goods)
        distr_bads.append(list_of_bads[i] / total_bads)

    # check whether there are no groups with 0 counts for good or bad - if there are drop the columns with that variable
    # all together
    flag = False
    if 0 in distr_goods or 0 in distr_bads:
        print("In at least one of the bins there is either no goods or bads distribution. Dropping that variable")
        flag = True

    for i in range(len(list_of_goods)):
        WoE.append(np.log(distr_goods[i] / distr_bads[i]) * 100)

    # Information Value of the whole characteristic
    distr_bads_nans = table['target'][table[column].isna()].sum()/total_bads
    # how many is nan and is not default
    distr_goods_nans = (table['target'][table[column].isna()].shape[0] - \
                       table['target'][table[column].isna()].sum())/total_goods
    WoE_nan = np.log(distr_goods_nans / distr_bads_nans) * 100
    WoE = WoE.insert(0, WoE_nan)  # inserting the value correspinding to NaNs in the first place

    # Information Value of the whole characteristic
    differences = [distr_goods[i] - distr_bads[i] for i in range(len(distr_goods))]
    differences.insert(0, distr_goods_nans-distr_bads_nans)
    IV = np.dot(differences, np.transpose(WoE))

    if ret_woe and not flag:
        return WoE, IV
    elif not ret_woe and not flag:
        return IV
    elif flag:
        return table.drop(columns=[column])
# consider correlation for all continuous data


def drop_columns_with_many_nans(table, threshold=0.2):
    """drops columns that contain over 20% of nan values"""
    for col in table.columns:
        if table[col].isna().sum() >= threshold * table[col].shape[0]:
            table = table.drop(columns=[col])
    return table


def woe_and_iv_continuous_data(table, column, number_of_bins, ret_woe=False):
    """assumes that target is provided in column 'target'
    1 - event, ie default
    0 - no default
    returns bins, woe - tuples, iv - scalar
    """

    bins = pd.qcut(table[str(column)], number_of_bins, retbins=True)[1]
    # bins = pd.cut(table[str(column)], number_of_bins, retbins=True)[1]
    bins[-1] += 1  # to include all points
    list_of_bads = [0] * number_of_bins # in each element of a list it contains woe for a corresponding interval
    list_of_goods = [0] * number_of_bins
    for i in range(len(table[str(column)])):
        for j in range(number_of_bins):
            if bins[j] <= table[column][i] < bins[j+1] and table['target'][i] == 1:  # default is an event here
                list_of_bads[j] += 1
            elif bins[j] <= table[column][i] < bins[j+1] and table['target'][i] == 0:
                list_of_goods[j] += 1


    # WoE = ln(distr_goods / distr_bads) * 100

    total_bads = table.target.sum()  # bad = default
    total_goods = len(table.target) - total_bads
    distr_goods = []
    distr_bads = []
    WoE = []

    for i in range(len(list_of_goods)):
        distr_goods.append(list_of_goods[i] / total_goods)
        distr_bads.append(list_of_bads[i] / total_bads)

    # check whether there are no groups with 0 counts for good or bad

    if 0 in distr_goods or 0 in distr_bads:
        warnings.warn("In at least one of the bins there is either no goods or bads distribution. Check the binning")
        exit()

    for i in range(len(list_of_goods)):
        WoE.append(np.log(distr_goods[i] / distr_bads[i]) * 100)

    # group also nans
    # how many is nan and is default
    distr_bads_nans = table['target'][table[column].isna()].sum()/total_bads
    # how many is nan and is not default
    distr_goods_nans = (table['target'][table[column].isna()].shape[0] - \
                       table['target'][table[column].isna()].sum())/total_goods
    # WoE_nan = np.log(distr_goods_nans / distr_bads_nans) * 100
    # WoE = WoE.insert(0, WoE_nan)  # inserting the value correspinding to NaNs in the first place

    # Information Value of the whole characteristic
    differences = [distr_goods[i] - distr_bads[i] for i in range(len(distr_goods))]
    # differences.insert(0, distr_goods_nans-distr_bads_nans)
    IV = np.dot(differences, np.transpose(WoE))

    if ret_woe:
        return bins, WoE, IV
    else:
        return IV/100


def input_missing_values(table, column, median=True, mode=False):
    if median:
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    elif mode:
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    imp_mean.fit(table[column].values.reshape(-1, 1))
    table[column] = imp_mean.transform(table[column].values.reshape(-1, 1))
    return table


def correlation(dataset, threshold=0.6):
    # deals only with numeric data, float64 and int64, so here the values

    col_corr = set()  # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname]  # deleting the column from the dataset

    return dataset


def exclude_data_few_unique_values(dataset):
    """drops columns that have either 0 distinct values (NANS) or only 1 distinct value"""
    col_to_drop = []

    for column in dataset.columns:
        if dataset[column].nunique() == 1 or dataset[column].nunique() == 0:
            col_to_drop.append(column)

    for column in col_to_drop:
        dataset = dataset.drop(columns=column)

    return dataset


def split_dataset(table):
    """categorical features object,
     numerical - float64
     ordinal - int64, the distinction between categorical and ordinal is to belooked into"""
    table_categorical = table.select_dtypes('object')
    table_numerical = table.select_dtypes('float64')
    table_ordinal = table.select_dtypes('int64')
    return table_numerical, table_categorical, table_ordinal


def bin_dataset(table, column):
    """ bins numerical value"""
    if column is not 'target':
        return pd.qcut(table[column], 4)


def drop_duplicated_ones_and_values_leaking_data_from_the_future(table):
    list_to_drop = ['id', 'member_id', 'url', 'emp_title', 'issue_d', 'funded_amnt', 'funded_amnt_inv',
                    'sub_grade', 'int_rate', 'addr_state', 'out_prncp', 'out_prncp_inv', 'total_pymnt',
                    'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries',
                    'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'zip_code',
                    'earliest_cr_line', 'next_pymnt_d', 'last_credit_pull_d', 'disbursement_method', 'delinq_amnt',
                    'open_rv_24m']

    for column in list_to_drop:
        try:  # as some of these columns might already have been dropped
            table = table.drop(columns=column)
        except:
            KeyError
    return table


def look_at_value_distribution(table, columns):
    """for later stage of analysis, to manually exclude data that has very few examples of some values"""
    print(table[columns].value_counts())


def fill_nans(table, column):  # only after values with too many nans were excluded
    return table[column].fillna("MISSING")


def calculate_woe_iv(dataset, feature):
    lst = []
    for i in range(dataset[feature].nunique()):
        val = list(dataset[feature].unique())[i]
        lst.append({
            'Value': val,
            'All': dataset[dataset[feature] == val].count()[feature],
            'Good': dataset[(dataset[feature] == val) & (dataset['target'] == 0)].count()[feature],
            'Bad': dataset[(dataset[feature] == val) & (dataset['target'] == 1)].count()[feature]
        })

    dset = pd.DataFrame(lst)
    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
    dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
    dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
    iv = dset['IV'].sum()

    dset = dset.sort_values(by='WoE')

    return dset, iv




def non_obvious_numerical_data(table):
    """some of the columns have highly unbalanced distribution of data, qcut would result in errors,
    also returning names of these columns to later pass it as value not to be considered again"""

    use_cut_no_qcut = ['pct_tl_nvr_dlq', 'num_tl_90g_dpd_24m', 'delinq_2yrs', 'open_act_il', 'tot_coll_amt',
                       'open_il_24m', 'total_bal_il', 'open_rv_12m', 'inq_fi', 'total_cu_tl', 'inq_last_12m',
                       'acc_open_past_24mths', 'mort_acc', 'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_bc_tl',
                       'percent_bc_gt_75', 'num_il_tl', 'target']
    for elem in use_cut_no_qcut:
        if elem != 'target':
            table[elem] = pd.cut(table[elem], 4)
    return table[use_cut_no_qcut], use_cut_no_qcut


def columns_not_to_bin(table):

    dont_bin_do_not_do_anything_treat_as_categorical = ['chargeoff_within_12_mths', 'open_il_12m',
                                                        'open_acc_6m', 'pub_rec', 'inq_last_6mths',
                                                        'collections_12_mths_ex_med', 'target']

    return table[dont_bin_do_not_do_anything_treat_as_categorical], dont_bin_do_not_do_anything_treat_as_categorical


def conc_tables(t1, t2, t3):  # concatenates tables
        return pd.concat([t1, t2, t3])


def standardize(table):
    from sklearn.preprocessing import StandardScaler

    for col in table.columns:
        if str(col)[:3] != 'bool':
            scaler = StandardScaler()
            scaler.fit(table[col].to_numpy().reshape(-1, 1))
            table[col] = scaler.transform(table[col].to_numpy().reshape(-1, 1))
    return table



def stack(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LogisticRegression
    from vecstack import stacking
    from sklearn import svm
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, classification_report, f1_score

    models = [LogisticRegression(), svm.SVC(), xgb.XGBClassifier(n_jobs=-1)]

    S_train, S_test = stacking(models, X_train, y_train, X_test, regression=False, verbose=2)

    model = xgb.XGBClassifier(seed=0, n_jobs=-1, learning_rate=0.1,n_estimators=100, max_depth=3)

    # Fit 2-nd level model
    model = model.fit(S_train, y_train)

    # Predict
    y_pred = model.predict(S_test)

    # Final prediction score
    print('Final prediction score: [%.8f]' % f1_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(roc_auc_score(y_test, y_pred))


def upsample_data(X_train, y_train):

    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res


"""for later"""
def bayes_optim(X_train, X_test, y_train, y_test):

    from bayes_opt import BayesianOptimization
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    def bayesian_optimization(X_train, X_test, y_train, y_test, function, parameters):
        n_iterations = 10
        gp_params = {"alpha": 1e-4}
        BO = BayesianOptimization(function, parameters)
        BO.maximize(n_iter=n_iterations, **gp_params)

        return BO.ma


    def rfc_optimization(cv_splits):
        def function(n_estimators, max_depth, min_samples_split):
            return cross_val_score(
                RandomForestClassifier(
                    n_estimators=int(max(n_estimators, 0)),
                    max_depth=int(max(max_depth, 1)),
                    min_samples_split=int(max(min_samples_split, 2)),
                    n_jobs=-1,
                    random_state=42,
                    class_weight="balanced"),
                X=X_train,
                y=y_train,
                cv=cv_splits,
                scoring="f1_macro",
                n_jobs=-1).mean()

        parameters = {"n_estimators": (10, 1000),
                      "max_depth": (1, 150),
                      "min_samples_split": (2, 10)}

        return function, parameters

        # Train model
    def train(X_train, y_train, X_test, y_test, function, parameters):
        cv_splits = 4

        best_solution = bayesian_optimization(X_train, y_train, X_test, y_test, function, parameters)
        params = best_solution["params"]

        model = RandomForestClassifier(
            n_estimators=int(max(params["n_estimators"], 0)),
            max_depth=int(max(params["max_depth"], 1)),
            min_samples_split=int(max(params["min_samples_split"], 2)),
            n_jobs=-1,
            random_state=42,
            class_weight="balanced")

        model.fit(X_train, y_train)

        return model

    function, parameters = rfc_optimization(10)
    print(train(X_train, y_train, X_test, y_test, function, parameters))


def encode_ordinal_category_encoders(table):
    import category_encoders
    for col in table.columns:
        if str(col)[:7] != 'encoded':
            encode = category_encoders.ordinal.OrdinalEncoder()
            encode.fit(table[col])
            table[f'encoded_ordinal{col}'] = encode.transform(table[col])
    return table


def encode_ordinal_sklearn(table, col):
    from sklearn.preprocessing import OrdinalEncoder
    enc = OrdinalEncoder()
    enc.fit(table[col].values.reshape(-1, 1))
    table[f'ordinal_encoded_{col}'] = enc.transform(table[col].values.reshape(-1, 1))
    return table


def encode_categorical_TargetEncoder(table, y):
    import category_encoders
    for col in table.columns:
        if str(col)[:7] != 'encoded':
            encode = category_encoders.target_encoder.TargetEncoder()
            encode.fit(table[col], y)
            table[f'encoded_target{col}'] = encode.transform(table[col])
    return table


def drop_cat_not_encoded(table):
    """drops non encoded categorical columns"""
    for col in table.columns:
        if str(col)[:7] != 'encoded':
            table = table.drop(columns=[col])
    return table


def fit_xgboost(X, y, upsample=True):
    from sklearn.metrics import roc_auc_score, classification_report, f1_score
    import xgboost as xgb

    model = xgb.XGBClassifier(seed=0, n_jobs=-1, learning_rate=0.1,
                              n_estimators=10, max_depth=5)

    # Fit 2-nd level model
    if upsample:
        X_res, y_res = upsample_data(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res)
        model = model.fit(X_train, y_train)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model = model.fit(X_train, y_train)


    # Predict
    y_pred = model.predict(X_test)
    print('f1:\t\n', f1_score(y_test, y_pred))
    print('roc_auc_score:\t\n', roc_auc_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model


def date_feature(date):
    """number of month, day of week, vacation, assuming US vacations"""
    from dateutil import parser
    import holidays
    us_holidays = holidays.UnitedStates()
    datetime_obj = parser.parse(date)
    weekday = datetime_obj.weekday()  # 0 is monday
    month = datetime_obj.month
    is_holiday = datetime_obj in us_holidays or weekday==6 or weekday==5  # includes weekends as holidays
    return weekday, month, is_holiday


def mean_encoding(table, column, target, drop=False):
    """assumes no nans"""
    """assumes target is a column in table"""
    table[f'mean_encoded_{column}'] = table[column].map(table.iloc[table.index].groupby(column)[target].mean())
    if drop:
        table = table.drop(columns=[column])
    return table


def frequency_encode(table, column, drop=False):
    """assumes no nans"""
    encoding = table.groupby(column).size()
    encoding = encoding / table.shape[0]
    table[f'freq_enc_{column}'] = table[column].map(encoding)
    if drop:
        table = table.drop(columns=[column])
    return table


def kfold_mean_encoding(table, column, target, nfolds=5):

    from sklearn.model_selection import KFold

    skf = KFold(nfolds, shuffle=False)

    for tr_ind, val_ind in skf.split(table[column]):

        X_tr, X_val = table[[column, target]].iloc[tr_ind], table[[column, target]].iloc[val_ind]

        table[column].iloc[val_ind] = table[column].iloc[val_ind].map(X_tr.groupby(column)[target].mean())

        table[column] = table[column].fillna(table[target].mean())

    return table


def train_test_split(X, y, test_size=0.2):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def log_transform(table, column, drop=True):
    table[f'log_{column}'] = pd.Series(np.log(1 + table[column]))
    if drop:
        table = table.drop(columns=[column])
    return table


def remove_duplicates(table):
    return table.drop_duplicates()


def compare_data_distributions(table1, table2):
    """This is a two-sided test for the null hypothesis that 2 independent samples are drawn from the same
    continuous distribution. """
    from scipy import stats
    pvalue = stats.ks_2samp(table1, table2)[1]
    if pvalue > .10:
        return True
    else:
        return False