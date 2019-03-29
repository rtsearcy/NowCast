# beach_model.py - Create NowCast models using beach-specific variable datasets
# RTS - 3/26/2018

# TODO - max variables, reorder performance spreadsheet, parameters in one place in inputs,

# For each FIB:
# - remove other FIB type data
# - identify technical feasibility of creating a model (more than X exceedances)
# - with appropriate variables, split dataset into train/test subsets
# - check for multicolinearity (remove least correlated/highest VIF)
# - apply MLR/BLR to training set
# - tune, and if passes performance criteria (sensitiviy/specificity), evaluate on test set
# - idenitfy passing models, and save info in separate dataframe for eval/presentations

import pandas as pd
import numpy as np
import os
import sys
import shutil
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix, roc_curve, r2_score
from sklearn.externals import joblib
from datetime import datetime


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for fil in self.files:
            fil.write(obj)

    def flush(self):
        pass


def check_corr(x, f, thresh):
    # Check if variables have correlations > thresh, and drop the one with least correlation to logFIB
    print('Checking variable correlations: ')
    c = x.corr()  # Pearson correlation coefs.
    to_drop = []

    for ii in c.columns:  # iterate through all variables in correlation matrix except dependant variable
        if ii == 'log' + f:  # skip FIB variable (Depepdnant var)
            continue
        temp = c.loc[ii]
        temp = temp[temp.abs() > thresh]  # .5 removed a lot of variables
        temp = temp.drop(ii, errors='ignore')  # Remove the variable itself
        temp = temp.drop('log' + f, errors='ignore')
        i_corr = c[ii].loc['log' + f]
        if len(temp) > 0:
            for j in temp.index:
                j_corr = c[j].loc['log' + f]
                if ii not in to_drop and abs(i_corr) < abs(j_corr):  # Drop variable if its corr. with logFIB is lower
                    to_drop.append(ii)

    x = x.drop(to_drop, axis=1, errors='ignore')  # Drop variables
    print('  Remaining variables (' + str(len(x.columns) - 1) + ')')
    print(x.columns.values)
    return x


def multicollinearity_check(X, thr):  # Check VIF of model variables, drop if any above 'thr'
    variables = list(X.columns)
    print('Checking multicollinearity of ' + str(len(variables)) + ' variables for VIF:')
    if len(variables) > 1:
        vif_model = LinearRegression()
        v = [1 / (1 - (r2_score(X[ix], vif_model.fit(X[variables].drop(ix, axis=1), X[ix]).
                                predict(X[variables].drop(ix, axis=1))))) for ix in variables]
        maxloc = v.index(max(v))  # Drop max VIF var if above 'thr'
        if max(v) > thr:
            print(' Dropped: ' + X[variables].columns[maxloc] + ' (VIF - ' + str(round(max(v), 3)) + ')')
            variables.pop(maxloc)  # remove variable with maximum VIF
        else:
            print('VIFs for all variables less than ' + str(thr))
        X = X[[i for i in variables]]
        return X
    else:
        return X


def data_splitter(df, method, seed, f, dir, season):  # Partitions data into calibration and validation subsets
    # Chronological Method - X in CX refers to the years previous to current in the validation subset.
    # The calibration subset contains the remaining past data
    if method in ['C1', 'C2', 'C3', 'C4']:
        if season == 'summer':
            # Split
            test_yr_e = str(max(df.index.year))  # End year
            test_yr_s = str(int(test_yr_e) - int(method[1]) + 1)  # Start year

            test_data = df[test_yr_s:test_yr_e].sort_index(ascending=False)
            train_data = df[~df.index.isin(test_data.index)].sort_index(ascending=False)

        elif season == 'winter':
            # Split
            test_yr_e = str(max(df.index.year))  # End year
            test_yr_s = str(int(test_yr_e) - int(method[1]))  # Start year

            temp_test = df[test_yr_s:test_yr_e]
            test_data = temp_test[~((temp_test.index.month.isin([1, 2, 3])) &
                                    (temp_test.index.year.isin([test_yr_s])))].sort_index(ascending=False)
            # Ensure winter seasons (which cross over years) are bundled together
            train_data = df[~df.index.isin(test_data.index)].sort_index(ascending=False)

        y_test = test_data['log' + f]
        X_test = test_data.drop('log' + f, axis=1)
        y_train = train_data['log' + f]
        X_train = train_data.drop('log' + f, axis=1)

        test_name = f + '_' + method + '_validation_dataset.csv'
        train_name = f + '_' + method + '_calibration_dataset.csv'

    # Jackknife - Randomly select a certain percentage for calibration and use the remaining in validation
    elif method in ['JK7525', 'JK7030', 'JK6040']:
        y = df['log' + f]  # Separate into mother dependent and independent datasets
        X = df.drop('log' + f, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float('0.' + method[-2:]),
                                                            random_state=seed)
        # test_size - YY% from JKXXYY in split method

        test_data = y_test.to_frame().merge(X_test, left_index=True, right_index=True).sort_index(ascending=False)
        train_data = y_train.to_frame().merge(X_train, left_index=True, right_index=True).sort_index(ascending=False)
        test_name = f + '_' + method + '_seed_' + str(seed) + '_validation_dataset.csv'
        train_name = f + '_' + method + '_seed_' + str(seed) + '_calibration_dataset.csv'

    # Account for NA samples
    y_test = y_test.dropna()
    X_test = X_test.reindex(y_test.index)
    y_train = y_train.dropna()
    X_train = X_train.reindex(y_train.index)

    # Save test and training datasets to seperate CSV files
    os.makedirs(dir + '\\' + f + '_' + method, exist_ok=True)  # create subfolder
    sub_dir = dir + '\\' + f + '_' + method
    train_data.to_csv(os.path.join(sub_dir, train_name))
    test_data.to_csv(os.path.join(sub_dir, test_name))

    return X_train, X_test, y_train, y_test


def model_fit(X, y, model_type, seed, thresh, c, cv):  # Fits model (model_type - MLR or BLR) to calibration data
    # X - calibration independent data; y - calibration dependant variable;
    # c - model regularization coefficient (smaller - more regularization of variables);\
    # cv - number of cross-validation steps
    if model_type == 'blr':
        y = (y > np.log10(thresh)).astype(int)
        lm = LogisticRegression(random_state=seed, C=c)
        scorer = 'roc_auc'
        #scorer = 'accuracy'
    elif model_type == 'mlr':
        lm = LinearRegression()
        scorer = 'neg_mean_squared_error'
        #scorer = 'r2'

    S = RFECV(lm, cv=cv, scoring=scorer, n_jobs=1).fit(np.array(X), np.array(y))
    # Recursive Feature Elimination - Stepwise variable selection
    # Creates multiple models to test
    # TODO - test n_jobs = 2, -1, -2
    features = list(np.where(S.support_)[0])
    return features, lm


def model_eval(true, predicted, thresh=0.5, tune=0):  # Evaluate Model Performance
    if true.dtype == 'float':
        true = (true > thresh).astype(int)  # Convert to binary
    if predicted.dtype == 'float':
        predicted = (predicted > thresh).astype(int)

    cm = confusion_matrix(true, predicted)  # Lists number of true positives, true negatives,false pos,and false negs.
    sens = cm[1, 1] / (cm[1, 1] + cm[1, 0])  # sensitivity - TP / TP + FN
    spec = cm[0, 0] / (cm[0, 1] + cm[0, 0])  # specificity - TN / TN + FP
    acc = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + cm[0, 1] + cm[1, 0])
    samples = len(true)  # number of samples
    exc = true.sum()  # number of exceedances

    if tune == 0:
        out = {'Sensitivity': round(sens, 3), 'Specificity': round(spec, 3), 'Accuracy': round(acc, 3),
               'Samples': samples, 'Exceedances': exc}
    else:
        out = [round(sens, 3), round(spec, 3)]

    return out


def tune_blr(X_train_sfs, y_train, perf_criteria, cm_perf, blr, ease, fine_tune=1):
    # Optimizes BLR model performance to achieve performance criteria (if possible) by adjusting threshold

    # X_train_sfs - Variable dataset with only model variables
    # y_train - must be in binary format
    # perf_criteria - dictionary with performance criteria
    # cm_perf - dataframe with calibration and validation performance for the Current Method
    # blr - fit model using sklearn
    # fine_tune - slower, but more precise tuning of threshold parameter
    # ease = use first index to pass

    y_prob_train = blr.predict_proba(X_train_sfs)[:, 1]  # Prob. of a post in calibration
    fpr, sens, thresh = roc_curve(y_train, y_prob_train)
    # Matrix of TPR and FPR by various thresholds, tpr = true positive rate, fpr = false positive rate
    spec = 1 - fpr
    A = np.vstack((sens, spec, thresh))

    if A.size == 0:
        print('\n* * * No tuning available that passes model performance criteria for training set * * *\n')
        return np.nan

    # Sensitivity/Specificity criteria
    meets_criteria = (A[0] > perf_criteria['sens_min']) & \
                     (A[0] > perf_criteria['sens_plus_cm'] + cm_perf.loc['Calibration']['Sensitivity']) & \
                     (A[1] >= perf_criteria['spec_min'])
    B = A[:, meets_criteria]  # Select all thresholds that enable tuned model to meet performance criteria

    if fine_tune == 1:
        if B.size == 0:
            tune_st = round(A[2, 0], 4)  # Maximum predicted prob in training set
        else: # select threshold that maxes sensitivity and minimizes FPR in original ROC curve
            # tune_st = round(B[-1, -1], 4)
            tune_st = round(B[-1, 0], 4)
        tune_range = np.arange(tune_st, 0, -0.0001)
        sens_spec = np.array([model_eval(y_train, (y_prob_train >= j).astype(int), tune=1) for j in tune_range])
        T = np.column_stack((sens_spec, tune_range))

        meets_criteria_t = (T[:, 0] > perf_criteria['sens_min']) & \
                           (T[:, 0] > perf_criteria['sens_plus_cm'] + cm_perf.loc['Calibration']['Sensitivity']) & \
                           (T[:, 1] >= perf_criteria['spec_min']) # TODO - add minimum(0.85, CM_spec) as criteria
        U = T[meets_criteria_t]
        if U.size == 0:
            print('\n* * * No tuning available that passes model performance criteria for training set * * *\n')
            return np.nan
        else:
            if ease == 0:
                return U[-1, -1]  # Returns threshold that maximizes sensitivity while maintaining specificity
            else:
                return U[0, -1]  # ease = use first index to pass

    elif B.size != 0:
        return B[-1, -1]  # threshold that maximizes sensitivity and minimizes FPR in original ROC curve

    else:
        print('\n* * * No tuning available that passes model performance criteria for training set * * *\n')
        return np.nan


def tune_mlr(X_train_sfs, y_train, perf_criteria, cm_perf, mlr, thresh, ease):
    # Optimizes MLR model performance to achieve performance criteria (if possible) by adjusting threshold

    # X_train_sfs - Variable dataset with only model variables
    # y_train - must be in binary format
    # perf_criteria - dictionary with performance criteria
    # cm_perf - dataframe with calibration and validation performance for the Current Method
    # mlr - fit model using sklearn
    # ease - use first passing index

    y_pred = mlr.predict(X_train_sfs)  # Prediction
    tune_range = np.arange(0.7, 2.25, 0.001)
    sens_spec = np.array([model_eval(y_train, (y_pred * j), thresh, tune=1) for j in tune_range])
    T = np.column_stack((sens_spec, tune_range))

    # Find all tuning factors (PM) that enable model to meet performance criteria
    meets_criteria_t = (T[:, 0] > perf_criteria['sens_min']) & \
                       (T[:, 0] > perf_criteria['sens_plus_cm'] + cm_perf.loc['Calibration']['Sensitivity']) & \
                       (T[:, 1] >= perf_criteria['spec_min'])  # TODO - add minimum(0.85, CM_spec) as criteria
    U = T[meets_criteria_t]

    if U.size == 0:
        print('\n* * * No tuning available that passes model performance criteria for training set * * *\n')
        return np.nan
    else:
        if ease == 0:
            return U[-1, -1]  # Select PM that maximizes sensitivity while maintaining specificity
        else:
            return U[0, -1]  # Select most conservative passing model (minimum sensitivity in order to max specificity)


# Inputs
# - Alter these inputs at the beginning of each season or if passing models cannot be created
base_folder = 'S:\SCIENCE & POLICY\\NowCast\Modeling\summer_2019\\'
beach_base_folder = base_folder + 'Beaches\\'
loc_file = os.path.join(base_folder, 'locations.csv')  # Beach metadata (name, angle, station info)

# Inputs
spec_beaches = ['']  # Fill if only to model certain beaches
fib = ['FC', 'ENT']  # Change if only trying to model a single FIB type or two
rs = 0  # random seed for data splitting
corr_thresh = 0.75  # Variable correlation max before dropping
VIF_max = 2.5  # maximum allowable VIF
ease = 0  # Select model tuning that just passes the sensitivity criterion (will have higher specificity) [0 or 1]
c = .001  # BLR model regularization
cv = 3  # cross-validation splits

fib_thresh = {'TC': 10000, 'FC': 400, 'ENT': 104}  # As of 2019, not modeling TC anymore
# Exclude prev. FIB vars, add any other variables to exclude from modeling
default_no_model = ['sample_time', 'TC', 'FC', 'ENT', 'TC1', 'FC1', 'ENT1', 'TC_exc', 'FC_exc', 'ENT_exc', 'TC1_exc',
                    'FC1_exc', 'ENT1_exc', 'wet']
# Variables that are automatically excluded [don't make sense to model]
vars_to_drop = ['logTC1', 'logFC1', 'logENT1'] + ['lograin3', 'lograin4', 'lograin5', 'lograin6', 'lograin7', 'wdir1',
                                                  'wdir_L1_max', 'wdir_L1_min', 'MWD1_b', 'MWD1_b_max', 'MWD1_b_min',
                                                  'SIN_MWD1_b_max', 'SIN_MWD1_b_min', 'APD1_max', 'APD1_min',
                                                  'DPD1_max', 'DPD1_min']
                                                  # 'WVHT1', 'WVHT1_max', 'WVHT1_min',  'DPD1', 'APD1', 'SIN_MWD1_b', 'q_dir1']
                                                  # 'wspd1', 'wspd1_min', 'wspd1_max',
                                                  # 'awind1', 'owind1']  # Additional variables to drop
no_model = default_no_model + vars_to_drop

split_methods = ['C1', 'C2', 'C3', 'C4', 'JK7525', 'JK7030', 'JK6040']  # C - chronological, JK - jackknife
s = 'summer'  # Season (for splitting) - 'summer' or 'winter'
tf = 4  # technical feasibility exceedance limit

model_types = ['blr', 'mlr']  # Binary logistic regression, multiple linear regression

perf_criteria = {  # Model Performance criteria
    'sens_min': 0.3,  # Model sensitivity must be at least this
    'sens_plus_cm': 0.1,  # Model sensitivity must also be at least this much greater than the current method
    'spec_min': 0.85  # Model specificity must be at least this
}

# Beaches - FIB where alternative performance criteria is acceptable (spec = CM spec)
# - List beach as key, and list of FIB as values
alt_spec_perf = {
    #'Santa Monica Pier': 'FC',
    'Cowell': ['FC', 'ENT'],
    'Capitola W Jetty': 'ENT',
    # 'Avalon 50ft W Pier':  ['FC', 'ENT'],
    'Cabrillo Harborside Bathrooms': 'ENT',
    'Clam Beach': 'ENT',
    'Pismo Beach Pier South': 'FC',
    # 'Malibu Surfrider Breach': ['FC', 'ENT'],
    'Poche Beach': 'ENT',
    # 'Baker Beach Lobos': 'ENT',
    # 'Doheny': 'ENT'
    'San Clemente Pier Point Zero': ['FC', 'ENT']
}

old_stdout = sys.stdout  # for logging
old_stderr = sys.stderr
debug = 0

# Get Beach Metadata
df_loc = pd.read_csv(loc_file)
df_loc.set_index('beach', inplace=True)
if len(spec_beaches) > 0:  # If specific beaches desired
    beach_list = spec_beaches
else:
    beach_list = list(df_loc.index)

print('Beach Modeler\nBase Directory: ' + base_folder)
for b in beach_list:  # Create models for each beach
    beach_folder = beach_base_folder + b.replace(' ', '_') + '\\'
    var_folder = beach_folder + 'variables\\'
    model_folder = beach_folder + 'models\\'
    if debug == 0:
        now_time = datetime.now().strftime("%m%d_%H%M")
        log_file = open(model_folder + b.replace(' ', '_') + '_modeling_logfile_' + now_time + '.log', 'w')
        sys.stdout = Tee(sys.stdout, log_file)
        sys.stderr = Tee(sys.stderr, log_file)
    print('\n\n- - - | Modeling ' + b + ' | - - -')

    # Grab beach-specific modeling dataset (which contains FIB and environmental variables)
    var_file = b.replace(' ', '_') + '_modeling_dataset.csv'
    df_vars = pd.read_csv(os.path.join(var_folder, var_file))
    df_vars['date'] = pd.to_datetime(df_vars['date'])
    df_vars.set_index('date', inplace=True)

    to_model = [x for x in df_vars.columns if x not in no_model]  # Drop excluded variables

    # Summary of Modeling Dataset
    print('\n- Dataset Summary -\nTotal Samples: ' + str(len(df_vars)))
    for f in fib:
        print('  ' + f + ' Exceedances: ' + str(df_vars[f + '_exc'].sum()))
    print('Number of Variables: ' + str(len(to_model)))
    print('\nParameters:\nRandom Seed: ' + str(rs) +
          '\nCorrelation Threshold: ' + str(corr_thresh) +
          '\nC (regularization): ' + str(c) +
          '\nCV: ' + str(cv) +
          '\nVIF: ' + str(VIF_max) +
          '\nDefault dropped Variables: ' + str(vars_to_drop))

    # Create current method df (used to reindex later); Remove non-model vars
    df_cm = df_vars[default_no_model]  # includes FIB_exc and FIB1_exc for CM eval
    df_vars = df_vars[to_model]

    # Create a model for each FIB
    working_models = {'TC': [], 'FC': [], 'ENT': []}  # List of passing models
    for f in fib:
        print('\n- - | ' + f + ' | - -')
        df_perf_fib = pd.DataFrame()

        # Test for technical feasibility
        total_exc = df_cm[f + '_exc'].sum()
        if total_exc < tf:
            print('Technically infeasible to model ' + f + ' (Less than ' + str(tf) + ' exceedances in dataset)')
            continue

        # Remove other FIB vars from dataset
        df_fib = df_vars.copy()
        other_fib = [x for x in fib_thresh if x != f]
        cols = df_fib.columns
        for i in range(0, len(other_fib)):
            cols = [x for x in cols if other_fib[i] not in x]
        df_fib = df_fib[cols]

        # Initial correlation check - remove highly correlated variables (Check VIF later)
        df_corr = df_fib.corr().loc['log' + f].sort_values(ascending=False)  # correlations
        df_fib.drop(list(df_corr[df_corr.isnull()].index), axis=1, inplace=True)
        df_fib = check_corr(df_fib, f, thresh=corr_thresh)

        # Model Iterator (as of Jan 2019)
        # ** Saves all models that pass sens/spec criteria (if none, prints technically infeasible)**
        # - Splits data into train/test subsets according to split hierarchy (C1, C2, C3, JK-7525, JK-7030, JK-6040)
        #     - Checks if sufficient amount of exceedances in train and test sets (At least one in each)
        # - Creates BLR model first, then MLR
        #   - Employs forward selection algorithm
        # - Checks performance on train subset, tunes to optimize, test on test subset
        # - Save model fit and performance to unique file, CONTINUE Looping
        for m in split_methods:
            print('\n- Modeling (' + f + '-' + m + ') -')
            df_perf = pd.DataFrame()
            cols_perf = ['Sensitivity', 'Specificity', 'Accuracy', 'Exceedances', 'Samples']

            # Split dataset into train/test datasets (CAL/VAL)
            X_train, X_test, y_train, y_test = data_splitter(df_fib, m, rs, f, model_folder, s)

            # Current Method and Performance
            cm_train = df_cm.reindex(y_train.index)  # Adjust current method dfs indices
            cm_test = df_cm.reindex(y_test.index)
            train_exc = cm_train[f + '_exc'].sum()
            test_exc = cm_test[f + '_exc'].sum()
            print('Training (calibration) dataset:\n' + '  Samples - ' + str(len(cm_train)) + '\n  Exc. - ' + str(train_exc))
            print('Test (validation) dataset:\n' + '  Samples - ' + str(len(cm_test)) + '\n  Exc. - ' + str(test_exc))

            model_subfolder = model_folder + '\\' + f + '_' + m
            if (train_exc < 2) | (test_exc == 0):  # If insufficient exceedances in cal/val sets, use new split method
                print('* Insufficient amount of exceedances in each dataset. Skipping this split method. *')
                shutil.rmtree(model_subfolder)  # Delete subdirectory
                continue

            df_perf = df_perf.append(pd.DataFrame(model_eval(cm_train[f + '_exc'], cm_train[f + '1_exc']),
                                                  index=[['Current Method'], ['Calibration']]))  # CM performance
            df_perf = df_perf.append(pd.DataFrame(model_eval(cm_test[f + '_exc'], cm_test[f + '1_exc']),
                                                  index=[['Current Method'], ['Validation']]))
            df_perf.index.names = ['Model', 'Dataset']  # Name multiindex
            df_perf = df_perf[cols_perf]
            cm_perf = df_perf.loc['Current Method']

            # Model with BLR/MLR
            for r in model_types:
                print('\n- Fitting ' + r.upper() + ' models (' + f + ') -')
                multi = True
                while multi:
                    try:
                        features, lm = model_fit(X_train, y_train, r, rs, fib_thresh[f], c=c, cv=cv)  # Fit models
                    except ValueError:
                        print('* Insufficient exceedances to cross-validate models *')
                        break
                    vars = X_train.columns[features]
                    X_multi = multicollinearity_check(X_train[vars], thr=VIF_max)  # Multicollinearity check for VIFs less than 5
                    if len(X_multi.columns) == len(vars):  # If VIF check passes, continue
                        multi = False
                    else:
                        X_train = X_multi  # Training dataset becomes set with high VIF variable removed
                if multi:  # If error in model fitting, skip to next iteration
                    continue

                X_train_sfs = np.array(X_train[vars])  # Reduce cal/val sets to only the selected variables
                X_test_sfs = np.array(X_test[vars])
                if r == 'blr':
                    y_train_blr = (y_train > np.log10(fib_thresh[f])).astype(int)  # Convert FIB to binary
                    lm.fit(X_train_sfs, y_train_blr)  # BLR fit
                elif r == 'mlr':
                    lm.fit(X_train_sfs, y_train)  # MLR fit
                coef = lm.coef_.reshape(-1)  # variable coefficients
                intercept = float(lm.intercept_)
                df_coef = pd.Series(coef, index=vars)
                df_coef.loc['constant'] = intercept
                print('\n' + r.upper() + ' Model Fit (' + str(len(features)) + ' variables): ')
                print('constant - ' + str(round(df_coef.loc['constant'], 5)))
                for i in range(0, len(vars)):
                    print(vars[i] + ' - ' + str(round(df_coef.iloc[i], 5)))

                # Initial Evaluation
                model_str = r.upper() + '_' + m + '_' + str(len(vars))
                df_perf = df_perf.append(
                    pd.DataFrame(model_eval(y_train, lm.predict(X_train_sfs), round(np.log10(fib_thresh[f]), 5)),
                                 index=[[model_str], ['Calibration']]))
                df_perf = df_perf.append(
                    pd.DataFrame(model_eval(y_test, lm.predict(X_test_sfs), round(np.log10(fib_thresh[f]), 5)),
                                 index=[[model_str], ['Validation']]))
                print('\nCurrent Method: \n')
                print(df_perf.loc['Current Method'][cols_perf])
                print('\n' + model_str + ': \n')
                print(df_perf.loc[model_str][cols_perf])

                # Tuning
                print('\nTuning model')
                df_perf.drop(model_str, level=0, inplace=True)  # Drop non_tuned model from perf dataframe
                pc = perf_criteria
                if b in alt_spec_perf:
                    if f in alt_spec_perf[b]:
                        pc['spec_min'] = df_perf.loc['Current Method', 'Calibration']['Specificity']

                if r == 'blr':
                    tune_param = tune_blr(X_train_sfs, y_train_blr, pc, cm_perf, lm, ease=ease)
                    if np.isnan(tune_param):
                        continue
                    else:
                        y_tune_train = (lm.predict_proba(X_train_sfs)[:, 1] >= tune_param).astype(int)
                        y_tune_test = (lm.predict_proba(X_test_sfs)[:, 1] >= tune_param).astype(int)
                        df_coef.loc['threshold'] = tune_param
                        print('\nProbability threshold after tuning: ' + str(round(tune_param, 4)))
                elif r == 'mlr':
                    tune_param = tune_mlr(X_train_sfs, y_train, pc, cm_perf, lm, round(np.log10(fib_thresh[f]), 5), ease=ease)
                    if np.isnan(tune_param):
                        continue
                    else:
                        y_tune_train = lm.predict(X_train_sfs) * tune_param
                        y_tune_test = lm.predict(X_test_sfs) * tune_param
                        df_coef.loc['PM'] = tune_param
                        print('\nPre-multiplier (PM) after tuning: ' + str(round(tune_param, 4)))

                # Re-evaluate tuned models
                model_str_t = model_str.replace(r.upper(), r.upper() + '-T')
                df_perf = df_perf.append(
                    pd.DataFrame(model_eval(y_train, y_tune_train, round(np.log10(fib_thresh[f]), 5)),
                                 index=[[model_str_t], ['Calibration']]))
                df_perf = df_perf.append(
                    pd.DataFrame(model_eval(y_test, y_tune_test, round(np.log10(fib_thresh[f]), 5)),
                                 index=[[model_str_t], ['Validation']]))

                print('\nCurrent Method: \n')
                print(df_perf.loc['Current Method'][cols_perf])
                print('\n' + model_str_t + ': \n')
                print(df_perf.loc[model_str_t][cols_perf])

                # Check if model meets performance criteria
                sens_min = df_perf.loc[model_str_t]['Sensitivity'] >= pc['sens_min']
                sens_plus_cm = df_perf.loc[model_str_t]['Sensitivity'] >= cm_perf['Sensitivity'] + pc['sens_plus_cm']
                spec_min = df_perf.loc[model_str_t]['Specificity'] >= pc['spec_min']  # or above CM spec
                if b in alt_spec_perf:
                    if f in alt_spec_perf[b]:
                        spec_min.loc['Validation'] = df_perf.loc[model_str_t, 'Validation']['Specificity'] >= \
                                                     df_perf.loc['Current Method', 'Validation']['Specificity']
                drop_mod = 0
                fail_str = ''
                for p in ['Calibration', 'Validation']:  # Check if model doesn't meet criteria in validation
                    if not sens_min.loc[p]:
                        drop_mod = 1
                        fail_str += '- Minimum Sensitivity (' + p + ')\n'
                    if not sens_plus_cm.loc[p]:
                        drop_mod = 1
                        fail_str += '- Current Method Sensitivity (' + p + ')\n'
                    if not spec_min.loc[p]:
                        drop_mod = 1
                        fail_str += '- Minimum Specificity (' + p + ')\n'

                if drop_mod == 1:
                    print('\n* * * Model fails to meet the performance criteria * * *\nReasons:\n' + fail_str)
                    df_perf.drop(model_str_t, level=0, inplace=True)  # Drop tuned model from perf dataframe
                    continue
                else:
                    print('\n* ^ * ^ MODEL PASSES CRITERIA IN BOTH CALIBRATION AND VALIDATION ^ * ^ *\n')
                    working_models[f].append(model_str_t)
                    # Save files
                    # Performance
                    df_out = df_perf.query('Model == "Current Method" or Model == "' + model_str_t + '"')
                    df_out = df_out[cols_perf]
                    out_file = 'performance_' + b.replace(' ', '_') + '_' + f + '_' + model_str_t + '.csv'
                    df_out.to_csv(os.path.join(model_subfolder, out_file), float_format='%.3f')

                    # Model Fit
                    lm.coef_ = lm.coef_[lm.coef_ != 0]  # .reshape(1, -1)  # Drop zero-coefficients
                    model_file = 'model_' + b.replace(' ', '_') + '_' + f + '_' + model_str_t + '.pkl'
                    joblib.dump(lm, os.path.join(model_subfolder, model_file))
                    # use joblib.load to load this file in the model runs script

                    # Variables, coefficients, intercepts, threshold
                    df_coef = df_coef[abs(df_coef) > 0]
                    coef_file = 'coefficients_' + b.replace(' ', '_') + '_' + f + '_' + str(model_str_t) + '.csv'
                    df_coef.to_csv(os.path.join(model_subfolder, coef_file))

            # Summary of model split method
            if len([x for x in os.listdir(model_subfolder) if x.startswith('model')]) == 0:
                print('\n * * * No passing models created for ' + f + ' using ' + m + ' * * *')
                shutil.rmtree(model_subfolder)  # Delete directory if no passing models for this split method
            else:
                # Save overall performance dataframe to model subfolder
                df_perf = df_perf[cols_perf]
                out_file_perf = 'performance_' + b.replace(' ', '_') + '_' + f + '_' + m + '_all_models.csv'
                df_perf.to_csv(os.path.join(model_subfolder, out_file_perf), float_format='%.3f')
                # Append to FIB specific performance DF
                df_perf['Split Method'] = m
                df_perf_fib = df_perf_fib.append(df_perf)

        # FIB Summary
        if len(df_perf_fib) > 2:
            df_perf_fib = df_perf_fib[['Split Method'] + cols_perf]
            out_file_perf_fib = 'performance_' + b.replace(' ', '_') + '_' + f + '_all_models.csv'
            df_perf_fib.to_csv(os.path.join(model_folder, out_file_perf_fib), float_format='%.3f')
            print('\n- - Total ' + f + ' Models: ' + str(len(working_models[f])) +
                  ' - -')

    # Beach Summary
    print('\n- - Summary of modeling for ' + b + ' - -')
    print('Number of working models:')
    for f in fib:
        print('  ' + f + ' - ' + str(len(working_models[f])) + ' ' + str(working_models[f]))
    if debug == 0:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_file.close()
