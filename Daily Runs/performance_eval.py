# performance_eval_2018.py - Harvests recent FIB data, and evaluates model and current method
# performance based on predictions made to date
# RTS - 04/27/2018

import pandas as pd
import numpy as np
import os
import sys
import warnings
import pyodbc
import datetime


class Tee:  # For logging
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for fil in self.files:
            fil.write(obj)

    def flush(self):
        pass


def model_eval(true, predicted, thresh=0.5):  # Model Performance
    if true.dtype == 'float':
        true = (true > thresh).astype(int)  # Convert to binary
    if predicted.dtype == 'float':
        predicted = (predicted > thresh).astype(int)

    samples = len(true)
    exc = true.sum()
    tp = np.sum((true > thresh) & (predicted > thresh))  # True positives
    tn = np.sum((true < thresh) & (predicted < thresh))  # True negatives
    fp = np.sum((true < thresh) & (predicted > thresh))  # False positives
    fn = np.sum((true > thresh) & (predicted < thresh))  # False negative

    sens = tp / (tp + fn)  # Sensitivity
    spec = tn / (tn + fp)  # Specificity
    acc = (tn + tp) / samples  # Accuracy

    out = {'Sensitivity': round(sens, 3), 'Specificity': round(spec, 3), 'Accuracy': round(acc, 3),
           'Samples': samples, 'Exceedances': exc}

    return out


# Inputs
sd_fib = '2018-09-01'  # To account for FIB1 for CM when collecting from BRC database
sd_perf = '2018-11-01'  # Start date for performance evaluation
ed = '2019-03-31'  # End date of season

brc_db_file = 'N:\BEACH 2018-19\BRC2015.mdb'  # BRC Database
base_folder = 'S:\SCIENCE & POLICY\\NowCast\Daily files\\winter_2018_2019'
beaches_folder = os.path.join(base_folder, 'beaches')
fib_folder = os.path.join(base_folder, 'data\\fib')
performance_folder = os.path.join(base_folder, 'performance')

debug = 0
fib_collect_only = 0
performance_eval_only = 0
dry_only = 1  # 1 - only dry samples included in evaluation, 0 - all samples included

# Beach list, metadata
loc_file = os.path.join(beaches_folder, 'nowcast_beaches_winter_2018_2019.csv')
df_loc = pd.read_csv(loc_file)
df_loc.set_index('beach', inplace=True)
beach_list = list(df_loc[df_loc['pilot'] == 0].index)  # non-pilot
#beach_list = list(df_loc.index)  # [all beaches] # or list of specific beaches

np.seterr(divide='ignore')  # Stop Divide By Zero warnings for 0 exceedance models
warnings.filterwarnings('ignore')
pd.set_option('display.expand_frame_repr', False)  # Don't limit print size

# FIB threshold
fib_thresh = {
    'TC': 10000,
    'FC': 400,
    'ENT': 104
}

# Collect FIB data
if performance_eval_only == 0:
    print('- - | Collecting FIB Data | - -\n')
    # Open BRC Database
    conn_str = (
        r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
        r'DBQ=' + brc_db_file + ';'
        )
    brc_db = pyodbc.connect(conn_str)
    d = brc_db.cursor()

    # For each BRC Beach ID, grab all sample data after minimum date
    brc_id = list(df_loc['brc_id'].dropna())
    brc_query = 'SELECT * FROM tblPoop WHERE fpkLocId IN {}'.format(tuple(brc_id)) + ' AND fpkDate >= #' + sd_fib + \
                '# AND fpkDate <= #' + ed + '#'
    df_brc = pd.read_sql_query(brc_query, brc_db)
    df_brc = df_brc[['fpkDate', 'fldTotalVal', 'fldFecalVal', 'fldEnteroVal', 'fldNotes', 'fpkLocId']]
    cols = ['date', 'TC', 'FC', 'ENT', 'sample_time', 'brc_LocId']  # Rename columns
    df_brc.columns = cols
    df_brc = df_brc.sort_values('date')

    for b in beach_list:  # Create/Add to individual beach datasets
        print(b + ' - ', end='')
        fib_file = b.replace(' ', '_') + '_fib_samples.csv'
        beach_folder = os.path.join(beaches_folder, b.replace(' ', '_'))
        try:
            # Old FIB Data (if exists)
            if fib_file in os.listdir(fib_folder):  # If file already exists, will append new data to old data
                df_old_fib = pd.read_csv(os.path.join(fib_folder, fib_file))
                df_old_fib['date'] = pd.to_datetime(df_old_fib['date'])
                df_old_fib.set_index('date', inplace=True)
            else:
                df_old_fib = pd.DataFrame()

            # BRC Data
            df_new_fib = df_brc[df_brc['brc_LocId'] == df_loc.loc[b]['brc_id']]
            df_new_fib['date'] = pd.to_datetime(df_new_fib['date'].loc[:])
            df_new_fib.set_index('date', inplace=True)
            df_new_fib = df_new_fib[['TC', 'FC', 'ENT', 'sample_time']]
            df_new_fib['supplemental'] = ''  # Column used to gage number of additional samples taken during the season

            # Add the sample previous of the first date (for CM eval later)
            day_one = df_new_fib[sd_perf:ed].index[0]
            day_one_idx = df_new_fib.index.get_loc(day_one)
            df_new_fib = df_new_fib.iloc[day_one_idx - 1:]

            df_combo_data = df_old_fib.append(df_new_fib)
            df_combo_data = df_combo_data[~df_combo_data.index.duplicated(keep='first')].sort_index()
            # Remove duplicate indices
            df_combo_data.to_csv(os.path.join(fib_folder, fib_file))  # Save file to base data folder
            df_combo_data.to_csv(os.path.join(os.path.join(beach_folder, 'fib'), fib_file))  # Save file to beach folder
            print(' COMPLETE')
        except:
            e = sys.exc_info()
            print(' ERROR')
            print(e)
            continue
    print('\n')


# Performance Eval
if fib_collect_only == 0:
    if datetime.datetime.strptime(ed, '%Y-%m-%d') > datetime.datetime.today():  # Adjust end date to match current day
        ed = datetime.date.today().strftime('%Y-%m-%d')

    if debug == 0:  # Log output if running (won't let you debug if logging)
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        if dry_only == 1:
            log_file = open(base_folder + '\\performance\\performance_log_DRY_' + sd_perf.replace('-', '')
                            + '_' + ed.replace('-', '') + '.log', 'w')
        else:
            log_file = open(base_folder + '\\performance\\performance_log_' + sd_perf.replace('-', '')
                            + '_' + ed.replace('-', '') + '.log', 'w')
        sys.stdout = Tee(sys.stdout, log_file)
        sys.stderr = Tee(sys.stderr, log_file)

    if dry_only == 1:
        print('** Dry Weather Data Only **   ')
    print('- - - | Performance Evaluation for ' + sd_perf + ' to ' + ed + ' | - - -\n')

    df_perf_all = pd.DataFrame()  # Cumulative performance metrics
    df_pred_stats = pd.DataFrame()  # Cumulative predictions stats (number, post/no posts, days above sampling)
    ps_cols = ['Beach Days Evaluated', 'Number of Predictions', '"Post" Predictions', '"No Post" Predictions',
               'Number of Samples', 'Missing Predictions', 'Wet Days', 'Additional Days from Predictions']
    # Pred stats column order

    # Performance by FIB and beach
    for b in beach_list:
        print('- - | ' + df_loc.loc[b]['display_name'] + '| - -')
        fib_file = b.replace(' ', '_') + '_fib_samples.csv'
        beach_folder = os.path.join(beaches_folder, b.replace(' ', '_'))
        pred_folder = os.path.join(beach_folder, 'predictions')
        perf_folder = os.path.join(beach_folder, 'performance')
        var_folder = os.path.join(beach_folder, 'variables')

        # Load Variables
        df_vars = pd.read_csv(os.path.join(var_folder, b.replace(' ', '_') + '_variables.csv'))
        df_vars['date'] = pd.to_datetime(df_vars['date'])
        df_vars.set_index('date', inplace=True)

        # Load Predictions
        df_pred = pd.DataFrame()
        for file in os.listdir(pred_folder):
            df_temp = pd.read_csv(os.path.join(pred_folder, file))
            df_pred = df_pred.append(df_temp)
        if len(df_pred) == 0:
            print('   No predictions found for ' + b + '\n')
            continue
        d = [d for d in df_pred.columns if d in ['Date', 'date']][0]
        df_pred[d] = pd.to_datetime(df_pred[d])
        df_pred = df_pred.set_index(d).sort_index()
        df_pred = df_pred[sd_perf:ed].dropna()
        df_pred.index.rename('date', inplace=True)

        # Load FIB for beach
        df_fib = pd.read_csv(os.path.join(fib_folder, fib_file))
        df_fib['date'] = pd.to_datetime(df_fib['date'])
        df_fib.set_index('date', inplace=True)

        for f in df_pred.columns:  # To binary
            coef_file = 'coefficients_' + b.replace(' ', '_') + '_' + f + '_' + df_loc.loc[b, f + '_model'] + '.csv'
            if 'MLR-T' in df_loc.loc[b, f + '_model']:  # MLR models
                df_pred[f] = df_pred[f].apply(lambda x: 1 if x > fib_thresh[f] else 0)

            else:  # BLR models
                df_coef = pd.read_csv(os.path.join(beach_folder, coef_file), header=None)
                df_coef.columns = ['Variable', 'Coefficient']
                df_coef.set_index('Variable', inplace=True)
                prob_thresh = float(df_coef.loc['threshold'])
                df_pred[f] = df_pred[f].apply(lambda x: 1 if x > prob_thresh else 0)

            # FIB dataframe
            df_fib[f] = df_fib[f].dropna().apply(lambda x: 1 if x > fib_thresh[f] else 0)  # Post/No-Post
            df_fib[f + '1'] = df_fib[f].shift(1)  # FIB1

        df_fib = df_fib[sd_perf:ed]

        if dry_only == 1:  # Remove Wet Day predictions and fib samples from evaluation
            df_pred = df_pred.reindex(df_vars[df_vars['wet'] != 1].index).dropna().astype(int)
            df_fib = df_fib.reindex(df_vars[df_vars['wet'] != 1].index).dropna(how='all').sort_index()

        # Prediction Stats (Number, post/no-post, days additional above samples
        # days_season = (df_pred.index[-1] - pd.to_datetime(sd_perf)).days + 1  # Number days in season
        days_season = (pd.to_datetime(ed) - pd.to_datetime(sd_perf)).days + 1  # Number days in season
        num_pred = len(df_pred)
        if df_loc.loc[b]['wet_exclude'] == 1:
            wet_days = 0
        else:
            wet_days = int(df_vars['wet'].sum())
        missing_pred = days_season - num_pred if dry_only == 0 else days_season - num_pred - wet_days
        # if dry only - missing predictions not counting wet days, if all weather - missing
        pred_stats_dict = {
            'Beach Days Evaluated': days_season,
            'Number of Predictions': num_pred,  # Excluding missed days, but including wet days
            '"Post" Predictions': max(df_pred.sum()),
            '"No Post" Predictions': num_pred - max(df_pred.sum()),
            'Number of Samples': len(df_fib),
            'Missing Predictions': missing_pred,  # days_season - num_pred,
            'Wet Days': wet_days,
            'Additional Days from Predictions': num_pred - len(df_fib) - missing_pred
        }
        df_pred_stats = df_pred_stats.append(
            pd.DataFrame(pred_stats_dict, index=[[df_loc.loc[b]['display_name']], [df_loc.loc[b]['county']]]))

        if len(df_fib) == 0:
            print('- - No FIB samples available found for ' + b + ' - -\n')
            for p in ps_cols:
                print(p + ' - ' + str(pred_stats_dict[p]))
            print('\n')
            continue
        else:
            print('Evaluation Period: ' + sd_perf + ' to ' + str(df_pred.index[-1].date()))
            print('  Samples through ' + str(df_fib.index[-1].date()) + '\n')
            for p in ps_cols:
                print(p + ' - ' + str(pred_stats_dict[p]))

        # Evaluate Models
        df_perf_beach = pd.DataFrame()
        for f in df_pred.columns.sort_values(ascending=False):
            df_fib_spec = df_fib[[f, f + '1']].dropna().astype(int)
            cm_perf = model_eval(df_fib_spec[f], df_fib_spec[f + '1'])
            df_pred_spec = df_pred[f].reindex(df_fib_spec.index)
            model_perf = model_eval(df_fib_spec[f], df_pred_spec)

            df_perf_beach = df_perf_beach.append(
                pd.DataFrame(model_perf,
                             index=[[df_loc.loc[b][f + '_model']], [f]]))
            df_perf_beach = df_perf_beach.append(
                pd.DataFrame(cm_perf,
                             index=[['Current Method'], [f]]))

        df_perf_beach = df_perf_beach[['Sensitivity', 'Specificity', 'Accuracy', 'Exceedances', 'Samples']]
        df_perf_beach.index.rename(['Model', 'FIB'], inplace=True)
        if dry_only == 1:
            perf_beach_file = 'performance_DRY_' + sd_perf.replace('-', '') + '_' + ed.replace('-', '') + '.csv'
        else:
            perf_beach_file = 'performance_' + sd_perf.replace('-', '') + '_' + ed.replace('-', '') + '.csv'
        df_perf_beach.to_csv(os.path.join(perf_folder, perf_beach_file))  # Save indiv. beach perf

        print('\nPerformance\n')
        print(df_perf_beach)
        df_perf_beach['Beach'] = df_loc.loc[b]['display_name']
        df_perf_beach['County'] = df_loc.loc[b]['county']
        df_perf_beach.set_index(['Beach', 'County'], append=True, inplace=True)
        df_perf_all = df_perf_all.append(df_perf_beach)
        print('\n\n')

    # Save number of predictions (post/no-post by FIB and by beach)
    df_pred_stats = df_pred_stats[ps_cols]
    df_pred_stats.index.rename(['Beach', 'County'], inplace=True)
    if dry_only == 1:
        pred_stats_file = 'prediction_statistics_all_models_DRY_' + sd_perf.replace('-', '') + \
                          '_' + ed.replace('-', '') + '.csv'
    else:
        pred_stats_file = 'prediction_statistics_all_models_' + sd_perf.replace('-', '') \
                          + '_' + ed.replace('-', '') + '.csv'
    df_pred_stats.to_csv(os.path.join(performance_folder, pred_stats_file))
    print('\n- - | Prediction Statistics (NowCast System) | - -\n')
    print('Total Number of Predictions: ' + str(df_pred_stats[ps_cols]['Number of Predictions'].sum()))
    print('Total Number of Samples: ' + str(df_pred_stats[ps_cols]['Number of Samples'].sum()))
    print('Total Additional Days from Predictions: ' +
          str(df_pred_stats[ps_cols]['Additional Days from Predictions'].sum()) + '\n')

    # Save performance to single spreadsheet
    df_perf_all = df_perf_all.reorder_levels(['Beach', 'County', 'Model', 'FIB'])
    if dry_only == 1:
        perf_all_file = 'performance_all_models_DRY_' + sd_perf.replace('-', '') + '_' + ed.replace('-', '') + '.csv'
    else:
        perf_all_file = 'performance_all_models_' + sd_perf.replace('-', '') + '_' + ed.replace('-', '') + '.csv'
    df_perf_all.to_csv(os.path.join(performance_folder, perf_all_file))

    # Evaluate median sens/spec for all models and CM
    print('\n- - | Performance (NowCast System) | - -\n')
    print('\nNowCast (Mean): ')
    print(df_perf_all.query('Model != "Current Method"').mean().loc[['Sensitivity', 'Specificity', 'Accuracy']].to_string())
    print('\nCurrent Method (Mean): ')
    print(df_perf_all.query('Model == "Current Method"').mean().loc[['Sensitivity', 'Specificity', 'Accuracy']].to_string())
    print('\n\nNowCast (Median): ')
    print(df_perf_all.query('Model != "Current Method"').median().loc[['Sensitivity', 'Specificity', 'Accuracy']].to_string())
    print('\nCurrent Method (Median): ')
    print(df_perf_all.query('Model == "Current Method"').median().loc[['Sensitivity', 'Specificity', 'Accuracy']].to_string())

    if debug == 0:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_file.close()
