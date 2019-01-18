#! python3
# rerun_for_new_models_2018.py - When a replacement model is used for a beach already in the system,
# this script uses the stored variable data to run the new model retroactively
# RTS -  5/29/2018

# 1. Using stored data, compute variables and run models
# 2. Save results to a dated csv. Also save a dated upload file for the current BRC site

import pandas as pd
import numpy as np
import os
import sys 
import datetime
from sklearn.externals import joblib


# SETUP
sd = '04/01/2018'  # Season start date
ed = '10/31/2018'  # end date

base_folder = 'S:\SCIENCE & POLICY\\NowCast\Daily files\\2018'
beaches_folder = os.path.join(base_folder, 'beaches')

# Beach list, metadata (model to use, angle, stations)
loc_file = os.path.join(beaches_folder, 'nowcast_beaches_2018.csv')  # beach metadata
df_loc = pd.read_csv(loc_file)
df_loc.set_index('beach', inplace=True)
beach_list = []  # list(df_loc.index)  # or all beaches

np.seterr(divide='ignore')  # Stop Divide vy zero warnings
pd.set_option('display.expand_frame_repr', False)  # Don't limit print size

# Run Models for Each Beach
for b in beach_list:
    print('\n- - NowCast Predictions for ' + b + '- -')
    model_folder = os.path.join(beaches_folder, b.replace(' ', '_'))
    var_folder = os.path.join(model_folder, 'variables')
    angle = int(df_loc.loc[b]['angle'])

    # Environmental Variables
    df_vars = pd.read_csv(os.path.join(var_folder, b.replace(' ', '_') + '_variables.csv'))
    if len(df_vars) == 0:
        print('   No variables found for ' + b + '\n')
        sys.exit()
    df_vars['date'] = pd.to_datetime(df_vars['date'])
    df_vars = df_vars.set_index('date').sort_index()
    df_vars = df_vars[sd:ed]

    # Run models
    df_pred_all = pd.DataFrame(index=df_vars.index)
    for f in ['TC', 'FC', 'ENT']:
        if type(df_loc.loc[b][f + '_model']) == float:  # If model is available
            print('\n' + f + ' Model (' + b + '): No Model')
            continue
        else:
            print('\n' + f + ' Model (' + b + '): ' + df_loc.loc[b][f + '_model'] + '\n')

            # Download model and coefficients
            model_file = 'model_' + b.replace(' ', '_') + '_' + f + '_' + df_loc.loc[b][f + '_model'] + '.pkl'
            lm = joblib.load(os.path.join(model_folder, model_file))

            coef_file = 'coefficients_' + b.replace(' ', '_') + '_' + f + '_' + df_loc.loc[b][f + '_model'] + '.csv'
            df_coef = pd.read_csv(os.path.join(model_folder, coef_file), header=None)
            df_coef.columns = ['Variable', 'Coefficient']
            df_coef.set_index('Variable', inplace=True)

            # Print model
            t = [t for t in df_coef.index if t in ['PM', 'threshold']][0]
            tuner = float(df_coef.loc[t])  # extract PM/thresh tuner and constant from coef dataframe
            constant = float(df_coef.loc['constant'])
            df_coef.drop([t, 'constant'], inplace=True)
            model_vars = list(df_coef.index)
            print(df_coef)
            print('\nConstant: ' + str(round(constant, 3)))
            print(t + ': ' + str(round(tuner, 3)))

            df_model_vars = df_vars[model_vars].dropna()  # Locate only model variables
            if 'MLR-T' in model_file:
                mod_results = 10 ** (tuner * lm.predict(df_model_vars.values))

            elif 'BLR-T' in model_file:
                lm.coef_ = lm.coef_.reshape(1, -1)
                mod_results = lm.predict_proba(df_model_vars.values)[:, 1]

            df_pred_all[f] = pd.Series(mod_results, index=[df_model_vars.index])

    print('\nPredictions:')
    print(df_pred_all)

    # Save predictions #
    for d in df_pred_all.index:
        date_str = datetime.date.strftime(d, '%m/%d/%Y')
        df_pred = df_pred_all[date_str:date_str]  # Save the day's predictions
        df_pred.index.rename('Date', inplace=True)
        df_pred.to_csv(model_folder + '\\predictions\\' + date_str.replace('/', '') + '_predictions.csv')
