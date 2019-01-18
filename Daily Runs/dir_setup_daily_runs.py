# dir_setup_daily_runs.py
# RTS - 4/3/2018

# - Sets up the daily runs directory for the season, starting with an existing base folder,
# and a list of beaches and their models

# NOTE: Use 'tideVarsDaily.py' to populate the tide folder with tide predictions for the season

import pandas as pd
import os
import shutil


# Inputs
daily_run_folder = 'S:\SCIENCE & POLICY\\NowCast\Daily files\\winter_2018_2019'  # Implementation folder

base_folder = 'Z:\Predictive Modeling\Phase III\Modeling\Winter_2018_2019\\'  # Location of beaches and model data
beach_base_folder = base_folder + 'Beaches\\'
model_list_file = os.path.join(daily_run_folder, 'beaches\\nowcast_beaches_winter_2018_2019.csv')
# Beach metadata and models for each FIB

spec_beaches = []  # Only setup certain beaches
overwrite = 0  # 1 - overwrite folders even if not empty; 0 - don't overwrite folder if it contains something

# Set up subdirectories (non-beaches)
for i in ['beaches', 'code', 'decisions', 'email', 'import', 'logs', 'performance', 'qaqc', 'data']:
    i_folder = os.path.join(daily_run_folder, i)
    os.makedirs(i_folder, exist_ok=True)  # create subfolder

    if i == 'email':
        os.makedirs(os.path.join(i_folder, 'sent_emails'), exist_ok=True)  # creates sent_emails subfolder
        os.makedirs(os.path.join(i_folder, 'tables'), exist_ok=True)  # creates tables subfolder
    elif i == 'data':
        for j in ['tide', 'met', 'wave', 'rad', 'flow', 'coop', 'fib']:
            os.makedirs(os.path.join(i_folder, j), exist_ok=True)  # creates data subfolder
            if j == 'met':
                os.makedirs(os.path.join(os.path.join(i_folder, j), 'raw'), exist_ok=True)
            elif j == 'tide':
                os.makedirs(os.path.join(os.path.join(i_folder, j), 'hourly'), exist_ok=True)
            elif j == 'rad':
                os.makedirs(os.path.join(os.path.join(i_folder, j), 'logs'), exist_ok=True)

# Get Beaches
df_loc = pd.read_csv(model_list_file)
df_loc.set_index('beach', inplace=True)
if len(spec_beaches) > 0:  # If specific beaches
    beach_list = spec_beaches
else:
    beach_list = list(df_loc.index)

beach_folder = os.path.join(daily_run_folder, 'beaches')
# Setup beach-specific subdirectories
for b in beach_list:
    print('Processing ' + b)
    model_folder = beach_base_folder + b.replace(' ', '_') + '\models\\'

    # Create beach subdirectory
    beach_subfolder = os.path.join(beach_folder, b.replace(' ', '_'))
    if overwrite == 1 and b.replace(' ', '_') in os.listdir(beach_folder):  # Delete if overwrite is true
        shutil.rmtree(beach_subfolder)

    os.makedirs(beach_subfolder, exist_ok=True)
    for k in ['fib', 'performance', 'predictions', 'variables']:
        os.makedirs(os.path.join(beach_subfolder, k), exist_ok=True)

    # Import model pickle files and coefficients
    for f in ['FC', 'ENT']:
        model = df_loc.loc[b][f + '_model']
        if type(model) == float:  # np.isnan(model):
            continue
        alg, split, var_no = model.split('_')
        model_subfolder = os.path.join(model_folder, f + '_' + split)
        pkl_file = 'model_' + b.replace(' ', '_') + '_' + f + '_' + model + '.pkl'
        shutil.copy(os.path.join(model_subfolder, pkl_file),
                    os.path.join(beach_subfolder, pkl_file))  # Copy pkl file
        coef_file = 'coefficients_' + b.replace(' ', '_') + '_' + f + '_' + model + '.csv'
        shutil.copy(os.path.join(model_subfolder, coef_file),
                    os.path.join(beach_subfolder, coef_file))  # Copy coef file
