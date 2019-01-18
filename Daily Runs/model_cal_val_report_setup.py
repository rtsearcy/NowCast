# model_cal_val_report_setup.py
# RTS - 04102018

# Collates modeling results and coefficients for each county in the system
# Includes variable list as well as copy of papers

import os
import pandas as pd
import shutil

# Inputs
base_model_folder = 'Z:\Predictive Modeling\Phase III\Modeling\Summer_2018\\'
base_beach_folder = base_model_folder + 'Beaches\\'
base_county_folder = base_model_folder + 'County Model Reports\\'

papers_folder = base_county_folder + 'papers\\'
variable_list_file = base_model_folder + 'Environmental Variables\\variables_list.pdf'

base_daily_folder = 'S:\SCIENCE & POLICY\\NowCast\Daily files\\2018\\beaches\\'
loc_file = os.path.join(base_daily_folder, 'nowcast_beaches_2018.csv')

# Compile folders
df_loc = pd.read_csv(loc_file)
df_loc.set_index('beach', inplace=True)
counties = [c for c in df_loc['county'].unique()]

for c in counties:
    df_county = df_loc[(df_loc['county'] == c)]
    if len(df_county) == 0:
        continue

    county_subfolder = base_county_folder + c
    os.makedirs(county_subfolder, exist_ok=True)  # Make base directory to store data

    try:
        shutil.copytree(papers_folder, county_subfolder + '\\papers\\')  # Copy papers
    except FileExistsError:
        pass

    os.makedirs(county_subfolder + '\\beaches', exist_ok=True)  # Make beaches folder
    for b in df_county.index:
        if df_county.loc[b]['pilot'] == 0:
            model_subfolder = county_subfolder + '\\beaches\\' + df_county.loc[b]['display_name']
        else:
            model_subfolder = county_subfolder + '\\beaches\\backup\\' + df_county.loc[b]['display_name']

        os.makedirs(model_subfolder, exist_ok=True)  # Make model subfolder

        shutil.copy(variable_list_file, county_subfolder + '\\beaches')  # Move variable list

        # Transfer modeling data file
        beach_folder = base_beach_folder + '\\' + b.replace(' ', '_') + '\\'
        dataset_file = beach_folder + 'variables\\' + b.replace(' ', '_') + '_modeling_dataset.csv'
        shutil.copy(dataset_file, model_subfolder)

        for f in ['TC', 'FC', 'ENT']:
            if type(df_county.loc[b][f + '_model']) == float:
                continue
            fib_folder = model_subfolder + '\\' + f
            os.makedirs(fib_folder, exist_ok=True)  # Make fib folder

            model = df_loc.loc[b][f + '_model']
            alg, split, var_no = model.split('_')
            coef_folder = os.path.join(beach_folder + 'models', f + '_' + split)
            coef_file = 'coefficients_' + b.replace(' ', '_') + '_' + f + '_' + model + '.csv'
            perf_file = 'performance_' + b.replace(' ', '_') + '_' + f + '_' + model + '.csv'

            shutil.copy(os.path.join(coef_folder, coef_file), fib_folder)  # Copy coef file
            shutil.copy(os.path.join(coef_folder, perf_file), fib_folder)  # Copy performance file
