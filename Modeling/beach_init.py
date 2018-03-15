# beach_init.py
# RTS - March 2018

# 1.Creates directory for beaches on list (if not already created)
#  In directory - variables folder, models folder, and saves raw FIB sample csv.
# 2.Creates FIB variable dataset (save in var folder, date and time will be used to match enviro vars)
#  Using raw FIB, create following vars:
#     date
#     sample_time
#     FIB
#     FIB1
#     FIB_exc
#     FIB_exc1
#     logFIB
#     logFIB1
#     weekend1
#     laborday

import pandas as pd
from numpy import log10
import os
import shutil
from datetime import date
from fib_thresholds import FIB

def labor_day(x):  # x must be a datetime.date
    lbd = [date(2008, 9, 1),
           date(2009, 9, 7),
           date(2010, 9, 6),
           date(2011, 9, 5),
           date(2012, 9, 3),
           date(2013, 9, 2),
           date(2014, 9, 1),
           date(2015, 9, 7),
           date(2016, 9, 5),
           date(2017, 9, 4),
           date(2018, 9, 3),
           date(2019, 9, 2),
           date(2020, 9, 7)]

    year = x.year
    short = [f for f in lbd if f.year in [year , year+1]]
    if short[0] <= x < short[1]:
        result = 1
    else:
        result = 0
    return result

# Inputs #
init_folder = 'Z:\Predictive Modeling\Phase III\Modeling\Summer_2018\Raw FIB\\'  # directory with all raw FIB files
init_file = init_folder + 'locations.csv'  # file to grab beach locations from ('beach' column must has space separated beach names)
base_folder = 'Z:\Predictive Modeling\Phase III\Modeling\Summer_2018\Beaches\\'  # directory to house beach folders

to_process = 'all' # all - all beaches in init file; list of beach names
to_skip = []  # Manually skip certain locations
skip_check = 0  # If 1, skip if directory is already created; if 0, replace directory, saving to old directory

season = 'Summer'  # Summer, Winter , All
sd = '20080101' # Start date (will parse out depending on season
ed = '20171231' # end date

# Create Beach Directories #
if to_process == 'all':
    df_init = pd.read_csv(init_file) # Open init file
    beach_list = list(df_init['beach'])
    beach_list = [b for b in beach_list if b not in to_skip]  # Remove manual skipped beaches
else:
    beach_list = to_process

print('base folder: ' + base_folder + '\n')
for b in beach_list:
    print('- ' + b)
    beach_dir = base_folder + b.replace(' ', '_') + '\\'
    fib_file = [file for file in os.listdir(init_folder) if b.replace(' ', '_') in file][0]  # raw sample file name
    # Check if exists
    if b.replace(' ','_') in os.listdir(base_folder): # If directory already exists...
        print(' Directory for ' + b + 'already exists in base folder...')
        if skip_check == 1:
            print('   Skipped')
            continue
        else:
            try:
                shutil.move(beach_dir, base_folder +'old\\')
            except shutil.Error:
                shutil.rmtree(base_folder +'old\\' + b.replace(' ','_'))
                shutil.move(beach_dir, base_folder + 'old\\')
            os.makedirs(beach_dir, exist_ok=True)
            os.makedirs(os.path.join(beach_dir, 'variables'))  # Make new variables subdirectory
            os.makedirs(os.path.join(beach_dir, 'models'))  # Make new models subdirectory
            print('   Replaced (see "old" directory for previous version)')
    else:
        os.makedirs(beach_dir) # Make new directory
        os.makedirs(os.path.join(beach_dir, 'variables'))  # Make new variables subdirectory
        os.makedirs(os.path.join(beach_dir, 'models'))  # Make new models subdirectory
        print(' New directory created: ' + beach_dir)

    try:  # Copy over sample file
        shutil.copy(init_folder + fib_file, beach_dir + fib_file)
        print(' Raw sample file copied to beach directory')
    except:
        print(' ERROR: Could not copy over raw FIB file')
        continue

    # FIB Variables
    fib = ['TC','FC','ENT']
    thresholds = FIB().fib_thresholds # from class
    df_raw = pd.read_csv(os.path.join(beach_dir, fib_file))
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_raw.set_index('date', inplace=True)

    df_vars = df_raw.copy()
    df_vars = df_vars[['sample_time'] + fib]
    for f in fib:
        df_vars[f + '1'] = df_vars[f].dropna().shift(1)  # previous sample, skipping any missed samples in dataset
        df_vars[f + '_exc'] = (df_vars[f] > thresholds[f]).astype(int)  # exceeds threshold? (binary)
        df_vars[f + '1_exc'] = (df_vars[f + '1'] > thresholds[f]).astype(int) # previous day exceeds threshold? (binary)
        df_vars['log' + f] = round(log10(df_vars[f]), 5)
        df_vars['log' + f + '1'] = round(log10(df_vars[f + '1']), 5)

    var_order = fib + [f + '1' for f in fib] + [f + '_exc' for f in fib] + \
                      [f + '1_exc' for f in fib] + ['log' + f for f in fib] + ['log' + f + '1'for f in fib]
    df_vars = df_vars[var_order]
    df_vars['weekend1'] = ((df_vars.index.weekday == 0) | (df_vars.index.weekday == 6) | (df_vars.index.weekday == 7)).astype(int) # Was yesterdau a weekend? (binary) Monday = 0, Sunday = 7

    # Account for time range and season
    df_vars = df_vars[sd:ed]
    if season == 'Summer':
        df_vars = df_vars[(df_vars.index.month >= 4) & (df_vars.index.month < 11)]
        df_vars['laborday'] = [labor_day(x) for x in df_vars.index.date] # Binary - has laborday passed?
    elif season == 'Winter':
        df_vars = df_vars[(df_vars.index.month <= 3) | (df_vars.index.month >= 11)]

    # Save variables
    print(' Variables (Season: ' + season + ', Sample Range: ' + str(df_vars.index.year[0]) + ' to ' + str(df_vars.index.year[-1]) + ')')
    print('   Number of Samples: ' + str(len(df_vars)))
    var_file = beach_dir + 'variables\\' + b.replace(' ', '_') + '_variables_fib.csv'
    df_vars.to_csv(var_file)
    print(' Saved to : ' + var_file + '\n')