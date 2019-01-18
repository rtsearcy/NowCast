# getRad_daily.py
# RTS 04-27-2018

# Grabs the previous day's solar radiation data [W/m^2] from CIMIS stations around CA, stores in csv.
# NOTE: this code should be run as early as possible in the morning to avoid server errors.
# http://www.cimis.water.ca.gov/

import requests
import json
import pandas as pd
from datetime import datetime, timedelta, date
import os
import sys
import numpy as np


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for fil in self.files:
            fil.write(obj)

    def flush(self):
        pass


# Inputs #
base_folder = 'S:\SCIENCE & POLICY\\NowCast\Daily files\\winter_2018_2019'
beaches_folder = os.path.join(base_folder, 'beaches')
rad_folder = os.path.join(base_folder, 'data\\rad')

units = 'E'  # 'E' English, 'M' Metric
api_key = '6216de17-d2ad-4f0f-b3d5-65ec3638c7c4'

day = date.today()
date_str = date.strftime(day, '%m/%d/%Y')
ed = day
sd = day - timedelta(days=1)

err = 0  # Number of errors
debug = 1
if debug == 0:  # Log output if running (won't let you debug if logging)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    log_file = open(rad_folder + '\\logs\\rad_run_log_' + date_str.replace('/', '') + '.log', 'w')
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

# Beach list, metadata (model to use, angle, stations)
loc_file = os.path.join(beaches_folder, 'nowcast_beaches_winter_2018_2019.csv')
df_loc = pd.read_csv(loc_file)
df_loc.set_index('beach', inplace=True)

# Station list
df_stations = pd.read_csv(os.path.join(rad_folder, 'rad_stations.csv'))  # enviro station DataFrame (stations list)
df_stations.set_index('name', inplace=True)  # Columns must be 'name' and 'station' only
stations = list(df_loc['rad_station'].dropna().unique())  # All stations used by beaches in system

print('- Updating Rad Data for ' + date_str + ' -')
print('\nRuntime: ' + str(datetime.now()) + '\n')

for s in stations:
    print(s + ' -', end='')

    # Open existing file
    var_file = s.replace(' ', '_') + '_rad_data.csv'
    if var_file in os.listdir(rad_folder):  # If file already exists, append new data to old data
        df_old_data = pd.read_csv(os.path.join(rad_folder, var_file))
        df_old_data['date'] = pd.to_datetime(df_old_data['date'])
        df_old_data.set_index('date', inplace=True)

    else:
        df_old_data = pd.DataFrame()

    if day in df_old_data.index:  # Check if getRad script ran successfully earlier in the morning
        if ~np.isnan(df_old_data.loc[day]['rad1']):  # If value is non-null
            print(' COMPLETE (Earlier Run)')
            continue
        else:
            df_old_data.drop(day, inplace=True)  # Drop NAN value for the day

    # Collect new data
    try:
        url = 'http://et.water.ca.gov/api/data?' \
              + 'appKey=' + api_key \
              + '&targets=' + str(df_stations.loc[s]['station']) \
              + '&startDate=' + sd.strftime('%Y-%m-%d') \
              + '&endDate=' + ed.strftime('%Y-%m-%d') \
              + '&dataItems=day-sol-rad-avg' \
              + '&unitOfMeasure=' + units  # + '&unitOfMeasure=\'' + units + '\'' quotations seems to work for now

        web = requests.get(url)
        try:
            web.raise_for_status()
        except Exception as exc:
            print(' ERROR (There was a problem: %s)' % exc)
            # return 0

        d = json.loads(web.text)
        d = d['Data']['Providers'][0]['Records']
        df_out = pd.DataFrame(d)
        df_out['date'] = pd.to_datetime(df_out['Date'])
        df_out.set_index('date', inplace=True)
        df_out['rad_avg'] = [s['Value'] for s in df_out['DaySolRadAvg']]  # Average solar radiation for day of
        df_out['rad1'] = df_out['rad_avg'].shift(1, freq='D')  # Previous day
        df_new_data = df_out[day:day]['rad1'].to_frame()

        # Save to file
        df_combo_data = df_old_data.append(df_new_data)
        df_combo_data = df_combo_data[~df_combo_data.index.duplicated(keep='first')].sort_index()
        # Remove duplicate indices
        df_combo_data.to_csv(os.path.join(rad_folder, var_file))
        print(' COMPLETE')

    except:
        e = sys.exc_info()
        print(' ERROR')
        print('* Data Collection Error: ' + str(e))
        err += 1
        continue

if debug == 0:
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    log_file.close()

if err > 0:
    exit(1)
