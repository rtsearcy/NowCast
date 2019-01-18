# tideVarsHourly.py - Computes hourly tidal variables from raw CO-OPS 6 min water level data for beaches with
# sample times available
# RS - 4/13/2018

# Raw data source/description: https://tidesandcurrents.noaa.gov/tide_predictions.html

# NOTE: raw csv files should have timestamps in LST

# Tide Variables: Tide, Tide_h, dTide_h

import pandas as pd
import os
import sys
from datetime import datetime

# Import raw data csv to pd DataFrame
infolder = 'Z:\Predictive Modeling\Phase III\Modeling\Summer_2018\Environmental Variables\Tides\\raw'
outfolder = 'S:\SCIENCE & POLICY\\NowCast\Daily files\\2018\data\\tide\\hourly'

loc_file = 'S:\SCIENCE & POLICY\\NowCast\Daily files\\2018\\beaches\\nowcast_beaches_2018.csv'
# location of beach metadata sheet

sd = '20180101'  # Start date (account for previous day, conservatuve)
ed = '20181231'  # End date

# TO ITERATE THROUGH ALL BEACHES WITH SAMPLE TIMES LISTED
df_loc = pd.read_csv(loc_file)  # beach metadata
df_loc.set_index('beach', inplace=True)
df_loc = df_loc[~df_loc['sample_time'].isnull()]
if len(df_loc) == 0:
    print('No beaches with sample times')
    sys.exit()

for b in df_loc.index:
    station = df_loc.loc[b]['tide_station']
    sample_time = datetime.strptime(df_loc.loc[b]['sample_time'], '%H:%M %p').time()
    raw_file = [f for f in os.listdir(infolder) if f.startswith(station.replace(' ', '_'))][0]
    df_raw = pd.read_csv(os.path.join(infolder, raw_file))
    df_raw.columns = ['date', 'tide']

    #  Convert Date Time to timestamp, set as index
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_raw.set_index('date', inplace=True)
    df_raw = df_raw[sd:ed]  # Only samples in time range (for speed)
    df = pd.DataFrame(index=df_raw.resample('D').mean().index)  # Preset index to days

    # Tide at the hour of sampling
    df['Tide'] = df_raw[df_raw.index.time == sample_time].resample('D').first().reindex(df.index,
                                                                                        method='nearest')

    for i in [1, 3, 6, 9, 12]:  # Tide level 'i' hours previous to sampling
        df['Tide_' + str(i) + 'h'] = df_raw[df_raw.index.shift(i, freq='H').time == sample_time].resample('D').first() \
            .reindex(df.index, method='nearest')  # Tide_h
        df['dTide_' + str(i) + 'h'] = df['Tide'] - df['Tide_' + str(i) + 'h']  # dTide_h

    # Save to file
    of_name = b.replace(' ', '_') + '_Hourly_Tide_Variables_' + sd + '_' + ed + '.csv'
    outfile = os.path.join(outfolder, of_name)
    df.index.rename('date', inplace=True)
    df.to_csv(outfile)
