# tideVarsDaily.py - Computes daily tidal variables from raw CO-OPS 6 min water level data
# RS - 1/5/2017
# RS - 3/14/2018 - Update

# Raw data source/description: https://tidesandcurrents.noaa.gov/tide_predictions.html

# NOTE: raw csv files should have timestamps in LST

# Tide Variables (Continuous): TideMax, TideMin, TideR (tidal range = max - min), TideMax1, TideMin1, TideR1
# Tide Variables (Binary): TideGT_# (tide greater than # m), TideLT_# (tide less than # m),
# TideGT1_#, TideLT1_# (1 signifies previous day's value)

import pandas as pd
import os
import re

def gt(df, thresh):
    a = df > thresh
    a = a.astype('int')
    return a

def lt(df, thresh):
    a = df < thresh
    a = a.astype('int')
    return a

# Import raw data csv to pd DataFrame
infolder = 'Z:\Predictive Modeling\Phase III\Modeling\Data\Tides\Raw'
outfolder = 'Z:\Predictive Modeling\Phase III\Modeling\Summer_2018\Environmental Variables\Tides'
# outfolder = 'Z:\Predictive Modeling\Phase III\Modeling\Data\Tides\Variables\Daily'

sd = '20070101'  # Start date (account for previous day, conservatuve)
ed = '20171231'  # End date

# TO PROCESS SINGLE FILE
# (change indent)
# file = 'NorthSplit_Tidal_Data_20080101_20201231.csv'
# infile = os.path.join(infolder, file)

# TO ITERATE THROUGH ALL FILES
df_means = pd.DataFrame()
for file in os.listdir(infolder):
    if not file.endswith('.csv'):
        continue
    infile = os.path.join(infolder, file)
    station = re.sub('_.+', '', file)
    print('Processing tidal data for ' + station + ' ...')
    df_raw = pd.read_csv(infile)
    df_raw.columns = ['Date Time', 'Water Level (m)']
    #Convert Date Time to timestamp, set as index
    df_raw['Date Time'] = pd.to_datetime(df_raw['Date Time'])
    df_raw.set_index('Date Time',inplace=True)
    mht = float(round(df_raw.resample('D').max().mean(),4))  # Mean high and low tides from entire infile (2000-2020)
    mlt = float(round(df_raw.resample('D').min().mean(),4))

    df_raw = df_raw[sd:ed]  # Only samples in time range (for speed)
    df_out = pd.DataFrame(index=df_raw.resample('D').mean().index) # Preset index to days

    # Daily vars: for each day, find variables
    df_out['TideMax'] = df_raw.resample('D').max()  # Max tide
    df_out['TideMin'] = df_raw.resample('D').min()  # Min tide
    df_out['TideR'] = df_out.TideMax - df_out.TideMin  # Tidal range
    # TideMax = df_raw.resample('D').max()
    # TideMin = df_raw.resample('D').min()
    # TideR = TideMax.subtract(TideMin)
    # TideMax.columns = ['TideMax']
    # TideMin.columns = ['TideMin']
    # TideR.columns = ['TideR']
    gt_list = [2, 2.25, 2.5, mht]
    for g in gt_list:
        if g != mht:
            df_out['TideGT_' + str(g).replace('.','_')] = gt(df_out.TideMax,g) # Was Tide today greater than # ?
        else:
            df_out['TideGT_mht'] = gt(df_out.TideMax, g) # Was the tide greater than the mean high tide in the entire dataset?
    # df_out['TideGT_2'] = gt(df_out.TideMax, 2)  # Was Tide today greater than # ?
    # df_out['TideGT_2_25'] = gt(df_out.TideMax, 2.25)
    # df_out['TideGT_2_5'] = gt(df_out.TideMax, 2.5)
    # df_out['TideGT_mean'] = gt(df_out.TideMax, mht) # Was the tide greater than the mean high tide in the entire dataset?
    # TideGT_2 = gt(TideMax, 2)
    # TideGT_2_5 = gt(TideMax, 2.5)
    # TideGT_mean = gt(TideMax, mht) # Was the tide greater than the mean high tide in the dataset?
    # TideGT_2.columns = ['TideGT_2']
    # TideGT_2_5.columns = ['TideGT_2_5']
    # TideGT_mean.columns = ['TideGT_mean']
    lt_list = [0, -0.25, -0.5, mlt]
    for g in lt_list:
        if g != mlt:
            df_out['TideLT_' + str(g).replace('-','').replace('.', '_')] = lt(df_out.TideMin, g)  # Was Tide today less than # ?
        else:
            df_out['TideLT_mlt'] = lt(df_out.TideMin, g)  # Was the tide less than the mean low tide in the entire dataset?
    # df_out['TideLT_0'] = lt(df_out.TideMin, 0)  # Was Tide today less than # ?
    # df_out['TideLT_0_25'] = lt(df_out.TideMin, -0.25)
    # df_out['TideLT_0_5'] = lt(df_out.TideMin, -0.5)
    # df_out['TideLT_mean'] = lt(df_out.TideMin, mlt)  # Was the tide less than the mean low tide in the entire dataset?
    # TideLT_0 = lt(TideMin, 0)
    # TideLT_0_5 = lt(TideMin, -0.5)
    # TideLT_mean = gt(TideMin, mlt) # Was the tide lower than the mean low tide in the dataset?
    # TideLT_0.columns = ['TideLT_0']
    # TideLT_0_5.columns = ['TideLT_0_5']
    # TideLT_mean.columns = ['TideLT_mean']

    # Prev day vars; NOTE: only works if you are SURE that days are consecutive (no gaps)
    for c in list(df_out.columns):
        if 'GT' in c:
            df_out[c.replace('GT','GT1')] = df_out[c].shift(1)  # Previous Day's value timestamped to today
        elif 'LT' in c:
            df_out[c.replace('LT', 'LT1')] = df_out[c].shift(1)  # Previous Day's value timestamped to today
        else:
            df_out[c + '1'] = df_out[c].shift(1)  # Previous Day's value timestamped to today
    # TideMax1 = TideMax.shift(1)
    # TideMin1 = TideMin.shift(1)
    # TideR1 = TideR.shift(1)
    # TideGT1_2 = TideGT_2.shift(1)
    # TideGT1_2_5 = TideGT_2_5.shift(1)
    # TideGT1_mean = TideGT_mean.shift(1)
    # TideLT1_0 = TideLT_0.shift(1)
    # TideLT1_0_5 = TideLT_0_5.shift(1)
    # TideLT1_mean = TideLT_mean.shift(1)
    #
    # TideMax1.columns = ['TideMax1']
    # TideMin1.columns = ['TideMin1']
    # TideR1.columns = ['TideR1']
    # TideGT1_2.columns = ['TideGT1_2']
    # TideGT1_2_5.columns = ['TideGT1_2_5']
    # TideGT1_mean.columns = ['TideGT1_mean']
    # TideLT1_0.columns = ['TideLT1_0']
    # TideLT1_0_5.columns = ['TideLT1_0_5']
    # TideLT1_mean.columns = ['TideLT1_mean']
    #
    # # Combine all vars into dataframe; export to csv; export means to separate file
    # df_out = pd.concat([TideMax, TideMin, TideR,
    #                   TideGT_2, TideGT_2_5, TideGT_mean,
    #                   TideLT_0,TideLT_0_5,TideLT_mean,
    #                   TideMax1, TideMin1, TideR1,
    #                   TideGT1_2, TideGT1_2_5, TideGT1_mean,
    #                   TideLT1_0,TideLT1_0_5,TideLT1_mean
    #                  ],axis = 1)

    # Save to file
    of_name = station + '_Tide_Variables_' + sd + '_' + ed + '.csv'
    outfile = os.path.join(outfolder, of_name)
    df_out.index.rename('date', inplace=True)
    df_out.to_csv(outfile)

    means = {'mean_High': mht, 'mean_Low': mlt}
    df_means = df_means.append(pd.DataFrame(means, index=[station]))
    print('Variables written to ' + outfile)

# Save means for each station
mf_name = 'tidal_means.csv'
mean_file = os.path.join(outfolder, mf_name)
df_means.index.rename('station', inplace=True)
df_means.to_csv(mean_file)
print('Tidal means written to ' + mean_file)
