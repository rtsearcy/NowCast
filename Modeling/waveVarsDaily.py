# waveVarsDaily.py - Computes daily wave variables for CDIP buoys using raw buoy data
# RTS - 3/15/2018

# NOTE: raw wave files should have timestamps in UTC -> TODO Need to convert here

# Variables included:
# 'WVHT', 'DPD', 'MWD', 'APD', 'Wtemp_B',

import pandas as pd
from math import sin, cos, pi, isnan
from datetime import datetime, time
import numpy as np
import os

def str2time(x):
    try:
        return pd.to_datetime(x)
    except ValueError:
        return x

def alongshore(x):
    if x == 0:# or isnan(x):
        return 0
    elif isnan(x):
        return np.nan
    elif x > 0:
        return -1
    else: return 1 # if sin < 0, - sin is > 0, current is +

# Inputs #
outfolder = 'Z:\Predictive Modeling\Phase III\Modeling\Summer_2018\Environmental Variables\Waves\\raw'
sd = '20080101'  # Must be YYYYMMDD format
ed = '20171101'  # Because UTC time, include extra day

# Import raw data csv to pd DataFrame
beach_angle = 235
file_raw = 'Santa_Monica_Raw_CDIP_Wave_Data20080101_20170331.csv'
infolder_raw = 'Z:\Predictive Modeling\Phase III\Modeling\Data\Waves\Raw' #No need to change

beach = 'Venice'
file_sam = beach + '_W_Vars.csv'
infolder_sam = 'Z:\Predictive Modeling\Phase III\Modeling\Winter_2017_2018\Beaches\\' + beach
outfolder = 'Z:\Predictive Modeling\Phase III\Modeling\Data\Waves\Variables'
outfolder_2 = infolder_sam

infile_raw = os.path.join(infolder_raw, file_raw)
infile_sam = os.path.join(infolder_sam, file_sam)
print('Processing wave data (' + file_raw + ') for ' + file_sam + ' samples ...')

df_raw = pd.read_csv(infile_raw)
base_params =['WVHT','DPD','MWD','APD','Wtemp_B']
cols = ['dt'] + base_params
df_raw.columns = cols
df_raw['dt'] = pd.to_datetime(df_raw['dt'])
df_raw = df_raw.set_index('dt')

df_sam = pd.read_csv(infile_sam)
names = [df_sam.columns[0],df_sam.columns[2]] # take the date and timeP column (imported as strings
df_sam = df_sam[names]
df_sam.columns = ['date','timeP']
df_sam['timeP'] = pd.to_timedelta(df_sam['timeP']) # hours minutes
df_sam['date'] = df_sam['date'].apply(str2time)
df_sam['date'] = df_sam['date'] + df_sam['timeP']
df_sam = df_sam['date']#use this df to filter through raw tide data
df_out = df_sam.copy()
df_out = df_out.to_frame()
df_out = df_out.set_index('date')

# 'WVHT', 'DPD', 'APD','MWD', 'Wtemp_B' (loop) + MWD_b : match nearest date #NOTE: WILL GRAB FROM MONTHS AWAY IF NEEDED
for i in base_params:
    df_raw_copy = df_raw.copy()
    df_out[i] = df_raw_copy[i].reindex(df_sam, method='nearest', tolerance= pd.tslib.Timedelta('2 days'))
    # Wave properties can change significantly in 2 days...limit search to two days

df_out['MWD_b'] = df_out['MWD'] - beach_angle

# COS & SIN : 'COS_MWD', 'SIN_MWD','COS_MWD_b','SIN_MWD_b' :
for i in ['MWD','MWD_b']:
    #df_out['COS_'+ i] = np.cos(((df_out[i]) / 180) * pi)
    df_out['SIN_'+ i] = np.sin(((df_out[i]) / 180) * pi)

# q_d - alongshore current
df_out['q_d'] = df_out['SIN_MWD_b'].apply(alongshore)

# WVHT_1/3/6/9/12 ( + all other params): hourly shifted variables
hr = [1,3,6,9,12]
for i in hr:
    for j in base_params:
        df_raw_copy = df_raw.shift(i, freq='h')  # shift DatetimeIndex i hrs forward
        df_out[j + '_' + str(i)] = df_raw_copy[j].reindex(df_sam, method='nearest', tolerance= pd.tslib.Timedelta('2 days'))
    df_out['MWD_b_' + str(i)] = df_out['MWD_' + str(i)] - beach_angle
    for k in ['MWD', 'MWD_b']:
        a = '_' + str(i)
        #df_out['COS_' + k + a] = np.cos(((df_out[k + a]) / 180) * pi)
        df_out['SIN_' + k + a] = np.sin(((df_out[k + a]) / 180) * pi)

    df_out['q_d' + a] = df_out['SIN_MWD_b' + a].apply(alongshore) # q_d_#

# Previous Day Average Variables
df_max = df_raw.resample('D').max().shift(1,freq='D') #previous day's max values
df_mean = df_raw.resample('D').mean().shift(1,freq='D') #previous day's mean values

#Mean
for i in base_params:
    df_mean_copy = df_mean.copy()
    df_out[i + '1'] = df_mean_copy[i].reindex(df_sam, method='nearest', tolerance= pd.tslib.Timedelta('2 days'))

df_out['MWD1_b'] = df_out['MWD1'] - beach_angle

for i in ['MWD1','MWD1_b']:
    #df_out['COS_'+ i] = np.cos(((df_out[i]) / 180) * pi)
    df_out['SIN_'+ i] = np.sin(((df_out[i]) / 180) * pi)

df_out['q_d1'] = df_out['SIN_MWD1_b'].apply(alongshore) #alongshore current for previous day

#Max
for i in ['WVHT','DPD','APD','Wtemp_B']:
    df_max_copy = df_max.copy()
    df_out[i + '1_max'] = df_max_copy[i].reindex(df_sam, method='nearest', tolerance= pd.tslib.Timedelta('2 days'))

# Export vars to spread sheet
outname = file_sam.replace('.csv','_Wave_Variables.csv')
out_file = os.path.join(outfolder, outname)
df_out.to_csv(out_file)
out_file2 = os.path.join(outfolder_2, outname)
df_out.to_csv(out_file2)
print('Wave variables written to ' + out_file2)
print('Missing Variables: ')
print(df_out.isnull().sum())