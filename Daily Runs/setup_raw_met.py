# setup_raw_met.py - Download raw met data from NCDC to setup files for daily run (used at start of season)
# RTS - 3/21/2018; UPDATE: 10/31/2018

# Grabs hourly METAR data (meteorological data) from airport stations along the coast, saves to file in daily run folder
# Parses raw data into met variables

# NOTE: Similar setup to getMet_NCDC.py in Modeling directory

import pandas as pd
import numpy as np
import requests


# Inputs
outfolder = 'S:\SCIENCE & POLICY\\NowCast\Daily files\\winter_2018_2019\data\met\\'
outfolder_raw = outfolder + 'raw\\'
airport_file = outfolder + 'met_stations.csv'  # airport metadata CSV

sd = '2018-09-25'  # in YYYY-MM-DD format (account for previous day)
ed = '2019-03-31'  # account for 8hr UTC shift

save_vars = 1  # 1 - replace variables; 0 - don't

# Import Airport Stations
df_air = pd.read_csv(airport_file)
df_air.set_index('name', inplace=True)
air_list = list(df_air.index)  # or custom list on airport locations

print('Setup Meteorological Data\nRaw Directory: ' + outfolder_raw)
for a in air_list:
    print('\nProcessing meteorological data for ' + a + ' (' + df_air.loc[a]['station'] + ')')
    USAF = str(df_air.loc[a]['USAF'])
    WBAN = str(df_air.loc[a]['WBAN'])
    if len(WBAN) != 5:
        WBAN = '0'*(5-len(WBAN)) + WBAN
    st_id = USAF + WBAN
    url = 'https://www.ncdc.noaa.gov/access-data-service/api/v1/data?dataset=global-hourly'
    payload = {
        'startDate': sd,
        'endDate': ed,
        'stations': st_id,
        'format': 'json',
        'includeAttributes': 'false'
        }
    print('  Searching for raw data via NCDC')
    r = requests.get(url, params=payload)
    try:
        r.raise_for_status()
        df_raw = pd.DataFrame(r.json())
        df_raw = df_raw[df_raw['REPORT_TYPE'] == 'FM-15']
        print('   ' + str(len(df_raw)) + ' METAR records found')
        df_raw['dt'] = pd.to_datetime(df_raw['DATE']) - pd.to_timedelta('8 hours')  # UTC to PST (-8hr)
        df_raw.set_index('dt', inplace=True)
        sd_new = str(df_raw.index[0].date())
        ed_new = str(df_raw.index[-1].date())
        print('Min. Date - ' + sd_new + '\nMax Date - ' + ed_new)
        cols = ['NAME', 'STATION', 'CALL_SIGN', 'REM', 'TMP', 'DEW', 'SLP', 'WND', 'AA1']
        df_raw = df_raw[cols]

    except Exception as exc:
        print('  There was a problem grabbing met data: %s' % exc)
        continue

    # Process Raw METAR Data #
    SF = 10  # scaling factor
    df_raw = df_raw.resample('H').last()  # Resample by hour, selecting last value
    # Note: timestamp now shows exactly on the hour, values are from the last METAR of that hour
    #     Ex. 8:00 - values from 8:56
    # This means that for calculation of sums and means grouped by day, the 0 hour will be included for that day
    # For rain_6h (first 6h of rain for the day, values from hour 0 - 5 should be included)

    # Temperature (degC)
    df_raw['temp'] = df_raw['TMP'][df_raw['TMP'].notnull()].apply(lambda x: x.split(',')[0])
    df_raw['temp'] = pd.to_numeric(df_raw['temp'], errors='coerce')/SF
    df_raw['temp'][df_raw['temp'] > 100] = np.nan  # Account for 99999 values
    print('Temperature parsed')

    # Dew Point Temperature (degC)
    df_raw['dtemp'] = df_raw['DEW'][df_raw['DEW'].notnull()].apply(lambda x: x.split(',')[0])
    df_raw['dtemp'] = pd.to_numeric(df_raw['dtemp'], errors='coerce') / SF
    df_raw['dtemp'][df_raw['dtemp'] > 100] = np.nan  # Account for 99999 values
    print('Dew point temperature parsed')

    # Sea Level Pressure (mbar)
    df_raw['pres'] = df_raw['SLP'][df_raw['SLP'].notnull()].apply(lambda x: x.split(',')[0])
    df_raw['pres'] = pd.to_numeric(df_raw['pres'], errors='coerce') / SF
    df_raw['pres'][df_raw['pres'] > 1500] = np.nan  # Account for 99999 values
    print('Sea level pressure parsed')

    # Wind Direction (deg) and Speed (m/s)
    df_raw['wdir'] = df_raw['WND'][df_raw['WND'].notnull()].apply(lambda x: x.split(',')[0])  # wind direction
    df_raw['wdir'] = pd.to_numeric(df_raw['wdir'], errors='coerce')
    df_raw['wdir'][df_raw['wdir'] > 360] = np.nan  # Account for 999 values
    print('Wind direction parsed')

    df_raw['wspd'] = df_raw['WND'][df_raw['WND'].notnull()].apply(lambda x: x.split(',')[3])  # wind speed
    df_raw['wspd'] = pd.to_numeric(df_raw['wspd'], errors='coerce') / SF
    df_raw['wspd'][df_raw['wspd'] > 90] = np.nan
    print('Wind speed parsed')

    # Precipitation (mm)
    df_raw['rain'] = df_raw['AA1'][df_raw['AA1'].notnull()].apply(lambda x: x.split(',')[1])
    df_raw['rain'] = pd.to_numeric(df_raw['rain'], errors='coerce') / SF
    df_raw['rain'][df_raw['rain'].isnull()] = 0
    df_raw['rain'][df_raw['rain'] > 900] = np.nan
    print('Rain parsed')

    # Parameterize met data into variables
    df_vals = df_raw[['temp', 'dtemp', 'pres', 'wspd', 'wdir', 'rain']]  # Values dataframe
    rounder = {'temp': 1,  # sig figs
               'dtemp': 1,
               'pres': 1,
               'wspd': 1,
               'wdir': 0,
               'rain': 1}

    df_daily = pd.DataFrame(index=df_vals.resample('D').mean().index)
    df_vars = pd.DataFrame(index=df_vals.resample('D').mean().index)

    # Mean
    for c in df_vals.columns:
        if c != 'rain':
            df_daily[c] = round(df_vals[c].resample('D').mean(), rounder[c])
            df_vars[c + '1'] = df_daily[c].shift(1, freq='D')
    # Max
    for c in df_vals.columns:
        if c not in ['wdir', 'rain']:
            df_daily[c + '_max'] = df_vals[c].resample('D').max()
            df_vars[c + '1_max'] = df_daily[c + '_max'].shift(1, freq='D')

    # Min
    for c in df_vals.columns:
        if c not in ['wdir', 'rain']:
            df_daily[c + '_min'] = df_vals[c].resample('D').min()
            df_vars[c + '1_min'] = df_daily[c + '_min'].shift(1, freq='D')

    rr = 4  # rain rounder
    # rain_6h - First 6 hours of rain
    df_vals_6h = df_vals[(df_vals.index.hour == 0) | (df_vals.index.hour == 1) | (df_vals.index.hour == 2) | (
        df_vals.index.hour == 3) | (df_vals.index.hour == 4) | (df_vals.index.hour == 5)]
    df_daily['rain_6h'] = df_vals_6h['rain'].resample('D').sum()
    df_vars['lograin_6h'] = round(np.log10(df_daily['rain_6h']), rr)
    df_vars['lograin_6h'][np.isneginf(df_vars['lograin_6h'])] = round(np.log10(0.005), rr)

    # rain
    df_daily['rain'] = df_vals['rain'].resample('D').sum()
    for i in range(1, 8):  # rain1 - rain7, lograin1-lograin7 [rainfall on i-th previous day]
        df_daily['rain' + str(i)] = df_daily['rain'].shift(i, freq='D')
        df_vars['lograin' + str(i)] = round(np.log10(df_daily['rain' + str(i)]), rr)
        df_vars['lograin' + str(i)][np.isneginf(df_vars['lograin' + str(i)])] = round(np.log10(0.005), rr)

    total_list = list(range(2, 8)) + [14, 30]
    for j in total_list:  # rain2T-rain7T [rainfall totals]
        df_daily['rain' + str(j) + 'T'] = 0.0
        for k in range(j, 0, -1):
            df_daily['rain' + str(j) + 'T'] += df_daily['rain'].shift(k, freq='D')
        df_vars['lograin' + str(j) + 'T'] = round(np.log10(df_daily['rain' + str(j) + 'T']), rr)
        df_vars['lograin' + str(j) + 'T'][np.isneginf(df_vars['lograin' + str(j) + 'T'])] = round(np.log10(0.005), rr)

    # Save to file
    dailyfile = a.replace(' ', '_') + '_raw_met_data.csv'
    df_daily.index.rename('date', inplace=True)
    df_daily.to_csv(outfolder_raw + dailyfile)  # Save daily file
    print('\nDaily data saved to: ' + dailyfile)

    # Save vars
    if save_vars == 1:
        outfile = a.replace(' ', '_') + '_met_data.csv'
        df_vars.index.rename('date', inplace=True)
        df_vars.to_csv(outfolder + outfile)  # Save vars file
        print('Meteorological variables saved to: ' + outfile)
