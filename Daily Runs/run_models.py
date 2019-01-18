#! python3
# run_models_2018.py - Runs NowCast models
# RTS - 11/1/2017, UPDATE: 5/11/2018

# TODO Official county rain advisory automation

# 1. Collects environmental data and stores it in a dated file
# 2. Using stored data, compute variables and run models
# 3. Save results to a dated csv. Also save a dated upload file for the current BRC site

# NOTE: Email and website API credentials in the below functions have been redacted - RTS 01/17/2019

import pandas as pd
import numpy as np
from math import isnan, pi
import os
import re
import sys 
import requests
import datetime
import json
from sklearn.externals import joblib
import smtplib
from string import Template
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class Tee:  # For logging
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for fil in self.files:
            fil.write(obj)

    def flush(self):
        pass


def getWaves(day, st):  # Import CDIP Wave Data
    yest = day - datetime.timedelta(days=1)
    yesterday = yest.strftime('%Y%m%d')
    # yest_in = yest.strftime('%Y-%m-%d')
    today = day.strftime('%Y%m%d')
    today_in = day.strftime('%Y-%m-%d')

    # date range using CDIP NDAR, grab wave parameters, no header
    if len(st) != 3:
        st = '0' * (3 - len(st)) + st
    url = 'http://cdip.ucsd.edu/data_access/ndar?' + st + '+pm+' + yesterday + '-' + today
    web = requests.get(url)

    try:
        web.raise_for_status()
    except Exception as exc:
        print(' ERROR (There was a problem getting wave data with the available URL: %s)' % exc)
        # return 0

    # Create DataFrame, index with timestamp, convert blanks/errors to NaN #
    data = [line.split() for line in web.text.splitlines()]
    data = data[:-1]  # exclude footer
    data[0][0] = data[0][0].replace('<pre>', '')  # remove header
    df = pd.DataFrame(data)
    df.columns = ['year', 'month', 'day', 'hour', 'minute', 'WVHT', 'DPD', 'MWD', 'APD', 'Wtemp_B']
    df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
    df.set_index('date', inplace=True)
    df.index = df.index.shift(-8, freq='h')  # convert from UTC to PST (- 8 hrs)
    df = df[['WVHT', 'DPD', 'MWD', 'APD', 'Wtemp_B']]
    df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

    # Create DF for daily variables
    wave_round = {'WVHT': 2, 'DPD': 2, 'MWD': 0, 'APD': 2, 'Wtemp_B': 1}  # sig figs on CDIP
    df_vars = pd.DataFrame(index=df.resample('D').mean().index)
    for c in df.columns:
        r = wave_round[c]
        df_vars[c + '1'] = round(df[c].resample('D').mean().shift(1, freq='D'), r)  # previous day mean
        df_vars[c + '1_max'] = df[c].resample('D').max().shift(1, freq='D')  # previous day max
        df_vars[c + '1_min'] = df[c].resample('D').min().shift(1, freq='D')  # previous day min
    df_vars = df_vars[today_in:today_in]

    return df_vars


def getRad(day, st):  # Solar Radiation Data from CIMIS
    # http://www.cimis.water.ca.gov/Default.aspx
    api_key = 'XXXX'
    units = 'E'  # 'E' English, 'M' Metric
    ed = day
    sd = day - datetime.timedelta(days=1)

    url = 'http://et.water.ca.gov/api/data?' \
          + 'appKey=' + api_key \
          + '&targets=' + str(st) \
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
    return df_out['rad1'].to_frame()[day:day]


def getFlow(day, st):  # Access USGS Flow Data
    yest = day - datetime.timedelta(days=1)  # day must be in datetime format
    yesterday = yest.strftime('%Y-%m-%d')
    day3 = day - datetime.timedelta(days=3)  # day must be in datetime format
    day3_str = day3.strftime('%Y-%m-%d')

    url = 'http://waterdata.usgs.gov/nwis/dv?cb_00060=on&format=rdb&site_no=' + str(st) + \
          '&referred_module=sw&period=&begin_date=' + day3_str + '&end_date=' + yesterday

    web = requests.get(url)
    try:
        web.raise_for_status()
    except Exception as exc:
        print(' ERROR (There was a problem grabbing flow data: %s)' % exc)

    data = [line.split() for line in web.text.splitlines()]
    while data[0][0].startswith('#'):  # delete comments from list
        del data[0]

    df = pd.DataFrame(data, columns=data[0]).drop([0, 1])  # Delete headers
    df = df[list(df.columns[2:])]  # Grab datetime, flow, and qualifier
    df.columns = ['date', 'flow', 'qual']
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # df_vars = pd.DataFrame(index=df.index)
    df_vars = pd.DataFrame(index=pd.DatetimeIndex(start=day3, end=yest, freq='D'))
    df_vars.index.rename('date', inplace=True)
    for i in range(0, 3):  # logflow1 - logflow3
        df_vars['logflow' + str(i + 1)] = round(np.log10(df['flow'].shift(i, freq='D').astype(float)), 5)
        df_vars['logflow' + str(i + 1)][np.isneginf(df_vars['logflow' + str(i + 1)])] = round(np.log10(0.005), 5)
    df_vars = df_vars.set_index(df_vars.index.shift(1, freq='D'))
    # NOTE: shift index after, diff than flowVarsDaily because same day not avail. on USGS'
    # fill any NA with mean of the other day's values
    return df_vars[day:day].fillna(value=round(float(df_vars[day:day].mean(axis=1)), 5))  # df_vars[day:day]


def getCOOP(day, st):  # Get NOAA CO-OP met data (temp_L, wspd_L, Wtemp_L, pres_L)
    bd = day - datetime.timedelta(days=1)
    # ed = day
    units = 'metric'  # Temp = C, Speed = m/s, Pressure = mbar
    time_zone = 'lst'  # Local Standard Time (ignore DLS)
    product = ['air_temperature', 'water_temperature', 'wind', 'air_pressure']
    format = 'json'

    # Get Raw CO-OPs
    df_raw = pd.DataFrame()
    for p in product:
        url = 'http://tidesandcurrents.noaa.gov/api/datagetter?' + \
              'begin_date=' + bd.strftime('%Y%m%d') + \
              '&range=25' + \
              '&station=' + st + \
              '&product=' + p + \
              '&units=' + units + \
              '&time_zone=' + time_zone + \
              '&format=' + format + \
              '&application=heal_the_bay'
        # Range = 25 ensures today's date ends up in call

        web = requests.get(url)
        try:
            web.raise_for_status()
        except Exception as exc:
            print(' ERROR (There was a problem with the NOAA URL: %s)' % exc)
            # return 0

        data = json.loads(web.text)
        try:
            data = data['data']
        except KeyError:  # If no data for this variable is found
            continue

        df = pd.DataFrame.from_dict(data)
        # df = df.append(pd.DataFrame.from_dict(data), ignore_index=True)

        if len(df) > 0:
            if p != 'wind':
                df = df[['t', 'v']]
                df.columns = ['date', p]
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            else:
                df = df[['t', 's', 'd']]
                df.columns = ['date', 'wspd', 'wdir']
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            df = df.apply(pd.to_numeric)
            df_raw = df_raw.join(df, how='outer')

    # CO-OPS Vars
    params_dict = {
        'temp_L': 'air_temperature',
        'Wtemp_L': 'water_temperature',
        'wspd_L': 'wspd',
        'wdir_L': 'wdir',
        'pres_L': 'air_pressure'
    }

    params = ['temp_L', 'Wtemp_L', 'wspd_L', 'wdir_L', 'pres_L']
    params = [p for p in params if params_dict[p] in df_raw.columns]  # Some do not have all params
    df_raw.columns = params

    df_vars = pd.DataFrame(index=df_raw.resample('D').mean().index)  # Preset index to days
    for p in params:
        df_vars[p + '1'] = round(df_raw[p].resample('D').mean().shift(1, freq='D'), 1)  # previous day mean
        df_vars[p + '1_max'] = df_raw[p].resample('D').max().shift(1, freq='D')  # previous day max
        df_vars[p + '1_min'] = df_raw[p].resample('D').min().shift(1, freq='D')  # previous day min

    return df_vars[day:day]


def getMet(day, met_folder, df_stations):  # Grab Weather data through most recent METARs
    # https://aviationweather.gov/
    yest = day - datetime.timedelta(days=1)
    sd = (day - datetime.timedelta(days=5)).strftime('%Y-%m-%d')
    ed = (day + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    all_st = ','.join([s for s in df_stations['station']])

    # Grab New Met Data
    url = 'https://www.aviationweather.gov/adds/dataserver_current/httpparam?datasource=metars' \
          '&requesttype=retrieve&format=csv' \
          '&startTime=' + sd + 'T00:00:00-0800' \
                               '&endTime=' + ed + 'T00:00:00-0800' \
                                                  '&stationString=' + all_st  # All airports at once

    web = requests.get(url)
    try:
        web.raise_for_status()
        data = [line.split(',') for line in web.text.splitlines()]
        while 'raw_text' not in data[0]:  # delete comments from list
            del data[0]
        df = pd.DataFrame(data[1:], columns=data[0])
        df['date'] = pd.to_datetime(df['observation_time'])
        df.set_index('date', inplace=True)
        df.index = df.index.shift(-8, freq='h')  # convert from UTC to PST (- 8 hrs)

        # df = df[df['metar_type'] == 'METAR']
        cols = ['temp_c', 'dewpoint_c', 'wind_speed_kt',
                'wind_dir_degrees', 'sea_level_pressure_mb', 'precip_in']
        df = df[['station_id', 'raw_text'] + cols]
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        df['wind_speed_kt'] *= 0.5144  # Knots to m/s
        df['precip_in'] *= 25.4  # In to mm
        df.loc[df['wind_dir_degrees'] == 0, 'wind_dir_degrees'] = np.nan  # Account for NaN values, North is 360
        df.columns = ['station_id', 'raw_text', 'temp', 'dtemp', 'wspd', 'wdir', 'pres', 'rain']

    except Exception as exc:
        e = sys.exc_info()
        print('* Data Collection Error: There was a problem grabbing met data: %s' % exc)
        dc_error_list.append('Data Collection Error - Grabbing Met Data [' + str(e) + ']')
        return

    # Create station variables
    rounder = {'temp': 1,  # sig figs
               'dtemp': 1,
               'pres': 1,
               'wspd': 1,
               'wdir': 0,
               'rain': 1}
    for s in list(df_stations.index):  # Process met data for each station
        st = df_stations.loc[s]['station']
        print('    ' + s + ' (' + st + ') - ', end='')

        # Import existing raw station data
        met_data_file = s.replace(' ', '_') + '_raw_met_data.csv'
        df_old_daily = pd.read_csv(os.path.join(met_folder + '\\raw', met_data_file))
        df_old_daily['date'] = pd.to_datetime(df_old_daily['date'])
        df_old_daily.set_index('date', inplace=True)

        try:
            # New raw data into variables
            df_raw = df[df['station_id'] == st].drop(['station_id', 'raw_text'], axis=1)
            df_raw = df_raw.resample('H').last()  # Get last observation of each hour (sometimes labelled as SPECI)

            df_daily = pd.DataFrame(index=df_raw.resample('D').mean().index)  # New daily data
            df_vars = pd.DataFrame(index=df_raw.resample('D').mean().index)  # New daily vars

            for c in df_raw.columns:
                if c != 'rain':
                    df_daily[c] = round(df_raw[c].resample('D').mean(), rounder[c])  # Mean
                    df_vars[c + '1'] = df_daily[c].shift(1, freq='D')
                    if c != 'wdir':
                        df_daily[c + '_max'] = round(df_raw[c].resample('D').max(), rounder[c])  # Max
                        df_vars[c + '1_max'] = df_daily[c + '_max'].shift(1, freq='D')

                        df_daily[c + '_min'] = round(df_raw[c].resample('D').min(), rounder[c])  # Min
                        df_vars[c + '1_min'] = df_daily[c + '_min'].shift(1, freq='D')

            df_daily['rain'] = round(df_raw['rain'].resample('D').sum().fillna(0), rounder['rain'])  # rain (sum)
            # rain_6h - First 6 hours of rain
            rr = 4  # lograin rounder
            df_vals_6h = df_raw[(df_raw.index.hour == 0) | (df_raw.index.hour == 1) | (df_raw.index.hour == 2) |
                                (df_raw.index.hour == 3) | (df_raw.index.hour == 4) | (df_raw.index.hour == 5)]
            df_daily['rain_6h'] = df_vals_6h['rain'].resample('D').sum().fillna(0)
            df_vars['lograin_6h'] = round(np.log10(df_daily['rain_6h']), rr)
            df_vars['lograin_6h'][np.isneginf(df_vars['lograin_6h'])] = round(np.log10(0.005), rr)

            # Previous day rain vars  Combine old met data with new data for the previous day
            df_daily = df_old_daily.append(df_daily[yest:day])
            df_daily = df_daily[~df_daily.index.duplicated(keep='last')].sort_index()

            for i in range(1, 8):  # rain1 - rain7, lograin1-lograin7
                df_daily['rain' + str(i)] = df_daily['rain'].shift(i, freq='D')
                df_vars['lograin' + str(i)] = round(np.log10(df_daily['rain' + str(i)]), rr)
                df_vars['lograin' + str(i)][np.isneginf(df_vars['lograin' + str(i)])] = round(np.log10(0.005), rr)

            total_list = list(range(2, 8)) + [14, 30]
            for j in total_list:  # rain2T-rain7T
                df_daily['rain' + str(j) + 'T'] = 0.0
                for k in range(j, 0, -1):
                    df_daily['rain' + str(j) + 'T'] += df_daily['rain'].shift(k, freq='D')
                df_vars['lograin' + str(j) + 'T'] = round(np.log10(df_daily['rain' + str(j) + 'T']), rr)
                df_vars['lograin' + str(j) + 'T'][np.isneginf(df_vars['lograin' + str(j) + 'T'])] = round(
                    np.log10(0.005), rr)

            # Save raw data file
            df_daily.drop(day, inplace=True)  # drop today's data (no complete in morning)
            df_daily.to_csv(os.path.join(met_folder + '\\raw', met_data_file))

            # Save var file
            met_var_file = s.replace(' ', '_') + '_met_data.csv'
            df_vars = df_vars[day:day]
            if met_var_file in os.listdir(met_folder):  # If file already exists, append new data to old data
                df_old_vars = pd.read_csv(os.path.join(met_folder, met_var_file))
                df_old_vars['date'] = pd.to_datetime(df_old_vars['date'])
                df_old_vars.set_index('date', inplace=True)

                df_combo_vars = df_old_vars.append(df_vars)
                df_combo_vars = df_combo_vars[~df_combo_vars.index.duplicated(keep='last')].sort_index()
                # Remove duplicate indices
                df_combo_vars.to_csv(os.path.join(met_folder, met_var_file))
            else:
                df_vars.to_csv(os.path.join(met_folder, met_var_file))
            print(' COMPLETE')

        except Exception as exc:
            print(' ERROR (There was a problem creating met variables: %s)' % exc)
            e = sys.exc_info()
            dc_error_list.append('Data Collection Error - Met Stations Data (' + s + ') [' + str(e) + ']')
            continue

    return


def angle_vars(df, ang, station_type):  # Calculate angle dependant enviro variables
    if station_type == 'met':
        if 'wspd1' in df.columns and 'wdir1' in df.columns:  # wind speed/direction
            df['awind1'] = df['wspd1'] * np.sin(((df['wdir1'] - ang) / 180) * pi)
            df['owind1'] = df['wspd1'] * np.cos(((df['wdir1'] - ang) / 180) * pi)
    elif station_type == 'coop':
        if 'wspd_L1' in df.columns and 'wdir_L1' in df.columns:  # local wspd/wdir
            df['awind_L1'] = df['wspd_L1'] * np.sin(((df['wdir_L1'] - ang) / 180) * pi)
            df['owind_L1'] = df['wspd_L1'] * np.cos(((df['wdir_L1'] - ang) / 180) * pi)
    elif station_type == 'wave':
        for w in ['MWD1', 'MWD1_max', 'MWD1_min']: # Wave direction
            if w in df.columns:
                df['MWD1_b' + w.replace('MWD1', '')] = df[w] - ang  # Direction relative to beach
                df['SIN_MWD1_b' + w.replace('MWD1', '')] = \
                    round(np.sin(((df['MWD1_b' + w.replace('MWD1', '')]) / 180) * pi), 5)
                if w == 'MWD1':
                    df['q_dir1'] = df['SIN_MWD1_b'].apply(qd)  # Longshore current direction
        df.drop(['MWD1', 'MWD1_max', 'MWD1_min'], axis=1, inplace=True)
    return df


def tide_vars_hourly(df, hourly_folder, b):  # Calculates hourly tide variables
    try:
        hourly_file = [rf for rf in os.listdir(hourly_folder) if rf.startswith(b.replace(' ', '_'))][0]
    except:
        print('No hourly tide file for ' + station + ' found...skipping hourly tide variables')
        return df

    df_raw = pd.read_csv(os.path.join(hourly_folder, hourly_file))
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_raw = df_raw.set_index('date')
    df = df.join(df_raw, how='left')

    return df


def qd(x):  # Current direction variable
    if x == 0:  # or isnan(x):
        return 0
    elif isnan(x):
        return np.nan
    elif x > 0:
        return -1
    else:
        return 1  # if sin < 0, - sin is > 0, current is +


# def rain_advisory():
#     rain_file = base_folder + '\weather\\rain\\' + b.replace(' ', '_') + '_rain_advisory.csv'
#     df_ra = pd.read_csv(rain_file, header=None, index_col=0)
#
#     if i in ['Venice (Windward Ave)', 'Manhattan (28th St)']:
#         try:
#             url = 'http://ladpw.org/wrd/precip/alert_rain/index.cfm?cont=72hr.cfm'
#             web = requests.get(url)
#             web.raise_for_status()
#             soup = bs4.BeautifulSoup(web.text, 'html.parser')
#             gauge = float(soup.find(title="375").getText())
#
#             if gauge > 0.1:
#                 print(' Rain gauge for LA County shows advisory')
#                 df_ra.loc['advisory'] = 1
#                 df_ra.loc['date'] = date_str
#             else:
#                 df_ra.loc['advisory'] = 0
#
#             df_ra.to_csv(rain_file, header=False)
#         except:
#             e = sys.exc_info()
#             print('LA Rain Gauge Error: ' + str(e))
#             error_list.append(i + ' - LA Rain Gauge Error: ' + str(e))
#
#     adv_date = str(df_ra.loc['date'].values[0])
#     adv_date_ts = pd.to_datetime(adv_date)
#
#     if (pd.to_datetime(date) < adv_date_ts + pd.to_timedelta('72 hours')) and (int(df_ra.loc['advisory']) == 1):
#         # advisory = 1 and  date < 72 hours past advisory -> rain advisory in effect
#         print(' ' + i + ' is under a rain advisory (as per county on ' + adv_date + ')')
#         return 1
#     else:  # advisory = 0 -> not in rain advisory
#         print(' ' + i + ' is not under a rain advisory')
#         return 0


def send_results_log(table, log_file_path, date, error_list, backup=0):
    # Results
    df_test = table
    df_html = df_test.to_html(justify='left')
    df_html = re.sub('<th></th>', '<th>Beach</th>', df_html)
    df_html = df_html.replace('<tr>', '<tr style="text-align: left;">')  # Left justify all cells

    # Date
    date_str = date

    # Errors
    if len(error_list) > 0:
        error_str = '  - ' + '<br>  - '.join(error_list)
    else:
        error_str = '<u>There were no errors during modeling.</u>'

    login_usr = 'XXXX'  # Credentials [don't post on github]
    login_pwd = 'XXXXX'

    template_file_html = base_folder + '\email\\qa_email_template_html.txt'

    # Initiate Email #
    msg = MIMEMultipart()  # create a message
    # setup the parameters of the message
    msg['From'] = login_usr
    if backup == 0:
        msg['To'] = login_usr + ', rsearcy@healthebay.org, lrieves@healthebay.org, lginger@healthebay.org'
    else:
        msg['To'] = login_usr + ', rsearcy@healthebay.org, lrieves@healthebay.org, lginger@healthebay.org, ' \
                                'hdolinh@healthebay.org, AWagner@healthebay.org'  # aboehm@stanford.edu,
    msg['Subject'] = '[UNPUBLISHED] NowCast Predictions - ' + date_str

    # Text
    # Read HTML template file
    temp_file = open(template_file_html)  # HTML Message
    temp = Template(temp_file.read())
    temp_file.close()
    html_message = temp.substitute(DATE=date_str, RESULTS=df_html, ERR=error_str)

    # Add in the message body and image
    lf = open(log_file_path, 'r')
    att = MIMEText(lf.read())
    att.add_header('Content-Disposition', 'attachment',
                   filename='run_log_' + date_str.replace('/', '') + '.txt')
    msg.attach(att)

    # Add in the message body and image
    msg.attach(MIMEText(html_message, 'html'))

    # Send the message
    s = smtplib.SMTP(host='smtp-mail.outlook.com', port=587)
    s.starttls()
    s.login(login_usr, login_pwd)
    s.send_message(msg)
    s.quit()


# SETUP
debug = 0  # 0 - production runs; 1 - debug mode (stop logging)
data_collect_only = 0  # 1 - use to only collect data and vars; 0 - run data collection and model
model_run_only = 0
backup = 0

date = datetime.date.today()  # - datetime.timedelta(days=1)
date_str = datetime.date.strftime(date, '%m/%d/%Y')

base_folder = 'S:\SCIENCE & POLICY\\NowCast\Daily files\\winter_2018_2019'
beaches_folder = os.path.join(base_folder, 'beaches')

# Beach list, metadata (model to use, angle, stations)
loc_file = os.path.join(beaches_folder, 'nowcast_beaches_winter_2018_2019.csv')
df_loc = pd.read_csv(loc_file)
df_loc.set_index('beach', inplace=True)
beach_list = list(df_loc.index)  # or list of specific beaches

# Initiate DataFrames, error list
df_decisions = pd.DataFrame()
df_pilot = pd.DataFrame()
df_pred_all = pd.DataFrame()
df_email = pd.DataFrame()
df_site = pd.DataFrame()
dc_error_list = []  # Data collection error list
model_error_list = []  # To collect model run errors, printed at the end of log

log_file_str = base_folder + '\\logs\\run_log_' + date_str.replace('/', '') + '.log'
if debug == 0:  # Log output if running (won't let you debug if logging)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    log_file = open(log_file_str, 'w')
    log_file_str = base_folder + '\\logs\\run_log_' + date_str.replace('/', '') + '.log'
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

np.seterr(divide='ignore')  # Stop Divide vy zero warnings
pd.set_option('display.expand_frame_repr', False)  # Don't limit print size

print('- - - | Running NowCast models for ' + date_str + ' | - - -\n')

# Grab All Environmental Data
enviro_stations = ['tide', 'wave', 'rad', 'flow', 'coop', 'met']
print('- - | Grabbing Recent Environmental Data | - -\n')
if model_run_only == 0:
    for es in enviro_stations:
        if es == 'tide':
            continue  # Tide data for the year already downloaded
        else:
            print('- Updating ' + es.capitalize() + ' Data:')

        es_folder = base_folder + '\data\\' + es
        df_es = pd.read_csv(os.path.join(es_folder, es + '_stations.csv'))  # enviro station DataFrame (stations list)
        df_es.set_index('name', inplace=True)  # Columns must be 'name' and 'station' only
        needed_stations = list(df_loc[es + '_station'].dropna().unique())  # All stations used by beaches in system

        if es != 'met':
            for ns in needed_stations:
                df_new_data = 0
                st = str(df_es.loc[ns]['station'])
                es_file = ns.replace(' ', '_') + '_' + es + '_data.csv'
                print('    ' + ns + ' (' + st + ') - ', end='')
                try:
                    # Open old data
                    if es_file in os.listdir(es_folder):  # If file already exists, will append new data to old data
                        df_old_data = pd.read_csv(os.path.join(es_folder, es_file))
                        df_old_data['date'] = pd.to_datetime(df_old_data['date'])
                        df_old_data.set_index('date', inplace=True)
                    else:
                        df_old_data = pd.DataFrame()

                    # Grab new data
                    if es == 'wave':
                        df_new_data = getWaves(date, st)
                    elif es == 'rad':
                        if date in df_old_data.index:  # Check if getRad script ran successfully earlier in the morning
                            if ~np.isnan(df_old_data.loc[date]['rad1']):  # If value is non-null
                                print(' COMPLETE (Earlier Run)')
                                continue
                            else:
                                df_old_data.drop(date, inplace=True)  # Drop NAN value for the day
                                df_new_data = getRad(date, st)
                        else:
                            df_new_data = getRad(date, st)
                    elif es == 'flow':
                        df_new_data = getFlow(date, st)
                    elif es == 'coop':
                        df_new_data = getCOOP(date, st)

                    # Save data if successful
                    df_combo_data = df_old_data.append(df_new_data)
                    df_combo_data = df_combo_data[~df_combo_data.index.duplicated(keep='first')].sort_index()
                    # Remove duplicate indices
                    df_combo_data.to_csv(os.path.join(es_folder, es_file))
                    print(' COMPLETE')

                except:
                    e = sys.exc_info()
                    print(' ERROR')
                    print('* Data Collection Error: ' + str(e))
                    dc_error_list.append('Data Collection Error - ' + es.capitalize() + ' Station '
                                                                                        '(' + ns + ') [' + str(e) + ']')
                    continue

        else:
            df_es = df_es[df_es.index.isin(needed_stations)]
            try:
                getMet(date, es_folder, df_es)

            except:
                e = sys.exc_info()
                print(' ERROR')
                print('* Data Collection Error: ' + str(e))
                dc_error_list.append('Data Collection Error: - Met Data [' + str(e) + ']')
                continue

    print('\n- - Environmental Data Collection Complete - -')
    print(' - Data Collection Errors: ' + str(len(dc_error_list)))

    if data_collect_only == 1:
        sys.exit()

# Run Models for Each Beach
print('\n\n- - | Making Predictions | - -')
var_folder = os.path.join(base_folder, 'data')
for b in beach_list:
    print('\n- - NowCast Predictions for ' + b + ' (Date: ' + date_str + ') - -')
    model_folder = os.path.join(beaches_folder, b.replace(' ', '_'))
    angle = int(df_loc.loc[b]['angle'])

    # Environmental Variables
    print('\n- Environmental Data Stations')
    df_vars = pd.DataFrame(index=[date])
    df_vars.index.rename('date', inplace=True)
    # Variable dataframe
    for s in enviro_stations:
        station = df_loc.loc[b][s + '_station']
        if pd.isnull(station):
            print('  ' + s.capitalize() + ': NA')
            continue
        print('  ' + s.capitalize() + ': ' + station)
        # Grab environmental data for given station type
        enviro_folder = os.path.join(base_folder + '\data\\', s)
        try:
            enviro_file = [x for x in os.listdir(enviro_folder) if x.startswith(station.replace(' ', '_'))][0]
        except IndexError as exc:
            e = sys.exc_info()
            model_error_list.append('Station Error: ' + str(e))
            print('    ERROR: No data for the following station: ' + station)
            continue
        df_enviro = pd.read_csv(os.path.join(enviro_folder, enviro_file))
        df_enviro['date'] = pd.to_datetime(df_enviro['date'])
        df_enviro.set_index('date', inplace=True)

        # For waves, met, coop, calculate angle variables
        if s in ['wave', 'met', 'coop']:
            df_enviro = angle_vars(df_enviro, angle, s)

        # Hourly variables
        if (type(df_loc.loc[b]['sample_time']) != float) and s in ['tide']:
            df_enviro = tide_vars_hourly(df_enviro, hourly_folder=os.path.join(enviro_folder, 'hourly'), b=b)

        df_vars = df_vars.join(df_enviro)

    # weekend1
    if date.isoweekday() in [6, 7, 1]:  # If yesterday was Friday, Saturday, or Sunday
        df_vars['weekend1'] = 1
    else:
        df_vars['weekend1'] = 1

    # laborday
    df_vars['laborday'] = int(date > datetime.date(2018, 9, 3))  # Has Labor Day passed [summer]

    # wet day ( If rain over last 72h > 0.1 in (2.54 mm)
    if (df_vars[['lograin_6h', 'lograin3T']] > 0.4048).any(axis=1).loc[date]:
        df_vars['wet'] = 1
        wet = 1
    else:
        df_vars['wet'] = 0
        wet = 0

    # Rain Advisory
    # print('Checking rain advisory status:')
    # try:
    #     ra = rain_advisory()  # raincheck
    # except:
    #     e = sys.exc_info()
    #     print('Rain Advisory Error: ' + str(e))
    #     error_list.append(i + ' - Rain Advisory Error: ' + str(e))
    #     ra = np.nan  # default to 'No Prediction' if rain advisory error [may need to change logic]
    #
    # all_vars.update({'rain_advisory': ra})

    # Save Variables
    var_folder = os.path.join(model_folder, 'variables')
    var_file = b.replace(' ', '_') + '_variables.csv'
    if var_file in os.listdir(var_folder):  # If file already exists, append new data to old data
        df_old_vars = pd.read_csv(os.path.join(var_folder, var_file))
        df_old_vars['date'] = pd.to_datetime(df_old_vars['date'])
        df_old_vars.set_index('date', inplace=True)

        df_combo_vars = df_old_vars.append(df_vars)
        df_combo_vars = df_combo_vars[~df_combo_vars.index.duplicated(keep='last')].sort_index()
        # Remove duplicate indices
        df_combo_vars.to_csv(os.path.join(var_folder, var_file))
    else:
        df_vars.to_csv(os.path.join(var_folder, var_file))
    print('Environmental variables for ' + date_str + ' saved to: ' + var_file)

    # Run models
    print('\n- Running Models')
    beach_decision_dict = {}  # Contains the posting decision for each FIB at the beach
    pred_dict = {}  # Contains FIB and their model prediction results
    site_post = 1  # For BRC, if 0, POST, if 1, NO POST
    missing_data = 0

    # if ra == 1:  # Rain Advisory
    #     print('\n- - - | RAIN ADVISORY for ' + i + ' - Predictions will not be released to the public. | - - -\n')
    #     beach_break = 1
    # if wet == 1:  # Wet Day
    #     print('\n- - - | WET DAY at ' + i + ' - Predictions will not be released to the public. | - - -\n')
    #     beach_break = 1
    # else:
    #     beach_break = 0
    beach_break = 0

    for f in ['FC', 'ENT']:
        if type(df_loc.loc[b][f + '_model']) == float:
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
            df_print = pd.DataFrame()

            for c in df_coef.index:  # Find all variables
                try:
                    var_value = float(df_vars[c])
                    df_print = df_print.append(pd.DataFrame({'Value': var_value}, index=[c]))
                except:
                    beach_break = 1

            print(df_coef.join(df_print))
            print('\nConstant: ' + str(round(constant, 3)))
            print(t + ': ' + str(round(tuner, 3)))

            df_day_vars = df_vars[model_vars]  # Locate only model variables
            try:
                if 'MLR-T' in model_file:
                    mod_result = 10 ** (tuner * lm.predict(df_day_vars.values.reshape(1, -1))[0])
                    print('\n' + f + ' (Predicted Concentration): ' + str(int(round(mod_result, 0))))

                    if f == 'TC':
                        thresh = 10000.0
                    elif f == 'FC':
                        thresh = 400.0
                    elif f == 'ENT':
                        thresh = 104.0

                elif 'BLR-T' in model_file:
                    lm.coef_ = lm.coef_.reshape(1, -1)
                    mod_result = lm.predict_proba(df_day_vars.values.reshape(1, -1))[0, 1]
                    print('\n' + f + ' (Predicted Probability): ' + str(round(mod_result, 3)))
                    thresh = tuner

            except ValueError:
                print('Some variables for ' + date_str +
                      ' not found in collected variable dataset. Run model manually')
                beach_break = 1
                pred_dict.update({f: np.nan})
                missing_data = 1
                model_error_list.append('Data Error - ' + b + ' - Missing Data')
                continue

            if mod_result > thresh:
                site_post = 0
                decision = 'Post'
            else:
                decision = 'No Post'

            print('Posting Decision: ' + decision + '\n')
            pred_dict.update({f: mod_result})

        # Add 'decision' for each FIB for each beach to dataframe for export to beach managers
        # elif ra == 1:
        #     beach_decision_dict.update({f: 'No Prediction (Rain Advisory)'})
        if wet == 1 and df_loc.loc[b]['wet_exclude'] == 0:
            beach_decision_dict.update({f: 'No Prediction (Wet Weather)'})
        elif beach_break == 0:
            beach_decision_dict.update({f: decision})
        else:
            beach_decision_dict.update({f: 'No Prediction'})

    # Post/ No Post Decisions #
    if df_loc.loc[b]['pilot'] != 1:
        df_decisions = df_decisions.append(pd.DataFrame(beach_decision_dict, index=[b]))

    # Email Dataframe #
    # if df_loc.loc[b]['pilot'] != 1:
    email_pred = 'No Prediction'
    email_comments = ''
    # if ra == 1:
    #     email_comments = 'Rain Advisory'
    if wet == 1 and df_loc.loc[b]['wet_exclude'] == 0:
        email_comments = 'Wet Weather'
    elif missing_data == 1:
        email_comments = 'Missing Environmental Data'
    elif 'Post' in beach_decision_dict.values():  # Post prediction
        email_pred = 'Post'
        email_comments = 'Triggered by ' + ', '.join(
            [k for k in beach_decision_dict.keys() if beach_decision_dict[k] == 'Post'])
    elif all(v == 'No Post' for v in beach_decision_dict.values()):  # No Post Prediction
        email_pred = 'No Post'

    df_email = df_email.append(pd.DataFrame({'Prediction': email_pred, 'Comments': email_comments},
                                            index=[b]))

    # BRC Site Decision Upload #
    if df_loc.loc[b]['brc_upload'] and beach_break == 0:
        if wet != 1 or df_loc.loc[b]['wet_exclude'] != 0:
            df_site = df_site.append(
                pd.DataFrame({'LocationID': df_loc.loc[b]['brc_upload_id'], 'NowcastPassFlag': site_post,
                              'NowcastDate': date_str}, index=[b]))

    # Prediction Update #
    df_pred = pd.DataFrame(pred_dict, index=[date_str])  # Save the day's predictions
    df_pred.index.rename('Date', inplace=True)
    df_pred.to_csv(model_folder + '\\predictions\\' + date_str.replace('/', '') + '_predictions.csv')

# After all models run
print('\n- - - | Finished daily run for ' + date_str + ' | - - -\n')

# Save decisions #
try:
    if len(df_decisions) > 0:
        # df_decisions = df_decisions[['TC', 'FC', 'ENT']]
        # Post/ No Post decisions #NOTE: 8/27/2018 - No TC results caused error in code 8/21/2018
        df_decisions.index.rename('Beach', inplace=True)
        df_decisions.to_csv(base_folder + '\\decisions\\NowCast_Predictions_' + date_str.replace('/', '') + '.csv')
except:
    e = sys.exc_info()
    print(' ERROR')
    print('* Save Decisions Error: ' + str(e))
    model_error_list.append('Save Decision Error: ' + str(e))

# Save Import File
try:
    if len(df_site) > 0:
        df_site = df_site[['LocationID', 'NowcastPassFlag', 'NowcastDate']]  # BRC upload file
        df_site.to_csv(base_folder + '\\import\\NowCast_Import_' + date_str.replace('/', '') + '.csv', index=False)
except:
    e = sys.exc_info()
    print(' ERROR')
    print('* Save Import Error: ' + str(e))
    model_error_list.append('Save Import Error: ' + str(e))

# Email predictions and log for QA
try:
    df_email['County'] = [df_loc.loc[b]['county'] for b in df_email.index]
    df_email = df_email[['County', 'Prediction', 'Comments']]

    df_email_pilot = df_email.loc[[b for b in df_email.index if df_loc.loc[b]['pilot'] == 1]]  # Not to be emailed
    df_email = df_email.loc[[b for b in df_email.index if df_loc.loc[b]['pilot'] != 1]]
    df_email_pilot.index = [df_loc.loc[b]['display_name'] for b in df_email_pilot.index]
    df_email.index = [df_loc.loc[b]['display_name'] for b in df_email.index]

    # Save Email Dataframe
    email_folder = base_folder + '\email\\tables\\'
    email_ext = 'NowCast_Email_Table' + date_str.replace('/', '') + '.csv'
    df_email.to_csv(os.path.join(email_folder, email_ext))
except:
    e = sys.exc_info()
    print(' ERROR')
    print('* Save Email Table Error: ' + str(e))
    model_error_list.append('Save Email Table Error: ' + str(e))

print('NowCast Predictions:')
print(df_email)
print('\nPilot Predictions:')
print(df_email_pilot)

error_list = dc_error_list + model_error_list
if len(error_list) > 0:
    print('\nErrors: \n')
    for e in error_list:
        print(e)
else:
    print('\n- - - No Errors - - -')

if debug == 0:
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    log_file.close()

# Email results and log for given day to QA check
send_results_log(df_email, log_file_str, date_str, error_list, backup=backup)
