#daily_runs_2017.py - Runs Summer 2017 NowCast models
# RTS - 3/14/2017

# 1. Creates backups of previous files
# 2. Collects environmental data and stores it in a dated file
# 3. Using stored data, compute variables and run models
# 4. Save results to a dated csv. Also save a dated upload file for the current BRC site

import pandas as pd
import numpy as np
from math import isnan, log10, sin, cos, pi
import os
import sys
import csv
import requests
import datetime
import time
import json
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

class Tee: # for logging
    def __init__(self,*files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        pass

def qd(x): # longshore current
    if x == 0:# or isnan(x):
        return 0
    elif isnan(x):
        return np.nan
    elif x > 0:
        return -1
    else: return 1 # if sin < 0, - sin is > 0, current is +

def grabWU(day,st,angle): # grab met data from WU
    yest = day - datetime.timedelta(days=1)
    yesterday = yest.strftime('%Y%m%d')
    today = day.strftime('%Y%m%d')
    wu_api = 'XXXX' # Redacted

    url_y = 'http://api.wunderground.com/api/' + wu_api + '/history_' + yesterday + '/q/CA/%s.json' % (st)
    url_t = 'http://api.wunderground.com/api/' + wu_api + '/history_' + today + '/q/CA/%s.json' % (st)

    web_y = requests.get(url_y)  # yesterday's weather data
    try:
        web_y.raise_for_status()
    except Exception as exc:
        print('There was a problem connecting to WU: %s' % exc)
        return 0

    ## Yesterday's parameters
    weatherData_y = json.loads(web_y.text)
    sum_y = weatherData_y['history']['dailysummary']

    yw = {'temp1': float(sum_y[0]['meantempi']),
          'dtemp1': float(sum_y[0]['meandewpti']),
          'wspd1': float(sum_y[0]['meanwindspdi']),
          'wdir1': float(sum_y[0]['meanwdird']),
          'pres1': float(sum_y[0]['meanpressurei'])}  # No cloud cover
    wspd = yw['wspd1']
    wdir = yw['wdir1']
    yw.update({'awind1': wspd * sin(((wdir - angle) / 180) * pi),
               'owind1': wspd * cos(((wdir - angle) / 180) * pi)})

    precip_y_total = sum_y[0]['precipi']
    if precip_y_total == 'T':
        precip_y_total = 0.0
    else:
        precip_y_total = float(precip_y_total)
    yw.update({'rain1': precip_y_total})

    if yw['rain1'] > 0.0:
        yw.update({'lograin1': log10(yw['rain1'])})
    else:
        yw.update({'lograin1': log10(0.005)})

    weather_vars = yw

    ## rainH
    web_t = requests.get(url_t)  # today's weather data
    try:
        web_t.raise_for_status()
    except Exception as exc:
        print('There was a problem getting rainH data from WU: %s' % exc)
        return 0

    weatherData_t = json.loads(web_t.text)
    sum_t = weatherData_t['history']['dailysummary']
    precip_t_total = sum_t[0]['precipi']

    if precip_t_total == 'T':  # If rainfall today is listed as trace
        first_8_hr = 0.0
    elif float(precip_t_total) > 0:
        O = weatherData_t['history']['observations']
        first_8_hr = 0.0
        for i in range(0, len(O)):
            if float(O[i]['date']['hour']) > 7:
                break
            # if O[i]['date']['hour'] == O[i + 1]['date']['hour']:
            if 'METAR' not in O[i]['metar']:
                continue
            else:
                if O[i]['precipi'] != '-9999.0' and O[i]['precipi'] != '-9999.00' and O[i]['precipi'] != 'T':
                    first_8_hr += float(O[i]['precipi'])
    else:
        first_8_hr = 0.0

    if first_8_hr > 0.0:
        rainH = {'rainH_b': 1, 'lograinH': log10(first_8_hr)}
    else:
        rainH = {'rainH_b': 0, 'lograinH': log10(0.005)}

    weather_vars.update(rainH)
    return weather_vars #dictionaryF

def getFlow(beach, day): # grab flow data from USGS
    # Select USGS Station
    if beach == 'Cowell':
        site_no = '11161000' #SAN LORENZO
    elif beach == 'East':
        site_no = '11119745'
    elif beach == 'Doheny':
        site_no = '11047300'
    elif beach == 'Huntington (Brookhurst)':
        site_no = '11078000'

    # Access USGS Flow Data
    yest = day - datetime.timedelta(days=1) # day must be in datetime format
    yesterday = yest.strftime('%Y-%m-%d')
    day2 = day - datetime.timedelta(days=2) # day must be in datetime format
    day2_str = day2.strftime('%Y-%m-%d')
    day3 = day - datetime.timedelta(days=3) # day must be in datetime format
    day3_str = day3.strftime('%Y-%m-%d')

    url = 'http://waterdata.usgs.gov/nwis/dv?cb_00060=on&format=rdb&site_no=' + site_no + \
          '&referred_module=sw&period=&begin_date=' + day3_str + '&end_date=' + yesterday

    web = requests.get(url)
    try:
        web.raise_for_status()
    except Exception as exc:
        print('There was a problem getting flow data with the available URL: %s' % exc)
        return 0

    reader = csv.reader(web.text.splitlines(), delimiter='\t')
    data = list(reader)

    flow1 = np.nan
    flow2 = np.nan
    flow3 = np.nan
    for j in range(-1, -10, -1):
       if '' not in data[j]:
           if yesterday in data[j]:
               flow1 = float(data[j][3])
               if flow1 == 0.0:
                   flow1 = 0.005
           elif day2_str in data[j]:
               flow2 = float(data[j][3])
               if flow2 == 0.0:
                   flow2 = 0.005
           elif day3_str in data[j]:
               flow3 = float(data[j][3])
               if flow3 == 0.0:
                   flow3 = 0.005

    # Missing Day (only one)
    if np.isnan(flow2):
        flow2 = flow3
        print('   Flow from two days ago missing. Replacement flow rate used for flow2')
    if np.isnan(flow1):
        flow1 = flow2
        print('   Flow from yesterday missing. Replacement flow rate used for flow1')

    flow_var = {'logflow1': log10(flow1), 'logflow2': log10(flow2), 'logflow3': log10(flow3)}

    return flow_var

def getWaves(beach, day, samt): # grab wave data from CDIP
    # Select CDIP buoy
    if beach == 'Cowell':  # SANTA CRUZ
        st = '157' #PT SUR
        angle = 130
        sam_time = samt
    elif beach == 'Main (Boardwalk)':  # SANTA CRUZ
        st = '157' #PT SUR
        angle = 170
        sam_time = samt
    elif beach == 'Arroyo Burro':  # SANTA BARBARA
        st = '111'
        angle = 185
        sam_time = samt
    elif beach == 'East':  # SANTA BARBARA
        st = '111'
        angle = 150
        sam_time = samt
    elif beach == 'Santa Monica Pier': # SANTA MONICA
        st = '028'
        angle = 225
        sam_time = samt
    elif beach == 'Redondo Beach Pier':  # Redondo Pier
        st = '028'
        angle = 260
        sam_time = samt
    elif beach == 'Belmont Pier': # LONG BEACH
        st = '092' # San Pedro
        angle = 195
        sam_time = samt
    elif beach == 'Huntington (Brookhurst)': # Huntington
        st = '092' # San Pedro
        angle = 215
        sam_time = samt
    elif beach == 'Doheny': # ORANGE COUNTY
        st = '045' # Oceanside
        angle = 190
        sam_time = samt
    elif beach == 'Moonlight': # ORANGE COUNTY
        st = '100' # Oceanside
        angle = 255
        sam_time = samt

    ## Import CDIP Data ##
    yest = day - datetime.timedelta(days = 1)
    yesterday = yest.strftime('%Y%m%d')
    yest_in = yest.strftime('%Y-%m-%d')
    today = day.strftime('%Y%m%d')
    today_in = day.strftime('%Y-%m-%d')
    #web = requests.get('http://cdip.ucsd.edu/data_access/justdar.cdip?' + st + '+pm+' + yesterday + '+' + today) JUSTDAR
    web = requests.get('http://cdip.ucsd.edu/data_access/ndar?' + st + '+pm+' + yesterday + '-' + today + '+h') #NDAR

    try:
        web.raise_for_status()
    except Exception as exc:
        print('There was a problem getting wave data with the available URL: %s' % exc)
        return 0

    data = []
    D = web.text.splitlines()
    for line in D:
        row = line.split()
        data.append(row)
    data = data[2:-1]
    df = pd.DataFrame(data)
    df.columns = ['year','month','day','hour','minute','WVHT','DPD','MWD','APD','Wtemp_B' ]
    df['dt'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
    df.set_index('dt', inplace = True)
    #df.index = df.index.tz_localize('UTC')
    df = df[['WVHT','DPD','MWD','APD','Wtemp_B']]
    df = df.apply(lambda x: pd.to_numeric(x, errors= 'coerce'))

    # Yesterday average/max 3
    df_yest = df[yest_in + ' 08:00:00': today_in + ' 08:00:00']
    yest_mean = df_yest.mean()
    wave_round = {'WVHT':2,'DPD':2,'MWD':0,'APD':2,'Wtemp_B':1}
    for m in wave_round: ## ROUND TO SITE's PRECISION
        yest_mean.loc[m] = yest_mean.loc[m].round(wave_round[m])
    yest_mean = yest_mean.rename({'WVHT': 'WVHT1', 'DPD': 'DPD1', 'MWD': 'MWD1', 'APD': 'APD1', 'Wtemp_B': 'Wtemp_B1'})
    yest_mean.set_value('MWD1_b', yest_mean.loc['MWD1'] - angle)
    yest_mean.set_value('SIN_MWD1_b', np.sin(pi*(yest_mean.loc['MWD1_b']/180)))
    yest_mean.set_value('q_d1', qd(yest_mean['SIN_MWD1_b']))

    yest_max = df_yest.max()
    yest_max = yest_max.reindex(['WVHT', 'DPD', 'APD', 'Wtemp_B'])
    yest_max = yest_max.rename({'WVHT': 'WVHT1_max', 'DPD': 'DPD1_max','APD': 'APD1_max', 'Wtemp_B': 'Wtemp_B1_max'})

    # Today's data
    hourly_var = pd.Series()
    df_copy = df.copy() # shift DatetimeIndex i hrs forward
    h, m = sam_time.split(':')
    sam_time = datetime.time(int(h) + 8, int(m)).strftime('%H:%M') # Adjust for UTC time
    now = df_copy.index.get_loc(pd.Timestamp(today_in + ' ' + sam_time), method='nearest',  tolerance=pd.Timedelta('4 hours')) #pd.tslib.Timedelta
    print(' Grabbing wave parameters at CDIP buoy ' + st + ' for a sample time of ' + df_copy.index[now].strftime('%H:%M') + ' UTC')
    for j in df.columns:
        hourly_var.set_value(j, df_copy[j].iloc[now])
        if np.isnan(hourly_var[j]):
            hourly_var[j] = df_copy[j].iloc[now - 1]
    hourly_var.set_value('MWD_b', hourly_var.loc['MWD'] - angle)
    hourly_var.set_value('SIN_MWD_b', np.sin(pi*(hourly_var.loc['MWD_b']/180)))
    hourly_var.set_value('q_d', qd(hourly_var.loc['SIN_MWD_b']))

    hr = [1, 3, 6, 9, 12]
    for i in hr:
        df_copy = df.shift(i, freq='h')  # shift DatetimeIndex i hrs forward
        L = df_copy.index.get_loc(pd.Timestamp(today_in + ' ' + sam_time), method= 'nearest', tolerance= pd.Timedelta('3 hours'))
        for j in df.columns:
            hourly_var.set_value(j + '_' + str(i), df_copy[j].iloc[L])
            if np.isnan(hourly_var[j + '_' + str(i)]):
                hourly_var[j + '_' + str(i)] = df_copy[j].iloc[L - 1]

        hourly_var.set_value('MWD_b_' + str(i), hourly_var.loc['MWD_' + str(i)] - angle)
        hourly_var.set_value('SIN_MWD_b_' + str(i), np.sin(pi*(hourly_var.loc['MWD_b_' + str(i)]/180)))
        hourly_var.set_value('q_d_' + str(i), qd(hourly_var.loc['SIN_MWD_b_' + str(i)]))

    # Combine data and return
    wave = yest_mean.append(yest_max)
    wave = wave.append(hourly_var)
    wave = dict(wave)
    return wave

def getTides(beach, day, samt, base_folder): # get tide data from pre-downloaded tidal predictions from NOAA
    # Select tide station
    if beach == 'Cowell':  # SANTA CRUZ
        st = 'Monterey'
        sam_time = samt
    elif beach == 'Main (Boardwalk)':  # SANTA CRUZ
        st = 'Monterey'
        sam_time = samt
    elif beach == 'Arroyo Burro':  # SANTA BARBARA
        st = 'SantaBarbara'
        sam_time = samt
    elif beach == 'East':  # SANTA BARBARA
        st = 'SantaBarbara'
        sam_time = samt
    elif beach == 'Santa Monica Pier':  # SANTA MONICA
        st = 'SantaMonica'
        sam_time = samt
    elif beach == 'Redondo Beach Pier':  # Redondo Beach Pier
        st = 'SantaMonica'
        sam_time = samt
    elif beach == 'Belmont Pier':  # LONG BEACH
        st = 'LongBeach'
        sam_time = samt
    elif beach == 'LB City Beach (5th Place)':  # LONG BEACH
        st = 'LongBeach'
        sam_time = samt
    elif beach == 'Huntington (Brookhurst)':  # ORANGE COUNTY
        st = 'NewportBay'
        sam_time = samt
    elif beach == 'Doheny':  # ORANGE COUNTY
        st = 'NewportBay'
        sam_time = samt
    elif beach == 'Moonlight':  # San Diego
        st = 'LaJolla'
        sam_time = samt

    tide_file = base_folder + 'tides\\' + st + '_Tidal_Data_20170101_20171231.csv'
    df = pd.read_csv(tide_file)
    df.columns = ['dt','Tide']
    df['dt'] = pd.to_datetime(df['dt'])
    df.set_index('dt', inplace= True)

    # Today average/max values #
    today_in = day.strftime('%Y-%m-%d')
    df_temp = df[today_in]
    t_max = float(df_temp.max())
    t_min = float(df_temp.min())
    t_r = t_max - t_min
    t_gt2 = int(t_max > 2)
    t_gt2_5 = int(t_max > 2.5)
    t_lt0 = int(t_min < 0)
    t_lt0_5 = int(t_max < -0.5)
    today_vars = {'TideMax': t_max, 'TideMin': t_min, 'TideR': t_r,
                  'TideGT_2': t_gt2, 'TideGT_2_5': t_gt2_5,
                  'TideLT_0': t_lt0, 'TideLT_0_5': t_lt0_5}

    # Previous day avg/ max values #
    yest = day - datetime.timedelta(days=1)  # no need to adjust, tide data in PST
    yest_in = yest.strftime('%Y-%m-%d')
    df_temp = df[yest_in]
    t_max = float(df_temp.max())
    t_min = float(df_temp.min())
    t_r = t_max - t_min
    t_gt2 = int(t_max > 2)
    t_gt2_5 = int(t_max > 2.5)
    t_lt0 = int(t_min < 0)
    t_lt0_5 = int(t_max < -0.5)
    yest_vars = {'TideMax1': t_max, 'TideMin1': t_min, 'TideR1': t_r,
                  'TideGT1_2': t_gt2, 'TideGT1_2_5': t_gt2_5,
                  'TideLT1_0': t_lt0, 'TideLT1_0_5': t_lt0_5}

    # Hourly related values #
    hourly_vars = {}
    now = df.index.get_loc(pd.Timestamp(today_in + ' ' + sam_time), method='nearest',
                              tolerance=pd.Timedelta('1 hours'))
    t_now = float(df.iloc[now])
    hourly_vars.update({'Tide': t_now})
    hr = [1,3,6,9,12]
    for i in hr:
        df_copy = df.shift(i, freq='h')  # shift DatetimeIndex i hr forward
        L = df_copy.index.get_loc(pd.Timestamp(today_in + ' ' + sam_time), method='nearest',
                               tolerance=pd.Timedelta('1 hours'))
        hourly_vars.update({'Tide_' + str(i): float(df_copy.iloc[L])})
        hourly_vars.update({'dTide_' + str(i): t_now - float(df_copy.iloc[L])})

    tide_vars = {}
    tide_vars.update(today_vars)
    tide_vars.update(yest_vars)
    tide_vars.update(hourly_vars)
    return tide_vars

def getWeather(beach, day, base_folder): # grab WU and stored weather data
    # Select weather station
    if beach == 'Cowell':  # SANTA CRUZ
        st = 'Watsonville'
        angle = 130

    elif beach == 'Main (Boardwalk)':  # SANTA CRUZ
        st = 'Watsonville'
        angle = 170

    elif beach == 'Arroyo Burro':  # SANTA BARBARA
        st = 'Santa_Barbara'
        angle = 185

    elif beach == 'East':  # SANTA BARBARA
        st = 'Santa_Barbara'
        angle = 150

    elif beach == 'Santa Monica Pier':  # SANTA MONICA
        st = 'Santa_Monica'
        angle = 225

    elif beach == 'Redondo Beach Pier':  # SANTA MONICA
        st = 'Los_Angeles_International'
        angle = 260

    elif beach == 'Belmont Pier':  # LONG BEACH
        st = 'Long_Beach'
        angle = 195

    elif beach == 'Mothers (LB)':  # LONG BEACH
        st = 'Long_Beach'
        angle = 195

    elif beach == 'LB City Beach (5th Place)':  # LONG BEACH
        st = 'Long_Beach'
        angle = 180

    elif beach == 'Huntington (Brookhurst)':  # ORANGE COUNTY
        st = 'Santa_Ana'
        angle = 215

    elif beach == 'Doheny':  # ORANGE COUNTY
        st = 'Santa_Ana'
        angle = 190

    elif beach == 'Moonlight':  # ORANGE COUNTY
        st = 'Carlsbad'
        angle = 255

    weather_file = base_folder + 'weather\\' +  beach.replace(' ','_') + '_weather.csv'

    weather_vars = grabWU(day, st, angle) #Get today's variables (rain1, temp1, wspd1, etc.)

    # Rain variables - pull from existing spreadsheet #
    df_weather = pd.read_csv(weather_file)
    df_weather['PDT'] = pd.to_datetime(df_weather['PDT'])
    df_weather.set_index('PDT', inplace= True)

    rain_30_t = weather_vars['rain1'] # rain30T ###REQUIRES 30 days of previous data
    for k in range(1,30):
        day_in = (day - datetime.timedelta(days=k)).strftime('%Y-%m-%d')
        if day_in in df_weather.index:
            L = int(df_weather.index.get_loc(day_in))
        else:
            print(' Weather data for ' + day_in + ' unavailable. Waiting one minute, then collecting past weather data...')
            time.sleep(60)
            df_weather.loc[pd.Timestamp(day_in)] = np.nan
            new_weather = grabWU(datetime.datetime.strptime(day_in,'%Y-%m-%d'),st,angle)
            for key in new_weather:
                df_weather[key].loc[day_in] = new_weather[key]
            L = int(df_weather.index.get_loc(day_in))
        df_temp = df_weather.iloc[L]
        rain_30_t += float(df_temp['rain1'])
    weather_vars.update({'rain30T': rain_30_t})
    if rain_30_t > 0.0:  # lograin2T-lograin7T
        weather_vars.update({'lograin30T': log10(rain_30_t)})
    else:
        weather_vars.update({'lograin30T': log10(0.005)})

    for i in range(2,8,1): # rain2-rain7, lograin2-lograin7
        day_in = (day - datetime.timedelta(days=i-1)).strftime('%Y-%m-%d')
        if day_in in df_weather.index:
            L = int(df_weather.index.get_loc(day_in))
        else:
            print(' Weather data for ' + day_in + ' unavailable. Waiting one minute, then collecting past weather data...')
            time.sleep(60)
            df_weather.loc[pd.Timestamp(day_in)] = np.nan
            new_weather = grabWU(datetime.datetime.strptime(day_in,'%Y-%m-%d'),st,angle)
            for key in new_weather:
                df_weather[key].loc[day_in] = new_weather[key]
            L = int(df_weather.index.get_loc(day_in))
        df_temp = df_weather.iloc[L]
        rain_d = float(df_temp['rain1'])
        weather_vars.update({'rain' + str(i): rain_d}) #rain2-rain7
        if rain_d > 0.0: # lograin2-lograin7
            weather_vars.update({'lograin' + str(i): log10(rain_d)})
        else:
            weather_vars.update({'lograin' + str(i): log10(0.005)})

        rain_tot = 0.0
        for j in range(i, 0,-1): # rain2T-7T
            rain_tot = rain_tot + weather_vars['rain' + str(j)]
        weather_vars.update({'rain' + str(i) + 'T': rain_tot})
        if rain_tot > 0.0:  # lograin2T-lograin7T
            weather_vars.update({'lograin' + str(i) + 'T': log10(rain_tot)})
        else:
            weather_vars.update({'lograin' + str(i) + 'T': log10(0.005)})

    df_update = pd.DataFrame(weather_vars, index= [pd.Timestamp(day)]) #update data file
    df_update.index.name = 'PDT'
    if df_update.index[0] in df_weather.index:
        df_weather.update(df_update)
    else:
        df_weather = df_weather.append(df_update)
    cols_ordered = ['rainH_b','rain1','rain2','rain3','rain4', 'rain5','rain6','rain7',
                          'rain2T','rain3T','rain4T','rain5T','rain6T','rain7T','rain30T',
                          'lograinH','lograin1','lograin2','lograin3','lograin4','lograin5','lograin6','lograin7',
                          'lograin2T','lograin3T','lograin4T','lograin5T','lograin6T','lograin7T','lograin30T',
                          'temp1','dtemp1','wspd1','wdir1','pres1','awind1','owind1']
    df_weather = df_weather[cols_ordered]
    df_weather = df_weather.sort_index(ascending=True)
    df_weather.to_csv(weather_file)
    return weather_vars

def getLocal(beach,day): # grab local parameters from NOAA CO-OPS
    ##Only Previous Day Means and Maxes##
    # Select station
    if beach == 'Santa Monica Pier':  # SANTA MONICA
        st = '9410840'
    elif beach == 'Belmont Pier':  # LONG BEACH
        st = '9410660'
    elif beach == 'Mothers (LB)':  # LONG BEACH
        st = '9410660'

    bd = day - datetime.timedelta(days=1)
    ed = day
    units = 'metric' # Temp = C, Speed = m/s, Pressure = mbar
    time_zone = 'lst' # Local Standard Time (ignore DLS)
    product = ['air_temperature', 'water_temperature', 'wind','air_pressure']
    format = 'json'
    local_vars = {}

    for p in product:
        df = pd.DataFrame()
        if p == 'air_temperature':
            var_name = 'temp_L'
        elif p == 'water_temperature':
            var_name = 'Wtemp_L'
        elif p == 'air_pressure':
            var_name = 'pres_L'

        url = 'http://tidesandcurrents.noaa.gov/api/datagetter?' + \
              'begin_date=' + bd.strftime('%Y%m%d') + \
              '&end_date=' + ed.strftime('%Y%m%d') + \
              '&station=' + st + \
              '&product=' + p + \
              '&units=' + units + \
              '&time_zone=' + time_zone + \
              '&format=' + format + \
              '&application=web_services'

        web = requests.get(url)
        try:
            web.raise_for_status()
        except Exception as exc:
            print('   There was a problem with the NOAA URL: %s' % exc)
            return 0

        data = json.loads(web.text)
        try:
            data = data['data']
        except KeyError:
            # print('   Could not find data in the date range for the following station: ' + station_name)
            # c += 1
            continue
        # print('   JSON data for ' + bd.strftime('%Y%m%d') + ' to ' + ed.strftime('%Y%m%d') + ' loaded. Parsing...')

        df = df.append(pd.DataFrame.from_dict(data), ignore_index=True)
        if len(df) > 0:
            if p != 'wind':
                df = df[['t', 'v']]
                df.columns = ['date', p]
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                df.columns = [var_name]
                df = df[bd.strftime('%Y-%m-%d')]
                #df = df.astype(float)
                #local_vars.update({var_name + '1': df[var_name].mean()})#Previous day mean
                #local_vars.update({var_name + '1_max': df[var_name].max()})#Previous day max
                df = pd.to_numeric(df[var_name], errors='coerce')
                local_vars.update({var_name + '1': df.mean().round(1)})  # Previous day mean
                local_vars.update({var_name + '1_max': df.max().round(1)})  # Previous day max

        else:
            print('No local ' + p + ' data found for ' + beach)
    return local_vars

def getFIB(beach,day, base_folder): # grab FIB data from stored files
    if beach == 'Santa Monica Pier':
        sample_delay = 2

    fib_file = base_folder + beach + '\\fib\\' + beach.replace(' ','_') + '_fib_samples_2017.csv'
    df = pd.read_csv(fib_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    yest_in = (day - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    df = df['2017-01-03': yest_in]
    fib_vars = {}

    # Last sample (FIB1) #
    print(' FIB variables:')
    if df.index[-1] < pd.to_datetime(day) - pd.to_timedelta(str(sample_delay) + ' days'):
        print('   Most recent sample from more than ' + str(sample_delay) + ' days ago. Update sample file with new samples, and rerun NowCast:')
    else:
        print('   Last Samples:')

    # Variables #
    #today_in = day.strftime('%Y-%m-%d')

    in_30 = (day - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
    in_60 = (day - datetime.timedelta(days=60)).strftime('%Y-%m-%d')
    for f in ['TC','FC','ENT']:
        df_fib = df[f]
        df_fib = df_fib[df_fib.notnull()]
        #fib_vars.update({f + '1': int(df.iloc[-1][f])}) #FIB1
        #fib_vars.update({'log' + f + '1': log10(int(df.iloc[-1][f]))})
        fib_vars.update({f + '1': int(df_fib.iloc[-1])})  # FIB1
        fib_vars.update({'log' + f + '1': log10(int(df_fib.iloc[-1]))})
        print('   ' + f + ' - ' + str(fib_vars[f + '1']) + ' (' + df_fib.index[-1].strftime('%m/%d/%Y') + ')')

        #df_30 = df[in_30:today_in][f] # 30-day geomean
        df_30 = df_fib[in_30:yest_in]  # 30-day geomean
        df_30 = df_30[df_30.notnull()]
        array_30 = np.array(df_30, dtype = 'float64')
        g_30 = array_30.prod() ** (1/len(df_30))
        fib_vars.update({f + 'g1': g_30})
        fib_vars.update({'log' + f + 'g1': log10(g_30)})

        #df_60 = df[in_60:today_in][f]  # 60-day geomean
        df_60 = df_fib[in_60:yest_in]  # 60-day geomean
        df_60 = df_60[df_60.notnull()]
        array_60 = np.array(df_60, dtype='float64')
        g_60 = array_60.prod() ** (1 / len(df_60))
        fib_vars.update({'log' + f + 'g1_60': log10(g_60)})

    return fib_vars

def upload_BRC(upload_csv): # upload predictions to BRC website/app

    htb_logon = 'XXX'
    htb_pwd = 'XXXX'  # redacted
    upload_file = upload_csv

    # Open Chrome #
    driver = webdriver.Chrome()
    driver.get('http://beachreportcard.org/admin/WaterQualityNowcastImporter.aspx')

    if driver.title == 'Heal the Bay - Beach Report Card - Administration - Logon' or \
                    driver.current_url == 'http://beachreportcard.org/Admin/brclogon.aspx?ReturnUrl=%2fadmin%2fWaterQualityNowcastImporter.aspx':
        # Logon Page #
        logon_elem = driver.find_element_by_id('txtUsername')
        logon_elem.send_keys(htb_logon)  # Username
        pwd_elem = driver.find_element_by_id('txtPassword')
        pwd_elem.send_keys(htb_pwd)  # Password
        pwd_elem.send_keys(Keys.RETURN)

    # Upload File #
    upload_elem = driver.find_element_by_xpath("//input[@type='file']")
    upload_elem.send_keys(upload_file)
    submit_elem = driver.find_element_by_xpath("//input[@type='submit']")
    submit_elem.send_keys(Keys.RETURN)
    if 'smashing success' in driver.page_source:
        print('\nFile uploaded to BRC successfully')
    driver.close()

date = datetime.date.today() #- datetime.timedelta(days= 1)
base_folder = '### INSERT BASE FOLDER HERE ###'

beaches = {'Cowell': {'Flow': 1,'Waves': 1,'Tides': 1,'Weather': 1,'Local':0,'FIB':0, 'sam_time': '09:35','Pilot':0,'BRC': 435}, #435
           'Main (Boardwalk)': {'Flow': 0,'Waves': 1,'Tides': 1,'Weather': 1,'Local':0,'FIB':0, 'sam_time': '09:13','Pilot':1,'BRC': 0},  #436
           'Arroyo Burro':{'Flow': 0,'Waves': 1,'Tides': 1, 'Weather': 1,'Local': 0,'FIB':0, 'sam_time': '09:00','Pilot':0,'BRC': 74}, # Sample times in PST
           'East':{'Flow': 1,'Waves': 1,'Tides': 1, 'Weather': 1,'Local':0,'FIB':0, 'sam_time': '09:30','Pilot':0,'BRC': 76},
           'Santa Monica Pier':{'Flow': 0,'Waves': 1,'Tides': 1, 'Weather': 1,'Local':1,'FIB':1, 'sam_time': '08:30','Pilot':0,'BRC': 23},
           'Redondo Beach Pier': {'Flow': 0,'Waves': 1,'Tides': 1, 'Weather': 1,'Local':0,'FIB':0, 'sam_time': '10:00','Pilot':0,'BRC': 640},
           'LB City Beach (5th Place)': {'Flow': 0,'Waves': 0,'Tides': 1, 'Weather': 1,'Local':0,'FIB':0, 'sam_time': '07:20','Pilot':0,'BRC': 191}, #191
           'Belmont Pier':{'Flow': 0,'Waves': 1,'Tides': 1,'Weather': 1,'Local':1,'FIB':0, 'sam_time': '07:00','Pilot':0,'BRC': 197},
           'Mothers (LB)':{'Flow': 0,'Waves': 0,'Tides': 0,'Weather': 1,'Local':1,'FIB':0, 'sam_time': '06:30','Pilot':1,'BRC': 0},#136
           'Huntington (Brookhurst)': {'Flow': 1,'Waves': 1,'Tides': 1,'Weather': 1,'Local':0,'FIB':0, 'sam_time': '06:52','Pilot':0,'BRC': 150}, #150
           'Doheny': {'Flow': 1,'Waves': 1,'Tides': 1,'Weather': 1,'Local':0,'FIB':0, 'sam_time': '07:42','Pilot':0,'BRC': 229},
           'Moonlight': {'Flow': 0, 'Waves': 1, 'Tides': 1, 'Weather': 1, 'Local': 0, 'FIB': 0, 'sam_time': '09:10', 'Pilot': 0, 'BRC': 290}}  #290}
beach_list = list(beaches.keys())

date_str = datetime.date.strftime(date, '%m/%d/%Y')
print('Running NowCast models for ' + date_str + '\n')
date_str_file = datetime.date.strftime(date, '%m%d%Y')
df_decisions = pd.DataFrame()
df_pilot = pd.DataFrame()
df_pred_all = pd.DataFrame()
df_site = pd.DataFrame()

old_stdout = sys.stdout # for logging
old_stderr = sys.stderr
log_file = open(base_folder +'logs\\run_log_' + date_str_file + '.log','w')
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

count = 0
for i in beach_list:
    if count % 5 == 0 and count>0: # WU 60s delay for API calls
        print('Five beaches modeled. Waiting 60 seconds...\n')
        time.sleep(60)
    #try:
    print('NowCast for ' + i + ' (Date: ' + date_str + ')')
    model_folder = base_folder + i

    ## Collect Environmental Data ##
    print('\nCollecting environmental data:')
    all_vars = {}
    # Flow Data #
    if beaches[i]['Flow'] == 1:
        try:
            flow = getFlow(i, date)
        except:
            e = sys.exc_info()
            print('Error: ' + str(e))
            flow = 0

        if flow == 0:#np.nan:
            print(' No flow values for ' + i + ' found')
        else:
            print(' Flow variables obtained')
            all_vars.update(flow)

    # Wave Data #
    if beaches[i]['Waves'] == 1:
        try:
            waves = getWaves(i, date, beaches[i]['sam_time'])
        except:
            e = sys.exc_info()
            print('Error: ' + str(e))
            waves = 0

        if waves == 0: #np.nan:
            print(' No waves values for ' + i + ' found')
        else:
            print(' Wave variables obtained')
            all_vars.update(waves)

    # Tide Data #
    if beaches[i]['Tides'] == 1:
        try:
            tides = getTides(i, date, beaches[i]['sam_time'], base_folder)
        except:
            e = sys.exc_info()
            print('Error: ' + str(e))
            tides = 0

        if tides == 0: #np.nan:
            print(' No tide values for ' + i + ' found')
        else:
            print(' Tide variables obtained')
            all_vars.update(tides)

    # Weather Data #
    if beaches[i]['Weather'] == 1:
        try:
            weather = getWeather(i, date, base_folder)
        except:
            e = sys.exc_info()
            print('Error: ' + str(e))
            weather = 0

        if weather == 0:# np.nan:
            print(' No weather data for ' + i + ' found')
        else:
            print(' Weather variables obtained')
            all_vars.update(weather)

    # Local Data #
    if beaches[i]['Local'] == 1:
        try:
            local = getLocal(i, date)
        except:
            e = sys.exc_info()
            print('Error: ' + str(e))
            local = 0

        if local == 0:#np.nan:
            print(' No local data for ' + i + ' found')
        else:
            print(' Local variables obtained')
            all_vars.update(local)

    # FIB Variables #
    if beaches[i]['FIB'] == 1:
        try:
            fib = getFIB(i, date, base_folder)
        except:
            e = sys.exc_info()
            print('Error: ' + str(e))
            fib = 0

        if fib == 0:#np.nan:
            print(' No FIB data for ' + i + ' found')
        else:
            print(' FIB variables obtained')
            all_vars.update(fib)

    #weekend1, laborday #
    all_vars.update({'laborday': int(date > datetime.date(2017,9,4))})
    if date.isoweekday() in [6,7,1]: # If yesterday was Friday, Saturday, or Sunday
        all_vars.update({'weekend1': 1})
    else:
        all_vars.update({'weekend1': 0})

    # SAVE ALL VARIABLES #
    df_all_vars = pd.DataFrame(all_vars, index= [date])
    all_vars_file = model_folder + '\\variables\\' + i.replace(' ', '_') + '_all_vars_' + date.strftime('%m%d%Y') + '.csv'
    df_all_vars.to_csv(all_vars_file)
    print('Environmental data collected and saved to ' + all_vars_file)

    ## Run Models ##
    site_post = 1
    beach_decision_dict = {}
    pred_dict ={}
    beach_break = 0
    for f in ['TC','FC','ENT']:
        coef_file = f + '_Model_Coefficients_PM.csv'
        decision = 'No Prediction' # No Prediction
        if coef_file in os.listdir(model_folder):
            print('\nRunning ' + f + ' model (' + i + '):')
            if f == 'TC':
                thresh = 10000.0
            elif f == 'FC':
                thresh = 400.0
            elif f == 'ENT':
                thresh = 104.0

            # Download model coefficients #
            df_coef = pd.read_csv(os.path.join(model_folder, coef_file), header=None)
            df_coef.columns = ['Variable','Coefficient']
            df_coef.set_index('Variable', inplace=True)

            # Plug in variables to model #
            PM = float(df_coef.loc['PM']) # exctract PM and constant from coef dataframe
            constant = float(df_coef.loc['(Constant)'])
            df_coef.drop(['PM', '(Constant)'], inplace=True)
            mod_sum = constant
            df_print = pd.DataFrame()
            break_val = 0
            for c in df_coef.index: #Sum all variables
                try:
                    var_value = float(df_all_vars[c])
                    mod_sum += (var_value * float(df_coef.loc[c]))
                    df_print = df_print.append(pd.DataFrame({'Value': var_value}, index=[c]))
                except:
                    break_val = 1
                    beach_break = 1
            if np.isnan(mod_sum):
                break_val = 1
                beach_break = 1
            print('\n')
            print(df_coef.join(df_print))
            print('\nConstant: ' + str(round(constant,3)))
            print('PM: ' + str(round(PM, 3)))

            if break_val == 1:
                print('\nSome variables for ' + date_str + ' not found in collected variable dataset. Run model manually')
                decision = 'No Prediction'
                beach_decision_dict.update({f: decision})
                continue

            mod_sum *= PM #multiply by PM
            mod_result = 10**mod_sum #MODEL PREDICTION
            if mod_result > thresh:
                site_post = 0
                decision = 'Post'
            else:
                decision = 'No Post'

            print('\n' + f + ' (Predicted): ' + str(round(mod_result)) + ' CFU/ 100 mL')
            print('Posting Decsion: ' + decision)
            pred_dict.update({f:mod_result})


        elif  f + '_BLR_Model_Coefficients.csv' in os.listdir(model_folder): ## BLR Model ##
            coef_file = f + '_BLR_Model_Coefficients.csv'
            model_pkl =  f + '_BLR_Model.pkl'
            model_file = os.path.join(model_folder, model_pkl)
            print('\nRunning ' + f + ' BLR model (' + i + '):')

            # Download model coefficients #
            df_coef = pd.read_csv(os.path.join(model_folder, coef_file), header=None)
            df_coef.columns = ['Variable', 'Coefficient']
            df_coef.set_index('Variable', inplace=True)

            # Print variables in mdel #
            thresh = float(df_coef.loc['threshold'])  # exctract PM and constant from coef dataframe
            constant = float(df_coef.loc['intercept'])
            df_coef.drop(['threshold', 'intercept'], inplace=True)
            model_vars = list(df_coef.index)
            df_print = pd.DataFrame()
            break_val = 0

            for c in df_coef.index: #Sum all variables
                try:
                    var_value = float(df_all_vars[c])
                    df_print = df_print.append(pd.DataFrame({'Value': var_value}, index=[c]))
                except:
                    break_val = 1
                    beach_break = 1
            if np.isnan(mod_sum):
                break_val = 1
                beach_break = 1
            print('\n')
            print(df_coef.join(df_print))
            print('\nConstant: ' + str(round(constant,3)))
            print('Tuning Threshold: ' + str(round(thresh, 3)))

            if break_val == 1:
                print('\nSome variables for ' + date_str + ' not found in collected variable dataset. Run model manually')
                decision = 'No Prediction'
                beach_decision_dict.update({f: decision})
                continue

            df_day_vars = df_all_vars[model_vars]  # Locate only model variables
            blr = joblib.load(model_file)

            try:
                mod_result = blr.predict_proba(df_day_vars.values.reshape(1, -1))[0, 1]
            except ValueError:
                print('Some variables for ' + str(
                    i.date()) + ' not found in collected variable dataset. Run model manually')
                continue

            if mod_result > thresh:
                site_post = 0
                decision = 'Post'
            else:
                decision = 'No Post'

            print('\n' + f + ' (Predicted Probability): ' + str(round(mod_result,3)))
            print('Posting Decsion: ' + decision)
            pred_dict.update({f:mod_result})

        else:
            print('\n' + f + ' model unavailable to run')

        ## Add 'decision' for each FIB for each beach to dataframe for export to beach managers
        if beach_break == 0:
            beach_decision_dict.update({f: decision})
        else: beach_decision_dict.update({f: 'No Prediction'})

    ## Post/ No Post Decisions ##
    if beaches[i]['Pilot'] != 1:
        df_decisions = df_decisions.append(pd.DataFrame(beach_decision_dict, index=[i]))
    else: #PILOT
        df_pilot = df_pilot.append(pd.DataFrame(beach_decision_dict, index=[i]))

    ## BRC Site Decision Upload ##
    if beaches[i]['BRC'] != 0 and beach_break == 0:
        df_site = df_site.append(pd.DataFrame({'LocationID': beaches[i]['BRC'],'NowcastPassFlag': site_post,'NowcastDate': date_str}, index = [i]))

    ## Prediction Update ##
    df_pred_all = df_pred_all.append(pd.DataFrame(pred_dict, index=[i]).round(0))

    df_pred = pd.DataFrame(pred_dict, index= [date_str])
    df_pred.index.rename('Date',inplace=True)
    df_pred.to_csv(base_folder + '\\' + i + '\\predictions\\' + date_str_file + '_' + i + '_predictions.csv')
    print('\n')
    count += 1

print('Finished daily run for ' + date_str)
## Save decisions ##
df_decisions = df_decisions[['TC','FC','ENT']] #Post/ No Post decisions
df_decisions.sort_index(ascending=True, inplace=True)
df_decisions.to_csv(base_folder + 'decisions\\NowCast_Predictions_' + date_str_file + '.csv')
print('\nPredictions:')
print(df_decisions)

if len(df_pilot) > 0:
    df_pilot = df_pilot[['TC','FC','ENT']] #Post/ No Post decisions PILOT
    df_pilot.sort_index(ascending=True, inplace=True)
    print('\nPredictions (Pilot):')
    print(df_pilot)

df_pred_all = df_pred_all[['TC','FC','ENT']] #Post/ No Post decisions
df_pred_all.sort_index(ascending=True, inplace=True)
print('\n')
print(df_pred_all)

## Save and Upload Import ##
df_site = df_site[['LocationID','NowcastPassFlag','NowcastDate']] # BRC upload file
df_site.to_csv(base_folder + 'import\\NowCast_Import_' + date_str_file + '.csv', index=False)
if len(df_site) > 0:
    upload_BRC(base_folder + 'import\\NowCast_Import_' + date_str_file + '.csv')

sys.stdout = old_stdout
sys.stderr = old_stderr
log_file.close()
time.sleep(600)