# getRawCOOPS.py - grab raw 'local' data from NOAA CO-OPS API (same source as Tide data)
# RS - 1/17/2017
# RTS - March 2018 UPDATE

# Note: Calls are only allowed for 30d of data at a time (slows things down a bit). Code can take up to an hour to run
# for all stations, depending on the length of time desired

import requests
import json
import os
from datetime import timedelta
from dateutil.parser import parse
import pandas as pd

# Inputs
path = 'Z:\Predictive Modeling\Phase III\Modeling\Winter_2018_2019\Environmental Variables\\NOAA CO-OPS\\raw'

begin_date_s = '20021231'
end_date_s = '20180331'

units = 'metric'  # Temp = C, Speed = m/s, Pressure = mbar
time_zone = 'lst'  # Local Standard Time (ignore DLS)
product = ['air_temperature', 'water_temperature', 'wind', 'air_pressure']
form = 'json'

stations_dict = {
    'San Diego': '9410170',
    'La Jolla': '9410230',
    # 'Newport Bay': '9410580', # Decommissioned
    'Long Beach': '9410660',
    'Santa Monica': '9410840',
    'Santa Barbara': '9411340',
    'Port San Luis': '9412110',
    'Monterey': '9413450',
    'San Francisco': '9414290',
    'Point Reyes': '9415020',
    'Green Cove': '9416409',  # Sonoma
    'North Split': '9418767',  # Humboldt
    'Crescent City': '9419750'  # Del Norte
}

for key in stations_dict:
    station_num = stations_dict[key]
    station_name = key
    begin_date = parse(begin_date_s)
    end_date = parse(end_date_s)
    df_out = pd.DataFrame()

    for p in product:
        bd = begin_date
        ed = begin_date + timedelta(days=30)  # NOAA-COOPS allows for up to 31 days of data per grab
        df = pd.DataFrame()
        c = 1

        print('Collecting ' + p + ' data for ' + station_name + '...')
        while ed < end_date:  # NOAA CO-OPS only allows collection of 31 days of data at a time
            if c != 1:  # change time parameters for the next API call
                bd = ed + timedelta(days=1)
                ed = ed + timedelta(days=30)
                if ed > end_date:
                    ed = end_date
            print('   Searching for ' + p + ' data from ' + bd.strftime('%Y%m%d') + ' to ' + ed.strftime('%Y%m%d'))

            url = 'http://tidesandcurrents.noaa.gov/api/datagetter?' + \
                'begin_date=' + bd.strftime('%Y%m%d') + \
                '&end_date=' + ed.strftime('%Y%m%d') + \
                '&station=' + station_num + \
                '&product=' + p + \
                '&units=' + units + \
                '&time_zone=' + time_zone + \
                '&format=' + form + \
                '&application=web_services'

            web = requests.get(url)
            try:
                web.raise_for_status()
            except Exception as exc:
                print('   There was a problem with the URL: %s' % exc)

            data = json.loads(web.text)
            try:
                data = data['data']
            except KeyError:
                print('   Could not find data in the date range for the following station: ' + station_name)
                c += 1
                continue
            print('   JSON data for ' + bd.strftime('%Y%m%d') + ' to ' + ed.strftime('%Y%m%d') + ' loaded. Parsing...')

            df = df.append(pd.DataFrame.from_dict(data), ignore_index=True)
            c += 1

        if len(df) > 0:
            if p != 'wind':
                df = df[['t', 'v']]  # t - datetime, v - value
                df.columns = ['date', p]
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            else:
                df = df[['t', 's', 'd']]  # t - datetime, s - speed, d - direction
                df.columns = ['date', 'wspd', 'wdir']
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')

            df_out = df_out.join(df, how='outer')
        else:
            print('No ' + p + ' data found for ' + station_name)

    # Save to file
    outname = station_name.replace(' ', '_') + '_' + begin_date_s + '_' + end_date_s + '_COOPS_Raw.csv'
    out_file = os.path.join(path, outname)
    df_out.to_csv(out_file)
    print('Raw NOAA CO-OPS data for ' + station_name + ' written to ' + out_file)
