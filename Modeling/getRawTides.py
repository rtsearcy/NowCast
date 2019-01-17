#! python3
# getTides.py - Download bulk tidal data from NOAA CO-OPS

# SET DATE RANGE IF NEEDED

import requests
import json
from datetime import timedelta
from dateutil.parser import parse
import pandas as pd

stations_dict = {
    'San Diego': '9410170',
    'La Jolla': '9410230',
    'Newport Bay': '9410580',
    'Long Beach': '9410680',
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

path = 'Z:\Predictive Modeling\Phase III\Modeling\Data\Tides\Raw\\'

for key in stations_dict:
    station_num = stations_dict[key]  # San Diego
    station_name = key
    begin_date = '20080101'
    end_date = '20201231'

    datum = 'MLLW'  # Mean Lower Low Water (Lowest tide if diurnal)
    units = 'metric'
    time_zone = 'lst'  # Local Standard Time (ignore DLS)
    product = 'predictions'
    form = 'json'

    begin_date = parse(begin_date)
    end_date = parse(end_date)
    bd = begin_date
    # ed = end_date - use if you only want a discrete amount of data (i.e. after 2020)
    ed = begin_date + timedelta(days=3649)  # NOAA-COOPS allows for up to 3650 days of data per grab
    c = 1

    print('Collecting ' + station_name + ' tidal data...')
    while ed < end_date:
        if c != 1:
            bd = ed + timedelta(days=1)
            ed = ed + timedelta(days=3649)  # Account for timestep limit
            if ed > end_date:
                ed = end_date
        print('   Searching for data from ' + bd.strftime('%Y%m%d') + ' to ' + ed.strftime('%Y%m%d'))

        url = 'http://tidesandcurrents.noaa.gov/api/datagetter?' + \
            'begin_date=' + bd.strftime('%Y%m%d') + \
            '&end_date=' + ed.strftime('%Y%m%d') + \
            '&station=' + station_num + \
            '&product=' + product + \
            '&datum=' + datum + \
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
            data = data['predictions']
        except KeyError:
            print('   Could not find data for the following station: ' + station_name)
            continue
        print('   JSON data loaded. Parsing')

        if c == 1:
            df = pd.DataFrame.from_dict(data)
        else:
            df = df.append(pd.DataFrame.from_dict(data), ignore_index=True)  # Add to exisiting df
        print('   Data parsed')
        c += 1

    # Save to file
    df.columns = ['Date Time', 'Water Level (m)']
    save_file = path + station_name.replace(' ', '') + '_Tidal_Data_' + begin_date.strftime('%Y%m%d') + '_' \
        + end_date.strftime('%Y%m%d') + '.csv'
    df.to_csv(save_file, index=False)
    print(station_name + ' tidal data written to file: ' + save_file)
