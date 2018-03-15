# getRawWaves.py
# RS - 10/6/2016
# RS - 03/15/2018 UPDATE

# Summary:
# - Find raw CDIP wave data for given time period, converts from UTC to PST time, save to raw file
# - Using raw data, calculate daily averages and max parameters

# NOTE: Output files may contain non-consecutive data, or data that does not go until the end date (decom. stations)
# NOTE: Output files may be offset from date in title do to UTC to PST shift of 8 hrs.

import requests
import os
import pandas as pd

# Inputs #
outfolder_vars = 'Z:\Predictive Modeling\Phase III\Modeling\Summer_2018\Environmental Variables\Waves'
outfolder_raw = outfolder_vars + '\\raw'
sd = '20080101'  # Must be YYYYMMDD format
ed = '20171101'  # Because UTC time, include extra day

stations = {
    'Point Loma South':	'191',
    'Torrey Pines Outer': '100',
    'Oceanside Offshore': '045',
    'San Pedro': '092',
    'Santa Monica Bay':	'028',
    'Anacapa Passage': '111',
    'Harvest': '071',
    'Diablo Canyon': '076',
    'Point Sur': '157',
    'Cabrillo Point Nearshore':	'158',
    'Monterey Bay West': '185',
    'Cape Mendocino': '094',
    'Humboldt Bay North Split':	'168'
}

# Find Raw CDIP Data #
print('CDIP Wave Data\nRaw Directory: ' + outfolder_raw + '\nVariable Directory: ' + outfolder_vars)
for key in stations:
    st_name = key
    st = stations[key]

    # Grab data from CDIP website #
    print('\nGrabbing CDIP wave data from ' + st_name + ' station (' + st + ')')
    url = 'http://cdip.ucsd.edu/data_access/ndar?' + st + '+pm+' + sd + '-' + ed  # date range using CDIP NDAR, grab wave parameters, no header
    web = requests.get(url)
    try:
        web.raise_for_status()
    except Exception as exc:
        print('  There was a problem grabbing wave data for Station ' + st + ': %s' % exc)

    # Create DataFrame, index with timestamp, convert blanks/errors to NaN #
    data = [line.split() for line in web.text.splitlines()]
    data = data[:-1]  # exclude footer
    data[0][0] = data[0][0].replace('<pre>', '')  # remove header
    df_raw = pd.DataFrame(data)
    df_raw.columns = ['year', 'month', 'day', 'hour', 'minute', 'WVHT', 'DPD', 'MWD', 'APD', 'Wtemp_B']
    df_raw['dt'] = pd.to_datetime(df_raw[['year', 'month', 'day', 'hour', 'minute']])
    df_raw.set_index('dt', inplace=True)
    df_raw.index = df_raw.index.shift(-8, freq='h')  # convert from UTC to PST (- 8 hrs)
    df_raw = df_raw[['WVHT', 'DPD', 'MWD', 'APD', 'Wtemp_B']]
    df_raw = df_raw.apply(lambda x: pd.to_numeric(x, errors='coerce'))

    # Save to raw outfile #
    outname_raw = st_name.replace(' ', '_') + '_raw_wave_data_' + sd + '_' + ed + '.csv'
    out_file_raw = os.path.join(outfolder_raw, outname_raw)
    df_raw.to_csv(out_file_raw)
    print('  Raw wave data saved to ' + outname_raw)

    # Create DF for daily variables
    wave_round = {'WVHT': 2, 'DPD': 2, 'MWD': 0, 'APD': 2, 'Wtemp_B': 1}  # sig figs on CDIP
    df_vars = pd.DataFrame(index=df_raw.resample('D').mean().index)
    for c in df_raw.columns:
        r = wave_round[c]
        df_vars[c + '1'] = round(df_raw[c].resample('D').mean().shift(1, freq='D'), r)  # previous day mean
        df_vars[c + '1_max'] = df_raw[c].resample('D').max().shift(1, freq='D')  # previous day max
        df_vars[c + '1_min'] = df_raw[c].resample('D').min().shift(1, freq='D')  # previous day min

    # Save to variable outfile #
    outname_vars = st_name.replace(' ', '_') + '_Wave_Variables_' + sd + '_' + ed + '.csv'
    out_file_vars = os.path.join(outfolder_vars, outname_vars)
    df_vars.to_csv(out_file_vars)
    print('  Daily wave variables saved to ' + outname_vars)
