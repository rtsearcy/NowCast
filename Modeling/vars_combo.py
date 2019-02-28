# var_combo.py - Combine available variable datasets into beach-specific datasets that ready for modeling
# RTS - 3/25/2018

# Uses beach FIB data and metadata to aggregate variables into a single dataset
# Adjust for wet/dry or dry-only datasets

import pandas as pd
import os
from numpy import sin, cos, pi, isnan, nan
from datetime import time


def alongshore(x):  # Calculates longshore current direction based on wave parameters
    if x == 0:   # or isnan(x):
        return 0
    elif isnan(x):
        return nan
    elif x > 0:
        return -1
    else:
        return 1  # if sin < 0, - sin is > 0, current is +


def angle_vars(df, ang, station_type):  # Calculates beach angle dependant variables
    if station_type == 'met_station':
        if 'wspd1' in df.columns and 'wdir1' in df.columns:  # Wind speed/direction
            df['awind1'] = df['wspd1'] * sin(((df['wdir1'] - ang) / 180) * pi)
            df['owind1'] = df['wspd1'] * cos(((df['wdir1'] - ang) / 180) * pi)
    elif station_type == 'coop_station':
        if 'wspd_L1' in df.columns and 'wdir_L1' in df.columns:  # Local wind speed/direction
            df['awind_L1'] = df['wspd_L1'] * sin(((df['wdir_L1'] - ang) / 180) * pi)
            df['owind_L1'] = df['wspd_L1'] * cos(((df['wdir_L1'] - ang) / 180) * pi)
    elif station_type == 'wave_station':
        for w in ['MWD1', 'MWD1_max', 'MWD1_min']:  # Wave direction
            if w in df.columns:
                df['MWD1_b' + w.replace('MWD1', '')] = df[w] - ang  # Direction relative to beach
                df['SIN_MWD1_b' + w.replace('MWD1', '')] = \
                    round(sin(((df['MWD1_b' + w.replace('MWD1', '')]) / 180) * pi), 5)
                if w == 'MWD1':
                    df['q_dir1'] = df['SIN_MWD1_b'].apply(alongshore)  # Longshore current direction
        df.drop(['MWD1', 'MWD1_max', 'MWD1_min'], axis=1, inplace=True)
    return df


def tide_vars_hourly(df, raw_folder, station, sample_time=time(10, 0)):
    # Calculates hourly tide variables (if applicable)
    try:
        raw_file = [rf for rf in os.listdir(raw_folder) if rf.startswith(station.replace(' ', '_'))][0]
    except:
        print('No tide file for ' + station + ' found...skipping hourly tide variables')
        return df

    df_raw = pd.read_csv(os.path.join(raw_folder, raw_file))
    df_raw.columns = ['date', 'tide']
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_raw = df_raw.set_index('date')

    df['Tide'] = df_raw[df_raw.index.time == sample_time].resample('D').first().reindex(df.index, method='nearest')
    # Tide at sampling time

    for i in [1, 3, 6, 9, 12]:  # Tide/Change in tide level 'i' hours previous to sampling
        df['Tide_' + str(i) + 'h'] = df_raw[df_raw.index.shift(i, freq='H').time == sample_time].resample('D').first()\
            .reindex(df.index, method='nearest')  # Tide_h
        df['dTide_' + str(i) + 'h'] = df['Tide'] - df['Tide_' + str(i) + 'h']  # dTide_h

    return df


# Inputs
base_folder = 'S:\SCIENCE & POLICY\\NowCast\Modeling\summer_2019\\'
beach_base_folder = base_folder + 'Beaches\\'
enviro_base_folder = base_folder + 'Environmental Variables\\'  # where enviro data is stored

loc_file = os.path.join(base_folder, 'locations.csv')  # Beach metadata (name, angle, station info)
range_file = os.path.join(enviro_base_folder, 'var_range.csv')

spec_beaches = ['Fitzgerald Marine Reserve','Surfers Beach','Venice Beach Frenchmans Creek','Francis']  # Only combine vars for certain beaches
percent_to_drop = 0.075  # Allowable percentage of missing values allowed before dropping variable entirely
dry_only = 0  # 1 - Dry samples only in the dataset; 0 - Dry and wet samples.
# Wet samples are those taken on days where at least 0.1 inch of rain occurred in previous 72 hours
hourly = False  # To calculate hourly values or not
station_types = {'tide_station': 'Tides',  # Station type: Subfolder
                 'wave_station': 'Waves',
                 'met_station': 'Meteorological',
                 'flow_station': 'Flow',
                 'rad_station': 'Solar Radiation',
                 'coop_station': 'NOAA CO-OPS'
                 }

# Get Beaches
df_loc = pd.read_csv(loc_file)
df_loc.set_index('beach', inplace=True)
if len(spec_beaches) > 0:  # If specific beaches
    beach_list = spec_beaches
else:
    beach_list = list(df_loc.index)
df_drop = pd.DataFrame()  # Record of number samples and dropped rows
df_range = pd.read_csv(range_file)
df_range.set_index('variable', inplace=True)

print('Variable Combination\nBase Directory: ' + base_folder + '\nPercentage Missing Before Drop: ' +
      str(100*percent_to_drop) + '%')
if dry_only == 1:
    print(' - - - | DRY SAMPLES ONLY | - - - ')

for b in beach_list:
    print('\n- - - | Processing variables for ' + b + ' | - - -')
    beach_folder = beach_base_folder + b.replace(' ', '_') + '\\'
    var_folder = beach_folder + 'variables\\'
    angle = int(df_loc.loc[b]['angle'])  # beach angle

    # Get FIB vars, start output DF
    df_fib = pd.read_csv(var_folder + b.replace(' ', '_') + '_variables_fib.csv')
    df_fib['date'] = pd.to_datetime(df_fib['date'])
    df_fib.set_index('date', inplace=True)
    df_fib = df_fib[~df_fib.index.duplicated(keep='first')]  # Remove duplicate index values
    if not isnan(df_loc.loc[b]['alt_start']):
        df_fib = df_fib[str(int(df_loc.loc[b]['alt_start'])):]
        # adjust for alternative start year (if not first year in dataset)
        print(' Alternative start date: ' + str(df_fib.index[0].date()))
    df_dirty = df_fib.copy()  # starts with FIB data

    for s in station_types:  # Iterate through environmental data types
        station = df_loc.loc[b][s]
        if pd.isnull(station):
            print('  ' + s.replace('_', ' ').capitalize() + ': NA')
            continue
        sub_folder = station_types[s]
        print('  ' + s.replace('_', ' ').capitalize() + ': ' + station)

        # Grab environmental data for given station type
        enviro_folder = os.path.join(enviro_base_folder, sub_folder)
        try:
            enviro_file = [x for x in os.listdir(enviro_folder) if x.startswith(station.replace(' ', '_'))][0]
        except IndexError:
            print('    ERROR: No data for the following station: ' + station)
            continue
        df_enviro = pd.read_csv(os.path.join(enviro_folder, enviro_file))
        d = [d for d in df_enviro.columns if d in ['date', 'dt', 'DATE']][0]  # Set datetimeindex
        df_enviro[d] = pd.to_datetime(df_enviro[d])
        df_enviro.set_index(d, inplace=True)

        # For waves, met, coop, calculate angle dependant variables
        if s in ['wave_station', 'met_station', 'coop_station']:
            df_enviro = angle_vars(df_enviro, angle, s)

        # Hourly variables
        if hourly and s in ['tide_station']:
            df_enviro = tide_vars_hourly(df_enviro, raw_folder=os.path.join(enviro_folder, 'raw'), station=station)

        df_dirty = df_dirty.join(df_enviro, how='left')  # Merge output and enviro dataframes

    # Check for errors in 'dirty' data (range)
    for c in df_dirty.columns:
        if c in df_range.index:  # Out of range = NAN
            df_dirty[c][~df_dirty[c].between(df_range.loc[c]['lower'], df_range.loc[c]['upper'])] = nan

    # Dry/Wet Samples
    if dry_only == 1:
        print('Dry samples kept. ' + str(int(df_dirty['wet'].sum())) + ' wet samples dropped.')
        df_dirty = df_dirty[df_dirty['wet'] == 0]  # Keep only dry samples

    # Check missing values
    before_rd = len(df_dirty)
    print('Length of dirty dataset: ' + str(before_rd))
    # df_dirty = df_dirty.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    df_missing = df_dirty.isnull().sum().sort_values(ascending=False)  # Find null values for each variable
    df_missing.to_csv(os.path.join(var_folder, 'missing_data_points.csv'))  # Missing data statistics

    # Drop columns with more missing values than % of length
    cols = list(df_dirty.columns)
    to_search = [x for x in cols if 'TC' not in x and 'FC' not in x and 'ENT' not in x and 'sample_time' not in x]
    df_missing = df_missing[to_search]  # search for non-FIB vars
    if not isnan(df_loc.loc[b]['alt_drop']):
        t = int(len(df_dirty) * df_loc.loc[b]['alt_drop'])
    else:
        t = int(len(df_dirty) * percent_to_drop)
        # t - number of allowable missing points before drop
    dropped_cols = list(df_missing[df_missing >= t].index)  # Find all columns with missing points above allowable
    new_cols = [x for x in cols if x not in dropped_cols]
    df_dirty = df_dirty[new_cols]  # Drop 'dropped_cols' from dirty dataset
    print('Dropped variables: ')
    print(df_missing[df_missing >= t])

    # Drop remaining rows with missing values
    to_search = [x for x in new_cols if 'TC' not in x and 'FC' not in x and 'ENT' not in x and 'sample_time' not in x]
    df_out = df_dirty.dropna(subset=to_search)  # Drop rows with missing values
    after_rd = len(df_out)
    print('     Dropped rows: ' + str(before_rd - after_rd))
    print('Final dataset length: ' + str(after_rd))
    drop_dict = {
        'dirty_samples': before_rd,
        'dirty_TC_exc': df_dirty.TC_exc.sum(),
        'dirty_FC_exc': df_dirty.FC_exc.sum(),
        'dirty_ENT_exc': df_dirty.ENT_exc.sum(),
        'dropped_cols': len(dropped_cols),
        'dropped_rows': before_rd - after_rd,
        'clean_samples': after_rd,
        'clean_TC_exc': df_out.TC_exc.sum(),
        'clean_FC_exc': df_out.FC_exc.sum(),
        'clean_ENT_exc': df_out.ENT_exc.sum()
    }

    # Save to output file in variable folder
    outfile = b.replace(' ', '_') + '_modeling_dataset.csv'
    df_out.to_csv(os.path.join(var_folder, outfile))
    print('Variables for ' + b + ' saved to ' + outfile)

    df_drop = df_drop.append(pd.DataFrame(drop_dict, index=[b]))

# Save drop df
dropfile = 'drop_stat.csv'
df_drop = df_drop[['dirty_samples', 'dirty_TC_exc', 'dirty_FC_exc', 'dirty_ENT_exc', 'dropped_cols', 'dropped_rows',
                  'clean_samples', 'clean_TC_exc', 'clean_FC_exc', 'clean_ENT_exc']]
df_drop.index.rename('beach', inplace=True)
df_drop.to_csv(os.path.join(beach_base_folder, dropfile))
