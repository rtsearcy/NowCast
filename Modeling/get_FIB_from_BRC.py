# get_FIB_from_BRC.py
# Grabs FIB data for specified sites and dates from the BRC database
# RTS - 02/18

import pyodbc
import pandas as pd

# Inputs
brc_db_file = 'N:\BEACH 2017-18\BRC2015.mdb'  # BRC Database
folder = 'Z:\Predictive Modeling\Phase III\Modeling\Summer_2018\Raw FIB\\' # Directory for output file
loc_file = folder + 'locations.csv' # CSV of BRC Site IDs [id] and Site Names [beach] (Names can be customized)
min_date = '2008-01-01' # For Summer 2018, 10y of data
season = 'Summer' # Summer, Winter, All

# Load Locations CSV
df_loc = pd.read_csv(loc_file, encoding= 'latin1')
brc_id = list(df_loc['id'].dropna())
df_loc.set_index('id', inplace=True)

# Open BRC Database
conn_str = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    r'DBQ=' + brc_db_file + ';'
    )
brc_db = pyodbc.connect(conn_str)
d = brc_db.cursor()

# For each BRC Beach ID, grab all sample data after minimum date
brc_query = 'SELECT * FROM tblPoop WHERE fpkLocId IN {}'.format(tuple(brc_id)) + 'AND fpkDate >= #' + min_date + '#'
df_brc = pd.read_sql_query(brc_query, brc_db)
df_brc = df_brc[['fpkDate', 'fldTotalVal', 'fldFecalVal', 'fldEnteroVal', 'fldNotes', 'fpkLocId']]
cols = ['date', 'TC', 'FC', 'ENT', 'sample_time', 'brc_LocId']  # Rename columns
df_brc.columns = cols
df_brc = df_brc.sort_values('date')

df_out = pd.DataFrame()
for b in brc_id:
    df = df_brc[df_brc['brc_LocId'] == b]
    df['date'] = pd.to_datetime(df['date'].loc[:])
    df.set_index('date', inplace=True)

    ##TODO Account for BLD samples
    # fib = ['TC','FC','ENT']
    # for f in fib:
    #     df[f].loc[df[f] < 10] = 10

    df.to_csv(folder + df_loc['beach'].loc[b].replace(' ', '_') + '_samples.csv', columns=['TC','FC','ENT','sample_time'])  # Save all samples to CSV, need all bc of FIB1

    if season == 'Summer':
        df = df[(df.index.month >= 4) & (df.index.month < 11)]
    elif season == 'Winter':
        df = df[(df.index.month <= 3) | (df.index.month >= 11)]

    if len(df)<2:
        continue
    N = len(df)
    sd = min(df.index)
    ed = max(df.index)
    years = round((ed-sd).total_seconds()/60/60/24/365,0)

    exc = ((df['TC'] > 10000)|(df['FC'] > 400)|(df['ENT'] > 104)).sum()
    TC_e = (df['TC'] > 10000).sum()
    FC_e = (df['FC'] > 400).sum()
    ENT_e = (df['ENT'] > 104).sum()
    TC_p = (df['TC'] > 10000).sum() / N
    FC_p = (df['FC'] > 400).sum() / N
    ENT_p = (df['ENT'] > 104).sum() / N

    if max(TC_p,FC_p,ENT_p) == TC_p:
        driver = 'TC'
    elif max(TC_p,FC_p,ENT_p) == FC_p:
        driver = 'FC'
    else:
        driver = 'ENT'

    out_dict = {
        'Name': df_loc['beach'].loc[b],
        'County': df_loc['county'].loc[b],
        'N': N,
        'Start Date': sd,
        'End Date': ed,
        'Years': years,
        'Post Days': exc,
        'TC Exc': TC_e,
        'FC Exc': FC_e,
        'ENT Exc': ENT_e,
        '% TC Exc': round(TC_p,3),
        '% FC Exc': round(FC_p,3),
        '% ENT Exc': round(ENT_p,3),
        # '% Exc TC Only': round(TC_o, 3),
        # '% Exc FC Only': round(FC_o, 3),
        # '% Exc ENT Only': round(ENT_o, 3),
        'Driver': driver  # max exc.
    }
    df_out = df_out.append(pd.DataFrame(out_dict, index=[b]))

df_out = df_out[['Name','County','N','Start Date','End Date','Years','Post Days','TC Exc','FC Exc','ENT Exc','% TC Exc','% FC Exc','% ENT Exc', 'Driver']]
#df_out = df_out[['Name','N','Start Date','End Date','Years','Post Days', '% TC Exc','% FC Exc','% ENT Exc','% Exc TC Only','% Exc FC Only','% Exc ENT Only', 'Driver']]
df_out.index.rename('BRC_ID',inplace=True)
df_out.to_csv(folder + 'sites_summary_' + season + '.csv')
print('yeah baby!')