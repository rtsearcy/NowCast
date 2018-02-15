# perfromance_eval2017.py - Outputs statistic on model performance from given start/stop dates
# - Sensitivity/Specificity/Pearson Correlation/RMSE for days with sample results by FIB AND by beach
# - Total post/no-post predictions by FIB/by beach

import pandas as pd
import numpy as np
import os
import sys
import datetime
import warnings

warnings.filterwarnings('ignore')

class Tee:
    def __init__(self,*files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        pass

def perf(df_model, df_obs, threshold):

    test = pd.concat([df_obs, df_model], axis=1).dropna()
    test.columns = ['Obs', 'Model']

    samples = len(test)
    exceedances = (test['Obs'] > threshold).sum()
    compliance = samples - exceedances

    tp = np.sum((test['Obs'] > threshold) & (test['Model'] > threshold))#True positives
    tn = np.sum((test['Obs'] < threshold) & (test['Model'] < threshold))#True negatives
    fp = np.sum((test['Obs'] < threshold) & (test['Model'] > threshold))#False positives
    fn = np.sum((test['Obs'] > threshold) & (test['Model'] < threshold))#False negative

    l_test= np.log10(test.astype(float))

    cor = l_test.corr()
    cor = cor.iloc[0][1]

    rmse = np.sqrt(np.mean((l_test['Model']-l_test['Obs'])**2))

    stats = {
        'Sensitivity': round(tp/exceedances,2),
        'Specificity': round(tn/compliance,2),
        'Total Correct': round(((tp + tn)/ samples),2),
        'Pearson Correlation': round(cor,3),
        'RMSE': round(rmse,3),
        'Exc.': exceedances,
        'Samples': int(samples)
    }
    return stats

##INPUTS##

base_folder = 'S:\SCIENCE & POLICY - work in progress\Beach Report Card\Predictive Modeling\Daily files\\2017\\'

beaches_dict = {'Cowell': 'Live as of 5/26/2017',
             'Arroyo Burro': 'Live as of 4/4/2017; BLR model for ENT as of 8/13/2017',
             'East':'Live as of 4/4/2017; BLR model for FC and ENT as of 8/13/2017',
             'Santa Monica Pier':'Live as of 4/4/2017',
             'Redondo Beach Pier':'Live as of 4/27/2017',
             'Belmont Pier':'Live as of 4/4/2017',
             'LB City Beach (5th Place)':'Live as of 4/27/2017',
             'Huntington (Brookhurst)': 'Emailing results as of 5/12/2017; Live as of 5/26/2017',
             'Doheny': 'Live as of 4/4/2017',
             'Moonlight':'Live as of 5/18/2017; BLR model for ENT as of 10/10/2017',
             'Main (Boardwalk)': 'PILOT'
             }

beaches = [   #'Cowell',
#                'Arroyo Burro',
#                'East',
#                'Santa Monica Pier',
#                'Redondo Beach Pier',
               'LB City Beach (5th Place)',
               'Belmont Pier',
              'Huntington (Brookhurst)',
              'Doheny',
              'Moonlight',
    'Main (Boardwalk)'

]

blr_models = [['Arroyo Burro','ENT'],['East','FC'],['East','ENT'],['Moonlight','ENT']]

date_str_file = datetime.date.strftime(datetime.date.today(), '%m%d%Y')
# old_stdout = sys.stdout
# old_stderr = sys.stderr
# log_file = open(base_folder +'logs\\performance_log_' + date_str_file + '.txt','w')
# sys.stdout = Tee(sys.stdout, log_file)
# sys.stderr = Tee(sys.stderr, log_file)

print('| - - -  Beach NowCast Status  - - - |\n')
c = 1
for b in beaches:
    if c < 10:
        print(str(c) + '.  ' + b + '  -  ' + beaches_dict[b])
    else:
        print(str(c) + '. ' + b + '  -  ' + beaches_dict[b])
    c+=1
print('\n\n')

total_pred = 0

for b in beaches:
    sd = '2017/04/04'
    #sd = '2017/05/26'
    ed = '2017/12/31'
    #ed = '2017/10/10'
    #print('Results for ' + b + ' for ' + sd + ' to ' + ed)
    fib_folder = base_folder + b + '\\fib'
    pred_folder = base_folder + b + '\\predictions'
    perf_folder = base_folder + b + '\\performance'
    var_folder = base_folder + b + '\\variables'
    beach_folder = base_folder + b

    # NowCast Predictions #
    df_pred = pd.DataFrame()
    for file in os.listdir(pred_folder):
        df_temp = pd.read_csv(os.path.join(pred_folder, file))
        df_pred = df_pred.append(df_temp)
    if len(df_pred) == 0:
        print('   No predictions found for ' + b +'\n')
        continue
    df_pred['Date'] = pd.to_datetime(df_pred['Date'])
    df_pred = df_pred.set_index('Date')
    df_pred = df_pred[sd:ed].dropna()  # date range indicated
    sd = datetime.date.strftime(df_pred.index[0],'%Y/%m/%d') #New start date if not exact with desired
    ed = datetime.date.strftime(df_pred.index[-1], '%Y/%m/%d')  # New end date if not exact with desired
    no_predictions = len(df_pred)
    total_pred += no_predictions
    #print('  Predictions found from ' + sd + ' to '+ ed)
    print('| - - -  ' + b + '  - - - |')
    print('\nResults for ' + sd + ' through ' + ed)
    print('\nDays of Predictions: ' + str(no_predictions))

    # FIB Samples #
    df_fib_all = pd.read_csv(os.path.join(fib_folder,b.replace(' ','_') + '_fib_samples_2017.csv'))
    df_fib_all['Date'] = pd.to_datetime(df_fib_all['Date'])
    df_fib_all = df_fib_all.set_index('Date')
    df_fib = df_fib_all[sd:ed].reindex(df_pred.index).dropna(how='all')#date range indicated
    df_fib_1 = df_fib_all.shift(1)[sd:ed].reindex(df_pred.index).dropna()
    no_samples = len(df_fib)
    if no_samples == 0:
        if 'TC' not in df_pred.columns:
            df_pred['TC'] = 0
        if 'FC' not in df_pred.columns:
            df_pred['FC'] = 0
        if 'ENT' not in df_pred.columns:
            df_pred['ENT'] = 0

        df_pred_post = (df_pred['TC'] > 10000) | (df_pred['FC'] > 400) | (df_pred['ENT'] > 104)
        pp_tc = (df_pred['TC'] > 10000).sum()
        pp_fc = (df_pred['FC'] > 400).sum()
        pp_ent = (df_pred['ENT'] > 104).sum()
        pred_post = df_pred_post.sum()
        pred_no_post = len(df_pred_post) - pred_post  # No. of predicted post days
        print('   Total ''Post'' Days Predicted at Beach: ' + str(pred_post))
        print('     TC : ' + str(pp_tc))
        print('     FC : ' + str(pp_fc))
        print('     ENT: ' + str(pp_ent))
        print('   Total ''No-Post'' Days Predicted at Beach: ' + str(pred_no_post))
        print('     TC : ' + str(len(df_pred['TC']) - pp_tc))
        print('     FC : ' + str(len(df_pred['FC']) - pp_fc))
        print('     ENT: ' + str(len(df_pred['ENT']) - pp_ent))
        print('\nNo sample data available for ' + b + ' in the desired range. Results cannot be calculated.\n\n')
        continue

    fib = list(df_pred.columns)

    # Performance Results by FIB #
    df_perf_fib = pd.DataFrame()
    for f in fib:
        if f == 'TC':
            thresh = 10000.0
        elif f == 'FC':
            thresh = 400.0
        elif f == 'ENT':
            thresh = 104.0

        if [b,f] in blr_models: ## BLR MODEL EVAL ##
            blr = 1
            blr_coef =  pd.read_csv(os.path.join(beach_folder, f + '_BLR_Model_Coefficients.csv'), header=None)
            blr_coef.columns = ['Variable', 'Coefficient']
            blr_coef.set_index('Variable', inplace=True)
            prob_thresh = float(blr_coef.loc['threshold'])

            #df_pred_spec = df_pred[f].copy()[df_pred[f] < 1]
            df_pred_spec = df_pred[f].copy()
            df_pred_spec[df_pred_spec > 1] = df_pred_spec[df_pred_spec > 1].apply(lambda x: 1 if x > thresh else 0)
            df_fib_spec = df_fib_all[f].copy()
            df_fib_sample = df_fib_spec.reindex(df_pred_spec.index).dropna()
            df_fib_sample = df_fib_sample.apply(lambda x: 1 if x > thresh else 0)
            df_fib_prev = df_fib_spec.shift(1).reindex(df_pred_spec.index).dropna()
            df_fib_prev = df_fib_prev.apply(lambda x: 1 if x > thresh else 0)

            cm_perf = perf(df_fib_prev, df_fib_sample, 0.5)
            cm_perf['RMSE'] = np.nan
            cm_perf['Pearson Correlation'] = np.nan

            df_nc_sample = pd.concat([df_pred_spec, df_fib_sample], axis=1).dropna()
            df_nc_sample.columns = ['NowCast', 'Obs']
            nc_perf = perf(df_nc_sample['NowCast'], df_nc_sample['Obs'], prob_thresh)
            nc_perf['RMSE'] = np.nan
            nc_perf['Pearson Correlation'] = np.nan

            #Prep for beach results
            print('   BLR Predictions (' + f + '): ' + str((df_pred[f] < 1).sum()) + ' (' + str(((df_pred[f] < 1) & (df_pred[f] > prob_thresh)).sum()) +' Post)')
            df_pred[f][df_pred[f] > 1] = df_pred[f][df_pred[f] > 1].apply(lambda x: 1 if x > thresh else 0)
            df_pred[f][df_pred[f] < 1] = df_pred[f][df_pred[f] < 1].apply(lambda x: 1 if x > prob_thresh else 0)


        else:
            blr = 0
        # Current Method #
            df_fib_spec = df_fib_all.copy()
            df_fib_spec = df_fib_spec[f]
            df_fib_sample = df_fib_spec[sd:ed].dropna()
            df_fib_prev = df_fib_spec.shift(1)[sd:ed].dropna()

        # NowCast #
            df_pred_spec = df_pred.copy()
            df_pred_spec = df_pred_spec[f]
            df_nc_sample = pd.concat([df_pred_spec,df_fib_sample], axis=1).dropna()
            df_nc_sample.columns = ['NowCast','Obs']

            df_fib_sample = df_fib_sample.reindex(df_pred_spec.index).dropna() # Drop samples on days where predictions were withheld
            df_fib_prev = df_fib_prev.reindex(df_fib_sample.index)

            cm_perf = perf(df_fib_prev, df_fib_sample, thresh)
            nc_perf = perf(df_nc_sample['NowCast'],df_nc_sample['Obs'], thresh)

            df_pred[f] = df_pred[f].apply(lambda x: 1 if x > thresh else 0)

        ind = [[f,f],['NowCast', 'Current Method']]
        df_out = pd.DataFrame([nc_perf,cm_perf], index=ind)
        df_perf_fib = df_perf_fib.append(df_out)

     # Performance by beach #

     # account for missing FIB
    skip = 0
    if 'TC' not in df_pred.columns:
        df_pred['TC'] = 0
        skip = 1
    if 'FC' not in df_pred.columns:
        df_pred['FC'] = 0
    if 'ENT' not in df_pred.columns:
        df_pred['ENT'] = 0

    # Current Method
    df_fib_post = (df_fib['TC'] > 10000) | (df_fib['FC'] > 400) | (df_fib['ENT'] > 104) #SAMPLES#
    df_fib_1_post = (df_fib_1['TC'] > 10000) | (df_fib_1['FC'] > 400) | (df_fib_1['ENT'] > 104)
    fib_post = df_fib_post.sum() #No. of observed postings
    fib_no_post = len(df_fib_post) - fib_post #No. of observed no-post days
    beach_fib_perf = perf(df_fib_1_post, df_fib_post, 0.5)  # Beach results for CM
    beach_fib_perf['RMSE'] = np.nan
    beach_fib_perf['Pearson Correlation'] = np.nan

    # NowCast
    df_pred_post = (df_pred['TC'] ==1) | (df_pred['FC'] ==1) | (df_pred['ENT'] ==1)
    pp_tc = int(df_pred['TC'].sum())
    pp_fc = int(df_pred['FC'].sum())
    pp_ent = int(df_pred['ENT'].sum())
    pred_post = int(df_pred_post.sum())
    pred_no_post = len(df_pred_post) - pred_post #No. of predicted post days
    print('\n   Total ''Post'' Days Predicted at Beach: ' + str(pred_post))
    if skip == 0:
        print('     TC : ' + str(pp_tc))
    print('     FC : ' + str(pp_fc))
    print('     ENT: ' + str(pp_ent))
    print('   Total ''No-Post'' Days Predicted at Beach: ' + str(pred_no_post))
    if skip == 0:
        print('     TC : ' + str(len(df_pred['TC']) - pp_tc))
    print('     FC : ' + str(len(df_pred['FC']) - pp_fc))
    print('     ENT: ' + str(len(df_pred['ENT']) - pp_ent))
    print('Number of Samples: ' + str(no_samples))

    df_pred_post = df_pred_post.loc[df_fib.index] #No. of predicted beach compliance days
    beach_pred_perf = perf(df_pred_post,df_fib_post,0.5) #Beach results for model
    beach_pred_perf['RMSE'] = np.nan
    beach_pred_perf['Pearson Correlation'] = np.nan

    df_out = pd.DataFrame([beach_pred_perf,beach_fib_perf], index = [['Overall','Overall'],['NowCast','Current Method']])
    df_perf_fib = df_perf_fib.append(df_out)

    cols = ['Sensitivity', 'Specificity', 'Total Correct', 'Pearson Correlation', 'RMSE', 'Exc.', 'Samples']
    df_perf_fib = df_perf_fib[cols]
    df_perf_fib = df_perf_fib.reindex(['TC', 'FC', 'ENT','Overall'], level=0) #reorder
    print('\nPerformance Results: ')
    pd.set_option('display.expand_frame_repr', False)
    print(df_perf_fib)
    print('\n')
    out_file = b.replace(' ','_') + '_FIB_Performance_' + sd.replace('/','') + '_' + ed.replace('/','') + '.csv'
    df_perf_fib.to_csv(os.path.join(perf_folder, out_file), float_format='%.3f')
    #print('\nFIB performance results saved to ' + out_file)

    ## Pearson Correlations of Model Variables to Log-Observed FIB
    #Loop through variable folder and collate variables
    df_vars = pd.DataFrame()
    for file in os.listdir(var_folder):
        df_temp = pd.read_csv(os.path.join(var_folder, file))
        df_vars = df_vars.append(df_temp)
    if len(df_pred) == 0:
        print('   No variables found for ' + b +'\n')
        continue
    df_vars['Unnamed: 0'] = pd.to_datetime(df_vars['Unnamed: 0'])
    df_vars = df_vars.set_index('Unnamed: 0')
    df_vars.index.rename('Date', inplace=True)

    #Match fib observations with variables
    df_vars = df_vars.reindex(df_fib.index)

    #LOOP - download model variables from FIB_Model_Coefficient_PM files
    for f in ['TC','FC','ENT']:
        coef_file = f + '_Model_Coefficients_PM.csv'
        if coef_file in os.listdir(beach_folder):
            print('\nModel Variable Pearson Correlations with log10(' + f + ') (' + b + ')')
            df_coef = pd.read_csv(os.path.join(beach_folder, coef_file), header=None)
            df_coef.columns = ['Variable', 'Coefficient']
            df_coef.set_index('Variable', inplace=True)
            df_coef.drop(['PM', '(Constant)'], inplace=True)

            for c in df_coef.index:  # Sum all variables
                test = pd.concat([np.log10(df_fib[f]), df_vars[c]], axis=1)
                cor = test.corr()
                cor = cor.iloc[0][1]
                if np.isnan(cor):
                    print('   ' + c + ': NA')
                else:
                    print('   ' + c + ': ' + str(round(cor, 3)))
        elif f + '_BLR_Model_Coefficients.csv' in os.listdir(beach_folder):
            print('\n' + f + ': Pearson Correlations not available for BLR models')

    #Calculate pearsons corr coef for all variables, Also save the all correlations
    df_corr = pd.DataFrame()
    for f in ['TC','FC','ENT']:
        df_corr_temp = pd.Series()
        for c in df_vars.columns:
            test = pd.concat([np.log10(df_fib[f]), df_vars[c]], axis=1)
            cor = test.corr()
            cor = cor.iloc[0][1]
            df_corr_temp.loc[c] = cor
        df_corr = pd.concat([df_corr, df_corr_temp], axis= 1)

    df_corr.columns = ['TC','FC','ENT']
    out_file = b.replace(' ','_') + '_Correlations_' + sd.replace('/','') + '_' + ed.replace('/','') + '.csv'
    df_corr.to_csv(os.path.join(perf_folder, out_file), float_format='%.3f')
    #print('\nAll Variable correlations saved to ' + out_file)
    print('\n\n')
    #Calculate p values too???

print('Total Predictions: ' + str(total_pred))

sys.stdout = old_stdout
sys.stderr = old_stderr
log_file.close()