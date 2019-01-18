# hypothetical_model_tracker.py - Delivers performance results for how a model may have performed
# this season if it were implemented. Can be used to test replacement models
# RTS - 5/23/2017
# UPDATED: 05/16/2018

import pandas as pd
import numpy as np
import os
import sys
import datetime
import warnings
from sklearn.externals import joblib


warnings.filterwarnings('ignore')


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)

    def flush(self):
        pass


def model_eval(true, predicted, thresh=0.5):  # Model Performance
    if true.dtype == 'float':
        true = (true > thresh).astype(int)  # Convert to binary
    if predicted.dtype == 'float':
        predicted = (predicted > thresh).astype(int)

    samples = len(true)
    exc = true.sum()
    tp = np.sum((true > thresh) & (predicted > thresh))  # True positives
    tn = np.sum((true < thresh) & (predicted < thresh))  # True negatives
    fp = np.sum((true < thresh) & (predicted > thresh))  # False positives
    fn = np.sum((true > thresh) & (predicted < thresh))  # False negative

    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    acc = (tn + tp) / samples

    out = {'Sensitivity': round(sens, 3), 'Specificity': round(spec, 3), 'Accuracy': round(acc, 3),
           'Samples': samples, 'Exceedances': exc}

    return out


# INPUTS
b = 'Long Beach 5th'  # beach name
f = 'ENT'  # FIB
model = 'JK7525'  # Model split
mn = 1  # model_number (1,2,...) if multiple models in directory

sd = '2018/04/01'  # Start date of evaluation
ed = '2018/10/31'  # End date of evaluation
debug = 0

# Model Folder
modeling_base_folder = 'Z:\Predictive Modeling\Phase III\Modeling\Summer_2018\Beaches'
modeling_beach_folder = os.path.join(modeling_base_folder, b.replace(' ', '_') + '\\models')
out_folder = os.path.join(modeling_beach_folder, 'hypothetical_models')

# Variable Data
base_folder = 'S:\SCIENCE & POLICY\\NowCast\Daily files\\2018'
beach_folder = os.path.join(base_folder, 'beaches\\' + b.replace(' ', '_'))
fib_folder = os.path.join(beach_folder, 'fib')
var_folder = os.path.join(beach_folder, 'variables')

fib_thresh = {'TC': 10000, 'FC': 400, 'ENT': 104}

# Create outfolder if it doesn't exists
os.makedirs(out_folder, exist_ok=True)

# Load Model
model_folder = os.path.join(modeling_beach_folder, f + '_' + model)
model_file = [x for x in os.listdir(model_folder) if x.startswith('model_')][mn-1]
coef_file = [x for x in os.listdir(model_folder) if x.startswith('coefficients_')][mn-1]

# Download model coefficients #
lm = joblib.load(os.path.join(model_folder, model_file))
df_coef = pd.read_csv(os.path.join(model_folder, coef_file), header=None)
df_coef.columns = ['Variable', 'Coefficient']
df_coef.set_index('Variable', inplace=True)

if debug == 0:  # Log output if running (won't let you debug if logging)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    log_file_name = model_file.replace('model_' + b.replace(' ', '_') + '_', '').replace('.pkl', '') + \
        '_hypothetical_performance_' + sd.replace('/', '') + '_' + ed.replace('/', '') + '.log'
    log_file = open(os.path.join(out_folder, log_file_name), 'w')
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

print('| - - - Hypothetical Model Performance for ' + b + ' (' + f + ')  - - - |\n')
# Print model
print('Model:  ' + model_file.replace('model_', '').replace('.pkl', '') + '\n')
print(df_coef)

# Load FIB data
df_fib = pd.read_csv(os.path.join(fib_folder, b.replace(' ', '_') + '_fib_samples.csv'))
if len(df_fib) == 0:
    print('   No fib data found for ' + b + '\n')
    sys.exit()
df_fib['date'] = pd.to_datetime(df_fib['date'])
df_fib = df_fib.set_index('date')
df_fib = df_fib[f].dropna().to_frame()
df_fib[f] = df_fib[f].dropna().apply(lambda x: 1 if x > fib_thresh[f] else 0)  # Post/No-Post
df_fib[f + '1'] = df_fib[f].shift(1)  # FIB1
if df_fib.index[0] > pd.Timestamp(sd):
    sd = datetime.date.strftime(df_fib.index[0], '%Y/%m/%d')  # New start date if not exact with desired
ed = datetime.date.strftime(df_fib.index[-1], '%Y/%m/%d')
df_fib = df_fib[sd:ed].astype(int)  # date range indicated
no_samples = len(df_fib)

# Load Variables
df_vars = pd.read_csv(os.path.join(var_folder, b.replace(' ', '_') + '_variables.csv'))
if len(df_vars) == 0:
    print('   No variables found for ' + b + '\n')
    sys.exit()
df_vars['date'] = pd.to_datetime(df_vars['date'])
df_vars = df_vars.set_index('date').sort_index()
df_vars = df_vars[sd:ed]

print('\nResults for ' + sd + ' through ' + ed + '\n')

# Hypothetical Predictions
tf = [x for x in df_coef.index if x in['PM', 'threshold']][0]
if tf == 'PM':
    tune = float(df_coef.loc['PM'])
    model_type = 'MLR-T'
else:
    tune = float(df_coef.loc['threshold'])
    model_type = 'BLR-T'
constant = float(df_coef.loc['constant'])
df_coef.drop([tf, 'constant'], inplace=True)
model_vars = list(df_coef.index)

check = False  # Check if all variables exist in dataset
for m in model_vars:
    if m not in df_vars.columns:
        print(m + ' not in variable dataset')
        check = True
if check:
    sys.exit()

df_vars = df_vars[model_vars].dropna()  # Remove variables not in model from dataset, drop days of missing data

# Number of Predictions (Post/No-Post) the model would have made
if model_type == 'MLR-T':
    thresh = np.log10(fib_thresh[f])
    #  all_preds = lm.predict(df_vars.values)
    all_preds = tune * lm.predict(df_vars.values)

else:  # BLR-T
    thresh = tune
    lm.coef_ = lm.coef_.reshape(1, -1)
    all_preds = lm.predict_proba(df_vars.values)[:, 1]

total_preds = len(all_preds)
post_preds = int((all_preds > thresh).sum())
nopost_preds = int(total_preds - post_preds)
print('Prediction Statistics:')
print('   Total Predictions - ' + str(total_preds))
print('   Post Predictions - ' + str(post_preds))
print('   No-Post Predictions - ' + str(nopost_preds) + '\n')

# Performance Results
# Current Method
cm_perf = model_eval(df_fib[f], df_fib[f + '1'])

# NowCast
df_vars = df_vars.reindex(df_fib.index)
if model_type == 'MLR-T':
    preds = (tune*lm.predict(df_vars.values) > np.log10(fib_thresh[f])).astype(int)
else:  # BLR-T
    lm.coef_ = lm.coef_.reshape(1, -1)
    preds = (lm.predict_proba(df_vars.values)[:, 1] > tune).astype(int)

nc_perf = model_eval(df_fib[f], preds)

ind = [model_type + '_' + model + '_' + str(len(df_coef)), 'Current Method']
df_out = pd.DataFrame([nc_perf, cm_perf], index=ind)
cols = ['Sensitivity', 'Specificity', 'Accuracy', 'Exceedances', 'Samples']
df_out = df_out[cols]
df_out.index.rename('Model', inplace=True)
print('Performance Results: ')
pd.set_option('display.expand_frame_repr', False)
print(df_out)
out_file = model_file.replace('model_' + b.replace(' ', '_') + '_', '').replace('.pkl', '') + \
           '_hypothetical_performance_' + sd.replace('/', '') + '_' + ed.replace('/', '') + '.csv'
df_out.to_csv(os.path.join(out_folder, out_file), float_format='%.3f')

if debug == 0:
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    log_file.close()
