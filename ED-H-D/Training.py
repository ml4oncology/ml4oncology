#!/usr/bin/env python
# coding: utf-8

# use <b>./kevin_launch_jupyter-notebook_webserver.sh</b> instead of <b>launch_jupyter-notebook_webserver.sh</b> if you want to increase buffer memory size so we can load greater filesizes

# In[1]:


get_ipython().run_line_magic('cd', '../')
# reloads all modules everytime before cell is executed (no need to restart kernel)
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import sys


# In[3]:


env = 'myenv'
user_path = 'XXXXXX'
for i, p in enumerate(sys.path):
    sys.path[i] = sys.path[i].replace("/software/anaconda/3/", f"{user_path}/.conda/envs/{env}/")
sys.prefix = f'{user_path}/.conda/envs/{env}/'


# In[4]:


import os
import tqdm
import pandas as pd
import numpy as np
import pickle
from functools import partial
from collections import defaultdict

from scripts.utilities import (load_ml_model, load_ensemble_weights, save_predictions, load_predictions,
                               most_common_by_category, 
                               data_splits_summary, subgroup_performance_summary,
                               feat_importance_plot, subgroup_performance_plot, 
                               get_clean_variable_names)
from scripts.preprocess import (split_and_parallelize, replace_rare_col_entries)
from scripts.config import (root_path, esas_ecog_cols, calib_param_logistic)
from scripts.prep_data import (PrepDataEDHD)
from scripts.train import (Train)

import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier


# In[5]:


# config
processes = 64
days = 30 # predict event within this number of days since chemo visit
target_keyword = f'_within_{days}days'
main_dir = f'{root_path}/ED-H-D'
output_path = f'{main_dir}/models/ML/within_{days}_days'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    os.makedirs(f'{output_path}/confidence_interval')
    os.makedirs(f'{output_path}/perm_importance')
    os.makedirs(f'{output_path}/best_params')
    os.makedirs(f'{output_path}/predictions')
    os.makedirs(f'{output_path}/figures')
    os.makedirs(f'{output_path}/tables')


# # More Preprocessing

# In[6]:


# Prepare Data for Model Input
prep = PrepDataEDHD()


# In[7]:


model_data = prep.get_data(main_dir, target_keyword, verbose=True)
model_data


# In[8]:


sorted(model_data.columns.tolist())


# In[9]:


catcols = model_data.dtypes[model_data.dtypes == object].index.tolist()
pd.DataFrame(model_data[catcols].nunique(), columns=['number of unique entries']).T


# In[10]:


num_missing = model_data.isnull().sum() # number of nans for each column
num_missing = num_missing[num_missing != 0] # remove columns without missing values
print("Missing values\n--------------")
print(num_missing)


# In[11]:


other = ['intent_of_systemic_treatment', 'lhin_cd', 'curr_morph_cd', 'curr_topog_cd', 'body_surface_area']
missing = {'lab tests': num_missing[num_missing.index.str.contains('baseline')],
           'symptom values': num_missing[esas_ecog_cols],
           'other data': num_missing[other] }
for name, miss in missing.items():
    missing_percentage = (miss / len(model_data) * 100).round(2)
    print(f'{missing_percentage.min()}-{missing_percentage.max()}% of {name} were missing before treatment sessions')


# In[12]:


model_data = prep.get_data(main_dir, target_keyword)
print(f"Size of model_data: {model_data.shape}\nNumber of unique patients: {model_data['ikn'].nunique()}")


# In[13]:


model_data, clip_thresholds = prep.clip_outliers(model_data, lower_percentile=0.001, upper_percentile=0.999)

clip_thresholds.columns = clip_thresholds.columns.str.replace('baseline_', '').str.replace('_count', '')
clip_thresholds


# In[14]:


model_data = prep.dummify_data(model_data)
print(f"Size of model_data: {model_data.shape}\nNumber of unique patients: {model_data['ikn'].nunique()}")


# In[15]:


train, valid, test = prep.split_data(model_data, target_keyword=target_keyword, convert_to_float=False)


# # Model Training

# In[16]:


pd.set_option('display.max_columns', None)


# In[17]:


X_train, Y_train = train
X_valid, Y_valid = valid
X_test, Y_test = test


# In[18]:


# sanity check - make sure all columns are normalized
for X in [X_train, X_valid, X_test]:
    mean = X.mean()
    mask = ((0 <= mean) & (mean <= 1))
    assert(all(mask))


# In[19]:


prep.get_label_distribution(Y_train, Y_valid, Y_test)


# In[20]:


Y_train.columns = Y_train.columns.str.replace(target_keyword, '')
Y_valid.columns = Y_valid.columns.str.replace(target_keyword, '')
Y_test.columns = Y_test.columns.str.replace(target_keyword, '')


# In[21]:


class TrainEDHD(Train):
    def __init__(self, dataset, clip_thresholds=None, n_jobs=32):
        super(TrainEDHD, self).__init__(dataset, clip_thresholds, n_jobs)
        
    def get_LR_model(self, C, max_iter=100):
        params = {'C': C, 
                  'class_weight': 'balanced',
                  'max_iter': max_iter,
                  'random_state': 42, 
                  'solver': 'sag',
                  'tol': 1e-3}
        model = MultiOutputClassifier(CalibratedClassifierCV(
                                        self.ml_models['LR'](**params), 
                                        n_jobs=9,
                                        **calib_param_logistic), 
                                      n_jobs=9)
        return model


# In[22]:


# Initialize Training class
dataset = (X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
train = TrainEDHD(dataset, n_jobs=processes)


# In[ ]:


# Conduct Baysian Optimization
# best_params = {}
# for algorithm, model in train.ml_models.items():
#     if algorithm in []: continue # put the algorithms already trained and tuned in this list
#     best_param = train.bayesopt(algorithm, save_dir=f'{output_path}/best_params')
#     best_params[algorithm] = best_param
#     if algorithm == 'NN': best_param['max_iter'] = 50
#     train.train_model_with_best_param(algorithm, model, best_param, save_dir=output_path)

# Since it will take a very long time, use slurm to submit bayes opt job
get_ipython().system('sbatch slurm_main')


# In[25]:


# Optional: Retrain model using best parameters
best_params = {}
for algorithm in train.ml_models:
    filename = f'{output_path}/best_params/{algorithm}_classifier_best_param.pkl'
    with open(filename, 'rb') as file:
        best_param = pickle.load(file)
    best_params[algorithm] = best_param

for algorithm, model in tqdm.tqdm(train.ml_models.items()):
    if algorithm in ['RF', 'NN']: continue # put the algorithms already trained in this list
    best_param = best_params[algorithm]
    if algorithm == 'NN': 
        best_param['max_iter'] = 100
        best_param['verbose'] = True
    # NOTE: Logistic Regression requires greater memory to train (e.g. 512 GB)
    train.train_model_with_best_param(algorithm, model, best_param, save_dir=output_path)
    print(f'{algorithm} training completed!')


# In[23]:


# Optional: Load saved predictions
train.preds = load_predictions(save_dir=f'{output_path}/predictions')


# In[30]:


# Find optimal ensemble weights
ensemble_weights = train.bayesopt('ENS', save_dir=f'{output_path}/best_params')


# In[24]:


# Extract ensemble weights
ensemble_weights = load_ensemble_weights(save_dir=f'{output_path}/best_params', ml_models=train.ml_models)


# In[32]:


# Get Model Performance Scores
score_df = train.get_evaluation_scores(model_dir=output_path, splits=['Valid', 'Test'], ensemble_weights=ensemble_weights, 
                                       display_ci=True, load_ci=True, verbose=False)
score_df.loc[[i for i in score_df.index if 'AUROC' in i[1] or 'AUPRC' in i[1]]]


# In[33]:


train.plot_curves(save_dir=output_path, curve_type='pr', legend_location='upper right', get_ensemble=True, figsize=(12,15))


# In[34]:


train.plot_curves(save_dir=output_path, curve_type='roc', legend_location='lower right', get_ensemble=True, figsize=(12,15))


# In[35]:


train.plot_calibs(save_dir=output_path, figsize=(12,12), n_bins=20, calib_strategy='quantile', include_pred_hist=True)


# In[36]:


# cumulative distribution function of model predictions
train.plot_cdf_pred(save_dir=output_path)


# In[37]:


# Optional: Save predictions
save_predictions(train.preds, save_dir=f'{output_path}/predictions')


# # Post-Training Analysis

# In[25]:


model_data = prep.get_data(main_dir, target_keyword)


# ## Study Characteristics

# In[40]:


data_splits_summary(train, model_data, save_dir=f'{main_dir}/models/ML')


# ## Ensemble Threshold Op Points

# In[41]:


pred_thresholds = np.arange(0.05, 0.51, 0.05)
thresh_df = train.threshold_op_points(None, pred_thresholds, algorithm='ENS')
thresh_df.to_csv(f'{output_path}/tables/threshold_performance.csv')


# In[42]:


thresh_df


# ## Ensemble All the Plots for ACU

# In[45]:


train.all_plots_for_single_target(save_dir=output_path, algorithm='ENS', target_type='ACU')


# ## Ensemble Most Important Features

# In[ ]:


get_ipython().system('python scripts/perm_importance.py')


# In[26]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
figsize = (6, 50)
feat_importance_plot('ENS', train.target_types, output_path, figsize, top=10)


# ## LR weights

# In[73]:


LR_model = load_ml_model(output_path, 'LR')
LR_weights_df = train.get_LR_weights(LR_model)
LR_weights_df.to_csv(f'{output_path}/tables/LR_weights.csv', index_label='index')


# In[60]:


LR_weights_df = pd.read_csv(f'{output_path}/tables/LR_weights.csv')
LR_weights_df = LR_weights_df.set_index('index')
LR_weights_df.head(n=60) # .style.background_gradient(cmap='Greens') # weird, this adds bunch of trailing 0s to the values


# ## Ensemble Performance on Subgroups

# In[60]:


# compute ensemble predictions
ens_preds = defaultdict(list)
for algorithm in train.ml_models:
    preds = load_predictions(f'{output_path}/predictions', filename=f'subgroup_predictions_{algorithm}')
    for name, pred in preds.items():
        ens_preds[name].append(pred)
ens_preds = {name: np.average(preds, axis=0, weights=ensemble_weights) for name, preds in ens_preds.items()}
save_predictions(ens_preds, f'{output_path}/predictions', filename=f'subgroup_predictions_ENS')


# In[26]:


df = subgroup_performance_summary(output_path, 'ENS', model_data, train, save_preds=False, 
                                  load_preds=True, display_ci=True, load_ci=True)


# In[27]:


df


# In[59]:


subgroup_performance_plot(df, save_dir=f'{output_path}/figures')


# # Scratch Notes

# ### hyperparameters

# In[28]:


from scripts.utilities import get_hyperparameters
get_hyperparameters(main_dir)


# ### how to read multiindex csvs

# In[39]:


pd.read_csv(f'{output_path}/subgroup_performance_summary.csv', header=[0,1], index_col=[0,1])


# ### over under sample

# In[19]:


def over_under_sample(X, Y, undersample_factor=5, oversample_min_samples=5000, seed=42):
    # undersample
    n = Y.shape[0]
    nsamples = int(n / undersample_factor)
    mask = Y.sum(axis=1) == 0 # undersample the examples with none of the 9 positive targets ~XXXXX rows
    Y = pd.concat([Y[~mask], Y[mask].sample(nsamples, replace=False, random_state=seed)])

    # oversample
    label_counts = prep.get_label_distribution(Y)
    label_counts = label_counts[('Train', 'True')] # Train is just the default name, ignore it, its not a bug
    for col, n_pos_samples in label_counts.iteritems():
        if n_pos_samples < oversample_min_samples:
            Y = pd.concat([Y, Y[Y[col]].sample(oversample_min_samples - n_pos_samples, replace=True, random_state=seed)])
    
    X = X.loc[Y.index]
    return X, Y
X_train, Y_train = over_under_sample(X_train, Y_train, undersample_factor=5, oversample_min_samples=5000)
X_valid, Y_valid = over_under_sample(X_valid, Y_valid, undersample_factor=5, oversample_min_samples=1000)
prep.get_label_distribution(Y_train, Y_valid, Y_test)

