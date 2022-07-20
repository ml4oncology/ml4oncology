"""
========================================================================
© 2018 Institute for Clinical Evaluative Sciences. All rights reserved.

TERMS OF USE:
##Not for distribution.## This code and data is provided to the user solely for its own non-commercial use by individuals and/or not-for-profit corporations. User shall not distribute without express written permission from the Institute for Clinical Evaluative Sciences.

##Not-for-profit.## This code and data may not be used in connection with profit generating activities.

##No liability.## The Institute for Clinical Evaluative Sciences makes no warranty or representation regarding the fitness, quality or reliability of this code and data.

##No Support.## The Institute for Clinical Evaluative Sciences will not provide any technological, educational or informational support in connection with the use of this code and data.

##Warning.## By receiving this code and data, user accepts these terms, and uses the code and data, solely at its own risk.
========================================================================
"""
#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('cd', '../')
# reloads all modules everytime before cell is executed (no need to restart kernel)
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib.pyplot as plt

from scripts.utility import (initialize_folders, load_predictions,
                             get_nunique_entries, get_nmissing, 
                             data_characteristic_summary, feature_summary, subgroup_performance_summary)
from scripts.visualize import (importance_plot, subgroup_performance_plot)
from scripts.config import (root_path, acu_folder)
from scripts.prep_data import (PrepDataEDHD)
from scripts.train import (TrainML, TrainRNN, TrainENS)
from scripts.evaluate import (Evaluate)


# In[3]:


# config
processes = 64
days = 30 # predict event within this number of days since chemo visit (the look ahead window)
target_keyword = f'_within_{days}days'
main_dir = f'{root_path}/{acu_folder}'
output_path = f'{main_dir}/models/within_{days}_days'
initialize_folders(output_path, extra_folders=['figures/important_groups'])


# # Prepare Data for Model Training

# In[4]:


# Prepare Data for Model Input
prep = PrepDataEDHD(adverse_event='acu')


# In[5]:


model_data = prep.get_data(target_keyword, verbose=True)
model_data


# In[6]:


sorted(model_data.columns.tolist())


# In[7]:


get_nunique_entries(model_data)


# In[8]:


get_nmissing(model_data, verbose=True)


# In[9]:


model_data = prep.get_data(target_keyword, missing_thresh=80, verbose=True)
print(f"Size of model_data: {model_data.shape}")
print(f"Number of unique patients: {model_data['ikn'].nunique()}")
N = model_data.loc[model_data['ACU_within_30days'], 'ikn'].nunique()
print(f"Number of unique patients that had ACU within 30 days after a treatment session: {N}")


# In[10]:


model_data, clip_thresholds = prep.clip_outliers(model_data, lower_percentile=0.001, upper_percentile=0.999)
clip_thresholds.columns = clip_thresholds.columns.str.replace('baseline_', '').str.replace('_count', '')
clip_thresholds


# In[11]:


# NOTE: any changes to X_train, X_valid, etc will also be seen in dataset
kwargs = {'target_keyword': target_keyword}
dataset = X_train, X_valid, X_test, Y_train, Y_valid, Y_test = prep.split_data(prep.dummify_data(model_data.copy()), **kwargs)


# In[12]:


prep.get_label_distribution(Y_train, Y_valid, Y_test)


# In[13]:


Y_train.columns = Y_train.columns.str.replace(target_keyword, '')
Y_valid.columns = Y_valid.columns.str.replace(target_keyword, '')
Y_test.columns = Y_test.columns.str.replace(target_keyword, '')


# # Train ML Models

# In[37]:


pd.set_option('display.max_columns', None)


# In[140]:


# Initialize Training class
train_ml = TrainML(dataset, output_path, n_jobs=processes)


# In[ ]:


skip_alg = []
train_ml.tune_and_train(run_bayesopt=False, run_training=True, save_preds=True, skip_alg=skip_alg)


# # Train RNN Model

# In[14]:


# Include ikn to the input data 
X_train['ikn'] = model_data['ikn']
X_valid['ikn'] = model_data['ikn']
X_test['ikn'] = model_data['ikn']

# Initialize Training class 
train_rnn = TrainRNN(dataset, output_path)


# In[15]:


# Distrubution of the sequence lengths in the training set
dist_seq_lengths = X_train.groupby('ikn').apply(len)
fig = plt.figure(figsize=(15, 3))
plt.hist(dist_seq_lengths, bins=100)
plt.grid()
plt.show()


# In[16]:


# A closer look at the samples of sequences with length 1 to 21
fig = plt.figure(figsize=(15, 3))
plt.hist(dist_seq_lengths[dist_seq_lengths < 21], bins=20)
plt.grid()
plt.xticks(range(1, 21))
plt.show()


# In[17]:


train_rnn.tune_and_train(run_bayesopt=False, run_training=False, run_calibration=True, save_preds=True)


# # Train ENS Model 
# Find Optimal Ensemble Weights

# In[14]:


# combine rnn and ml predictions
preds = load_predictions(save_dir=f'{output_path}/predictions')
preds_rnn = load_predictions(save_dir=f'{output_path}/predictions', filename='rnn_predictions')
for split, pred in preds_rnn.items(): preds[split]['RNN'] = pred
del preds_rnn
# Initialize Training Class
train_ens = TrainENS(dataset, output_path, preds)


# In[15]:


train_ens.tune_and_train(run_bayesopt=False, run_calibration=False, calibrate_pred=True)


# # Evaluate Models

# In[16]:


eval_models = Evaluate(output_path=output_path, preds=train_ens.preds, labels=train_ens.labels, orig_data=model_data)


# In[22]:


kwargs = {'get_baseline': True, 'display_ci': True, 'load_ci': True, 'save_ci': False, 'verbose': False}
eval_models.get_evaluation_scores(**kwargs)


# In[23]:


eval_models.plot_curves(curve_type='pr', legend_location='lower left', figsize=(12,18), save=False)
eval_models.plot_curves(curve_type='roc', legend_location='lower right', figsize=(12,18), save=False)
eval_models.plot_curves(curve_type='pred_cdf', figsize=(12,18), save=False) # cumulative distribution function of model prediction
eval_models.plot_calibs(legend_location='upper left', figsize=(12,18), save=False) 
# eval_models.plot_calibs(include_pred_hist=True, legend_location='upper left', figsize=(12,28), padding={'pad_y1': 0.3, 'pad_y0': 3.0})


# # Post-Training Analysis

# ## Study Characteristics

# In[98]:


data_characteristic_summary(eval_models, save_dir=f'{main_dir}/models')


# ## Feature Characteristics

# In[132]:


feature_summary(eval_models, prep, target_keyword, save_dir=f'{main_dir}/models').head(60)


# ## Threshold Operating Points

# In[38]:


pred_thresholds = np.arange(0.05, 0.51, 0.05)
thresh_df = eval_models.operating_points(algorithm='ENS', points=pred_thresholds, metric='threshold',
                                         include_outcome_recall=True, event_dates=prep.event_dates)
thresh_df


# ## Most Important Features/Feature Groups

# In[ ]:


get_ipython().system('python scripts/perm_importance.py --adverse-event ACU')


# In[17]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot('ENS', eval_models.target_types, output_path, figsize=(6,50), top=10, importance_by='feature', padding={'pad_x0': 4.0})


# In[18]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot('ENS', eval_models.target_types, output_path, figsize=(6,50), top=10, importance_by='group', padding={'pad_x0': 1.2})


# ## Performance on Subgroups

# In[134]:


df = subgroup_performance_summary('ENS', eval_models, pred_thresh=0.25, display_ci=False, load_ci=False, save_ci=False)
df


# ## Decision Curve Plot

# In[141]:


eval_models.plot_decision_curve_analysis('ENS', padding={'pad_x0': 1.0})


# ## ACU

# ### All the Plots

# In[26]:


eval_models.all_plots_for_single_target(algorithm='ENS', target_type='ACU')


# ### Subgroup Performance Plot

# In[136]:


subgroups = ['Entire Test Cohort', 'Age', 'Sex', 'Immigration', 'Regimen', 'Days Since Starting Regimen']
padding = {'pad_y0': 1.2, 'pad_x1': 2.6, 'pad_y1': 0.2}
subgroup_performance_plot(df, subgroups=subgroups, padding=padding, figsize=(12,24), save=True, save_dir=f'{output_path}/figures')
# PPV = 0.3 means roughly for every 3 alarms, 2 are false alarms and 1 is true alarm
# Sesnsitivity = 0.5 means roughly for every 2 true alarms, the model predicts 1 of them correctly
# Event Rate = 0.15 means true alarms occur 15% of the time


# # Scratch Notes

# ## Brooks 2 Variable Based Model

# In[137]:


df = prep.get_data(target_keyword)
print(f'Size of data = {len(df)}, Number of patients = {df["ikn"].nunique()}')
df = df.loc[Y_test.index]
print(f'Size of test data = {len(df)}, Number of patients = {df["ikn"].nunique()}')
df = df[df['baseline_sodium_count'].notnull() & df['baseline_albumin_count'].notnull()]
print(f'Size of test data with both sodium and albumin count = {len(df)}, Number of patients = {df["ikn"].nunique()}')
df = df[df['days_since_starting_chemo'] == 0] # very first treatment
print(f'Size of test data with only first day chemos = {len(df)}, Number of patients = {df["ikn"].nunique()}')


# In[6]:


def predict(df):
    x = 10.392 - 0.472*0.1*df['baseline_albumin_count'] - 0.075*df['baseline_sodium_count']
    return 1 / (1 + np.exp(-x))


# In[68]:


split = 'Test'
pred = predict(df)
labels = {split: Y_test.loc[df.index]}
preds = {split: {'ENS': train_ens.preds[split]['ENS'].loc[df.index],
                 'BRK': pd.DataFrame({col: pred for col in Y_test.columns})}}
eval_brooks_model = Evaluate(output_path='', preds=preds, labels=labels, orig_data=df)


# In[69]:


# label distribtuion
labels[split].apply(pd.value_counts)


# In[70]:


kwargs = {'algorithms': ['ENS', 'BRK'], 'splits': ['Test'], 'display_ci': True, 'save_score': False}
result = eval_brooks_model.get_evaluation_scores(**kwargs)
result


# In[71]:


eval_brooks_model.all_plots_for_single_target(algorithm='BRK', target_type='H', split='Test',
                                              n_bins=20, calib_strategy='quantile', figsize=(12,12), save=False)


# In[72]:


points = np.arange(0.05, 0.51, 0.05)
eval_brooks_model.operating_points('BRK', points, metric='threshold', target_types=['H'], split='Test', save=False)


# ### Compare with ENS

# In[73]:


eval_brooks_model.all_plots_for_single_target(algorithm='ENS', target_type='H', split='Test',
                                              n_bins=20, calib_strategy='quantile', figsize=(12,12), save=False)


# In[74]:


points = np.arange(0.05, 0.51, 0.05)
eval_brooks_model.operating_points('ENS', points, metric='threshold', target_types=['H'], split='Test', save=False)


# ## Hyperparameters

# In[16]:


from scripts.utility import get_hyperparameters
get_hyperparameters(output_path, days=days)
