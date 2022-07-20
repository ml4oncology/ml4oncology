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


import os
import tqdm
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt

from scripts.utility import (initialize_folders, load_predictions,
                             get_nunique_entries, get_nmissing, 
                             data_characteristic_summary, feature_summary, subgroup_performance_summary)
from scripts.visualize import (importance_plot, subgroup_performance_plot)
from scripts.config import (root_path, death_folder)
from scripts.prep_data import (PrepDataEDHD)
from scripts.train import (TrainML, TrainRNN, TrainENS)
from scripts.evaluate import (Evaluate)


# In[3]:


# config
processes = 64
target_keyword = 'Mortality'
main_dir = f'{root_path}/{death_folder}'
output_path = f'{main_dir}/models'
initialize_folders(output_path, extra_folders=['figures/important_groups'])


# # Prepare Data for Model Training

# In[4]:


# Prepare Data for Model Input
prep = PrepDataEDHD(adverse_event='death')


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
N = model_data.loc[model_data['30d Mortality'], 'ikn'].nunique()
print(f"Number of unique patients that died within 30 days after a treatment session: {N}")


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


# # Train ML Models

# In[14]:


pd.set_option('display.max_columns', None)


# In[15]:


# Initialize Training class
train_ml = TrainML(dataset, output_path, n_jobs=processes)


# In[18]:


skip_alg = []
train_ml.tune_and_train(run_bayesopt=False, run_training=False, save_preds=True, skip_alg=skip_alg)


# # Train RNN Model

# In[14]:


# Include ikn to the input data 
X_train['ikn'] = model_data['ikn']
X_valid['ikn'] = model_data['ikn']
X_test['ikn'] = model_data['ikn']

# Initialize Training class 
train_rnn = TrainRNN(dataset, output_path)


# In[17]:


# Distrubution of the sequence lengths in the training set
dist_seq_lengths = X_train.groupby('ikn').apply(len)
fig = plt.figure(figsize=(15, 3))
plt.hist(dist_seq_lengths, bins=100)
plt.grid()
plt.show()


# In[18]:


# A closer look at the samples of sequences with length 1 to 21
fig = plt.figure(figsize=(15, 3))
plt.hist(dist_seq_lengths[dist_seq_lengths < 21], bins=20)
plt.grid()
plt.xticks(range(1, 21))
plt.show()


# In[50]:


train_rnn.tune_and_train(run_bayesopt=False, run_training=False, run_calibration=False, save_preds=True)


# # Train ENS Model 
# Find Optimal Ensemble Weights

# In[13]:


# combine rnn and ml predictions
preds = load_predictions(save_dir=f'{output_path}/predictions')
preds_rnn = load_predictions(save_dir=f'{output_path}/predictions', filename='rnn_predictions')
for split, pred in preds_rnn.items(): preds[split]['RNN'] = pred
del preds_rnn
# Initialize Training Class
train_ens = TrainENS(dataset, output_path, preds)


# In[14]:


train_ens.tune_and_train(run_bayesopt=False, run_calibration=False, calibrate_pred=True)


# # Evaluate Models

# In[15]:


eval_models = Evaluate(output_path=output_path, preds=train_ens.preds, labels=train_ens.labels, orig_data=model_data)


# In[101]:


kwargs = {'get_baseline': True, 'display_ci': True, 'load_ci': True, 'save_ci': False, 'verbose': False, 
          'baseline_cols': ['regimen', 'intent_of_systemic_treatment']}
eval_models.get_evaluation_scores(**kwargs)


# In[102]:


eval_models.plot_curves(curve_type='pr', legend_location='lower left', figsize=(12,18))
eval_models.plot_curves(curve_type='roc', legend_location='lower right', figsize=(12,18))
eval_models.plot_curves(curve_type='pred_cdf', figsize=(12,18)) # cumulative distribution function of model prediction
eval_models.plot_calibs(legend_location='upper left', figsize=(12,18)) 
# eval_models.plot_calibs(include_pred_hist=True, legend_location='upper left', figsize=(12,28), padding={'pad_y1': 0.3, 'pad_y0': 3.0})


# # Post-Training Analysis

# ## Study Characteristics

# In[103]:


data_characteristic_summary(eval_models, save_dir=f'{main_dir}/models/tables')


# ## Feature Characteristics

# In[104]:


feature_summary(eval_models, prep, target_keyword, save_dir=f'{main_dir}/models/tables').head(60)


# ## Threshold Operating Points

# In[110]:


pred_thresholds = np.arange(0.05, 0.51, 0.05)
thresh_df = eval_models.operating_points(algorithm='ENS', points=pred_thresholds, metric='threshold',
                                         include_outcome_recall=True, event_dates=prep.event_dates)
thresh_df


# ## Ensemble Most Important Features/Feature Groups

# In[ ]:


get_ipython().system('python scripts/perm_importance.py --adverse-event DEATH')


# In[119]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot('ENS', eval_models.target_types, output_path, figsize=(6,30), top=10, importance_by='feature', padding={'pad_x0': 4.0})


# In[126]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot('ENS', eval_models.target_types, output_path, figsize=(6,30), top=10, importance_by='group', padding={'pad_x0': 1.2})


# ## Performance on Subgroups

# In[132]:


df = subgroup_performance_summary('ENS', eval_models, pred_thresh=0.2, display_ci=False, load_ci=False, save_ci=False)
df


# ## 30 day Mortality

# ### Ensemble All the Plots

# In[128]:


eval_models.all_plots_for_single_target(algorithm='ENS', target_type='30d Mortality', calib_ci=False)


# ### Ensemble Subgroup Performance Plot

# In[134]:


subgroups = ['Entire Test Cohort', 'Age', 'Sex', 'Immigration', 'Regimen', 'Days Since Starting Regimen']
padding = {'pad_y0': 1.2, 'pad_x1': 2.6, 'pad_y1': 0.2}
subgroup_performance_plot(df, target_type='30d Mortality', subgroups=subgroups, 
                          padding=padding, figsize=(12,24), save=True, save_dir=f'{output_path}/figures')
# PPV = 0.3 means roughly for every 3 alarms, 2 are false alarms and 1 is true alarm
# Sesnsitivity = 0.5 means roughly for every 2 true alarms, the model predicts 1 of them correctly
# Event Rate = 0.15 means true alarms occur 15% of the time


# # Scratch Notes

# ## Kaplan Meier Curve and Lifeline Plot

# In[23]:


from scripts.visualize import plot_lifeline, plot_km_curve
from scripts.survival import compute_survival


# In[17]:


event_dates = prep.event_dates
event_dates['ikn'] = model_data['ikn']
df = compute_survival(event_dates)


# In[18]:


plot_lifeline(df, n=100) # 100 patients


# In[19]:


plot_km_curve(df.sample(n=100, random_state=2)) # TESTING TESTING


# In[20]:


plot_km_curve(df)


# ## Treatment Recommender System

# In[41]:


from scripts.survival import get_subgroup, get_recommendation, evaluate_recommendation


# In[42]:


df = model_data # all data 
test_df = model_data.loc[Y_test.index] # test split data


# ### Pancreas cancer (C25), Palliative Intent (P), First Treatment Course, GEMCNPAC(W) vs FOLFIRINOX/MFOLFIRINOX

# In[43]:


regimens = ['gemcnpac(w)', 'folfirinox']
panc_cancer_codes = ['C25']
panc_cancer = get_subgroup(test_df, name='pancreas cancer', regimens=regimens, cancer_codes=panc_cancer_codes)
recommended_regimens = get_recommendation(train_ens, panc_cancer)
evaluate_recommendation(prep.event_dates, panc_cancer, recommended_regimens)


# ### Melanoma (872-879), Palliative Intent (P), First Treatment Course, NIVL+IPIL vs NIVL

# In[44]:


"""
Thoughts: probably hard for the model to find a distinguishable pattern 
          between patient data for NIVL and NIVL+IPIL, since they have very similar treatment regimen
"""
melanoma_codes = [f'87{i}' for i in range(2,10)]
regimens = ['nivl', 'nivl+ipil']
melanoma = get_subgroup(test_df, name='melanoma', regimens=regimens, cancer_codes=melanoma_codes, cancer_col='curr_morph_cd')
recommended_regimens = get_recommendation(train_ens, melanoma)
evaluate_recommendation(prep.event_dates, melanoma, recommended_regimens)


# ### Renal Cancer (C64, C65), Palliative Intent (P), First Treatment Course, NIVI+IPIL vs AXIT+PEMB

# In[45]:


"""
Thoughts: only 10 observations for axit+pemb, not enough data
"""
renal_cancer_codes = ['C64', 'C65']
regimens = ['nivl+ipil', 'axit+pemb']
renal_cancer = get_subgroup(df, name='renal cancer', regimens=regimens, cancer_codes=renal_cancer_codes)
recommended_regimens = get_recommendation(train_ens, renal_cancer)
evaluate_recommendation(prep.event_dates, renal_cancer, recommended_regimens)


# ### Stomach Cancer (C16), ECX vs ECF vs FOLFOX vs FOLFIRI

# In[50]:


stomach_cancer_codes = ['C16']
regimens = ['ecx', 'ecf', 'folfiri'] # 'folfox' DOES NOT EXIST
stomach_cancer = get_subgroup(test_df, name='stomach cancer', regimens=regimens, cancer_codes=stomach_cancer_codes, 
                              palliatative_intent=False, first_treatment_course=False)
recommended_regimens = get_recommendation(train_ens, stomach_cancer)
evaluate_recommendation(prep.event_dates, stomach_cancer, recommended_regimens)


# In[ ]:
