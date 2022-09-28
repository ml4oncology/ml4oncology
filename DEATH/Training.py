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

# In[33]:


get_ipython().run_line_magic('cd', '../')
# reloads all modules everytime before cell is executed (no need to restart kernel)
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[34]:


import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 150)
pd.set_option('display.max_rows', 150)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocess import (filter_ohip_data, process_ohip_data)
from src.utility import (twolevel, initialize_folders, load_predictions,
                         get_nunique_entries, get_nmissing, 
                         time_to_target_after_alarm, time_to_alarm_after_service)
from src.summarize import (data_characteristic_summary, feature_summary, 
                           subgroup_performance_summary, serivce_request_summary)
from src.visualize import (importance_plot, subgroup_performance_plot, service_request_plot)
from src.config import (root_path, death_folder, split_date)
from src.prep_data import (PrepDataEDHD)
from src.train import (TrainML, TrainRNN, TrainENS)
from src.evaluate import (Evaluate)


# In[35]:


# config
processes = 64
target_keyword = 'Mortality'
main_dir = f'{root_path}/{death_folder}'
output_path = f'{main_dir}/models'
initialize_folders(output_path, extra_folders=['figures/important_groups'])


# # Prepare Data for Model Training

# In[4]:


prep = PrepDataEDHD(adverse_event='death')
model_data = prep.get_data(target_keyword, treatment_intent=['P'], verbose=True)
model_data


# In[5]:


sorted(model_data.columns.tolist())


# In[6]:


get_nunique_entries(model_data)


# In[7]:


get_nmissing(model_data, verbose=True)


# In[8]:


model_data = prep.get_data(target_keyword, missing_thresh=75, treatment_intent=['P'], verbose=True)
print(f"Size of model_data: {model_data.shape}")
print(f"Number of unique patients: {model_data['ikn'].nunique()}")
N = model_data.loc[model_data['30d Mortality'], 'ikn'].nunique()
print(f"Number of unique patients that died within 30 days after a treatment session: {N}")


# In[9]:


# NOTE: any changes to X_train, X_valid, etc will also be seen in dataset
kwargs = {'target_keyword': target_keyword, 'split_date': split_date}
dataset = X_train, X_valid, X_test, Y_train, Y_valid, Y_test = prep.split_data(prep.dummify_data(model_data.copy()), **kwargs)


# In[10]:


prep.get_label_distribution(Y_train, Y_valid, Y_test)


# # Train ML Models

# In[13]:


# Initialize Training class
train_ml = TrainML(dataset, output_path, n_jobs=processes)


# In[15]:


skip_alg = []
train_ml.tune_and_train(run_bayesopt=False, run_training=True, save_preds=True, skip_alg=skip_alg)


# # Train RNN Model

# In[130]:


# Include ikn to the input data 
X_train['ikn'] = model_data['ikn']
X_valid['ikn'] = model_data['ikn']
X_test['ikn'] = model_data['ikn']

# Initialize Training class 
train_rnn = TrainRNN(dataset, output_path)


# In[157]:


# Distrubution of the sequence lengths in the training set
dist_seq_lengths = X_train.groupby('ikn').apply(len)
dist_seq_lengths = dist_seq_lengths.clip(upper=dist_seq_lengths.quantile(q=0.999))
fig, ax = plt.subplots(figsize=(15, 3))
ax.grid(zorder=0)
sns.histplot(dist_seq_lengths, ax=ax, zorder=2, bins=int(dist_seq_lengths.max()))


# In[19]:


train_rnn.tune_and_train(run_bayesopt=False, run_training=True, run_calibration=True, save_preds=True)


# # Train ENS Model 
# Find Optimal Ensemble Weights

# In[11]:


# combine rnn and ml predictions
preds = load_predictions(save_dir=f'{output_path}/predictions')
preds_rnn = load_predictions(save_dir=f'{output_path}/predictions', filename='rnn_predictions')
for split, pred in preds_rnn.items(): preds[split]['RNN'] = pred
del preds_rnn
# Initialize Training Class
train_ens = TrainENS(dataset, output_path, preds)


# In[12]:


train_ens.tune_and_train(run_bayesopt=False, run_calibration=False, calibrate_pred=True)


# # Evaluate Models

# In[13]:


eval_models = Evaluate(output_path=output_path, preds=train_ens.preds, labels=train_ens.labels, orig_data=model_data)


# In[14]:


kwargs = {'get_baseline': True, 'display_ci': True, 'load_ci': True, 'save_ci': False, 'verbose': False}
eval_models.get_evaluation_scores(**kwargs)


# In[18]:


eval_models.plot_curves(curve_type='pr', legend_location='upper right', figsize=(12,18))
eval_models.plot_curves(curve_type='roc', legend_location='lower right', figsize=(12,18))
eval_models.plot_curves(curve_type='pred_cdf', figsize=(12,18)) # cumulative distribution function of model prediction
eval_models.plot_calibs(legend_location='upper left', figsize=(12,18)) 
# eval_models.plot_calibs(include_pred_hist=True, legend_location='upper left', figsize=(12,28), padding={'pad_y1': 0.3, 'pad_y0': 3.0})


# # Post-Training Analysis

# ## Study Characteristics

# In[19]:


data_characteristic_summary(eval_models, save_dir=f'{main_dir}/models/tables')


# ## Feature Characteristics

# In[20]:


feature_summary(eval_models, prep, target_keyword, save_dir=f'{main_dir}/models/tables').head(60)


# ## Threshold Operating Points

# In[21]:


pred_thresholds = np.arange(0.05, 0.51, 0.05)
thresh_df = eval_models.operating_points(algorithm='ENS', points=pred_thresholds, metric='threshold',
                                         include_outcome_recall=True, event_dates=prep.event_dates)
thresh_df


# ## Most Important Features/Feature Groups

# In[ ]:


get_ipython().system('python scripts/perm_importance.py --adverse-event DEATH')


# In[15]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot('ENS', eval_models.target_events, output_path, figsize=(6,30), top=10, importance_by='feature', padding={'pad_x0': 4.0})


# In[16]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot('ENS', eval_models.target_events, output_path, figsize=(6,30), top=10, importance_by='group', padding={'pad_x0': 1.2})


# ## Performance on Subgroups

# In[22]:


df = subgroup_performance_summary('ENS', eval_models, pred_thresh=0.2, display_ci=False, load_ci=False, save_ci=False)
df


# ## Decision Curve Plot

# In[23]:


result = eval_models.plot_decision_curve_analysis('ENS')
result['30d Mortality'].tail(n=100)


# ## 30 day Mortality

# ### All the Plots

# In[24]:


eval_models.all_plots_for_single_target(algorithm='ENS', target_event='30d Mortality', calib_ci=False)


# ### Subgroup Performance Plot

# In[25]:


groupings = {'Demographic': ['Entire Test Cohort', 'Age', 'Sex', 'Immigration', 'Neighborhood Income Quintile'],
             'Treatment': ['Entire Test Cohort', 'Regimen', 'Cancer Location', 'Days Since Starting Regimen']}
padding = {'pad_y0': 1.2, 'pad_x1': 2.6, 'pad_y1': 0.2}
for name, subgroups in groupings.items():
    subgroup_performance_plot(df, target_event='30d Mortality', subgroups=subgroups, padding=padding,
                              figsize=(12,24), save=True, save_dir=f'{output_path}/figures/subgroup_performance/{name}')
# PPV = 0.3 means roughly for every 3 alarms, 2 are false alarms and 1 is true alarm
# Sesnsitivity = 0.5 means roughly for every 2 true alarms, the model predicts 1 of them correctly
# Event Rate = 0.15 means true alarms occur 15% of the time


# ## 365 day Mortality

# ### Time to Death After First Alarm

# In[146]:


pred_thresh = 0.5
time_to_death = time_to_target_after_alarm(eval_models, prep.event_dates,
                                           target_event='365d Mortality', target_date_col='D_date', 
                                           split='Test', algorithm='ENS', pred_thresh=pred_thresh)
fig, ax = plt.subplots(figsize=(10, 4))
ax.grid(zorder=0)
sns.histplot(time_to_death, ax=ax, zorder=2)
ax.set(xlabel='Days', title=f'Time to Death After Risk Prediction > {pred_thresh}')
plt.show()


# # Palliative Consultation Service Analysis

# In[18]:


# Sensitivity Analysis: Get Service Request Summary of Only Billing Codes A945 and C945 
# Extract and Preprocess the OHIP Data
ohip = pd.read_csv(f'{root_path}/data/ohip.csv')
ohip = filter_ohip_data(ohip)
ohip = ohip[ohip['feecode'].isin(['A945', 'C945'])]
ohip['ikn'] = ohip['ikn'].astype(int)
# Process the OHIP Data
df = prep.event_dates[['visit_date']]
df['ikn'] = model_data['ikn']
event_dates = process_ohip_data(df, ohip)


# ## Summary of All Billing Codes

# In[29]:


# By Sessions
summary = serivce_request_summary(eval_models, prep.event_dates, target_event='365d Mortality', 
                                  thresholds=[0.1, 0.25, 0.5], by_patients=False)
service_request_plot(summary)
summary


# In[30]:


# By Patients
summary = serivce_request_summary(eval_models, prep.event_dates, target_event='365d Mortality', 
                                  thresholds=[0.1, 0.25, 0.5], by_patients=True)
service_request_plot(summary)
summary


# ## Summary of C945 & A945 Billing Codes

# In[31]:


# By Sessions
summary = serivce_request_summary(eval_models, event_dates, target_event='365d Mortality', 
                                  thresholds=[0.1, 0.25, 0.5], days_ago=365*5, by_patients=False)
service_request_plot(summary)
summary


# In[32]:


# By Patients
summary = serivce_request_summary(eval_models, event_dates, target_event='365d Mortality', 
                                  thresholds=[0.1, 0.25, 0.5], days_ago=365*5, by_patients=True)
service_request_plot(summary)
summary


# ## Time to Alarm After Palliative Consultation Service (C945 & A945 Billing Codes)
# Most recent service prior to treatment session

# In[26]:


# Cumulative Distribution of Months from Service Date to First Risk of Target Event
time_to_alarm = time_to_alarm_after_service(eval_models, event_dates, target_event='365d Mortality', 
                                            split='Test', algorithm='ENS', pred_thresh=0.5)
fig, ax = plt.subplots(figsize=(8, 8))
N = len(time_to_alarm)
ax.plot(time_to_alarm,  np.arange(N) / float(N))
ax.set(xlabel='Months', ylabel="Cumulative Proportion of Patients",
       title=f'Time to First Alarm After A945/C945 Palliative Consultation Service')
plt.show()


# # Scratch Notes

# ## Hyperparameters

# In[26]:


from src.utility import get_hyperparameters
get_hyperparameters(output_path)


# In[ ]:
