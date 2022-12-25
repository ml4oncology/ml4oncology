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
pd.set_option('display.max_columns', 150)
pd.set_option('display.max_rows', 150)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocess import (filter_ohip_data, process_ohip_data)
from src.utility import (
    twolevel, initialize_folders, load_predictions,
    get_nunique_categories, get_nmissing, 
    equal_rate_pred_thresh,
    time_to_target_after_alarm, time_to_alarm_after_pccs
)
from src.summarize import (
    data_characteristic_summary, feature_summary, 
    subgroup_performance_summary, 
    pccs_receival_summary, eol_chemo_receival_summary
)
from src.visualize import (
    get_bbox, importance_plot, subgroup_performance_plot,                
    time_to_event_plot, pccs_receival_plot,                   
    pccs_impact_plot, eol_chemo_impact_plot
)
from src.config import (root_path, death_folder, split_date)
from src.prep_data import (PrepDataEDHD)
from src.train import (TrainML, TrainRNN, TrainENS)
from src.evaluate import (Evaluate)


# In[3]:


# config
processes = 64
target_keyword = 'Mortality'
main_dir = f'{root_path}/{death_folder}'
output_path = f'{main_dir}/models'
initialize_folders(output_path, extra_folders=['figures/important_groups'])


# # Prepare Data for Model Training

# In[4]:


prep = PrepDataEDHD(adverse_event='death')
model_data = prep.get_data(target_keyword, treatment_intents=['P'], verbose=True)
model_data


# In[5]:


sorted(model_data.columns.tolist())


# In[6]:


get_nunique_categories(model_data)


# In[7]:


get_nmissing(model_data, verbose=True)


# In[8]:


prep = PrepDataEDHD(adverse_event='death') # need to reset
model_data = prep.get_data(target_keyword, missing_thresh=75, treatment_intents=['P'], verbose=True)
print(f"Size of model_data: {model_data.shape}")
print(f"Number of unique patients: {model_data['ikn'].nunique()}")
N = model_data.loc[model_data['30d Mortality'], 'ikn'].nunique()
print(f"Number of unique patients that died within 30 days after a treatment session: {N}")
N = model_data.loc[model_data['365d Mortality'], 'ikn'].nunique()
print(f"Number of unique patients that died within 365 days after a treatment session: {N}")


# In[9]:


# NOTE: any changes to X_train, X_valid, etc will also be seen in dataset
kwargs = {'target_keyword': target_keyword, 'split_date': split_date}
dataset = X_train, X_valid, X_test, Y_train, Y_valid, Y_test = prep.split_data(prep.dummify_data(model_data.copy()), **kwargs)


# In[10]:


prep.get_label_distribution(Y_train, Y_valid, Y_test)


# # Train ML Models

# In[11]:


# Initialize Training class
train_ml = TrainML(dataset, output_path, n_jobs=processes)


# In[13]:


skip_alg = []
train_ml.tune_and_train(run_bayesopt=False, run_training=True, save_preds=True, skip_alg=skip_alg)


# # Train RNN Model

# In[13]:


# Include ikn to the input data 
X_train['ikn'] = model_data['ikn']
X_valid['ikn'] = model_data['ikn']
X_test['ikn'] = model_data['ikn']

# Initialize Training class 
train_rnn = TrainRNN(dataset, output_path)


# In[14]:


# Distrubution of the sequence lengths in the training set
dist_seq_lengths = X_train.groupby('ikn').apply(len)
dist_seq_lengths = dist_seq_lengths.clip(upper=dist_seq_lengths.quantile(q=0.999))
fig, ax = plt.subplots(figsize=(15, 3))
ax.grid(zorder=0)
sns.histplot(dist_seq_lengths, ax=ax, zorder=2, bins=int(dist_seq_lengths.max()))


# In[15]:


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


kwargs = {'baseline_cols': ['regimen'], 'display_ci': True, 'load_ci': True, 'save_ci': True, 'verbose': False}
eval_models.get_evaluation_scores(**kwargs)


# In[15]:


eval_models.plot_curves(curve_type='pr', legend_location='upper right', figsize=(12,18), save=False)
eval_models.plot_curves(curve_type='roc', legend_location='lower right', figsize=(12,18), save=False)
eval_models.plot_curves(curve_type='pred_cdf', figsize=(12,18), save=False) # cumulative distribution function of model prediction
eval_models.plot_calibs(legend_location='upper left', figsize=(12,18), save=False) 
# eval_models.plot_calibs(include_pred_hist=True, legend_location='upper left', figsize=(12,28), padding={'pad_y1': 0.3, 'pad_y0': 3.0})


# # Post-Training Analysis

# In[24]:


event_dates = prep.event_dates[['visit_date', 'D_date', 'PCCS_date']]
event_dates['ikn'] = model_data['ikn']


# ## Study Characteristics

# In[125]:


data_characteristic_summary(eval_models, save_dir=f'{main_dir}/models/tables', partition='cohort', target_event='30d Mortality')


# ## Feature Characteristics

# In[21]:


feature_summary(eval_models, prep, target_keyword, save_dir=f'{main_dir}/models/tables').head(60)


# ## Most Important Features/Feature Groups

# In[ ]:


get_ipython().system('python scripts/perm_importance.py --adverse-event DEATH')


# In[20]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot('ENS', eval_models.target_events, output_path, figsize=(6,30), top=10, importance_by='feature', padding={'pad_x0': 4.0})


# In[21]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot('ENS', eval_models.target_events, output_path, figsize=(6,30), top=10, importance_by='group', padding={'pad_x0': 1.2})


# ## All the Plots

# In[20]:


eval_models.all_plots_for_single_target(alg='ENS', target_event='365d Mortality')


# In[21]:


alg, target_events, split = 'ENS', ['30d Mortality', '365d Mortality'], 'Test'
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,12))
axes = axes.flatten()
plt.subplots_adjust(hspace=0.2, wspace=0.2)
eval_models.plot_auc_curve(axes[0], alg, target_events, split, curve_type='roc')
eval_models.plot_calib(axes[1], alg, target_events, split, legend_location='lower right')
eval_models.plot_decision_curve(axes[2], alg, '30d Mortality', split, colors={'System': '#1f77b4', 'All': '#bcbd22', 'None': '#7f7f7f'})
eval_models.plot_decision_curve(axes[3], alg, '365d Mortality', split, colors={'System': '#ff7f0e', 'All': '#bcbd22', 'None': '#7f7f7f'})
for idx, filename in enumerate(['roc','calib', 'dc_30d_death', 'dc_365d_death']):
    filepath = f'{output_path}/figures/curves/{alg}_{filename}.jpg'
    pad_x0 = 0.90 if filename == 'dc_30d_death' else 0.75
    fig.savefig(filepath, bbox_inches=get_bbox(axes[idx], fig, pad_x0=pad_x0), dpi=300) 
    axes[idx].set_title(filename) # set title AFTER saving individual figures


# ## Threshold Operating Points

# In[19]:


pred_thresholds = np.arange(0.05, 1.01, 0.05)
thresh_df = eval_models.operating_points(
    algorithm='ENS', points=pred_thresholds, metric='threshold', include_outcome_recall=True, event_dates=event_dates
)
thresh_df


# ## Threshold Selection
# Select prediction threshold at which alarm rate and event rate are equal

# In[25]:


# year_mortality_thresh = 0.5
year_mortality_thresh = equal_rate_pred_thresh(eval_models, event_dates, target_event='365d Mortality')


# In[26]:


# month_mortality_thresh = 0.2
month_mortality_thresh = equal_rate_pred_thresh(eval_models, event_dates, target_event='30d Mortality')


# ## Performance on Subgroups

# In[27]:


df = subgroup_performance_summary(
    'ENS', eval_models, pred_thresh=[0.15, month_mortality_thresh, 0.3, 0.5, year_mortality_thresh], 
    display_ci=False, load_ci=False, save_ci=False, include_outcome_recall=True, event_dates=event_dates
)
df


# In[28]:


groupings = {
    'Demographic': [
        'Entire Test Cohort', 'Age', 'Sex', 'Immigration', 'Immigrant World Region of Birth', 
        'Neighborhood Income Quintile', 'Area of Residence'
    ],
    'Treatment': [
        'Entire Test Cohort', 'Regimen', 'Cancer Topography ICD-0-3', 'Days Since Starting Regimen'
    ]
}


# ## 30 day Mortality

# ### Subgroup Performance Plot

# In[29]:


padding = {'pad_y0': 1.2, 'pad_x1': 2.6, 'pad_y1': 0.2}
for name, subgroups in groupings.items():
    width = min(max(12, len(df.loc[subgroups])), 20)
    subgroup_performance_plot(
        df, target_event='30d Mortality', subgroups=subgroups, padding=padding, figsize=(width,30), 
        xtick_rotation=75, save=True, save_dir=f'{output_path}/figures/subgroup_performance/{name}'
    )
# PPV = 0.3 means roughly for every 3 alarms, 2 are false alarms and 1 is true alarm
# Sesnsitivity = 0.5 means roughly for every 2 true alarms, the model predicts 1 of them correctly
# Event Rate = 0.15 means true alarms occur 15% of the time


# ### Time to Death After First Alarm

# In[30]:


time_to_death = time_to_target_after_alarm(
    eval_models, event_dates, target_event='30d Mortality', target_date_col='D_date', 
    split='Test', algorithm='ENS', pred_thresh=month_mortality_thresh
)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
time_to_event_plot(time_to_death, ax=axes[0], plot_type='hist', xlabel='Months')
time_to_event_plot(time_to_death, ax=axes[1], plot_type='cdf', xlabel='Months', ylabel="Cumulative Proportion of Patients")
plt.savefig(f'{output_path}/figures/time_to_event/30d_Mortality.jpg', bbox_inches='tight', dpi=300)
for ax in axes: ax.set_title('Time to Death After First Alarm')


# ## 365 day Mortality

# ### Subgroup Performance Plot

# In[31]:


padding = {'pad_y0': 1.2, 'pad_x1': 2.6, 'pad_y1': 0.2}
for name, subgroups in groupings.items():
    width = min(max(12, len(df.loc[subgroups])), 20)
    subgroup_performance_plot(
        df, target_event='365d Mortality', subgroups=subgroups, padding=padding, figsize=(width,30), 
        xtick_rotation=75, save=True, save_dir=f'{output_path}/figures/subgroup_performance/{name}'
    )
# PPV = 0.3 means roughly for every 3 alarms, 2 are false alarms and 1 is true alarm
# Sesnsitivity = 0.5 means roughly for every 2 true alarms, the model predicts 1 of them correctly
# Event Rate = 0.15 means true alarms occur 15% of the time


# ### Time to Death After First Alarm

# In[32]:


time_to_death = time_to_target_after_alarm(
    eval_models, event_dates, target_event='365d Mortality', target_date_col='D_date', 
    split='Test', algorithm='ENS', pred_thresh=year_mortality_thresh
)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
xticks = range(0,int(time_to_death.max()),3)
time_to_event_plot(time_to_death, ax=axes[0], plot_type='hist',xticks=xticks, xlabel='Months')
time_to_event_plot(time_to_death, ax=axes[1], plot_type='cdf', xticks=xticks, xlabel='Months', ylabel="Cumulative Proportion of Patients")
plt.savefig(f'{output_path}/figures/time_to_event/365d_Mortality.jpg', bbox_inches='tight', dpi=300)
for ax in axes: ax.set_title('Time to Death After First Alarm')


# # Palliative Care Consultation Service (PCCS) Analysis

# In[33]:


pccs_result = pccs_receival_summary(
    eval_models, event_dates, pred_thresh=year_mortality_thresh, 
    algorithm='ENS', split='Test', target_event='365d Mortality'
)


# In[34]:


# For each patient, we take the first alarm incident or very last treatment session if no alarm incidents occured
# We check if palliative consultation was requested within the appropriate time frame (e.g. 5 years before the session to 3 months after the session)
for outcome_type, matrices in pccs_result.items():
    for subgroup_name, matrix in matrices.items():
        print(matrix.astype(int).to_string(), end='\n\n')


# In[35]:


pccs_receival_plot(pccs_result, target_event='365d Mortality')


# ## Time to Alarm After PCCS
# Most recent service prior to treatment session

# In[36]:


# Cumulative Distribution of Months from PCCS Date to First Risk of Target Event
time_to_alarm = time_to_alarm_after_pccs(
    eval_models, event_dates, target_event='365d Mortality', split='Test', algorithm='ENS', 
    pred_thresh=year_mortality_thresh
)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
xticks = range(0,int(time_to_alarm.max()),6)
time_to_event_plot(time_to_alarm, ax=axes[0], plot_type='hist', xticks=xticks, xlabel='Months')
time_to_event_plot(time_to_alarm, ax=axes[1], plot_type='cdf', xticks=xticks, xlabel='Months', ylabel="Cumulative Proportion of Patients")
for ax in axes: ax.set_title('Time to First Alarm After A945/C945')


# ## Model Impact

# In[37]:


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 12))
pccs_impact_plot(eval_models, event_dates, axes[0], pred_thresh=year_mortality_thresh, verbose=True)
pccs_impact_plot(eval_models, event_dates, axes[1], eval_method='count', pred_thresh=year_mortality_thresh, verbose=False)
plt.savefig(f'{output_path}/figures/model_impact/365d_Mortality.jpg', bbox_inches='tight', dpi=300)


# # Chemo at End-of-Life Analysis

# In[38]:


eol_chemo_result = eol_chemo_receival_summary(eval_models, event_dates, pred_thresh=month_mortality_thresh)
eol_chemo_result.astype(int)


# In[39]:


result = eol_chemo_result.copy()
old_col = 'Received Chemo Near EOL (%)'
new_col = 'Proportion of Patients that Received\nChemotherapy Near End-Of-Life'
result = result.T[[old_col]]
result[new_col] = result.pop(old_col)
result.columns = pd.MultiIndex.from_product([['30d Mortality'], result.columns])
padding = {'pad_y0': 1.2, 'pad_x1': 2.6, 'pad_y1': 0.2}
subgroup_performance_plot(result/100, target_event='30d Mortality', padding=padding, figsize=(18,3))


# ## Model Impact

# In[40]:


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 12))
eol_chemo_impact_plot(eval_models, event_dates, axes[0], pred_thresh=month_mortality_thresh, verbose=True)
eol_chemo_impact_plot(eval_models, event_dates, axes[1], eval_method='count', pred_thresh=month_mortality_thresh, verbose=False)
plt.savefig(f'{output_path}/figures/model_impact/30d_Mortality.jpg', bbox_inches='tight', dpi=300)


# # Scratch Notes

# ## PCCS Numbers Graph Plot

# In[41]:


import graphviz
from src.summarize import get_pccs_analysis_data

df = get_pccs_analysis_data(eval_models, event_dates, pred_thresh=year_mortality_thresh, verbose=False)
N = len(df)

mask = df['observed'] # mask of which patients experienced 365 day mortality
died, survived = df[mask], df[~mask]
n_died, n_survived  = len(died), len(survived)

mask = died['received_pccs']
died_without_pccs = died[~mask]
n_died_with_pccs = sum(mask) 

mask = survived['received_pccs']
survived_without_pccs = survived[~mask]
n_survived_with_pccs = sum(mask) 

f = lambda n: f'{n} ({n/N*100:.1f}%)'
total_str = f'Total\n{N} (100%)'
died_str = f'Died\n{f(n_died)}'
surv_str = f'Survived\n{f(n_survived)}'
died_with_pccs_str = f'Received PCCS\n{f(n_died_with_pccs)}'
died_without_pccs_str = f'Not Received PCCS\n{f(len(died_without_pccs))}'
surv_with_pccs_str = f'Received PCCS\n {f(n_survived_with_pccs)}'
surv_without_pccs_str = f'Not Received PCCS\n{f(len(survived_without_pccs))}'
died_without_pccs_pred_str = f'Alerted\n{f(sum(died_without_pccs["predicted"]))}'
died_without_pccs_no_pred_str = f'Not Alerted\n{f(sum(~died_without_pccs["predicted"]))}'
surv_without_pccs_pred_str = f'Alerted\n{f(sum(survived_without_pccs["predicted"]))}'
surv_without_pccs_no_pred_str = f'Not Alerted\n{f(sum(~survived_without_pccs["predicted"]))}'

d = graphviz.Digraph()
d.edge(total_str, died_str)
d.edge(died_str, died_with_pccs_str)
d.edge(died_str, died_without_pccs_str)
d.edge(died_without_pccs_str, died_without_pccs_pred_str)
d.edge(died_without_pccs_str, died_without_pccs_no_pred_str)
d.edge(total_str, surv_str)
d.edge(surv_str, surv_with_pccs_str)
d.edge(surv_str, surv_without_pccs_str)
d.edge(surv_without_pccs_str, surv_without_pccs_pred_str)
d.edge(surv_without_pccs_str, surv_without_pccs_no_pred_str)
d.render(filename='pccs_num_graph', directory=f'{output_path}/figures/output', format='png')
d


# ## Bias Among Subgroups Plot
# And How Model Can Mitigate Bias

# In[42]:


split = 'Test'
algorithm = 'ENS'
idxs = eval_models.labels[split].index
catcol = 'subgroup'

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18,12))
axes = axes.flatten()

# Urban vs Rural for Timely Receival of Early Palliative Care Consultation Service
pccs_col = 'Received PCCS (%)'
df = pd.DataFrame(
    data = [
        ['Overall', pccs_result['observed'][f'Entire {split} Cohort'].loc[pccs_col, 'Dies']],
        ['Rural', pccs_result['observed']['Area of Residence'].loc[pccs_col, ('Rural', 'Dies')]],
        ['Urban', pccs_result['observed']['Area of Residence'].loc[pccs_col, ('Urban', 'Dies')]],
    ], 
    columns=[catcol, pccs_col]
)
df[pccs_col] /= 100
sns.barplot(data=df, x=catcol, y=pccs_col, ax=axes[0])
axes[0].set_xlabel('Subgroup Population - Area of Residence')
axes[0].set_ylabel('Proportion of Patients that\nRecieved Specialized Palliative Care')

# Urban vs Rural for 365d Mortality AUPRC
target_event = '365d Mortality'
mask = model_data.loc[idxs, 'rural']
args = (axes[1], algorithm, [target_event])
kwargs = {'split': split, 'curve_type': 'pr'}
eval_models.plot_auc_curve(*args, **kwargs, mask_name='Overall')
eval_models.plot_auc_curve(*args, **kwargs, mask=mask, mask_name='Rural')
eval_models.plot_auc_curve(*args, **kwargs, mask=~mask, mask_name='Urban')

# Urban vs Rural for 365d Mortality Calibration
args = (axes[2], algorithm, [target_event])
kwargs = {'split': split}
eval_models.plot_calib(*args, **kwargs, mask_name='Overall', show_perf_calib=False)
eval_models.plot_calib(*args, **kwargs, mask=mask, mask_name='Rural', show_perf_calib=False)
eval_models.plot_calib(*args, **kwargs, mask=~mask, mask_name='Urban')

# Carrib/SSA vs EU/NA/A+NZ (World Region of Birth) for Receival of Chemotherapy near End-of-Life
eol_chemo_col = 'Received Chemo Near EOL (%)'
df = pd.DataFrame(
    data = [
        ['Overall', eol_chemo_result.loc[eol_chemo_col, (f'Entire {split} Cohort', '')]],
        # ['Immigrant', eol_chemo_result.loc[eol_chemo_col, (f'Immigration', 'Immigrant')]],
        ['Carrib/SSA', eol_chemo_result.loc[eol_chemo_col, ('World Region of Birth', 'Carrib/SSA')]],
        ['EU/NA/A+NZ', eol_chemo_result.loc[eol_chemo_col, ('World Region of Birth', 'EU/NA/A+NZ')]],
    ], 
    columns=[catcol, eol_chemo_col]
)
df[eol_chemo_col] /= 100
sns.barplot(data=df, x=catcol, y=eol_chemo_col, ax=axes[3])
axes[3].set_xlabel('Subgroup Population - World Region of Birth')
axes[3].set_ylabel('Proportion of Patients that\nRecieved Chemotherapy at End-of-Life')
text = 'Carrib - Carribean, SSA - Sub-Saharan Africa, EU - Europe, NA - North America, A+NZ - Australia + New Zealand'
axes[3].text(0, -0.2, text, transform=axes[3].transAxes)

# Carrib/SSA vs EU/NA/A+NZ for 30d Mortality AUPRC
target_event = '30d Mortality'
carrib_ssa_mask = model_data.loc[idxs, 'world_region_of_birth'] == 'Carrib/SSA'
eu_na_anz_mask = model_data.loc[idxs, 'world_region_of_birth'] == 'EU/NA/A+NZ'
args = (axes[4], algorithm, [target_event])
kwargs = {'split': split, 'curve_type': 'pr', 'legend_location': 'lower left'}
eval_models.plot_auc_curve(*args, **kwargs, mask_name='Overall')
# eval_models.plot_auc_curve(*args, **kwargs, mask=model_data.loc[idxs, 'is_immigrant'], mask_name='Immigrant')
eval_models.plot_auc_curve(*args, **kwargs, mask=carrib_ssa_mask, mask_name='Carrib/SSA')
eval_models.plot_auc_curve(*args, **kwargs, mask=eu_na_anz_mask, mask_name='EU/NA/A+NZ')

# Carrib/SSA vs EU/NA/A+NZ for 30d Mortality Calibration
args = (axes[5], algorithm, [target_event])
kwargs = {'split': split, 'legend_location': 'upper right'}
eval_models.plot_calib(*args, **kwargs, mask_name='Overall', show_perf_calib=False)
# eval_models.plot_calib(*args, **kwargs, mask=model_data.loc[idxs, 'is_immigrant'], mask_name='Immigrant', show_perf_calib=False)
eval_models.plot_calib(*args, **kwargs, mask=carrib_ssa_mask, mask_name='Carrib/SSA', show_perf_calib=False)
eval_models.plot_calib(*args, **kwargs, mask=eu_na_anz_mask, mask_name='EU/NA/A+NZ')

# save results
plt.savefig(f'{output_path}/figures/output/bias.jpg', bbox_inches='tight', dpi=300)


# ## Placing Dots on AUC Plots

# In[65]:


from sklearn.metrics import precision_score, recall_score
split = 'Test'
target_event = '365d Mortality'
idxs = eval_models.labels[split].index
df = event_dates.loc[idxs, ['ikn', 'PCCS_date']]
df['died'] = eval_models.labels[split][target_event]
df['received_pccs'] = event_dates.loc[df.index, 'PCCS_date'].notnull()
df = df.groupby('ikn').last() # Take the last session of each patient
ppv = precision_score(df['died'], df['received_pccs'])
sensitivity = recall_score(df['died'], df['received_pccs'])
specificity = recall_score(df['died'], df['received_pccs'], pos_label=False)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
eval_models.plot_auc_curve(axes[0], alg, [target_event], split, curve_type='roc')
axes[0].plot(1 - specificity, sensitivity, 'go')
eval_models.plot_auc_curve(axes[1], alg, [target_event], split, curve_type='pr')
axes[1].plot(sensitivity, ppv, 'go')


# ## Hyperparameters

# In[149]:


from src.utility import get_hyperparameters
get_hyperparameters(output_path)


# In[ ]:
