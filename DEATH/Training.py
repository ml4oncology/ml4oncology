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

from src.utility import (
    twolevel, initialize_folders, load_predictions,
    get_nunique_categories, get_nmissing, 
    time_to_x_after_y,
)
from src.summarize import (
    data_characteristic_summary, feature_summary, 
    subgroup_performance_summary, 
    get_pccs_analysis_data, get_eol_treatment_analysis_data,
    pccs_receival_summary, eol_treatment_receival_summary,
    epc_impact_summary,
)
from src.visualize import (
    get_bbox, importance_plot, subgroup_performance_plot,
    time_to_event_plot, pccs_receival_plot, post_pccs_survival_plot,
    remove_top_right_axis
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

# In[43]:


event_dates = prep.event_dates[['visit_date', 'D_date', 'first_PCCS_date', 'last_seen_date']]
event_dates['ikn'] = model_data['ikn']


# ## Study Characteristics

# In[17]:


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

# In[153]:


pred_thresholds = np.arange(0.05, 1.01, 0.05)
thresh_df = eval_models.operating_points(
    algorithm='ENS', points=pred_thresholds, metric='threshold', include_outcome_recall=True, event_dates=event_dates
)
thresh_df


# ## Threshold Selection

# In[16]:


# Select prediction threshold at which alarm rate and usual intervention rate is equal 
# (resource utility is equal for both usual care and model-guided care)
# year_mortality_thresh = equal_rate_pred_thresh(eval_models, event_dates, split='Valid', alg='ENS', target_event='365d Mortality')
year_mortality_thresh = 0.5


# In[17]:


month_mortality_thresh = 0.2


# ## Performance on Subgroups

# In[29]:


df = subgroup_performance_summary(
    'ENS', eval_models, pred_thresh=[0.15, month_mortality_thresh, 0.3, 0.5, year_mortality_thresh], 
    display_ci=False, load_ci=False, save_ci=False, include_outcome_recall=True, event_dates=event_dates
)
df


# In[30]:


groupings = {
    'Demographic': [
        'Entire Test Cohort', 'Age', 'Sex', 'Immigration', 'Neighborhood Income Quintile', 'Area of Residence'
    ],
    'Treatment': [
        'Entire Test Cohort', 'Regimen', 'Cancer Topography ICD-0-3', 'Days Since Starting Regimen'
    ]
}


# ## 30 day Mortality

# ### Subgroup Performance Plot

# In[31]:


padding = {'pad_y0': 1.2, 'pad_x1': 2.7, 'pad_y1': 0.2}
width = {'Demographic': 18, 'Treatment': 12}
for name, subgroups in groupings.items():
    subgroup_performance_plot(
        df, target_event='30d Mortality', subgroups=subgroups, padding=padding, figsize=(width[name],30), 
        save=True, save_dir=f'{output_path}/figures/subgroup_performance/{name}'
    )
# PPV = 0.3 means roughly for every 3 alarms, 2 are false alarms and 1 is true alarm
# Sesnsitivity = 0.5 means roughly for every 2 true alarms, the model predicts 1 of them correctly
# Event Rate = 0.15 means true alarms occur 15% of the time


# ### Time to Death After First Alarm

# In[59]:


time_to_death = time_to_x_after_y(
    eval_models, event_dates, x='death', y='first_alarm', target_event='30d Mortality', 
    split='Test', algorithm='ENS', pred_thresh=month_mortality_thresh
)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
time_to_event_plot(time_to_death, ax=axes[0], plot_type='hist', xlabel='Months')
time_to_event_plot(time_to_death, ax=axes[1], plot_type='cdf', xlabel='Months', ylabel="Cumulative Proportion of Patients")
plt.savefig(f'{output_path}/figures/time_to_event/30d_Mortality.jpg', bbox_inches='tight', dpi=300)
for ax in axes: ax.set_title('Time to Death After First Alarm')


# ## 365 day Mortality

# ### Subgroup Performance Plot

# In[33]:


padding = {'pad_y0': 1.2, 'pad_x1': 2.7, 'pad_y1': 0.2}
width = {'Demographic': 18, 'Treatment': 12}
for name, subgroups in groupings.items():
    subgroup_performance_plot(
        df, target_event='365d Mortality', subgroups=subgroups, padding=padding, figsize=(width[name],30), 
        save=True, save_dir=f'{output_path}/figures/subgroup_performance/{name}'
    )
# PPV = 0.3 means roughly for every 3 alarms, 2 are false alarms and 1 is true alarm
# Sesnsitivity = 0.5 means roughly for every 2 true alarms, the model predicts 1 of them correctly
# Event Rate = 0.15 means true alarms occur 15% of the time


# ### Time to Death After First Alarm

# In[61]:


time_to_death = time_to_x_after_y(
    eval_models, event_dates, x='death', y='first_alarm', target_event='365d Mortality', 
    split='Test', algorithm='ENS', pred_thresh=year_mortality_thresh
)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
xticks = range(0,int(time_to_death.max()),3)
time_to_event_plot(time_to_death, ax=axes[0], plot_type='hist',xticks=xticks, xlabel='Months')
time_to_event_plot(time_to_death, ax=axes[1], plot_type='cdf', xticks=xticks, xlabel='Months', ylabel="Cumulative Proportion of Patients")
plt.savefig(f'{output_path}/figures/time_to_event/365d_Mortality.jpg', bbox_inches='tight', dpi=300)
for ax in axes: ax.set_title('Time to Death After First Alarm')


# # Palliative Care Consultation Service (PCCS) Analysis

# In[18]:


pccs_df = get_pccs_analysis_data(
    eval_models, event_dates, pred_thresh=year_mortality_thresh, 
    algorithm='ENS', split='Test', target_event='365d Mortality'
)


# In[19]:


pccs_result = pccs_receival_summary(pccs_df, split='Test')
for subgroup_name, matrix in pccs_result.items():
    print(matrix.astype(int).to_string(), end='\n\n')


# In[20]:


pccs_receival_plot(pccs_result, target_event='365d Mortality')


# ## Model Impact

# In[21]:


# EPC - Early Palliative Care (receival of early PCCS)
impact = epc_impact_summary(pccs_df, no_alarm_strategy='uc')
impact


# In[22]:


impact / len(pccs_df)


# In[23]:


def epc_impact_plot(impact):
    rate = impact.loc['Died Without EPC'] / len(pccs_df)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
    bar = ax.bar(x=rate.index, height=rate, color=['#1f77b4', '#ff7f0e'])
    ax.margins(0.1)
    ax.set_ylabel('Propotion Of Patients Who Died\nWithout Early Palliative Care')
    remove_top_right_axis(ax)
    
epc_impact_plot(impact)
plt.savefig(f'{output_path}/figures/model_impact/Died_Without_EPC_Rate.jpg', bbox_inches='tight', dpi=300)


# In[26]:


kmfs = post_pccs_survival_plot(eval_models, event_dates, verbose=True, pred_thresh=year_mortality_thresh)
plt.savefig(f'{output_path}/figures/model_impact/KM_curve.jpg', bbox_inches='tight', dpi=300)


# In[27]:


def get_surv_prob(kmfs, months_after=24):
    df = pd.DataFrame()
    for kmf, care_name in kmfs:
        ci = kmf.confidence_interval_survival_function_
        lower, upper = ci[ci.index > months_after].iloc[0]
        surv_prob = kmf.predict(months_after)
        df.loc['Upper CI', care_name] = upper
        df.loc['Survival Probability', care_name] = surv_prob
        df.loc['Lower CI', care_name] = lower
    df['Difference'] = df['Model-Guided Care'] - df['Usual Care']
    return df

get_surv_prob(kmfs, months_after=24)


# # Chemo at End-of-Life Analysis

# In[28]:


eol_treatment_df = get_eol_treatment_analysis_data(eval_models, event_dates, pred_thresh=month_mortality_thresh)
eol_chemo_result = eol_treatment_receival_summary(eol_treatment_df)
eol_chemo_result.astype(int)


# In[29]:


result = eol_chemo_result.copy()
old_col = 'Received Chemo Near EOL (%)'
new_col = 'Proportion of Patients Who Received\nTreatment Near End-Of-Life'
result = result.T[[old_col]]
result[new_col] = result.pop(old_col)
result.columns = pd.MultiIndex.from_product([['30d Mortality'], result.columns])
padding = {'pad_y0': 1.2, 'pad_x1': 2.6, 'pad_y1': 0.2}
subgroup_performance_plot(result/100, target_event='30d Mortality', padding=padding, figsize=(18,3))


# # Scratch Notes

# ## PCCS Graph Plot

# In[31]:


from src.visualize import pccs_graph_plot
d = pccs_graph_plot(pccs_df)
d.render(filename='pccs_num_graph', directory=f'{output_path}/figures/output', format='png')
d


# ## Bias Among Subgroups Plot
# And How Model Can Mitigate Bias

# In[45]:


from src.visualize import epc_bias_mitigation_plot
idxs = eval_models.labels['Test'].index
rural_mask = model_data.loc[idxs, 'rural']
immigrant_mask = model_data.loc[idxs, 'is_immigrant']
female_mask = model_data.loc[idxs, 'sex'] == 'F'
subgroup_masks = {
    'Area of Residence': {'Urban': ~rural_mask, 'Rural': rural_mask},
    'Immigration': {'Recent Immigrant': immigrant_mask, 'Long-Term Resident': ~immigrant_mask},
    'Sex': {'Female': female_mask, 'Male': ~female_mask},
}
epc_bias_mitigation_plot(
    eval_models, pccs_result, subgroup_masks, save=True, save_path=f'{output_path}/figures/output'
)


# ## All Billing Codes
# TODO: Table of Billing Code Freq

# In[28]:


from src.preprocess import filter_ohip_data
ohip = pd.read_csv(f'{root_path}/data/ohip.csv')
ohip = filter_ohip_data(ohip)
initial_pccs_date = ohip.groupby('ikn')['servdate'].first()
initial_pccs_date.index = initial_pccs_date.index.astype(int)


# In[29]:


pccs_df['first_PCCS_date'] = pccs_df.index.map(initial_pccs_date)
pccs_df['received_early_pccs'] = pccs_df['first_PCCS_date'] < pccs_df['D_date'] - pd.Timedelta(days=180)
event_dates['first_PCCS_date'] = event_dates['ikn'].map(pccs_df['first_PCCS_date'])


# In[33]:


impact = epc_impact_summary(pccs_df, no_alarm_strategy='uc')
impact


# In[34]:


impact / len(pccs_df)


# In[42]:


epc_impact_plot(impact)
kmfs = post_pccs_survival_plot(eval_models, event_dates, verbose=True, pred_thresh=year_mortality_thresh)
plt.show()
get_surv_prob(kmfs, months_after=24)


# ## Equal Resource Utility Threshold

# In[53]:


from src.utility import equal_rate_pred_thresh
year_mortality_thresh = equal_rate_pred_thresh(eval_models, event_dates, split='Valid', alg='ENS', target_event='365d Mortality')


# In[54]:


pccs_df = get_pccs_analysis_data(
    eval_models, event_dates, pred_thresh=year_mortality_thresh, 
    algorithm='ENS', split='Test', target_event='365d Mortality'
)


# In[55]:


impact = epc_impact_summary(pccs_df, no_alarm_strategy='uc')
impact


# In[56]:


impact / len(pccs_df)


# In[58]:


epc_impact_plot(impact)
plt.savefig(f'{output_path}/figures/model_impact/Died_Without_EPC_Rate_equal_resource.jpg', bbox_inches='tight', dpi=300)
kmfs = post_pccs_survival_plot(eval_models, event_dates, verbose=True, pred_thresh=year_mortality_thresh)
plt.savefig(f'{output_path}/figures/model_impact/KM_curve_equal_resource.jpg', bbox_inches='tight', dpi=300)
plt.show()
get_surv_prob(kmfs, months_after=24)


# ## 9-month Before Death as EPC

# In[49]:


pccs_df = get_pccs_analysis_data(
    eval_models, event_dates, days_before_death=270, pred_thresh=year_mortality_thresh, 
    algorithm='ENS', split='Test', target_event='365d Mortality'
)


# In[50]:


impact = epc_impact_summary(pccs_df, no_alarm_strategy='uc')
impact


# In[51]:


impact / len(pccs_df)


# In[52]:


epc_impact_plot(impact)


# ## Time to First Alarm After First PCCS

# In[84]:


time_to_alarm = time_to_x_after_y(
    eval_models, event_dates, x='first_alarm', y='first_pccs', target_event='365d Mortality', 
    split='Test', algorithm='ENS', pred_thresh=year_mortality_thresh, clip=True,
)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
xticks = range(-60,61,6)
time_to_event_plot(time_to_alarm, ax=axes[0], plot_type='hist', xticks=xticks, xlabel='Months')
time_to_event_plot(time_to_alarm, ax=axes[1], plot_type='cdf', xticks=xticks, xlabel='Months', ylabel="Cumulative Proportion of Patients")
for ax in axes: ax.set_title('Time to First Alarm After First PCCS')


# ## Time to Death After First PCCS

# In[83]:


time_to_death = time_to_x_after_y(
    eval_models, event_dates, x='death', y='first_pccs', target_event='365d Mortality', 
    split='Test', algorithm='ENS', pred_thresh=year_mortality_thresh, clip=True,
)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
xticks = np.arange(0,81,6)
time_to_event_plot(time_to_death, ax=axes[0], plot_type='hist', xticks=xticks, xlabel='Months')
time_to_event_plot(time_to_death, ax=axes[1], plot_type='cdf', xticks=xticks, xlabel='Months', ylabel="Cumulative Proportion of Patients")
for ax in axes: ax.set_title('Time to Death After First PCCS')


# ## Experimental Visualizations

# In[76]:


impact_rate = impact.T / impact['Usual Care'].sum()
impact_rate.columns = impact_rate.columns.str.replace('Died ', '')
impact_rate.columns = impact_rate.columns.str.replace('EPC', 'Early Palliative Care')
ax = impact_rate.plot.barh(stacked=True, color=['#3d85c6', '#9fc5e8'], figsize=(6,6), width=0.7)
ax.set_xlabel('Propotion Of Patients Who Died')
ax.margins(0.1)
ax.legend(frameon=False, loc='upper right')
remove_top_right_axis(ax)


# In[77]:


from statsmodels.graphics.mosaicplot import mosaic
data = {(care, status): number for care, metrics in impact.to_dict().items() for status, number in metrics.items()} 
colors = ['#d9ead3', '#ff3636', '#91c579', '#ff6f65']
props = {key: {'color': colors[idx]} for idx, key in enumerate(data.keys())}
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
mosaic(data, labelizer=lambda k: data[k], properties=lambda k: props[k], ax=ax)
plt.show()


# ## Placing Dots on AUC Plots

# In[50]:


from sklearn.metrics import precision_score, recall_score, precision_recall_curve, average_precision_score, roc_curve, roc_auc_score
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))

Y_true = pccs_df['status'] == 'Dead'
Y_pred_prob = pccs_df['predicted_prob']
# Y_pred_bool = Y_pred_prob > year_mortality_thresh
Y_pred_bool = pccs_df['early_pccs_by_alert']

ppv = precision_score(Y_true, Y_pred_bool)
sensitivity = recall_score(Y_true, Y_pred_bool)
specificity = recall_score(Y_true, Y_pred_bool, pos_label=False)

fpr, tpr, thresholds = roc_curve(Y_true, Y_pred_prob)
score = roc_auc_score(Y_true, Y_pred_prob)
label = f'AUROC={score:.2f}'
axes[0].plot(fpr, tpr, label=label)
axes[0].plot(1 - specificity, sensitivity, 'go')
axes[0].set(xlabel= '1 - Specificity', ylabel='Sensitivity')
axes[0].legend()

precision, recall, thresholds = precision_recall_curve(Y_true, Y_pred_prob)
score = average_precision_score(Y_true, Y_pred_prob)
label = f'AUPRC={score:.2f}'
axes[1].plot(recall, precision, label=label)
axes[1].plot(sensitivity, ppv, 'go')
axes[1].set(xlabel='Sensitivity', ylabel='Positive Predictive Value')
axes[1].legend()


# ## Hyperparameters

# In[149]:


from src.utility import get_hyperparameters
get_hyperparameters(output_path)


# In[ ]:
