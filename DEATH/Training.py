"""
========================================================================
© 2023 Institute for Clinical Evaluative Sciences. All rights reserved.

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
pd.set_option('display.max_columns', 150)
pd.set_option('display.max_rows', 150)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import root_path, death_folder, split_date
from src.evaluate import EvaluateClf
from src.impact import (
    get_intervention_analysis_data, get_pccs_analysis_data, get_eol_treatment_analysis_data,
    get_pccs_receival_by_subgroup, get_eol_treatment_receival_by_subgroup,
    get_pccs_impact
)
from src.model import SimpleBaselineModel
from src.prep_data import PrepDataEDHD
from src.summarize import data_description_summary, feature_summary
from src.train import Ensembler, LASSOTrainer, Trainer
from src.utility import (
    initialize_folders, load_pickle,
    get_nunique_categories, get_nmissing, get_clean_variable_names, get_units,
    equal_rate_pred_thresh,
    time_to_x_after_y, 
)
from src.visualize import (
    importance_plot, subgroup_performance_plot,
    time_to_event_plot, post_pccs_survival_plot, 
    epc_subgroup_plot, epc_impact_plot, epc_bias_mitigation_plot
)


# In[3]:


# config
processes = 64
target_keyword = 'Mortality'
main_dir = f'{root_path}/{death_folder}'
output_path = f'{main_dir}/models'
initialize_folders(output_path, extra_folders=['figures/time_to_event', 'figures/model_impact', 'figures/output'])


# # Prepare Data for Model Training

# In[4]:


prep = PrepDataEDHD(adverse_event='death', target_keyword=target_keyword)
model_data = prep.get_data(treatment_intents=['P'], verbose=True)
model_data


# In[5]:


sorted(model_data.columns.tolist())


# In[6]:


get_nunique_categories(model_data)


# In[7]:


get_nmissing(model_data, verbose=True)


# In[8]:


prep = PrepDataEDHD(adverse_event='death', target_keyword=target_keyword) # need to reset
model_data = prep.get_data(missing_thresh=80, treatment_intents=['P'], verbose=True)
X, Y, tag = prep.split_and_transform_data(model_data, split_date=split_date, remove_immediate_events=True)
# remove sessions in model_data that were excluded during split_and_transform
model_data = model_data.loc[tag.index]


# In[9]:


prep.get_label_distribution(Y, tag, with_respect_to='sessions')


# In[10]:


prep.get_label_distribution(Y, tag, with_respect_to='patients')


# In[11]:


# Convenience variables
train_mask, valid_mask, test_mask = tag['split'] == 'Train', tag['split'] == 'Valid', tag['split'] == 'Test'
X_train, X_valid, X_test = X[train_mask], X[valid_mask], X[test_mask]
Y_train, Y_valid, Y_test = Y[train_mask], Y[valid_mask], Y[test_mask]


# ## Study Characteristics

# In[13]:


subgroups = [
    'sex', 'immigration', 'birth_region', 'language', 'income', 'area_density',
    'regimen', 'cancer_type', 'cancer_location', 'target'
]
data_description_summary(
    model_data, Y, tag, save_dir=f'{output_path}/tables', partition_method='cohort', 
    target_event='30d Mortality', subgroups=subgroups
)


# ## Feature Characteristics

# In[14]:


df = prep.ohe.encode(model_data.copy(), verbose=False) # get original (non-normalized, non-imputed) data one-hot encoded
df = df[train_mask].drop(columns=['ikn'] + Y.columns.tolist())
feature_summary(
    df, save_dir=f'{output_path}/tables', deny_old_survey=True, event_dates=prep.event_dates[train_mask]
).head(60)


# # Train Models

# In[ ]:


get_ipython().system('python scripts/cross_validate_pipeline.py --adverse-event DEATH --algorithm NN --output-path {output_path} --bayesopt')


# ## LASSO Model

# In[32]:


lasso_trainer = LASSOTrainer(X, Y, tag, output_path, target_event='365d Mortality')


# In[23]:


lasso_trainer.run(grid_search=True, train=True, save=True, C_search_space=np.geomspace(3.9e-5, 1, 100))


# In[35]:


C_search_space = [3.9e-5, 4.3e-5, 4.52e-5, 4.8e-5, 5.9e-5, 8.4e-5]
lasso_trainer.run(grid_search=True, train=False, save=False, C_search_space=C_search_space)


# In[25]:


gs_results = pd.read_csv(f'{output_path}/tables/grid_search.csv')
gs_results = gs_results.sort_values(by='n_feats')
params = lasso_trainer.select_param(gs_results) # model with least complexity while AUROC upper CI >= max AUROC
ax = gs_results.plot(x='n_feats', y='AUROC', marker='o', markersize=4, legend=False, figsize=(6,6))
ax.set(xlabel='Number of Features', ylabel='AUROC')
plt.savefig(f'{output_path}/figures/LASSO_score_vs_num_feat.jpg', dpi=300)


# In[27]:


model = load_pickle(output_path, 'LASSO')
idx = list(Y.columns).index(lasso_trainer.target_event)
coef = lasso_trainer.get_coefficients(model, estimator_idx=idx)
assert len(coef) == params['n_feats']
intercept = lasso_trainer.get_intercept(model, estimator_idx=idx)

# Clean the feature names
rename_map = {name: f'{name} ({unit})' for name, unit in get_units().items()}
coef = coef.rename(index=rename_map)
coef.index = get_clean_variable_names(coef.index)

# Include the intercept
coef['Intercept'] = intercept

coef.to_csv(f'{output_path}/tables/LASSO_coefficients.csv')
coef.head(n=100)


# ## Main Models

# In[37]:


# Distrubution of the sequence lengths in the training set
dist_seq_lengths = X_train.groupby(tag.loc[train_mask, 'ikn']).apply(len)
dist_seq_lengths = dist_seq_lengths.clip(upper=dist_seq_lengths.quantile(q=0.999))
fig, ax = plt.subplots(figsize=(15, 3))
ax.grid(zorder=0)
sns.histplot(dist_seq_lengths, ax=ax, zorder=2, bins=int(dist_seq_lengths.max()))


# In[ ]:


trainer = Trainer(X, Y, tag, output_path)
trainer.run(bayesopt=False, train=False, save_preds=False)


# ## ENS Model 
# Find Optimal Ensemble Weights

# In[35]:


# combine predictions
preds = {} # load_pickle(f'{output_path}/preds', 'all_pred')
for alg in ['LR', 'XGB', 'RF', 'NN', 'RNN']:
    preds.update(load_pickle(f'{output_path}/preds', f'{alg}_preds'))


# In[36]:


ensembler = Ensembler(X, Y, tag, output_path, preds)
ensembler.run(bayesopt=False, calibrate=False)


# # Evaluate Models

# In[37]:


# setup the final prediction and labels
preds, labels = ensembler.preds.copy(), ensembler.labels.copy()
preds.update(SimpleBaselineModel(model_data[['regimen']], labels).predict())
preds.update(load_pickle(f'{output_path}/preds', filename='LASSO_preds'))


# In[40]:


evaluator = EvaluateClf(output_path, preds, labels)
evaluator.get_evaluation_scores(display_ci=True, load_ci=True, save_ci=True)


# In[39]:


evaluator.plot_curves(curve_type='pr', legend_loc='upper right', figsize=(12,24), save=False)
evaluator.plot_curves(curve_type='roc', legend_loc='lower right', figsize=(12,24), save=False)
evaluator.plot_curves(curve_type='pred_cdf', figsize=(12,24), save=False) # cumulative distribution function of model prediction
evaluator.plot_calibs(legend_loc='upper left', figsize=(12,24), save=False) 
# evaluator.plot_calibs(
#     include_pred_hist=True, legend_loc='upper left', figsize=(12,28), padding={'pad_y1': 0.3, 'pad_y0': 3.0}, save=False
# )


# # Post-Training Analysis

# In[16]:


cols = ['visit_date', 'death_date', 'first_PCCS_date', 'first_visit_date', 'last_seen_date']
event_dates = prep.event_dates[cols].copy()
evaluator.orig_data = model_data


# ## Most Important Features/Feature Groups

# In[ ]:


get_ipython().system('python scripts/feat_imp.py --adverse-event DEATH --output-path {output_path}')


# In[21]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot('ENS', evaluator.target_events, output_path, figsize=(6,30), top=10, importance_by='feature', padding={'pad_x0': 4.0})


# In[22]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot('ENS', evaluator.target_events, output_path, figsize=(6,30), top=10, importance_by='group', padding={'pad_x0': 1.2})


# In[33]:


importance = pd.read_csv(f'{output_path}/perm_importance/ENS_feature_importance.csv').set_index('index')
auroc_scores = evaluator.get_evaluation_scores(save_score=False).loc[('ENS', 'AUROC Score'), 'Test']
pd.concat([importance.sum(), 1 - auroc_scores], axis=1, keys=['Cumulative Importance', '1 - AUROC'])


# ## All the Plots

# In[145]:


evaluator.all_plots_for_single_target(alg='ENS', target_event='365d Mortality')


# ## Decision Curves

# In[146]:


output = evaluator.plot_decision_curves(alg='ENS', save=False)
output['365d Mortality'].query('0.5 < Threshold < 0.7')


# ## Threshold Operating Points

# In[17]:


pred_thresholds = np.arange(0.05, 1.01, 0.05)
perf_metrics = [
    'warning_rate', 'precision', 'recall', 'NPV', 'specificity', 
    'outcome_level_recall', 'first_warning_rate', 'first_warning_precision'
]
thresh_df = evaluator.operating_points(
    pred_thresholds, op_metric='threshold', perf_metrics=perf_metrics, event_df=event_dates.join(tag)
)
thresh_df


# ## Threshold Selection

# In[18]:


# Select prediction threshold at which alarm rate and usual intervention rate is equal 
# (resource utility is constant for both usual care and system-guided care)
year_mortality_thresh = equal_rate_pred_thresh(evaluator, event_dates, split='Test', alg='ENS', target_event='365d Mortality')
# year_mortality_thresh += 0.001 # manual adjustment
month_mortality_thresh = 0.2


# In[19]:


perf_metrics = ['precision', 'outcome_level_recall', 'first_warning_rate', 'first_warning_precision']
evaluator.operating_points(
    [year_mortality_thresh], op_metric='threshold', perf_metrics=perf_metrics, event_df=event_dates.join(tag), save=False
)


# In[20]:


# above is same as threshold 0.6
year_mortality_thresh = 0.6


# ## Performance on Subgroups

# In[21]:


subgroups = [
    'all', 'age', 'sex', 'immigrant', 'language', 'arrival', 'income', 'world_region_of_birth',
    'area_density', 'regimen', 'cancer_location', 'days_since_starting'
]


# In[22]:


perf_kwargs = {
    'event_df': event_dates.join(tag), 
    'perf_metrics': ['precision', 'recall', 'outcome_level_recall', 'event_rate']
}
subgroup_performance = evaluator.get_perf_by_subgroup(
    model_data, subgroups=subgroups, pred_thresh=[0.15, month_mortality_thresh, 0.3, 0.5, year_mortality_thresh], 
    alg='ENS', save=True, display_ci=True, load_ci=True, perf_kwargs=perf_kwargs
)
subgroup_performance


# In[40]:


subgroup_performance['365d Mortality']


# In[23]:


subgroup_performance = pd.read_csv(f'{output_path}/tables/subgroup_performance.csv', index_col=[0,1], header=[0,1])
subgroup_plot_groupings = {
    'Demographic': [
        'Entire Test Cohort', 'Age', 'Sex', 'Immigration', 'Language', 'Neighborhood Income Quintile', 
        'Immigrant World Region of Birth', 'Area of Residence'
    ],
    'Treatment': [
        'Entire Test Cohort', 'Regimen', 'Topography ICD-0-3', 'Days Since Starting Regimen'
    ]
}
subgroup_plot_padding = {'pad_y0': 1.2, 'pad_x1': 2.8, 'pad_y1': 0.2}
subgroup_plot_width = {'Demographic': 18, 'Treatment': 12}


# ## 30 day Mortality

# ### Subgroup Performance Plot

# In[24]:


for name, grouping in subgroup_plot_groupings.items():
    subgroup_performance_plot(
        subgroup_performance, target_event='30d Mortality', subgroups=grouping, padding=subgroup_plot_padding,
        figsize=(subgroup_plot_width[name],30), save_dir=f'{output_path}/figures/subgroup_perf/{name}'
    )
# PPV = 0.3 means roughly for every 3 alarms, 2 are false alarms and 1 is true alarm
# Sesnsitivity = 0.5 means roughly for every 2 true alarms, the model predicts 1 of them correctly
# Event Rate = 0.15 means true alarms occur 15% of the time


# ### Time to Death After First Alarm

# In[25]:


analysis_df = get_intervention_analysis_data(
    evaluator, event_dates, pred_thresh=month_mortality_thresh, target_event='30d Mortality'
)
time_to_death = time_to_x_after_y(analysis_df, x='death', y='first_alarm')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
time_to_event_plot(time_to_death, ax=axes[0], plot_type='hist', xlabel='Months')
time_to_event_plot(time_to_death, ax=axes[1], plot_type='cdf', xlabel='Months', ylabel="Cumulative Proportion of Patients")
plt.savefig(f'{output_path}/figures/time_to_event/30d_Mortality.jpg', bbox_inches='tight', dpi=300)


# ## 365 day Mortality

# ### Subgroup Performance Plot

# In[26]:


for name, grouping in subgroup_plot_groupings.items():
    subgroup_performance_plot(
        subgroup_performance, target_event='365d Mortality', subgroups=grouping, padding=subgroup_plot_padding,
        figsize=(subgroup_plot_width[name],30), save_dir=f'{output_path}/figures/subgroup_perf/{name}'
    )


# ### Time to Death After First Alarm

# In[27]:


analysis_df = get_intervention_analysis_data(evaluator, event_dates, pred_thresh=year_mortality_thresh, target_event='365d Mortality')
time_to_death = time_to_x_after_y(analysis_df, x='death', y='first_alarm')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
xticks = range(0,int(time_to_death.max()),3)
time_to_event_plot(time_to_death, ax=axes[0], plot_type='hist',xticks=xticks, xlabel='Months')
time_to_event_plot(time_to_death, ax=axes[1], plot_type='cdf', xticks=xticks, xlabel='Months', ylabel="Cumulative Proportion of Patients")
plt.savefig(f'{output_path}/figures/time_to_event/365d_Mortality.jpg', bbox_inches='tight', dpi=300)


# # Palliative Care Consultation Service (PCCS) Analysis

# In[28]:


def get_num_consults(df):
    usual_care = sum(df['first_PCCS_date'].notnull())
    system_care_no_alarm_no_pccs = sum(df['predicted'])
    system_care_no_alarm_usual_care = usual_care + sum(df['predicted'] & df['first_PCCS_date'].isnull())
    print(f'Usual Care: {usual_care} consultations')
    print(f'System-Guided Care (No Alarm = No PCCS): {system_care_no_alarm_no_pccs} consultations')
    print(f'System-Guided Care (No Alarm = Usual Care): {system_care_no_alarm_usual_care} consultations')


# In[29]:


pccs_df = get_pccs_analysis_data(
    evaluator, event_dates, 
    days_before_death=180, pred_thresh=year_mortality_thresh,
    alg='ENS', split='Test', target_event='365d Mortality'
)
get_num_consults(pccs_df)


# In[30]:


mask = pccs_df['first_PCCS_date'] < pccs_df['first_visit_date']
early_pccs_mask = pccs_df['received_early_pccs']
print('Usual Care: ')
print(f'Total patients = {len(mask)}')
print(f'Received consultation before first treatment = {sum(mask)}')
print(f'Received early consultation (prior to 6 months before death) = {sum(early_pccs_mask)}')
print(f'Received early consultation before first treatment = {sum(early_pccs_mask & mask)}')


# ## Model Impact

# ### All Patients

# In[31]:


impact = get_pccs_impact(pccs_df, no_alarm_strategy='no_pccs')
print(epc_impact_plot(impact, len(pccs_df), epc=True))
plt.savefig(f'{output_path}/figures/model_impact/EPC_rate_all_patients_eq_res.jpg', bbox_inches='tight', dpi=300)
print(post_pccs_survival_plot(pccs_df, no_alarm_strategy='no_pccs'))
plt.savefig(f'{output_path}/figures/model_impact/KM_curve_all_patients_eq_res.jpg', bbox_inches='tight', dpi=300)
impact.astype(str) + ' (' + (impact / len(pccs_df) * 100).round(2).astype(str) + '%)'


# ### Viable Patients

# In[32]:


def remove_unviable_patients(df):
    """Remove patients whose death occured within 6 months after their first 
    visit date (the system has no chance to deliver early palliative care to
    these patients)
    """
    mask = df['first_visit_near_death']
    print(f'Removing {sum(mask)} patients whose death occured within 6 months '
          'after their first treatment session')
    tmp = df[mask]
    tmp = tmp['predicted'] & (tmp['visit_date'] == tmp['first_visit_date'])
    print(f'Among those patients, {sum(tmp)} had an alarm trigger on their '
          'first treatment session\n')
    return df[~mask]


# In[33]:


viable_pccs_df = remove_unviable_patients(pccs_df)
impact = get_pccs_impact(viable_pccs_df, no_alarm_strategy='no_pccs')
print(epc_impact_plot(impact, len(viable_pccs_df), epc=True))
plt.savefig(f'{output_path}/figures/model_impact/EPC_rate_viable_patients_eq_res.jpg', bbox_inches='tight', dpi=300)
print(post_pccs_survival_plot(viable_pccs_df, no_alarm_strategy='no_pccs'))
plt.savefig(f'{output_path}/figures/model_impact/KM_curve_viable_patients_eq_res.jpg', bbox_inches='tight', dpi=300)
impact.astype(str) + ' (' + (impact / len(viable_pccs_df) * 100).round(2).astype(str) + '%)'


# ### Impact by Subgroup

# In[34]:


pccs_result = {}
pccs_result['usual'] = get_pccs_receival_by_subgroup(pccs_df, subgroups=subgroups, care_name='usual')
pccs_result['system'] = get_pccs_receival_by_subgroup(pccs_df, subgroups=subgroups, care_name='system')
for care_name in pccs_result:
    print(f'######################### {care_name.upper()} #########################')
    for subgroup, result in pccs_result[care_name].items():
        print(result.round(1).to_string(), end='\n\n')


# In[35]:


padding = {'pad_x0': 0.8, 'pad_y0': 1.2, 'pad_x1': 2.8, 'pad_y1': 0.3}
for name, grouping in subgroup_plot_groupings.items():
    output = epc_subgroup_plot(
        pccs_result, target_event='365d Mortality', subgroups=grouping, padding=padding,
        figsize=(subgroup_plot_width[name], 10), save_dir=f'{output_path}/figures/model_impact/{name}'
    )


# In[36]:


diff = output.iloc[:, 1].str[:5].astype(float) - output.iloc[:, 0].str[:5].astype(float)
output[('365d Mortality', 'Difference')] = diff
output


# ### Bias Mitigation Plot

# In[38]:


rural_mask = model_data.loc[test_mask, 'rural']
immigrant_mask = model_data.loc[test_mask, 'recent_immigrant']
subgroup_masks = {
    'Area of Residence': {'Urban': ~rural_mask, 'Rural': rural_mask},
    'Immigration': {'Recent Immigrant': immigrant_mask, 'Long-Term Resident': ~immigrant_mask}
}
# hmmmm if only one row, aspect ratio goes wacky
results = epc_bias_mitigation_plot(
    evaluator, pccs_result, subgroup_masks, save_path=f'{output_path}/figures/output', 
    padding={'pad_x0': 0.8, 'pad_x1': 0.05, 'pad_y0': 0.7}
)


# In[39]:


pd.concat(results.values()).drop_duplicates()


# # Chemo at End-of-Life Analysis

# In[40]:


eol_treatment_df = get_eol_treatment_analysis_data(evaluator, event_dates, pred_thresh=month_mortality_thresh)
eol_treatment_result = get_eol_treatment_receival_by_subgroup(eol_treatment_df, subgroups=subgroups)
eol_treatment_result


# In[41]:


result = eol_treatment_result.copy()
target_event = '30d Mortality'
old_col = 'Received Treatment Near EOL (Rate)'
new_col = 'Proportion of Patients Who Received\nTreatment Near End-Of-Life'
result = result.T[[old_col]]
result[new_col] = result.pop(old_col)
result.columns = pd.MultiIndex.from_product([[target_event], result.columns])
for name, grouping in subgroup_plot_groupings.items():
    subgroup_performance_plot(result, target_event=target_event, subgroups=grouping, figsize=(subgroup_plot_width[name],3.5))


# # Scratch Notes

# ## 50% Threshold

# In[42]:


year_mortality_thresh = 0.5


# In[43]:


pccs_df = get_pccs_analysis_data(
    evaluator, event_dates, 
    days_before_death=180, pred_thresh=year_mortality_thresh, 
    alg='ENS', split='Test', target_event='365d Mortality'
)
get_num_consults(pccs_df)


# ### All Patients

# In[44]:


impact = get_pccs_impact(pccs_df, no_alarm_strategy='no_pccs')
print(epc_impact_plot(impact, len(pccs_df), epc=True))
plt.savefig(f'{output_path}/figures/model_impact/EPC_rate_all_patients.jpg', bbox_inches='tight', dpi=300)
print(post_pccs_survival_plot(pccs_df, verbose=True))
plt.savefig(f'{output_path}/figures/model_impact/KM_curve_all_patients.jpg', bbox_inches='tight', dpi=300)
impact.astype(str) + ' (' + (impact / len(pccs_df) * 100).round(2).astype(str) + '%)'


# ### Viable Patients

# In[45]:


viable_pccs_df = remove_unviable_patients(pccs_df)
impact = get_pccs_impact(viable_pccs_df, no_alarm_strategy='no_pccs')
print(epc_impact_plot(impact, len(viable_pccs_df), epc=True))
plt.savefig(f'{output_path}/figures/model_impact/EPC_rate_viable_patients.jpg', bbox_inches='tight', dpi=300)
print(post_pccs_survival_plot(viable_pccs_df, no_alarm_strategy='no_pccs'))
plt.savefig(f'{output_path}/figures/model_impact/KM_curve_viable_patients.jpg', bbox_inches='tight', dpi=300)
impact.astype(str) + ' (' + (impact / len(viable_pccs_df) * 100).round(2).astype(str) + '%)'


# ### PCCS Graph Plot

# In[46]:


from src.visualize import pccs_graph_plot
d = pccs_graph_plot(pccs_df)
d.render(filename='pccs_num_graph', directory=f'{output_path}/figures/output', format='png')
d


# ## All Billing Codes Equal Resource Utility Thresh

# In[47]:


from src.preprocess import filter_ohip_data
ohip = pd.read_parquet(f'{root_path}/data/ohip.parquet.gzip')
ohip = filter_ohip_data(ohip)
pd.DataFrame(
    [ohip['feecode'].value_counts(), 
     ohip.groupby('ikn').first()['feecode'].value_counts()],
    index=['All Consultations', 'First Consulations']
).T


# In[48]:


initial_pccs_date = ohip.groupby('ikn')['servdate'].first()
initial_pccs_date.index = initial_pccs_date.index.astype(int)
event_dates['first_PCCS_date'] = tag['ikn'].map(initial_pccs_date)


# In[49]:


# Select prediction threshold at which alarm rate and usual intervention rate is equal 
# (resource utility is constant for both usual care and system-guided care)
year_mortality_thresh = equal_rate_pred_thresh(evaluator, event_dates, split='Test', alg='ENS', target_event='365d Mortality')
year_mortality_thresh = year_mortality_thresh + 0.02 # manual adjustment


# In[50]:


pccs_df = get_pccs_analysis_data(
    evaluator, event_dates, 
    days_before_death=180, pred_thresh=year_mortality_thresh, 
    alg='ENS', split='Test', target_event='365d Mortality'
)
get_num_consults(pccs_df)


# ### All Patients

# In[51]:


impact = get_pccs_impact(pccs_df, no_alarm_strategy='no_pccs')
print(epc_impact_plot(impact, len(pccs_df), epc=True))
plt.savefig(f'{output_path}/figures/model_impact/EPC_rate_all_patients&billing_eq_res.jpg', bbox_inches='tight', dpi=300)
print(post_pccs_survival_plot(pccs_df, verbose=True))
plt.savefig(f'{output_path}/figures/model_impact/KM_curve_all_patients&billing_eq_res.jpg', bbox_inches='tight', dpi=300)
impact.astype(str) + ' (' + (impact / len(pccs_df) * 100).round(2).astype(str) + '%)'


# ### Viable Patients

# In[52]:


viable_pccs_df = remove_unviable_patients(pccs_df)
impact = get_pccs_impact(viable_pccs_df, no_alarm_strategy='no_pccs')
print(epc_impact_plot(impact, len(viable_pccs_df), epc=True))
plt.savefig(f'{output_path}/figures/model_impact/EPC_rate_viable_patients&all_billing_eq_res.jpg', bbox_inches='tight', dpi=300)
print(post_pccs_survival_plot(viable_pccs_df, no_alarm_strategy='no_pccs'))
plt.savefig(f'{output_path}/figures/model_impact/KM_curve_viable_patients&all_billing_eq_res.jpg', bbox_inches='tight', dpi=300)
impact.astype(str) + ' (' + (impact / len(viable_pccs_df) * 100).round(2).astype(str) + '%)'


# ## Changes in Treatment Between Developement and Test Cohort

# In[53]:


test_treatment_count = model_data.loc[test_mask, 'regimen'].value_counts()
dev_treatment_count = model_data.loc[~test_mask, 'regimen'].value_counts()
count = pd.concat([test_treatment_count, dev_treatment_count], axis=1, keys=['test_regimen', 'dev_regimen'])
mask = (count > 10).all(axis=1)
count = count[mask]
fig, ax = plt.subplots(figsize=(20,3))
count.plot.bar(stacked=True, ax=ax)
plt.show()


# ## Time to First Alarm After First PCCS

# In[54]:


analysis_df = get_intervention_analysis_data(evaluator, event_dates, pred_thresh=0.5, target_event='365d Mortality')
time_to_alarm = time_to_x_after_y(analysis_df, x='first_alarm', y='first_pccs', clip=True, care_name='Usual Care')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
xticks = range(-60,81,6)
time_to_event_plot(time_to_alarm, ax=axes[0], plot_type='hist', xticks=xticks, xlabel='Months')
time_to_event_plot(time_to_alarm, ax=axes[1], plot_type='cdf', xticks=xticks, xlabel='Months', ylabel="Cumulative Proportion of Patients")


# ## Time to Death After First PCCS

# In[55]:


analysis_df = get_intervention_analysis_data(evaluator, event_dates, pred_thresh=0.5, target_event='365d Mortality')
time_to_death = time_to_x_after_y(analysis_df, x='death', y='first_pccs', clip=True, care_name='Usual Care')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
xticks = np.arange(0,91,6)
time_to_event_plot(time_to_death, ax=axes[0], plot_type='hist', xticks=xticks, xlabel='Months')
time_to_event_plot(time_to_death, ax=axes[1], plot_type='cdf', xticks=xticks, xlabel='Months', ylabel="Cumulative Proportion of Patients")


# ## Regimen performance

# In[56]:


regimen_performance = evaluator.get_perf_by_subgroup(
    model_data,
    subgroups=['regimen'], 
    pred_thresh=year_mortality_thresh, 
    alg='ENS', 
    target_events=['365d Mortality'], 
    split='Test',
    save=False, 
    display_ci=False, 
    load_ci=False,
    top=evaluator.orig_data['regimen'].nunique(),
    perf_kwargs={'perf_metrics': ['precision', 'recall','event_rate', 'count']}
)['365d Mortality']
mask = regimen_performance['N'] > 100
regimen_performance[mask].sort_values(by='AUROC')


# ## Hyperparameters

# In[57]:


from src.utility import get_hyperparameters
get_hyperparameters(output_path)
