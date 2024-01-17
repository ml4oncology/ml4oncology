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


get_ipython().run_line_magic('cd', '../../')
# reloads all modules everytime before cell is executed (no need to restart kernel)
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 150)
pd.set_option('display.max_rows', 150)
import seaborn as sns

from src.config import root_path, can_folder, split_date, SCr_rise_threshold
from src.evaluate import EvaluateClf
from src.model import SimpleBaselineModel
from src.prep_data import PrepDataCAN
from src.summarize import data_description_summary, feature_summary
from src.train import Ensembler, LASSOTrainer, Trainer
from src.utility import (
    initialize_folders, load_pickle, 
    get_nunique_categories, get_nmissing, most_common_categories,
    get_hyperparameters
)
from src.visualize import importance_plot, subgroup_performance_plot


# In[3]:


processes = 64
target_keyword = 'SCr|dialysis|next'
main_dir = f'{root_path}/projects/{can_folder}'
adverse_event = 'aki'
output_path = f'{main_dir}/models/{adverse_event.upper()}'
initialize_folders(output_path)


# # Prepare Data for Model Training

# In[4]:


prep = PrepDataCAN(adverse_event=adverse_event, target_keyword=target_keyword)
model_data = prep.get_data(verbose=True)
model_data


# In[5]:


most_common_categories(model_data, catcol='regimen', with_respect_to='patients', top=10)


# In[6]:


sorted(model_data.columns.tolist())


# In[7]:


get_nunique_categories(model_data)


# In[8]:


get_nmissing(model_data)


# In[9]:


prep = PrepDataCAN(adverse_event=adverse_event, target_keyword=target_keyword) # need to reset
model_data = prep.get_data(missing_thresh=80, include_comorbidity=True, verbose=True)
X, Y, tag = prep.split_and_transform_data(model_data, split_date=split_date)
# remove sessions in model_data that were excluded during split_and_transform
model_data = model_data.loc[tag.index]


# In[10]:


prep.get_label_distribution(Y, tag, with_respect_to='sessions')


# In[11]:


prep.get_label_distribution(Y, tag, with_respect_to='patients')


# In[12]:


# Convenience variables
train_mask, valid_mask, test_mask = tag['split'] == 'Train', tag['split'] == 'Valid', tag['split'] == 'Test'
X_train, X_valid, X_test = X[train_mask], X[valid_mask], X[test_mask]
Y_train, Y_valid, Y_test = Y[train_mask], Y[valid_mask], Y[test_mask]


# ## Study Characteristics

# In[13]:


subgroups = [
    'sex', 'immigration', 'birth_region', 'language', 'income', 'area_density',
    'regimen', 'cancer_type', 'cancer_location', 'target', 'comorbidity', 'dialysis', 'ckd'
]
data_description_summary(
    model_data, Y, tag, save_dir=f'{output_path}/tables', partition_method='cohort', target_event='AKI', 
    subgroups=subgroups
)


# ## Feature Characteristic

# In[14]:


df = prep.ohe.encode(model_data.copy(), verbose=False) # get original (non-normalized, non-imputed) data one-hot encoded
df = df[train_mask].drop(columns=['ikn'])
feature_summary(
    df, save_dir=f'{output_path}/tables', deny_old_survey=True, event_dates=prep.event_dates[train_mask]
).head(60)


# # Train Models

# ## Main Models

# In[15]:


# Distrubution of the sequence lengths in the training set
dist_seq_lengths = X_train.groupby(tag.loc[train_mask, 'ikn']).apply(len)
dist_seq_lengths = dist_seq_lengths.clip(upper=dist_seq_lengths.quantile(q=0.999))
fig, ax = plt.subplots(figsize=(15, 3))
ax.grid(zorder=0)
sns.histplot(dist_seq_lengths, ax=ax, zorder=2, bins=int(dist_seq_lengths.max()))


# In[13]:


trainer = Trainer(X, Y, tag, output_path)
trainer.run(bayesopt=True, train=True, save_preds=True)


# ## ENS Model 
# Find Optimal Ensemble Weights

# In[16]:


preds = load_pickle(f'{output_path}/preds', 'all_preds')
ensembler = Ensembler(X, Y, tag, output_path, preds)
ensembler.run(bayesopt=True, calibrate=True)


# # Evaluate Models

# In[18]:


# setup the final prediction and labels
preds, labels = ensembler.preds, ensembler.labels
preds.update(SimpleBaselineModel(model_data[['regimen', 'baseline_eGFR']], labels).predict())


# In[22]:


evaluator = EvaluateClf(output_path, preds, labels)
evaluator.get_evaluation_scores(display_ci=True, load_ci=True, save_ci=True)


# In[23]:


evaluator.plot_curves(curve_type='pr', legend_loc='lower left', save=False)
evaluator.plot_curves(curve_type='roc', legend_loc='lower right', save=False)
evaluator.plot_curves(curve_type='pred_cdf', save=False) # cumulative distribution function of model prediction
evaluator.plot_calibs(legend_loc='upper left', save=False) 
# evaluator.plot_calibs(include_pred_hist=True, legend_loc='upper left', figsize=(12,28), padding={'pad_y1': 0.3, 'pad_y0': 3.0})


# # Post-Training Analysis

# ## Threshold Op Points

# In[24]:


pred_thresholds = np.arange(0.05, 0.51, 0.05)
perf_metrics = ['warning_rate', 'precision', 'recall', 'NPV', 'specificity', ]
thresh_df = evaluator.operating_points(pred_thresholds, op_metric='threshold', perf_metrics=perf_metrics)
thresh_df


# ## All the Plots

# In[25]:


evaluator.all_plots_for_single_target(alg='ENS', target_event='AKI')


# ## Most Important Features/Feature Groups

# In[27]:


get_ipython().system('python scripts/feat_imp.py --adverse-event AKI --output-path {output_path} ')


# In[28]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot('ENS', evaluator.target_events, output_path, figsize=(6,5), top=10, importance_by='feature', padding={'pad_x0': 2.7})


# In[29]:


get_ipython().system('python scripts/feat_imp.py --adverse-event AKI --output-path {output_path} --permute-group')


# In[30]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot('ENS', evaluator.target_events, output_path, figsize=(6,5), top=10, importance_by='group', padding={'pad_x0': 1.2})


# ## Performance on Subgroups

# In[26]:


subgroups = [
    'all', 'age', 'sex', 'immigrant', 'language', 'arrival', 'income', 
    'area_density', 'ckd', 'regimen', 'cancer_location', 'days_since_starting', 
]
perf_kwargs = {'perf_metrics': ['precision', 'recall', 'event_rate']}
subgroup_performance = evaluator.get_perf_by_subgroup(
    model_data, subgroups=subgroups, pred_thresh=0.1, alg='ENS', display_ci=True, load_ci=True, 
    perf_kwargs=perf_kwargs
)
subgroup_performance


# In[27]:


subgroup_performance = pd.read_csv(f'{output_path}/tables/subgroup_performance.csv', index_col=[0,1], header=[0,1])
groupings = {
    'Demographic': ['Entire Test Cohort', 'Age', 'Sex', 'Immigration', 'Language', 'Neighborhood Income Quintile'],
    'Treatment': ['Entire Test Cohort', 'Regimen', 'Topography ICD-0-3', 'Days Since Starting Regimen', 'CKD Prior to Treatment']
}
padding = {'pad_y0': 1.2, 'pad_x1': 2.7, 'pad_y1': 0.2}
for name, subgroups in groupings.items():
    subgroup_performance_plot(
        subgroup_performance, target_event='AKI', subgroups=subgroups, padding=padding,
        figsize=(12,30), save_dir=f'{output_path}/figures/subgroup_perf/{name}'
    )
# PPV = 0.3 means roughly for every 3 alarms, 2 are false alarms and 1 is true alarm
# Sesnsitivity = 0.5 means roughly for every 2 true alarms, the model predicts 1 of them correctly
# Event Rate = 0.15 means true alarms occur 15% of the time


# ## Decision Curve Plot

# In[28]:


result = evaluator.plot_decision_curves('ENS')
result['AKI'].tail(n=100)


# In[29]:


get_hyperparameters(output_path)


# # Scratch Notes

# ## Motwani Score Based Model

# In[31]:


prep = PrepDataCAN(adverse_event='aki', target_keyword=target_keyword)
df = prep.get_data(include_comorbidity=True)
X, Y, tag = prep.split_and_transform_data(df, split_date=split_date, verbose=False, ohe_kwargs={'verbose': False})
df = df.loc[tag.index]
print(f'Size of data = {len(df)}, Number of patients = {df["ikn"].nunique()}')
df['cisplatin_dosage'] *= df['body_surface_area'] # convert from mg/m^2 to mg
df = df.loc[tag['split']=='Test']
print(f'Size of test data = {len(df)}, Number of patients = {df["ikn"].nunique()}')
df = df[df['baseline_albumin_value'].notnull()]
print(f'Size of test data with albumin = {len(df)}, Number of patients = {df["ikn"].nunique()}')
df = df.query('days_since_starting_chemo == 0') # very first treatment
print(f'Size of test data with only first day chemos = {len(df)}, Number of patients = {df["ikn"].nunique()}')


# In[32]:


def compute_score(data):
    score = pd.Series(0, index=data.index)
    score[data['age'].between(61, 70)] += 1.5
    score[data['age'] > 70] += 2.5
    score[data['baseline_albumin_value'] < 35] += 2.0
    score[data['cisplatin_dosage'].between(101, 150)] += 1.0
    score[data['cisplatin_dosage'] > 150] += 3.0
    score[data['hypertension']] += 2.0
    score /= score.max()
    return score


# In[38]:


score = compute_score(df)
labels = {'Test': Y.loc[df.index]}
preds = {'ENS': {'Test': ensembler.preds['ENS']['Test'].loc[df.index]}, 'MSB': {'Test': pd.DataFrame({'AKI': score})}}
eval_motwani_model = EvaluateClf(output_path='', preds=preds, labels=labels)


# In[39]:


# label distribtuion
labels['Test'].apply(pd.value_counts)


# In[40]:


kwargs = {'algs': ['ENS', 'MSB'], 'splits': ['Test'], 'display_ci': True, 'save_score': False}
result = eval_motwani_model.get_evaluation_scores(**kwargs)
result


# In[41]:


eval_motwani_model.all_plots_for_single_target(alg='MSB', target_event='AKI', n_bins=20, figsize=(12,16), save=False)


# In[42]:


points = np.arange(0, 8.6, 0.5)/8.5 # 8.5 is the highest score possible, 0 is lowest score possible
eval_motwani_model.operating_points(
    points, op_metric='threshold', alg='MSB', target_events=['AKI'], 
    perf_metrics=['warning_rate', 'precision', 'recall', 'NPV', 'specificity'], save=False
)


# ### Compare with ENS

# In[43]:


eval_motwani_model.all_plots_for_single_target(alg='ENS', target_event='AKI', n_bins=20,figsize=(12,16), save=False)


# In[44]:


points = np.arange(0, 8.6, 0.5)/8.5 # 8.5 is the highest score possible, 0 is lowest score possible
eval_motwani_model.operating_points(
    points, op_metric='threshold', alg='ENS', target_events=['AKI'], 
    perf_metrics=['warning_rate', 'precision', 'recall', 'NPV', 'specificity'], save=False
)


# ## Missingness By Splits

# In[45]:


from src.utility import get_nmissing_by_splits


# In[48]:


missing = get_nmissing_by_splits(model_data, ensembler.labels)
missing.sort_values(by=(f'Test (N={sum(test_mask)})', 'Missing (N)'), ascending=False)


# In[ ]:
