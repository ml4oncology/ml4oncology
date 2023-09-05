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
pd.set_option('display.max_columns', 150)
pd.set_option('display.max_rows', 150)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from src.config import root_path, acu_folder
from src.evaluate import EvaluateClf
from src.model import SimpleBaselineModel
from src.prep_data import PrepDataEDHD
from src.summarize import data_description_summary, feature_summary
from src.train import TrainLASSO, TrainML, TrainRNN, TrainENS
from src.utility import (
    initialize_folders, load_pickle,
    get_nunique_categories, get_nmissing, get_clean_variable_names, get_units
)
from src.visualize import importance_plot, subgroup_performance_plot


# In[3]:


# config
processes = 64
days = 30 # predict event within this number of days since chemo visit (the look ahead window)
target_keyword = f'within_{days}_days'
main_dir = f'{root_path}/{acu_folder}'
output_path = f'{main_dir}/models/{target_keyword}'
initialize_folders(output_path)


# # Prepare Data for Model Training

# In[4]:


prep = PrepDataEDHD(adverse_event='acu', target_keyword=target_keyword)
model_data = prep.get_data(verbose=True)
model_data


# In[5]:


sorted(model_data.columns.tolist())


# In[6]:


get_nunique_categories(model_data)


# In[7]:


get_nmissing(model_data, verbose=True)


# In[8]:


prep = PrepDataEDHD(adverse_event='acu', target_keyword=target_keyword) # need to reset
model_data = prep.get_data(missing_thresh=80, verbose=True)
X, Y, tag = prep.split_and_transform_data(model_data, remove_immediate_events=True)
# remove sessions in model_data that were excluded during split_and_transform
model_data = model_data.loc[tag.index]


# In[9]:


prep.get_label_distribution(Y, tag, with_respect_to='sessions')


# In[10]:


prep.get_label_distribution(Y, tag, with_respect_to='patients')


# In[11]:


Y.columns = Y.columns.str.replace(f' {target_keyword}', '')
# Convenience variables
train_mask, valid_mask, test_mask = tag['split'] == 'Train', tag['split'] == 'Valid', tag['split'] == 'Test'
X_train, X_valid, X_test = X[train_mask], X[valid_mask], X[test_mask]
Y_train, Y_valid, Y_test = Y[train_mask], Y[valid_mask], Y[test_mask]


# ## Study Characteristics

# In[14]:


subgroups = [
    'sex', 'immigration', 'birth_region', 'language', 'income', 'area_density',
    'regimen', 'cancer_type', 'cancer_location', 'target'
]
data_description_summary(
    model_data, Y, tag, save_dir=f'{output_path}/tables', partition_method='split', target_event='ACU', subgroups=subgroups
)


# ## Feature Characteristics

# In[13]:


df = prep.ohe.encode(model_data.copy(), verbose=False) # get original (non-normalized, non-imputed) data one-hot encoded
df = df[train_mask].drop(columns=['ikn'])
feature_summary(
    df, save_dir=f'{output_path}/tables', deny_old_survey=True, event_dates=prep.event_dates[train_mask]
).head(60)


# # Train Models

# ## Lasso Model

# In[12]:


train_lasso = TrainLASSO(X, Y, tag, output_path=output_path, target_event='ACU')


# In[41]:


# train_lasso.tune_and_train(run_grid_search=True, run_training=True, save=True, C_search_space=np.geomspace(0.000025, 1, 100))
C_search_space = [0.000025, 0.000028, 0.000030, 0.000034, 0.000038, 0.0000408, 0.000042, 0.000043, 0.000044, 0.000048]
train_lasso.tune_and_train(run_grid_search=True, run_training=False, save=False, C_search_space=C_search_space)


# In[14]:


gs_results = pd.read_csv(f'{output_path}/tables/grid_search.csv')
gs_results = gs_results.sort_values(by='n_feats')
train_lasso.select_param(gs_results) # model with least complexity while AUROC upper CI >= max AUROC
ax = gs_results.plot(x='n_feats', y='AUROC', marker='o', markersize=4, legend=False, figsize=(6,6))
ax.set(xlabel='Number of Features', ylabel='AUROC')
plt.savefig(f'{output_path}/figures/LASSO_score_vs_num_feat.jpg', dpi=300)


# In[15]:


model = load_pickle(output_path, 'LASSO')
coef = train_lasso.get_coefficients(model, non_zero=False)

# Clean the feature names
rename_map = {name: f'{name} ({unit})' for name, unit in get_units().items()}
coef = coef.rename(index=rename_map)
coef.index = get_clean_variable_names(coef.index)

coef.to_csv(f'{output_path}/tables/LASSO_coefficients.csv')
coef.head(n=100)


# ## Main ML Models

# In[14]:


train_ml = TrainML(X, Y, tag, output_path, n_jobs=processes)
train_ml.tune_and_train(run_bayesopt=False, run_training=True, save_preds=True)


# ## RNN Model

# In[27]:


# Distrubution of the sequence lengths in the training set
dist_seq_lengths = X_train.groupby(tag.loc[train_mask, 'ikn']).apply(len)
dist_seq_lengths = dist_seq_lengths.clip(upper=dist_seq_lengths.quantile(q=0.999))
fig, ax = plt.subplots(figsize=(15, 3))
ax.grid(zorder=0)
sns.histplot(dist_seq_lengths, ax=ax, zorder=2, bins=int(dist_seq_lengths.max()))


# In[17]:


train_rnn = TrainRNN(X, Y, tag, output_path)
train_rnn.tune_and_train(run_bayesopt=False, run_training=False, run_calibration=True, save_preds=True)


# ## ENS Model 
# Find Optimal Ensemble Weights

# In[15]:


# combine lasso, rnn, and ml predictions
preds = load_pickle(f'{output_path}/preds', 'ML_preds')
preds_lasso = load_pickle(f'{output_path}/preds', 'LASSO_preds')
preds_rnn = load_pickle(f'{output_path}/preds', filename='RNN_preds')
for split, pred in preds_rnn.items(): preds[split]['RNN'] = pred
for split, pred in preds_lasso.items(): preds[split]['LR'] = pred['LR']
del preds_rnn, preds_lasso


# In[16]:


train_ens = TrainENS(X, Y, tag, output_path, preds)
train_ens.tune_and_train(run_bayesopt=False, run_calibration=False, calibrate_pred=True, random_state=0)


# # Evaluate Models

# In[ ]:


# setup the final prediction and labels
preds, labels = train_ens.preds.copy(), train_ens.labels.copy()
base_model = SimpleBaselineModel(model_data[['regimen']], labels)
base_preds = base_model.predict()
for split, pred in base_preds.items(): preds[split].update(pred)


# In[19]:


target_order = ['ACU', 'ED', 'H', 'TR_ACU', 'TR_ED', 'TR_H', 'INFX_ED', 'INFX_H','GI_ED', 'GI_H']
eval_models = EvaluateClf(output_path, preds, labels)
df = eval_models.get_evaluation_scores(target_events=target_order, display_ci=True, load_ci=True, save_ci=False)
df


# In[35]:


df.loc[(slice(None), slice('AUROC Score')), df.columns].T.loc['Test']


# In[36]:


eval_models.plot_curves(curve_type='pr', legend_loc='upper right', figsize=(12,18), save=False)
eval_models.plot_curves(curve_type='roc', legend_loc='lower right', figsize=(12,18), save=False)
eval_models.plot_curves(curve_type='pred_cdf', figsize=(12,18), save=False) # cumulative distribution function of model prediction
eval_models.plot_calibs(legend_loc='upper left', figsize=(12,18), save=False) 
# eval_models.plot_calibs(include_pred_hist=True, legend_loc='upper left', figsize=(12,28), padding={'pad_y1': 0.3, 'pad_y0': 3.0})


# # Post-Training Analysis

# In[12]:


event_dates = prep.event_dates[['visit_date', 'next_ED_date', 'next_H_date']].copy()


# ## Threshold Operating Points

# In[34]:


pred_thresholds = np.arange(0.05, 0.51, 0.05)
perf_metrics = [
    'warning_rate', 'precision', 'recall', 'NPV', 'specificity', 'outcome_level_recall',
]
thresh_df = eval_models.operating_points(
    pred_thresholds, op_metric='threshold', perf_metrics=perf_metrics, event_df=event_dates.join(tag)
)
thresh_df


# In[35]:


thresh_df['ACU'][['Warning Rate', 'PPV', 'Outcome-Level Recall']]


# ## Most Important Features/Feature Groups

# In[ ]:


get_ipython().system('python scripts/feat_imp.py --adverse-event ACU --output-path {output_path} --days {days}')


# In[19]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot('ENS', eval_models.target_events, output_path, figsize=(6,50), top=10, importance_by='feature', padding={'pad_x0': 4.0})


# In[20]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot('ENS', eval_models.target_events, output_path, figsize=(6,50), top=10, importance_by='group', padding={'pad_x0': 1.2})


# ## Performance on Subgroups

# In[34]:


subgroups = [
    'all', 'age', 'sex', 'immigrant', 'language', 'arrival', 'income', 
    'area_density', 'regimen', 'cancer_location', 'days_since_starting'
]
perf_kwargs = {
    'event_df': event_dates.join(tag), 
    'perf_metrics': ['precision', 'recall', 'outcome_level_recall', 'event_rate']
}
subgroup_performance = eval_models.get_perf_by_subgroup(
    model_data, subgroups=subgroups, pred_thresh=0.35, alg='ENS', 
    display_ci=False, load_ci=False, perf_kwargs=perf_kwargs
)
subgroup_performance


# ## Decision Curve Plot

# In[37]:


result = eval_models.plot_decision_curves('ENS', padding={'pad_x0': 1.0}, save=False)


# In[38]:


# find threshold range where net benefit for system is better than treat all and treat none (aka 0)
thresh_range = {}
for target_event, df in result.items():
    mask = df['System'] > df['All'].clip(0)
    thresh = df.loc[mask, 'Threshold']
    thresh_range[target_event] = (thresh.min(), thresh.max())
pd.DataFrame(thresh_range, index=['Threshold Min', 'Threshold Max']).T


# ## ACU

# ### All the Plots

# In[42]:


eval_models.all_plots_for_single_target(alg='ENS', target_event='ACU')


# ### Subgroup Performance Plot

# In[32]:


subgroup_performance = pd.read_csv(f'{output_path}/tables/subgroup_performance_summary.csv', index_col=[0,1], header=[0,1])
groupings = {
    'Demographic': ['Entire Test Cohort', 'Age', 'Sex', 'Immigration', 'Language', 'Neighborhood Income Quintile'],
    'Treatment': ['Entire Test Cohort', 'Regimen', 'Topography ICD-0-3', 'Days Since Starting Regimen']
}
padding = {'pad_y0': 1.2, 'pad_x1': 2.6, 'pad_y1': 0.2}
for name, subgroups in groupings.items():
    subgroup_performance_plot(
        subgroup_performance, target_event='ACU', subgroups=subgroups, padding=padding,
        figsize=(12,30), save_dir=f'{output_path}/figures/subgroup_perf/{name}'
    )
# PPV = 0.3 means roughly for every 3 alarms, 2 are false alarms and 1 is true alarm
# Sesnsitivity = 0.5 means roughly for every 2 true alarms, the model predicts 1 of them correctly
# Event Rate = 0.15 means true alarms occur 15% of the time


# In[33]:


subgroups = ['Entire Test Cohort', 'Age', 'Sex', 'Immigration', 'Regimen', 'Days Since Starting Regimen']
subgroup_performance_plot(
    subgroup_performance, target_event='ACU', subgroups=subgroups, padding=padding,
    figsize=(12,30), save_dir=f'{output_path}/figures/subgroup_perf'
)


# # Scratch Notes

# ## Brooks 2 Variable Based Model

# In[109]:


prep = PrepDataEDHD(adverse_event='acu')
df = prep.get_data(target_keyword)
print(f'Size of data = {len(df)}, Number of patients = {df["ikn"].nunique()}')
df = df.loc[Y_test.index]
print(f'Size of test data = {len(df)}, Number of patients = {df["ikn"].nunique()}')
df = df[df['baseline_sodium_value'].notnull() & df['baseline_albumin_value'].notnull()]
print(f'Size of test data with both sodium and albumin count = {len(df)}, Number of patients = {df["ikn"].nunique()}')
df = df[df['days_since_starting_chemo'] == 0] # very first treatment
print(f'Size of test data with only first day chemos = {len(df)}, Number of patients = {df["ikn"].nunique()}')


# In[110]:


def predict(df):
    x = 10.392 - 0.472*0.1*df['baseline_albumin_value'] - 0.075*df['baseline_sodium_value']
    return 1 / (1 + np.exp(-x))


# In[111]:


split = 'Test'
pred = predict(df)
labels = {split: Y_test.loc[df.index]}
preds = {split: {'ENS': train_ens.preds[split]['ENS'].loc[df.index],
                 'BRK': pd.DataFrame({col: pred for col in Y_test.columns})}}
eval_brooks_model = EvaluateClf(output_path='', preds=preds, labels=labels)


# In[112]:


# label distribtuion
labels[split].apply(pd.value_counts)


# In[70]:


kwargs = {'algs': ['ENS', 'BRK'], 'splits': ['Test'], 'display_ci': True, 'save_score': False}
result = eval_brooks_model.get_evaluation_scores(**kwargs)
result


# In[71]:


eval_brooks_model.all_plots_for_single_target(
    alg='BRK', target_event='H', split='Test', n_bins=20, calib_strategy='quantile', figsize=(12,18), save=False
)


# In[74]:


points = np.arange(0.05, 0.51, 0.05)
perf_metrics = ['warning_rate', 'precision', 'recall', 'NPV', 'specificity']
eval_brooks_model.operating_points(points, op_metric='threshold', alg='BRK', target_events=['H'], save=False, perf_metrics=perf_metrics)


# ### Compare with ENS

# In[113]:


eval_brooks_model.all_plots_for_single_target(
    alg='ENS', target_event='H', split='Test', n_bins=20, calib_strategy='quantile', figsize=(12,18), save=False
)


# In[114]:


points = np.arange(0.05, 0.51, 0.05)
perf_metrics = ['warning_rate', 'precision', 'recall', 'NPV', 'specificity']
eval_brooks_model.operating_points(points, op_metric='threshold', target_events=['H'], save=False, perf_metrics=perf_metrics)


# ## Hyperparameters

# In[43]:


from src.utility import get_hyperparameters
get_hyperparameters(output_path, days=days)


# In[ ]:
