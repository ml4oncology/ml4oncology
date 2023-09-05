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
from src.train import TrainML, TrainRNN, TrainENS
from src.utility import (
    initialize_folders, load_pickle, 
    get_nunique_categories, get_nmissing, most_common_categories,
    get_hyperparameters
)
from src.visualize import importance_plot, subgroup_performance_plot


# In[3]:


processes = 64
target_keyword = 'SCr|dialysis|next'
main_dir = f'{root_path}/{can_folder}'
adverse_event = 'ckd'
output_path = f'{main_dir}/models/{adverse_event.upper()}'
initialize_folders(output_path)


# # Prepare Data for Model Training

# In[4]:


prep = PrepDataCAN(adverse_event=adverse_event, target_keyword=target_keyword)
model_data = prep.get_data(include_comorbidity=True, verbose=True)
model_data


# In[5]:


most_common_categories(model_data, catcol='regimen', with_respect_to='patients', top=10)


# In[6]:


sorted(model_data.columns)


# In[7]:


get_nunique_categories(model_data)


# In[8]:


get_nmissing(model_data)


# In[9]:


prep = PrepDataCAN(adverse_event=adverse_event, target_keyword=target_keyword)
model_data = prep.get_data(missing_thresh=80, include_comorbidity=True, verbose=True)
model_data['next_eGFR'].hist(bins=100)
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

# In[17]:


subgroups = [
    'sex', 'immigration', 'birth_region', 'language', 'income', 'area_density',
    'regimen', 'cancer_type', 'cancer_location', 'target', 'comorbidity', 'dialysis', 'ckd'
]
data_description_summary(
    model_data, Y, tag, save_dir=f'{output_path}/tables', partition_method='cohort', target_event='CKD', subgroups=subgroups
)


# ## Feature Characteristic

# In[18]:


df = prep.ohe.encode(model_data.copy(), verbose=False) # get original (non-normalized, non-imputed) data one-hot encoded
df = df[train_mask].drop(columns=['ikn'])
feature_summary(
    df, save_dir=f'{output_path}/tables', deny_old_survey=True, event_dates=prep.event_dates[train_mask]
).head(60)


# # Train Models

# ## Main ML Models

# In[14]:


train_ml = TrainML(X, Y, tag, output_path, n_jobs=processes)
train_ml.tune_and_train(run_bayesopt=False, run_training=True, save_preds=True)


# ## RNN Model

# In[9]:


# Distrubution of the sequence lengths in the training set
dist_seq_lengths = X_train.groupby(tag.loc[train_mask, 'ikn']).apply(len)
dist_seq_lengths = dist_seq_lengths.clip(upper=dist_seq_lengths.quantile(q=0.999))
fig, ax = plt.subplots(figsize=(15, 3))
ax.grid(zorder=0)
sns.histplot(dist_seq_lengths, ax=ax, zorder=2, bins=int(dist_seq_lengths.max()))


# In[13]:


train_rnn = TrainRNN(X, Y, tag, output_path)
train_rnn.tune_and_train(run_bayesopt=False, run_training=True, run_calibration=True, save_preds=True)


# ## ENS Model 

# In[14]:


# combine rnn and ml predictions
preds = load_pickle(f'{output_path}/preds', 'ML_preds')
preds_rnn = load_pickle(f'{output_path}/preds', 'RNN_preds')
for split, pred in preds_rnn.items(): preds[split]['RNN'] = pred
del preds_rnn


# In[15]:


train_ens = TrainENS(X, Y, tag, output_path, preds=preds)
train_ens.tune_and_train(run_bayesopt=True, run_calibration=True, calibrate_pred=True)


# # Evaluate Models

# In[16]:


# setup the final prediction and labels
preds, labels = train_ens.preds, train_ens.labels
base_model = SimpleBaselineModel(model_data[['regimen', 'baseline_eGFR']], labels)
base_preds = base_model.predict()
for split, pred in base_preds.items(): preds[split].update(pred)


# In[19]:


eval_models = EvaluateClf(output_path, preds, labels)
eval_models.get_evaluation_scores(display_ci=True, load_ci=False, save_ci=True)


# In[20]:


eval_models.plot_curves(curve_type='pr', legend_loc='lower left', save=False)
eval_models.plot_curves(curve_type='roc', legend_loc='lower right', save=False)
eval_models.plot_curves(curve_type='pred_cdf', save=False) # cumulative distribution function of model prediction
eval_models.plot_calibs(legend_loc='upper left', save=False) 
# eval_models.plot_calibs(include_pred_hist=True, legend_loc='upper left', figsize=(12,28), padding={'pad_y1': 0.3, 'pad_y0': 3.0})


# # Post-Training Analysis

# ## Threshold Op Points

# In[21]:


pred_thresholds = np.arange(0.05, 0.51, 0.05)
perf_metrics = [
    'warning_rate', 'precision', 'recall', 'NPV', 'specificity', 
]
thresh_df = eval_models.operating_points(points=pred_thresholds, alg='ENS', op_metric='threshold', perf_metrics=perf_metrics)
thresh_df


# ## All the Plots

# In[22]:


eval_models.all_plots_for_single_target(alg='ENS', target_event='CKD')


# ## Most Important Features/Feature Groups

# In[ ]:


get_ipython().system('python scripts/feat_imp.py --adverse-event CKD --output-path {output_path} ')


# In[24]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot('ENS', eval_models.target_events, output_path, figsize=(6,15), top=10, importance_by='feature', padding={'pad_x0': 2.7})


# In[ ]:


get_ipython().system('python scripts/feat_imp.py --adverse-event CKD --output-path {output_path} --permute-group')


# In[26]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot('ENS', eval_models.target_events, output_path, figsize=(6,15), top=10, importance_by='group', padding={'pad_x0': 1.2})


# ## Performance on Subgroups

# In[27]:


subgroups = [
    'all', 'age', 'sex', 'immigrant', 'language', 'arrival', 'income', 
    'area_density', 'ckd', 'regimen', 'cancer_location', 'days_since_starting', 
]
perf_kwargs = {'perf_metrics': ['precision', 'recall', 'event_rate']}
subgroup_performance = eval_models.get_perf_by_subgroup(
    model_data, subgroups=subgroups, pred_thresh=0.1, alg='ENS', display_ci=True, load_ci=True, perf_kwargs=perf_kwargs
)
subgroup_performance


# In[28]:


subgroup_performance = pd.read_csv(f'{output_path}/tables/subgroup_performance.csv', index_col=[0,1], header=[0,1])
groupings = {
    'Demographic': ['Entire Test Cohort', 'Age', 'Sex', 'Immigration', 'Language', 'Neighborhood Income Quintile'],
    'Treatment': ['Entire Test Cohort', 'Regimen', 'Topography ICD-0-3', 'Days Since Starting Regimen', 'CKD Prior to Treatment']
}
padding = {'pad_y0': 1.2, 'pad_x1': 2.7, 'pad_y1': 0.2}
for name, subgroups in groupings.items():
    subgroup_performance_plot(
        subgroup_performance, target_event='CKD', subgroups=subgroups, padding=padding,
        figsize=(12,30), save_dir=f'{output_path}/figures/subgroup_perf/{name}'
    )
# PPV = 0.3 means roughly for every 3 alarms, 2 are false alarms and 1 is true alarm
# Sesnsitivity = 0.5 means roughly for every 2 true alarms, the model predicts 1 of them correctly
# Event Rate = 0.15 means true alarms occur 15% of the time


# ## Decision Curve Plot

# In[29]:


result = eval_models.plot_decision_curves('ENS')
result['CKD'].tail(n=100)


# In[30]:


get_hyperparameters(output_path)


# # Scratch Notes

# ## Spline Baseline Model

# In[31]:


from src.train import TrainLOESSModel, TrainPolynomialModel
from src.evaluate import EvaluateBaselineModel


# In[32]:


def run(X, Y, tag, base_vals, output_path, alg='SPLINE', split='Test', task_type='C'):
    Trains = {'LOESS': TrainLOESSModel, 'SPLINE': TrainPolynomialModel, 'POLY': TrainPolynomialModel}
    train = Trains[alg](X, Y, tag, output_path, base_vals.name, alg, task_type=task_type)
    best_param = train.bayesopt(alg=alg, verbose=0)
    model = train.train_model(**best_param)
    Y_preds, Y_preds_min, Y_preds_max = train.predict(model, split=split)
    mask = tag['split'] == split
    preds, pred_ci, labels = {split: {alg: Y_preds}}, {split: {alg: (Y_preds_min, Y_preds_max)}}, {split: Y[mask]}
    eval_base = EvaluateBaselineModel(base_vals[mask], preds, labels, output_path, pred_ci=pred_ci)
    eval_base.all_plots(alg=alg)
    return Y_preds, Y_preds_min, Y_preds_max

preds, preds_min, preds_max = run(X, Y, tag, model_data['baseline_eGFR'], output_path)


# ### Save the CKD Spline Baseline Model as a Threshold Table

# In[34]:


cols = ['baseline_creatinine_value', 'baseline_eGFR', 'next_eGFR', 'ikn']
df = pd.concat([preds, model_data.loc[preds.index, cols]], axis=1)


# In[35]:


# Assign bins to the baseline eGFR and combine bins with less than 10 unique patients
df['baseline_eGFR'] = df['baseline_eGFR'].round(1)
tmp = df.groupby('baseline_eGFR')['ikn'].unique()
assert all(tmp.index == sorted(tmp.index))
bins, seen = list(), set()
for base_val, ikns in tmp.items():
    seen.update(ikns)
    if len(seen) > 10:
        bins.append(base_val)
        seen = set()
df['baseline_eGFR'] = pd.cut(df['baseline_eGFR'], bins=bins)


# In[37]:


df = df.groupby('baseline_eGFR').agg({
    'ikn': 'nunique',
    'baseline_creatinine_value': 'mean',
    'next_eGFR': 'mean',
    **{col: 'mean' for col in preds.columns}
}).round(3)
df.to_csv(f'{output_path}/SPLINE_model.csv')
df


# ## Missingness By Splits

# In[38]:


from src.utility import get_nmissing_by_splits


# In[39]:


missing = get_nmissing_by_splits(model_data, train_ens.labels)
missing.sort_values(by=(f'Test (N={sum(test_mask)})', 'Missing (N)'), ascending=False)


# In[ ]:
