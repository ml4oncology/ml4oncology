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
from src.evaluate import EvaluateClf, EvaluateBaselineModel
from src.model import SimpleBaselineModel
from src.prep_data import PrepDataCAN
from src.summarize import data_description_summary, feature_summary
from src.train import Ensembler, LASSOTrainer, PolynomialModelTrainer, Trainer
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
adverse_event = 'ckd'
output_path = f'{main_dir}/models/{adverse_event.upper()}'
initialize_folders(output_path)


# # Prepare Data for Model Training

# In[18]:


prep = PrepDataCAN(adverse_event=adverse_event, target_keyword=target_keyword)
model_data = prep.get_data(include_comorbidity=True, verbose=True, first_course_treatment=True)
model_data


# In[19]:


most_common_categories(model_data, catcol='regimen', with_respect_to='patients', top=10)


# In[20]:


sorted(model_data.columns)


# In[21]:


get_nunique_categories(model_data)


# In[22]:


get_nmissing(model_data)


# In[27]:


prep = PrepDataCAN(adverse_event=adverse_event, target_keyword=target_keyword)
model_data = prep.get_data(missing_thresh=80, include_comorbidity=True, verbose=True, first_course_treatment=True)
model_data['next_eGFR'].hist(bins=100)
X, Y, tag = prep.split_and_transform_data(model_data, split_date=split_date)
# remove sessions in model_data that were excluded during split_and_transform
model_data = model_data.loc[tag.index]


# In[28]:


prep.get_label_distribution(Y, tag, with_respect_to='sessions')


# In[29]:


prep.get_label_distribution(Y, tag, with_respect_to='patients')


# In[30]:


# Convenience variables
train_mask, valid_mask, test_mask = tag['split'] == 'Train', tag['split'] == 'Valid', tag['split'] == 'Test'
X_train, X_valid, X_test = X[train_mask], X[valid_mask], X[test_mask]
Y_train, Y_valid, Y_test = Y[train_mask], Y[valid_mask], Y[test_mask]


# ## Study Characteristics

# In[31]:


subgroups = [
    'sex', 'immigration', 'birth_region', 'language', 'income', 'area_density',
    'regimen', 'cancer_type', 'cancer_location', 'target', 'comorbidity', 'dialysis', 'ckd'
]
data_description_summary(
    model_data, Y, tag, save_dir=f'{output_path}/tables', partition_method='cohort', target_event='CKD', 
    subgroups=subgroups
)


# ## Feature Characteristic

# In[32]:


df = prep.ohe.encode(model_data.copy(), verbose=False) # get original (non-normalized, non-imputed) data one-hot encoded
df = df[train_mask].drop(columns=['ikn'])
feature_summary(
    df, save_dir=f'{output_path}/tables', deny_old_survey=True, event_dates=prep.event_dates[train_mask]
).head(60)


# # Train Models

# ## Spline Baseline Model

# In[18]:


trainer = PolynomialModelTrainer(X, Y, tag, output_path, base_col='baseline_eGFR', alg='SPLINE')
trainer.run(bayesopt=True, train=True, save=True)


# In[59]:


# save the model as a table
df = trainer.model_to_table(
    model=load_pickle(output_path, 'SPLINE'),
    base_vals=model_data['baseline_eGFR'],
    extra_info=model_data[['baseline_creatinine_value', 'next_eGFR']]
)
df.to_csv(f'{output_path}/SPLINE_model.csv')
df


# ## Main Models

# In[35]:


trainer = Trainer(X, Y, tag, output_path)
trainer.run(bayesopt=True, train=True, save_preds=True, algs=['LR', 'RF', 'XGB', 'NN'])


# ## ENS Model 
# Find Optimal Ensemble Weights

# In[36]:


preds = load_pickle(f'{output_path}/preds', 'all_preds')
ensembler = Ensembler(X, Y, tag, output_path, preds)
ensembler.run(bayesopt=True, calibrate=True)


# # Evaluate Models

# In[37]:


# setup the final prediction and labels
preds, labels = ensembler.preds, ensembler.labels
# Include the baseline models
preds.update(SimpleBaselineModel(model_data[['regimen', 'baseline_eGFR']], labels).predict())
preds.update(load_pickle(f'{output_path}/preds', 'SPLINE_preds'))


# In[24]:


evaluator = EvaluateClf(output_path, preds, labels)
scores = evaluator.get_evaluation_scores(display_ci=True, load_ci=True, save_ci=True)
scores


# In[26]:


scores.loc[['ENS', 'SPLINE']]


# In[40]:


evaluator.plot_curves(curve_type='pr', legend_loc='lower left', save=False)
evaluator.plot_curves(curve_type='roc', legend_loc='lower right', save=False)
evaluator.plot_curves(curve_type='pred_cdf', save=False) # cumulative distribution function of model prediction
evaluator.plot_calibs(legend_loc='upper left', save=False) 
# evaluator.plot_calibs(include_pred_hist=True, legend_loc='upper left', figsize=(12,28), padding={'pad_y1': 0.3, 'pad_y0': 3.0})


# # Post-Training Analysis

# ## Threshold Op Points

# In[41]:


pred_thresholds = np.arange(0.05, 0.51, 0.05)
perf_metrics = ['warning_rate', 'precision', 'recall', 'NPV', 'specificity']
thresh_df = evaluator.operating_points(pred_thresholds, alg='ENS', op_metric='threshold', perf_metrics=perf_metrics)
thresh_df


# ## All the Plots

# In[42]:


evaluator.all_plots_for_single_target(alg='ENS', target_event='CKD')


# ## Most Important Features/Feature Groups

# In[ ]:


get_ipython().system('python scripts/feat_imp.py --adverse-event CKD --output-path {output_path} ')


# In[24]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot('ENS', evaluator.target_events, output_path, figsize=(6,15), top=10, importance_by='feature', padding={'pad_x0': 2.7})


# In[ ]:


get_ipython().system('python scripts/feat_imp.py --adverse-event CKD --output-path {output_path} --permute-group')


# In[26]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot('ENS', evaluator.target_events, output_path, figsize=(6,15), top=10, importance_by='group', padding={'pad_x0': 1.2})


# ## Performance on Subgroups

# In[48]:


subgroups = [
    'all', 'age', 'sex', 'immigrant', 'language', 'arrival', 'income', 
    'area_density', 'ckd', 'regimen', 'cancer_location', 'days_since_starting', 
]
perf_kwargs = {'perf_metrics': ['precision', 'recall', 'event_rate']}
subgroup_performance = evaluator.get_perf_by_subgroup(
    model_data, subgroups=subgroups, pred_thresh=0.1, alg='ENS', display_ci=True, load_ci=False, perf_kwargs=perf_kwargs
)
subgroup_performance


# In[49]:


subgroup_performance = pd.read_csv(f'{output_path}/tables/subgroup_performance.csv', index_col=[0,1], header=[0,1])
groupings = {
    'Demographic': ['Entire Test Cohort', 'Age', 'Sex', 'Immigration', 'Language', 'Neighborhood Income Quintile'],
    'Treatment': ['Entire Test Cohort', 'Regimen', 'Topography ICD-0-3', 'CKD Prior to Treatment']
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

# In[50]:


result = evaluator.plot_decision_curves('ENS')
result['CKD'].tail(n=100)


# In[51]:


get_hyperparameters(output_path)


# ## Prediction vs Baseline Plots

# In[70]:


preds = load_pickle(f'{output_path}/preds', 'SPLINE_preds')
pred_ci = load_pickle(f'{output_path}/preds', 'SPLINE_preds_ci')
baseline_evaluator = EvaluateBaselineModel(
    model_data['baseline_eGFR'][test_mask], preds, labels, output_path, pred_ci=pred_ci
)
target_events = Y.columns
fig, axes = plt.subplots(nrows=1, ncols=len(target_events), figsize=(6*len(target_events), 6))
for i, target_event in enumerate(target_events):
    baseline_evaluator.plot_pred_vs_base(axes[i], alg='SPLINE', target_event=target_event, split='Test')
plt.savefig(f'{output_path}/figures/baseline/SPLINE_pred_vs_baseline.jpg', bbox_inches='tight', dpi=300)


# Scratch Notes

# ## Missingness By Splits

# In[61]:


from src.utility import get_nmissing_by_splits


# In[63]:


missing = get_nmissing_by_splits(model_data, ensembler.labels)
missing.sort_values(by=(f'Test (N={sum(test_mask)})', 'Missing (N)'), ascending=False)


# In[ ]:
