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


# In[93]:


import copy

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 150)
pd.set_option('display.max_rows', 150)
import seaborn as sns

from src.config import root_path, can_folder, split_date, SCr_rise_threshold
from src.evaluate import EvaluateReg, EvaluateBaselineModel
from src.prep_data import PrepDataCAN
from src.train import Ensembler, Trainer, PolynomialModelTrainer
from src.utility import initialize_folders, load_pickle, get_hyperparameters
from src.visualize import importance_plot


# In[3]:


processes = 64
target_keyword = 'SCr|dialysis|next'
main_dir = f'{root_path}/projects/{can_folder}'
output_path = f'{main_dir}/models/eGFR'
initialize_folders(output_path)


# # Prepare Data for Model Training

# In[84]:


prep = PrepDataCAN(adverse_event='ckd', target_keyword=target_keyword)
model_data = prep.get_data(missing_thresh=80, include_comorbidity=True, verbose=True, first_course_treatment=True)
(model_data['next_eGFR'] - model_data['baseline_eGFR']).hist(bins=100)
X, _, tag = prep.split_and_transform_data(model_data, split_date=split_date)
# remove sessions in model_data that were excluded during split_and_transform
model_data = model_data.loc[tag.index]


# In[85]:


# Convenience variables
train_mask, valid_mask, test_mask = tag['split'] == 'Train', tag['split'] == 'Valid', tag['split'] == 'Test'
X_train, X_valid, X_test = X[train_mask], X[valid_mask], X[test_mask]


# In[86]:


# setup regression label
Y = pd.DataFrame()
Y['next_eGFR'] = model_data['next_eGFR']
Y['eGFR_change'] = model_data['next_eGFR'] - model_data['baseline_eGFR']
# scale the target
scaler = StandardScaler()
Y[train_mask] = Y_train = scaler.fit_transform(Y[train_mask])
Y[valid_mask] = Y_valid = scaler.transform(Y[valid_mask])
Y[test_mask] = Y_test = scaler.transform(Y[test_mask])


# # Train Models

# ## Spline Baseline Model

# In[18]:


trainer = PolynomialModelTrainer(X, Y, tag, output_path, base_col='baseline_eGFR', alg='SPLINE', task_type='R')
trainer.run(bayesopt=True, train=True, save=True)


# In[59]:


# save the model as a table
df = trainer.model_to_table(
    model=load_pickle(output_path, 'SPLINE'),
    base_vals=model_data['baseline_eGFR'],
    extra_info=model_data[['baseline_creatinine_value', 'next_eGFR']].rename(columns={'next_eGFR': 'true_next_eGFR'})
)
df[Y.columns] = scaler.inverse_transform(df[Y.columns])
df.to_csv(f'{output_path}/SPLINE_model.csv')
df


# ## Main Models

# In[87]:


trainer = Trainer(X, Y, tag, output_path, task_type='R')
trainer.run(bayesopt=True, train=True, save_preds=True, algs=['LR', 'RF', 'XGB', 'NN'], allow_duplicate_points=True)


# ## ENS Model 
# Find Optimal Ensemble Weights

# In[88]:


preds = load_pickle(f'{output_path}/preds', 'all_preds')
ensembler = Ensembler(X, Y, tag, output_path, preds, task_type='R')
ensembler.run(bayesopt=True, calibrate=False)


# # Evaluate Models

# In[89]:


preds, labels = copy.deepcopy(ensembler.preds), copy.deepcopy(ensembler.labels)
# Include the baseline models
preds.update(load_pickle(f'{output_path}/preds', 'SPLINE_preds'))


# In[90]:


for split, label in labels.items():
    # inverse scale the labels
    labels[split][:] = scaler.inverse_transform(label)
    # inverse scale the predictions
    for alg, pred in preds.items():
        preds[alg][split][:] = scaler.inverse_transform(pred[split])


# In[91]:


evaluator = EvaluateReg(output_path, preds, labels)
evaluator.get_evaluation_scores(display_ci=True, load_ci=False, save_ci=True)


# In[92]:


evaluator.plot_err_dist(alg='LR', target_event='next_eGFR')
evaluator.plot_err_dist(alg='LR', target_event='eGFR_change')

