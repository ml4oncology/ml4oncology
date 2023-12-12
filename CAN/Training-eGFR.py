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
from src.evaluate import EvaluateReg
from src.prep_data import PrepDataCAN
from src.train import Ensembler, Trainer
from src.utility import initialize_folders, load_pickle, get_hyperparameters
from src.visualize import importance_plot


# In[3]:


processes = 64
target_keyword = 'SCr|dialysis|next'
main_dir = f'{root_path}/{can_folder}'
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


# In[90]:


for split, label in labels.items():
    # scale the labels
    labels[split][:] = scaler.inverse_transform(label)
    # scale the predictions
    for alg in ensembler.models:
        pred = preds[alg][split]
        preds[alg][split][:] = scaler.inverse_transform(pred)


# In[91]:


evaluator = EvaluateReg(output_path, preds, labels)
evaluator.get_evaluation_scores(display_ci=True, load_ci=False, save_ci=True)


# In[92]:


evaluator.plot_err_dist(alg='LR', target_event='next_eGFR')
evaluator.plot_err_dist(alg='LR', target_event='eGFR_change')


# ## Most Important Features

# In[28]:


get_ipython().run_line_magic('run', "scripts/feat_imp.py --adverse-event CKD --output-path {output_path} --task-type 'R'")


# In[94]:


# importance score is defined as the increase in MSE when feature value is randomly shuffled
importance_plot(
    'ENS', ['next_eGFR'], output_path, figsize=(6,5), top=10, importance_by='feature', padding={'pad_x0': 2.7}, 
    smaller_is_better=True
)


# # Scratch Notes

# ## Spline Baseline Model

# In[95]:


from src.train import LOESSModelTrainer, PolynomialModelTrainer
from src.evaluate import EvaluateBaselineModel


# In[100]:


def run(X, Y, tag, base_vals, output_path, scale_func, alg='SPLINE', split='Test', task_type='R', save=True):
    Trainers = {'LOESS': LOESSModelTrainer, 'SPLINE': PolynomialModelTrainer, 'POLY': PolynomialModelTrainer}
    trainer = Trainers[alg](X, Y, tag, output_path, base_vals.name, alg, task_type=task_type)
    best_param = trainer.bayesopt(alg=alg)
    model = trainer.train_model(save=save, **best_param)
    Y_preds, Y_preds_min, Y_preds_max = trainer.predict(model, split=split)
    Y_preds[:], Y_preds_min[:], Y_preds_max[:] = scale_func(Y_preds), scale_func(Y_preds_min), scale_func(Y_preds_max)
    mask = tag['split'] == split
    preds, labels = {alg: {split: Y_preds}}, {split: Y[mask]}
    eval_base = EvaluateBaselineModel(base_vals[mask], preds, labels, output_path)
    for target_event in Y:
        fig, ax = plt.subplots(figsize=(6,6))
        eval_base.plot_prediction(ax, alg, target_event, split=split, show_diagonal=True)
        plt.savefig(f'{output_path}/figures/baseline/{target_event}_{alg}.jpg', bbox_inches='tight', dpi=300)
        
    return Y_preds, Y_preds_min, Y_preds_max

preds, preds_min, preds_max = run(X, Y, tag, model_data['baseline_eGFR'], output_path, scaler.inverse_transform)


# In[101]:


base_eGFR = model_data.loc[test_mask, 'baseline_eGFR']
for eGFR in [40, 60, 80, 100]:
    mask = base_eGFR.round(1) == eGFR
    pred = preds.loc[mask, 'next_eGFR'].mean()
    print(f'Pre-treatment eGFR={eGFR}. Post-treatment eGFR Prediction={pred:.2f}')


# In[102]:


eval_models = EvaluateReg(output_path, preds={'SPLINE': {'Test': preds}}, labels=labels)
eval_models.get_evaluation_scores(splits=['Test'], display_ci=True, load_ci=False, save_ci=False)


# ### Save the Spline Baseline Model as a Threshold Table

# In[109]:


preds.columns = 'predicted_'+preds.columns
cols = ['baseline_creatinine_value', 'baseline_eGFR', 'next_eGFR', 'ikn']
df = pd.concat([preds, model_data.loc[preds.index, cols]], axis=1)


# In[110]:


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


# In[111]:


df = df.groupby('baseline_eGFR').agg({
    'ikn': 'nunique',
    'baseline_creatinine_value': 'mean',
    'next_eGFR': 'mean',
    **{col: 'mean' for col in preds.columns}
}).round(3)
df.to_csv(f'{output_path}/SPLINE_model.csv')
df


# In[ ]:
