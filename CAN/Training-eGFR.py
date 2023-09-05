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
from src.train import TrainML, TrainENS
from src.utility import initialize_folders, load_pickle, get_hyperparameters


# In[3]:


processes = 64
target_keyword = 'SCr|dialysis|next'
main_dir = f'{root_path}/{can_folder}'
output_path = f'{main_dir}/models/eGFR'
initialize_folders(output_path)


# # Prepare Data for Model Training

# In[4]:


prep = PrepDataCAN(adverse_event='ckd', target_keyword=target_keyword)
model_data = prep.get_data(missing_thresh=80, include_comorbidity=True, verbose=True)
(model_data['next_eGFR'] - model_data['baseline_eGFR']).hist(bins=100)
X, _, tag = prep.split_and_transform_data(model_data, split_date=split_date)
# remove sessions in model_data that were excluded during split_and_transform
model_data = model_data.loc[tag.index]


# In[5]:


# Convenience variables
train_mask, valid_mask, test_mask = tag['split'] == 'Train', tag['split'] == 'Valid', tag['split'] == 'Test'
X_train, X_valid, X_test = X[train_mask], X[valid_mask], X[test_mask]


# In[6]:


# setup regression label
Y = model_data[['next_eGFR']].copy()
# scale the target
scaler = StandardScaler()
Y[train_mask] = Y_train = scaler.fit_transform(Y[train_mask])
Y[valid_mask] = Y_valid = scaler.transform(Y[valid_mask])
Y[test_mask] = Y_test = scaler.transform(Y[test_mask])


# # Train Models

# ## Main ML Models

# In[21]:


train_ml = TrainML(X, Y, tag, output_path, n_jobs=processes, task_type='R')
train_ml.tune_and_train(run_bayesopt=False, run_training=True, save_preds=True, allow_duplicate_points=True)


# ## ENS Model 

# In[22]:


# Initialize Training Class
preds = load_pickle(f'{output_path}/preds', 'ML_preds')
train_ens = TrainENS(X, Y, tag, output_path, preds, task_type='R')


# In[23]:


train_ens.tune_and_train(run_bayesopt=True, run_calibration=True, calibrate_pred=True)


# # Evaluate Models

# In[24]:


preds, labels = train_ens.preds, train_ens.labels


# In[25]:


# used for baseline prediction
mean_change = (model_data.loc[train_mask, 'next_eGFR'] - model_data.loc[train_mask, 'baseline_eGFR']).mean()
for split, label in labels.items():
    # scale the labels
    labels[split][:] = scaler.inverse_transform(label)
    # scale the predictions
    for alg, pred in preds[split].items():
        preds[split][alg][:] = scaler.inverse_transform(pred)
    # add baseline predictions
    base_eGFR = model_data.loc[label.index, 'baseline_eGFR'].to_numpy()
    kwargs = {'index': label.index, 'columns': ['next_eGFR']}
    preds[split].update({
        'Baseline - Pretreatment eGFR': pd.DataFrame(base_eGFR, **kwargs),
        'Baseline - Pretreatment eGFR + Mean Change': pd.DataFrame(base_eGFR + mean_change, **kwargs)
    })


# In[26]:


eval_models = EvaluateReg(output_path, preds, labels)
eval_models.get_evaluation_scores(display_ci=True, load_ci=False, save_ci=True)


# In[27]:


eval_models.plot_err_dist(alg='ENS', target_event='next_eGFR')


# ## Most Important Features

# In[28]:


get_ipython().run_line_magic('run', "scripts/feat_imp.py --adverse-event CKD --output-path {output_path} --task-type 'R'")


# In[29]:


# importance score is defined as the increase in MSE when feature value is randomly shuffled
importance_plot(
    'ENS', ['next_eGFR'], output_path, figsize=(6,5), top=10, importance_by='feature', padding={'pad_x0': 2.7}, 
    smaller_is_better=True
)


# # Scratch Notes

# ## Spline Baseline Model

# In[ ]:


from src.train import TrainLOESSModel, TrainPolynomialModel
from src.evaluate import EvaluateBaselineModel


# In[ ]:


def run(X, Y, tag, base_vals, output_path, scale_func, alg='SPLINE', split='Test', task_type='R'):
    Trains = {'LOESS': TrainLOESSModel, 'SPLINE': TrainPolynomialModel, 'POLY': TrainPolynomialModel}
    train = Trains[alg](X, Y, tag, output_path, base_vals.name, alg, task_type=task_type)
    best_param = train.bayesopt(alg=alg, verbose=0)
    model = train.train_model(**best_param)
    Y_preds, Y_preds_min, Y_preds_max = train.predict(model, split=split)
    Y_preds[:], Y_preds_min[:], Y_preds_max[:] = scale_func(Y_preds), scale_func(Y_preds_min), scale_func(Y_preds_max)
    mask = tag['split'] == split
    preds, labels = {split: {alg: Y_preds}}, {split: Y[mask]}
    eval_base = EvaluateBaselineModel(base_vals[mask], preds, labels, output_path)
    for target_event in Y:
        fig, ax = plt.subplots(figsize=(6,6))
        eval_base.plot_prediction(ax, alg, target_event, split=split, show_diagonal=True)
        plt.savefig(f'{output_path}/figures/baseline/{target_event}_{alg}.jpg', bbox_inches='tight', dpi=300)
        
    return Y_preds, Y_preds_min, Y_preds_max

preds, preds_min, preds_max = run(X, Y, tag, model_data['baseline_eGFR'], output_path, scaler.inverse_transform)


# In[ ]:


base_eGFR = model_data.loc[test_mask, 'baseline_eGFR']
for eGFR in [40, 60, 80, 100]:
    mask = base_eGFR.round(1) == eGFR
    pred = preds.loc[mask, 'next_eGFR'].mean()
    print(f'Pre-treatment eGFR={eGFR}. Post-treatment eGFR Prediction={pred:.2f}')


# In[ ]:


preds = {'Test': {'SPLINE': preds}} 
eval_models = EvaluateReg(output_path, preds, labels)
eval_models.get_evaluation_scores(splits=['Test'], display_ci=True, load_ci=False, save_ci=False)
