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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from src.config import root_path, can_folder, split_date, SCr_rise_threshold
from src.evaluate import EvaluateClf, EvaluateReg
from src.model import SimpleBaselineModel
from src.prep_data import PrepData, PrepDataCAN
from src.train import TrainML, TrainRNN, TrainENS
from src.utility import initialize_folders, load_pickle, get_eGFR


# In[3]:


# config
processes = 64
target_keyword = 'SCr|dialysis|next'
main_dir = f'{root_path}/{can_folder}'


# In[46]:


def load_prediction(output_path, tag, include_rnn_pred=True):
    # Need to match the original predictions to the new train-valid-test split by combining and reallocating :(
    preds = {}
    preds_ml = load_pickle(f'{output_path}/preds', 'ML_preds')
    for split, values in preds_ml.items():
        for alg, pred in values.items():
            preds[alg] = pd.concat([preds.get(alg, pd.DataFrame()), pred])
            
    if include_rnn_pred:
        preds_rnn = load_pickle(f'{output_path}/preds', 'RNN_preds')
        preds['RNN'] = pd.concat(preds_rnn.values())
        
    for alg in list(preds.keys()):
        pred = preds.pop(alg)
        for split, group in tag.groupby('split'):
            if split not in preds: preds[split] = {}
            preds[split][alg] = pred.loc[group.index]
    return preds


# In[79]:


def get_ckd_evaluator(X, Y, tag, model_data, result_path):
    initialize_folders(result_path)
    
    preds = load_prediction(f'{main_dir}/models/CKD', tag)
    train_ens = TrainENS(X, Y, tag, f'{main_dir}/models/CKD', preds=preds)
    train_ens.tune_and_train(run_bayesopt=False, run_calibration=False, calibrate_pred=True)
    
    preds, labels = train_ens.preds, train_ens.labels
    base_model = SimpleBaselineModel(model_data[['regimen', 'baseline_eGFR']], labels)
    base_preds = base_model.predict()
    for split, pred in base_preds.items(): preds[split].update(pred)
    eval_models = EvaluateClf(result_path, preds, labels)
    return eval_models


# In[72]:


def get_eGFR_evaluator(X, tag, model_data, result_path):
    initialize_folders(result_path)
    
    # setup regression label
    train_mask, valid_mask, test_mask = tag['split'] == 'Train', tag['split'] == 'Valid', tag['split'] == 'Test'
    Y = model_data[['next_eGFR']].copy()
    scaler = StandardScaler()
    Y[train_mask] = scaler.fit_transform(Y[train_mask])
    Y[valid_mask] = scaler.transform(Y[valid_mask])
    Y[test_mask] = scaler.transform(Y[test_mask])
    
    preds = load_prediction(f'{main_dir}/models/eGFR', tag, include_rnn_pred=False)
    train_ens = TrainENS(X, Y, tag, f'{main_dir}/models/eGFR', preds, task_type='R')
    train_ens.tune_and_train(run_bayesopt=False, run_calibration=False, calibrate_pred=True)
    
    preds, labels = train_ens.preds.copy(), train_ens.labels.copy()
    mean_change = (model_data.loc[train_mask, 'next_eGFR'] - model_data.loc[train_mask, 'baseline_eGFR']).mean()
    for split, label in labels.items():
        # scale the labels and predictions
        labels[split][:] = scaler.inverse_transform(label)
        for alg, pred in preds[split].items(): preds[split][alg][:] = scaler.inverse_transform(pred)
        # add baseline predictions
        base_eGFR = model_data.loc[label.index, 'baseline_eGFR'].to_numpy()
        kwargs = {'index': label.index, 'columns': ['next_eGFR']}
        preds[split].update({
            'Baseline - Pretreatment eGFR': pd.DataFrame(base_eGFR, **kwargs),
            'Baseline - Pretreatment eGFR + Mean Change': pd.DataFrame(base_eGFR + mean_change, **kwargs)
        })
    eval_models = EvaluateReg(result_path, preds, labels)
    return eval_models


# # Label uses average of 2 measurements

# In[83]:


prep = PrepDataCAN(adverse_event='ckd', target_keyword=target_keyword)
model_data = prep.get_data(use_target_average=True, missing_thresh=80, include_comorbidity=True, verbose=True)
model_data['next_eGFR'].hist(bins=100)
X, Y, tag = prep.split_and_transform_data(model_data, split_date=split_date)
# remove sessions in model_data that were excluded during split_and_transform
model_data = model_data.loc[tag.index]


# In[84]:


# Insepct the difference between first future measurement target average of two future measurement target
prep = PrepDataCAN(adverse_event='ckd', target_keyword=target_keyword)
tmp = prep.get_data(missing_thresh=80, include_comorbidity=True, verbose=False)
diff = tmp.loc[model_data.index, 'next_eGFR'] - model_data['next_eGFR']
diff.hist(bins=100)


# ## CKD

# In[85]:


result_path = f'{main_dir}/experiment/ckd_target_avg'
eval_models = get_ckd_evaluator(X, Y, tag, model_data, result_path)


# In[86]:


"""
Scores increased
"""
eval_models.get_evaluation_scores(display_ci=True, load_ci=True, save_ci=False)


# In[87]:


eval_models.operating_points(
    points=np.arange(0.05, 0.51, 0.05), alg='ENS', op_metric='threshold', 
    perf_metrics=['warning_rate', 'precision', 'recall', 'NPV', 'specificity']
)


# ## eGFR

# In[88]:


result_path = f'{main_dir}/experiment/eGFR_target_avg'
eval_models = get_eGFR_evaluator(X, tag, model_data, result_path)


# In[89]:


"""
Scores increased
"""
eval_models.get_evaluation_scores(display_ci=True, load_ci=True, save_ci=False)


# # Only First Treatment Session

# In[65]:


prep = PrepDataCAN(adverse_event='ckd', target_keyword=target_keyword)
model_data = prep.get_data(missing_thresh=80, include_comorbidity=True, verbose=True, first_course_treatment=True)
# model_data = model_data.reset_index().groupby('ikn').first().reset_index().set_index('index')
model_data['next_eGFR'].hist(bins=100)
X, Y, tag = prep.split_and_transform_data(model_data, split_date=split_date)
# remove sessions in model_data that were excluded during split_and_transform
model_data = model_data.loc[tag.index]

# Convenience variables
train_mask, valid_mask, test_mask = tag['split'] == 'Train', tag['split'] == 'Valid', tag['split'] == 'Test'


# ## CKD

# In[80]:


result_path = f'{main_dir}/experiment/ckd_first_trt'
eval_models = get_ckd_evaluator(X, Y, tag, model_data, result_path)


# In[81]:


"""
Scores decreased
"""
eval_models.get_evaluation_scores(display_ci=True, load_ci=True, save_ci=False)


# In[82]:


eval_models.operating_points(
    points=np.arange(0.05, 0.51, 0.05), alg='ENS', op_metric='threshold', 
    perf_metrics=['warning_rate', 'precision', 'recall', 'NPV', 'specificity']
)


# ## eGFR

# In[76]:


result_path = f'{main_dir}/experiment/eGFR_first_trt'
eval_models = get_eGFR_evaluator(X, tag, model_data, result_path)


# In[77]:


"""
Scores are worse (errors are larger), as expected
"""
eval_models.get_evaluation_scores(display_ci=True, load_ci=True, save_ci=False)


# # Label is the Change in eGFR

# In[90]:


prep = PrepDataCAN(adverse_event='ckd', target_keyword=target_keyword)
model_data = prep.get_data(missing_thresh=80, include_comorbidity=True, verbose=True)
(model_data['next_eGFR'] - model_data['baseline_eGFR']).hist(bins=100)
X, Y, tag = prep.split_and_transform_data(model_data, split_date=split_date)
# remove sessions in model_data that were excluded during split_and_transform
model_data = model_data.loc[tag.index]


# In[98]:


# setup regression label
train_mask, valid_mask, test_mask = tag['split'] == 'Train', tag['split'] == 'Valid', tag['split'] == 'Test'
Y = pd.DataFrame(model_data['next_eGFR'] - model_data['baseline_eGFR'], columns=['eGFR_change'])
scaler = StandardScaler()
Y[train_mask] = scaler.fit_transform(Y[train_mask])
Y[valid_mask] = scaler.transform(Y[valid_mask])
Y[test_mask] = scaler.transform(Y[test_mask])


# In[99]:


result_path = f'{main_dir}/experiment/eGFR_change'
initialize_folders(result_path)


# In[101]:


get_ipython().system('cp {main_dir}/models/eGFR/best_params/*.pkl {result_path}/best_params')


# In[102]:


# Train ML Models
train_ml = TrainML(X, Y, tag, result_path, n_jobs=processes, task_type='R')
train_ml.tune_and_train(run_bayesopt=False, run_training=True, save_preds=True, allow_duplicate_points=True)


# In[103]:


# Get Evaluator
preds = load_pickle(f'{result_path}/preds', 'ML_preds')
train_ens = TrainENS(X, Y, tag, result_path, preds, task_type='R')
train_ens.tune_and_train(run_bayesopt=True, run_calibration=True, calibrate_pred=True)
preds, labels = train_ens.preds, train_ens.labels
for split, label in labels.items():
    labels[split][:] = scaler.inverse_transform(label)
    for alg, pred in preds[split].items(): preds[split][alg][:] = scaler.inverse_transform(pred)
eval_models = EvaluateReg(result_path, preds, labels)


# In[104]:


eval_models.get_evaluation_scores(display_ci=True, load_ci=False, save_ci=True)


# In[ ]:
