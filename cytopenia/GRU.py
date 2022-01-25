#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('cd', '../')
# reloads all modules everytime before cell is executed (no need to restart kernel)
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import sys


# In[3]:


env = 'myenv'
user_path = 'XXXXXX'
for i, p in enumerate(sys.path):
    sys.path[i] = sys.path[i].replace("/software/anaconda/3/", f"{user_path}/.conda/envs/{env}/")
sys.prefix = f'{user_path}/.conda/envs/{env}/'


# In[4]:


import os
import tqdm
import pandas as pd
import numpy as np
import warnings
import pickle
# warnings.filterwarnings('ignore')

from scripts.utilities import (most_common_by_category, pred_thresh_binary_search)
from scripts.config import (root_path, blood_types, cytopenia_thresholds, cancer_location_mapping, cancer_type_mapping)
from scripts.prep_data import (PrepData)
from scripts.train import (TrainGRU, GRU)

import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, accuracy_score, precision_score, recall_score,
                            confusion_matrix, roc_auc_score, average_precision_score)
from bayes_opt import BayesianOptimization

from eli5.permutation_importance import get_score_importances

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

torch.manual_seed(0)
np.random.seed(0)


# In[5]:


# config
chemo_df = pd.read_csv(f'{root_path}/cytopenia/data/chemo_processed2.csv', dtype={'curr_morph_cd': str, 'lhin_cd': str})
output_path = f'{root_path}/cytopenia/models/GRU'


# # More Preprocessing

# In[6]:


def read_data(blood_type):
    df = pd.read_csv(f'{root_path}/cytopenia/data/{blood_type}.csv')
    # turn string of numbers columns into integer column 
    df.columns = df.columns.values.astype(int)
    # include all the features
    df = pd.concat([df, chemo_df], axis=1)
    # convert each day into a row
    df = df.melt(id_vars=df.columns[~df.columns.isin(range(-5,29))], var_name='Day', value_name=f'{blood_type}_count')
    df = df[~df[f'{blood_type}_count'].isnull()]
    df = df.sort_values(by=['ikn', 'prev_visit'])
    # remove negative days (they are duplicates of prev blood values mostly)
    # - BONUS: also ensures no time conflict with baseline extra blood work
    df = df[~df['Day'] < 0]
    return df

def get_data():
    data = {blood_type: read_data(blood_type) for blood_type in blood_types}
    
    # keep only rows where all blood types are present
    n_indices = data['neutrophil'].index
    h_indices = data['hemoglobin'].index
    p_indices = data['platelet'].index
    keep_indices = n_indices[n_indices.isin(h_indices) & n_indices.isin(p_indices)]
    data = {blood_type: data[blood_type].loc[keep_indices] for blood_type in blood_types}
    return data

def organize_data(data):
    model_data = data['neutrophil'] # all blood types have the same values
    model_data['hemoglobin_count'] = data['hemoglobin']['hemoglobin_count']
    model_data['platelet_count'] = data['platelet']['platelet_count']
    
    model_data['Day'] = model_data['Day'].astype(int)
    model_data['prev_visit'] = pd.to_datetime(model_data['prev_visit'])
    
    target_cols = ['target_neutrophil_count', 'target_hemoglobin_count', 
                   'target_platelet_count', 'target_day_since_starting']
    values = []
    for ikn, group in tqdm.tqdm(model_data.groupby('ikn')):
        start_date = group['prev_visit'].iloc[0]
        group['days_since_starting'] = (group['prev_visit'] - start_date + 
                                        pd.to_timedelta(group['Day'], unit='D')).dt.days
        
        group[target_cols] = group[['neutrophil_count', 'hemoglobin_count', 
                                    'platelet_count', 'days_since_starting']].shift(-1)
        
        # discard last row as there are no target blood counts to predict
        group = group.iloc[:-1]
        
        values.extend(group.values.tolist())
    model_data = pd.DataFrame(values, columns=group.columns)
    
    # remove entries where patient were administered blood transfusions and ODB growth factors
    model_data = model_data[~(model_data['H_blood_tranfused'] | model_data['ER_blood_tranfused'] | model_data['ODBGF_given'])]
    
    drop_columns = ['visit_date', 'prev_visit', 'chemo_interval', 'Day', 
                    'H_blood_tranfused', 'ER_blood_tranfused', 'ODBGF_given']
    model_data = model_data.drop(columns=drop_columns)
    
    return model_data


# In[7]:


model_data = organize_data(get_data())
model_data.to_csv(f'{output_path}/model_data.csv', index=False)


# In[7]:


model_data = pd.read_csv(f'{output_path}/model_data.csv', dtype={'curr_morph_cd': str, 'lhin_cd': str})
model_data


# In[8]:


class PrepDataGRU(PrepData):   
    def extra_norm_cols(self):
        return ['days_since_starting', 'target_day_since_starting'] + [f'{bt}_count' for bt in blood_types]
    
    def clean_feature_target_cols(self, feature_cols, target_cols):
        feature_cols = feature_cols.tolist() + ['target_day_since_starting']
        target_cols = target_cols.drop('target_day_since_starting')
        return feature_cols, target_cols


# In[9]:


# Prepare Data for Model Input
prep = PrepDataGRU()

model_data, clip_thresholds = prep.clip_outliers(model_data)
model_data = prep.dummify_data(model_data)
print(f'Size of model_data after one-hot encoding: {model_data.shape}')
train, valid, test = prep.split_data(model_data)


# # Model Training - Classification

# In[9]:


X_train, Y_train = train
X_valid, Y_valid = valid
X_test, Y_test = test


# In[10]:


Y_train = prep.regression_to_classification(Y_train, cytopenia_thresholds)
Y_valid = prep.regression_to_classification(Y_valid, cytopenia_thresholds)
Y_test = prep.regression_to_classification(Y_test, cytopenia_thresholds)


# In[11]:


# Distrubution of the sequence lengths in the training set (most patients have less than 5 blood work measurements)
dist_seq_lengths = X_train.groupby('ikn').apply(len)
fig = plt.figure(figsize=(15, 5))
plt.hist(dist_seq_lengths, bins=100)
plt.grid()
plt.show()


# In[12]:


# A closer look at the samples of sequences with length 1 to 21
fig = plt.figure(figsize=(15, 5))
plt.hist(dist_seq_lengths[dist_seq_lengths < 21], bins=20)
plt.grid()
plt.xticks(range(1, 21))
plt.show()


# In[13]:


# Initialize Training class
dataset = (X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
train = TrainGRU(dataset, output_path)


# In[22]:


save_path = f'{output_path}/gru_classifier'
model, train_losses, valid_losses, train_scores, valid_scores = train.train_classification(save=True, save_path=save_path, 
                                                                                batch_size=128, learning_rate=0.001)
np.save(f"{output_path}/loss_and_acc/train_losses.npy", train_losses)
np.save(f"{output_path}/loss_and_acc/valid_losses.npy", valid_losses)
np.save(f"{output_path}/loss_and_acc/train_scores.npy", train_scores)
np.save(f"{output_path}/loss_and_acc/valid_scores.npy", valid_scores)


# In[24]:


train_losses = np.load(f"{output_path}/loss_and_acc/train_losses.npy")
valid_losses = np.load(f"{output_path}/loss_and_acc/valid_losses.npy")
train_scores = np.load(f"{output_path}/loss_and_acc/train_scores.npy")
valid_scores = np.load(f"{output_path}/loss_and_acc/valid_scores.npy")
save_path = f'{output_path}/loss_and_acc/loss_and_acc.jpg'
train.plot_training_curve(train_losses, valid_losses, train_scores, valid_scores, save=True, save_path=save_path)


# In[26]:


model = GRU(n_features=X_train.shape[1]-1, n_targets=Y_train.shape[1], hidden_size=20, hidden_layers=3,
                      batch_size=512, dropout=0.5, pad_value=-999)

if torch.cuda.is_available():
    model.cuda()
    
save_path = f'{output_path}/gru_classifier'
model.load_state_dict(torch.load(save_path))


# In[27]:


save_path = f'{output_path}/classification_result.csv'
score_df = train.get_model_scores(model, save=True, save_path=save_path)
score_df


# # Hyperparam Tuning

# In[17]:


def load_model():
    filename = f'{output_path}/hyperparam_tuning/best_param.pkl'
    with open(filename, 'rb') as file:
        best_param = pickle.load(file)
    del best_param['learning_rate']
    model = GRU_model(n_features=len(X_train.columns)-1, n_targets=3, pad_value=-999, **best_param)
    if torch.cuda.is_available():
        model.cuda()
    save_path = f'{output_path}/gru_best_classifier'
    model.load_state_dict(torch.load(save_path))
    return model


# In[ ]:


# Conduct Bayesian Optimization 
best_param = train.bayesopt('gru', save_dir=f'{output_path}/hyperparam_tuning')


# In[28]:


# Train final model using the best parameters
filename = f'{output_path}/hyperparam_tuning/gru_classifier_best_param.pkl'
with open(filename, 'rb') as file:
    best_param = pickle.load(file)
    
save_path = f'{output_path}/gru_best_classifier'
model, train_losses, valid_losses, train_scores, valid_scores = train_classification(save=True, save_path=save_path, 
                                                                                     **best_param)
np.save(f"{output_path}/loss_and_acc/best_train_losses.npy", train_losses)
np.save(f"{output_path}/loss_and_acc/best_valid_losses.npy", valid_losses)
np.save(f"{output_path}/loss_and_acc/best_train_scores.npy", train_scores)
np.save(f"{output_path}/loss_and_acc/best_valid_scores.npy", valid_scores)


# In[29]:


train_losses = np.load(f"{output_path}/loss_and_acc/best_train_losses.npy")
valid_losses = np.load(f"{output_path}/loss_and_acc/best_valid_losses.npy")
train_scores = np.load(f"{output_path}/loss_and_acc/best_train_scores.npy")
valid_scores = np.load(f"{output_path}/loss_and_acc/best_valid_scores.npy")
save_path = f'{output_path}/loss_and_acc/best_loss_and_acc.jpg'
train.plot_training_curve(train_losses, valid_losses, train_scores, valid_scores, save=True, save_path=save_path)


# In[31]:


model = load_model()


# In[32]:


save_path = f'{output_path}/best_classification_result.csv'
score_df = train.get_model_scores(model, save=True, save_path=save_path)
score_df


# # GRU Post-Training Analysis

# In[22]:


model = load_model()


# ## Most Important Features
# I need to one-hot encode AFTER doing permutation importance<br>
# Note: After one-hot encoding<br>
# - X_train - 669 columns
# - X_valid - 562 columns
# - total dataset - 717 columns
# 
# The model must have all 717 columns, so for feature importance calculations on training set, I need to add the missing columns and fill it with zeros

# In[24]:


model_data = pd.read_csv(f'{output_path}/model_data.csv', dtype={'curr_morph_cd': str, 'lhin_cd': str})
model_data, clip_thresholds = clip_outliers(model_data)
dummy_cols = dummify_data(model_data).columns
dummy_cols = dummy_cols.drop([f'target_{bt}_count' for bt in blood_types])
(X_train, Y_train), (X_valid, Y_valid), _, _ = split_data(model_data, gru=True, convert_to_float=False, verbose=False)
Y_valid = regression_to_classification(Y_valid, cytopenia_thresholds)


# In[25]:


def score(X, Y):
    X = pd.DataFrame(X, columns=X_valid.columns)
    Y = pd.DataFrame(Y, columns=Y_valid.columns, dtype=bool)
    for col, dtype in X_valid.dtypes.items():
        X[col] = X[col].astype(dtype)
    X = dummify_data(X)
    # add the missing columns
    X[dummy_cols.difference(X.columns)] = 0
    X = X[dummy_cols]
    X = X.astype(float)
    mapping = {}
    for ikn, group in tqdm.tqdm(X.groupby('ikn')):
        group = group.drop(columns=['ikn'])
        mapping[ikn] = (group, Y.loc[group.index])
    dataset = seq_data(mapping=mapping, ids=X['ikn'].unique())
    return get_evaluation_score(model, dataset)


# In[ ]:


param = {'random_state': 42, 'n_iter':2, 'columns_to_shuffle': range(1, X_valid.shape[1])}
base_score, score_decreases = get_score_importances(score, X_valid.values, Y_valid.values, **param)
feature_importances = np.mean(score_decreases, axis=0)
np.save(f'{output_path}/feat_importance.npy', feature_importances)


# In[29]:


feature_importances = np.load(f'{output_path}/feat_importance.npy')
sorted_idx = (-feature_importances).argsort()
sorted_idx = sorted_idx[0:20] # get the top 20 important features

fig = plt.figure(figsize=(6, 4))
plt.barh(X_valid.columns[sorted_idx], feature_importances[sorted_idx])
plt.gca().invert_yaxis()
plt.xlabel('Feature Importance Score')
plt.savefig(f'{output_path}/feat_importance.jpg', bbox_inches='tight') #dpi=300


# ## GRU performance for most common cancer regimens/cancer type

# In[29]:


def GRU_results_by_category(model, most_common, category='regimen', save=True, pad_value=-999):
    indices = pd.MultiIndex.from_product([[],[]])
    score_df = pd.DataFrame(index=indices)

    for entry in most_common: 
        mask = X_test[f'{category}_{entry}'] == 1
        X = X_test[mask]
        Y = Y_test[mask]
        
        mapping = {}
        for ikn, group in tqdm.tqdm(X.groupby('ikn')):
            group = group.drop(columns=['ikn'])
            mapping[ikn] = (group, Y.loc[group.index])
        dataset = seq_data(mapping=mapping, ids=X['ikn'].unique())
        
        pred, target = get_model_predictions(model, dataset)
        if category == 'curr_morph_cd': entry = cancer_type_mapping[entry]
        if category == 'curr_topog_cd': entry = cancer_location_mapping[entry]
        for idx, blood_type in enumerate(blood_types):
            Y_true = target[:, idx]
            # if no positive examples exists, skip
            if not Y_true.any(axis=None):
                print(f'Found no positive examples in {category} {entry} for blood type {blood_type} - Skipping')
                continue
            
            Y_pred_prob = pred[:, idx]
            score_df.loc[(entry, 'AUROC Score'), blood_type] = roc_auc_score(Y_true, Y_pred_prob)
            score_df.loc[(entry, 'AUPRC Score'), blood_type] = average_precision_score(Y_true, Y_pred_prob)

    if save:
        name_map = {'regimen': 'regimen', 'curr_morph_cd': 'cancer_type', 'curr_topog_cd': 'cancer_location'}
        score_df.to_csv(f'{output_path}/common_{name_map[category]}_results.csv')
    return score_df


# In[20]:


model_data = pd.read_csv(f'{output_path}/model_data.csv', dtype={'curr_morph_cd': str, 'lhin_cd': str})

top_regimens = most_common_by_category(model_data, category='regimen')
most_common_regimens = list(top_regimens.index)

top_cancer_types = most_common_by_category(model_data, category='curr_morph_cd')
most_common_cancer_types = list(top_cancer_types.index)

top_cancer_locations = most_common_by_category(model_data, category='curr_topog_cd')
most_common_cancer_locations = list(top_cancer_locations.index)


# In[25]:


GRU_results_by_category(model, most_common_regimens, category='regimen')


# In[26]:


GRU_results_by_category(model, most_common_cancer_types, category='curr_morph_cd')


# In[30]:


GRU_results_by_category(model, most_common_cancer_locations, category='curr_topog_cd')


# ## GRU Precision vs Sensitivity Operating Points

# In[32]:


pred_thresholds = np.arange(0, 1.01, 0.1)
train.threshold_op_points(pred_thresholds, train.test_dataset)


# In[33]:


def precision_op_points(precisions, dataset):
    cols = pd.MultiIndex.from_product([blood_types, ['Prediction Threshold', 'Sensitivity']])
    df = pd.DataFrame(columns=cols)
    df.index.name = 'Precision'

    pred, target = get_model_predictions(model, dataset)
    for idx, blood_type in enumerate(blood_types):
        Y_true = target[:, idx]
        Y_pred_prob = pred[:, idx]
        for desired_precision in precisions:
            # LOL binary search the threshold to get desired precision
            threshold = pred_thresh_binary_search(desired_precision, Y_pred_prob, Y_true)
            Y_pred_bool = Y_pred_prob > threshold
            df.loc[desired_precision, (blood_type, 'Prediction Threshold')] = threshold
            df.loc[desired_precision, (blood_type, 'Sensitivity')] = recall_score(Y_true, Y_pred_bool, zero_division=1)
    return df


# In[34]:


precisions = [0.2, 0.25, 0.33, 0.4, 0.5, 0.6, 0.75]
precision_op_points(precisions, test_dataset)


# In[ ]:




