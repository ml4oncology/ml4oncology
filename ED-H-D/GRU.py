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
import pickle
import matplotlib.pyplot as plt

from scripts.preprocess import (replace_rare_col_entries)
from scripts.config import (root_path)
from scripts.prep_data import (PrepDataEDHD)
from scripts.train import (TrainGRU, GRU)

import torch


# In[5]:


# config
days = 30 # predict event within this number of days since chemo visit
target_keyword = f'_within_{days}days'
main_dir = f'{root_path}/ED-H-D'
output_path = f'{main_dir}/models/GRU/within_{days}_days'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    os.makedirs(f'{output_path}/hyperparam_tuning')
    os.makedirs(f'{output_path}/loss_and_acc')


# # More Preprocessing

# In[6]:


class PrepDataGRU(PrepDataEDHD):   
    def clean_feature_target_cols(self, feature_cols, target_cols):
        return feature_cols, target_cols
    def extra_norm_cols(self):
        return super().extra_norm_cols() + ['days_since_true_prev_chemo']


# In[7]:


# Prepare Data for Model Input
prep = PrepDataGRU()


# In[9]:


model_data = prep.get_data(main_dir, target_keyword, rem_days_since_prev_chemo=False)
model_data


# In[10]:


model_data, clip_thresholds = prep.clip_outliers(model_data, lower_percentile=0.001, upper_percentile=0.999)
model_data = prep.dummify_data(model_data)
print(f'Size of model_data after one-hot encoding: {model_data.shape}')
train, valid, test = prep.split_data(model_data, target_keyword=target_keyword, convert_to_float=False)


# In[11]:


X_train, Y_train = train
X_valid, Y_valid = valid
X_test, Y_test = test


# In[12]:


prep.get_label_distribution(Y_train, Y_valid, Y_test)


# In[13]:


Y_train.columns = Y_train.columns.str.replace(target_keyword, '')
Y_valid.columns = Y_valid.columns.str.replace(target_keyword, '')
Y_test.columns = Y_test.columns.str.replace(target_keyword, '')


# In[14]:


# Distrubution of the sequence lengths in the training set
dist_seq_lengths = X_train.groupby('ikn').apply(len)
fig = plt.figure(figsize=(15, 5))
plt.hist(dist_seq_lengths, bins=100)
plt.grid()
plt.show()


# In[15]:


# A closer look at the samples of sequences with length 1 to 21
fig = plt.figure(figsize=(15, 5))
plt.hist(dist_seq_lengths[dist_seq_lengths < 21], bins=20)
plt.grid()
plt.xticks(range(1, 21))
plt.show()


# # Training

# In[16]:


pd.set_option('display.max_columns', None)


# In[17]:


# Initialize Training class 
dataset = (X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
train = TrainGRU(dataset, output_path)


# In[ ]:


# Conduct Bayesian Optimization 
# best_param = train.bayesopt('gru', save_dir=f'{output_path}/hyperparam_tuning')

# Train final model using the best parameters
# filename = f'{output_path}/hyperparam_tuning/GRU_classifier_best_param.pkl'
# with open(filename, 'rb') as file:
#     best_param = pickle.load(file)

# save_path = f'{output_path}/gru_best_classifier'
# model, train_losses, valid_losses, train_scores, valid_scores = train.train_classification(save=True, save_path=save_path, 
#                                                                                            **best_param)
# np.save(f"{output_path}/loss_and_acc/best_train_losses.npy", train_losses)
# np.save(f"{output_path}/loss_and_acc/best_valid_losses.npy", valid_losses)
# np.save(f"{output_path}/loss_and_acc/best_train_scores.npy", train_scores)
# np.save(f"{output_path}/loss_and_acc/best_valid_scores.npy", valid_scores)
    
# Submit a slurm job since bayes opt takes a long time
get_ipython().system('sbatch slurm_main')


# In[18]:


# Get the best parameters
filename = f'{output_path}/hyperparam_tuning/GRU_classifier_best_param.pkl'
with open(filename, 'rb') as file:
    best_param = pickle.load(file)
best_param


# In[21]:


# Optional: Retrain model using best parameters
save_path = f'{output_path}/gru_classifier'
model, train_losses, valid_losses, train_scores, valid_scores = train.train_classification(save=True, save_path=save_path, 
                                                                                           early_stopping=30, **best_param)
np.save(f"{output_path}/loss_and_acc/train_losses.npy", train_losses)
np.save(f"{output_path}/loss_and_acc/valid_losses.npy", valid_losses)
np.save(f"{output_path}/loss_and_acc/train_scores.npy", train_scores)
np.save(f"{output_path}/loss_and_acc/valid_scores.npy", valid_scores)


# In[22]:


train_losses = np.load(f"{output_path}/loss_and_acc/train_losses.npy")
valid_losses = np.load(f"{output_path}/loss_and_acc/valid_losses.npy")
train_scores = np.load(f"{output_path}/loss_and_acc/train_scores.npy")
valid_scores = np.load(f"{output_path}/loss_and_acc/valid_scores.npy")
save_path = f'{output_path}/loss_and_acc/loss_and_acc.jpg'
train.plot_training_curve(train_losses, valid_losses, train_scores, valid_scores, save=True, save_path=save_path)


# In[23]:


model = GRU(n_features=train.n_features, 
            n_targets=train.n_targets, 
            hidden_size=best_param['hidden_size'], 
            hidden_layers=best_param['hidden_layers'],
            batch_size=best_param['batch_size'], 
            dropout=best_param['dropout'], 
            pad_value=-999)

if torch.cuda.is_available():
    model.cuda()
    
save_path = f'{output_path}/gru_classifier'
model.load_state_dict(torch.load(save_path))


# In[24]:


# Get Model Performance Scores
save_path = f'{output_path}/classification_result.csv'
score_df = train.get_model_scores(model, save=True, save_path=save_path, splits=['Valid', 'Test'], verbose=False)
score_df.loc[[i for i in score_df.index if 'AUROC' in i[1] or 'AUPRC' in i[1]]]


# In[25]:


pred_thresholds = np.arange(0.05, 0.51, 0.05)
train.threshold_op_points(model, pred_thresholds)


# # Scratch Notes

# ### Experiment with longitudinal lab data

# In[8]:


from scripts.preprocess import (parallelize, shared_dict)
from scripts.config import (all_observations)
obs_names = set(all_observations.values())


# In[31]:


model_data = pd.read_csv(f'{main_dir}/data/model_data.csv', dtype={'curr_morph_cd': str, 'lhin_cd': str})
model_data = model_data.set_index('index')

def read_data(obs_name):
    obs_data = pd.read_csv(f'{main_dir}/data/experiment/{obs_name}.csv')
    obs_data = obs_data.set_index('index')
    # turn string of numbers columns into integer column 
    obs_data.columns = obs_data.columns.values.astype(int)
    # convert each day into a row
    cols = obs_data.columns
    obs_data = obs_data.melt(id_vars=cols[~cols.isin(range(-28,1))], var_name='Day', 
                             value_name=f'{obs_name}_count', ignore_index=False)
    return obs_data

def get_data():
    for i, obs_name in enumerate(tqdm.tqdm(obs_names)):
        obs_data = read_data(obs_name)
        if i == 0: 
            data = obs_data
        else:
            col = f'{obs_name}_count'
            data[col] = obs_data[col]

    # remove rows with zero observations
    mask = data.iloc[:, 1:].isnull().all(axis=1)
    data = data[~mask]

    # include all the features
    data = data.join(model_data, how='left')
    
    return data

def organize_data(df, verbose=False):
    # adjust days since
    cols = df.columns
    cols = cols[cols.str.contains('days_since')]
    for col in cols: df[col] += df['Day']
        
    # fill null values with 0 or max value
    df['line_of_therapy'] = df['line_of_therapy'].fillna(0) # the nth different chemotherapy taken
    df['num_prior_EDs'] = df['num_prior_EDs'].fillna(0)
    df['num_prior_Hs'] = df['num_prior_Hs'].fillna(0)
    for col in ['days_since_prev_chemo', 'days_since_true_prev_chemo']:
        df[col] = df[col].fillna(df[col].max())

    # reduce sparse matrix by replacing rare col entries with less than 6 patients with 'Other'
    cols = ['regimen', 'curr_morph_cd', 'curr_topog_cd']
    df = replace_rare_col_entries(df, cols, verbose=verbose)

    # get visit month 
    df['visit_date'] = pd.to_datetime(df['visit_date'])
    df['visit_month'] = df['visit_date'].dt.month

    # create features for missing entries
    cols_with_nan = df.columns[df.isnull().any()]
    df[cols_with_nan + '_is_missing'] = df[cols_with_nan].isnull()

    # create column for acute care (ACU = ED + H) and treatment related acute care (TR_ACU = TR_ED + TR_H)
    df['ACU'+target_keyword] = df['ED'+target_keyword] | df['H'+target_keyword] 
    df['TR_ACU'+target_keyword] = df['TR_ED'+target_keyword] | df['TR_H'+target_keyword] 
    
    # sort by ikn and visit date
    df = df.sort_values(by=['ikn', 'days_since_starting_chemo'])

    cols = df.columns
    drop_columns = cols[cols.str.contains('within') | cols.str.contains('baseline')]
    drop_columns = drop_columns[~drop_columns.str.contains(target_keyword)]
    drop_columns = ['visit_date', 'Day'] + drop_columns.tolist()
    df = df.drop(columns=drop_columns)
    return df


# In[32]:


data = organize_data(get_data(), verbose=False)
data.to_csv(f'{main_dir}/data/experiment/model_data.csv', index_label='index')


# In[6]:


model_data = pd.read_csv(f'{main_dir}/data/experiment/model_data.csv', dtype={'curr_morph_cd': str, 'lhin_cd': str})
model_data = model_data.set_index('index')
model_data


# In[9]:


class PrepDataGRU(PrepDataEDHD):  
    def __init__(self):
        super(PrepDataEDHD, self).__init__()
        self.observation_cols = [f'{obs_name}_count' for obs_name in obs_names]
    
    def clean_feature_target_cols(self, feature_cols, target_cols):
        return feature_cols, target_cols

prep = PrepDataGRU()
model_data, clip_thresholds = prep.clip_outliers(model_data, lower_percentile=0.001, upper_percentile=0.999)
model_data = prep.dummify_data(model_data)
print(f'Size of model_data after one-hot encoding: {model_data.shape}')
train, valid, test = prep.split_data(model_data, target_keyword=target_keyword, convert_to_float=False)


# In[12]:


X_train, Y_train = train
X_valid, Y_valid = valid
X_test, Y_test = test
X_train,  Y_train = X_train.reset_index(drop=True),  Y_train.reset_index(drop=True)
X_valid,  Y_valid = X_valid.reset_index(drop=True),  Y_valid.reset_index(drop=True)
X_test,  Y_test = X_test.reset_index(drop=True),  Y_test.reset_index(drop=True)


# In[13]:


"""
LOL oops, this is more like within 30 + 28 = 58 days
"""
prep.get_label_distribution(Y_train, Y_valid, Y_test)


# In[14]:


Y_train.columns = Y_train.columns.str.replace(target_keyword, '')
Y_valid.columns = Y_valid.columns.str.replace(target_keyword, '')
Y_test.columns = Y_test.columns.str.replace(target_keyword, '')


# In[15]:


pd.set_option('display.max_columns', None)

# Initialize Training class 
dataset = (X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
train = TrainGRU(dataset, output_path)

# Get the best parameters
filename = f'{output_path}/hyperparam_tuning/gru_classifier_best_param.pkl'
with open(filename, 'rb') as file:
    best_param = pickle.load(file)
    
# Optional: Retrain model using best parameters
save_path = f'{main_dir}/data/experiment/gru_classifier'
model, train_losses, valid_losses, train_scores, valid_scores = train.train_classification(save=True, save_path=save_path, 
                                                                                           early_stopping=30, **best_param)
save_path = f'{main_dir}/data/experiment/loss_and_acc.jpg'
train.plot_training_curve(train_losses, valid_losses, train_scores, valid_scores, save=True, save_path=save_path)


# In[16]:


score_df = train.get_model_scores(model, save=False, splits=['Valid', 'Test'], verbose=False)
score_df.loc[[i for i in score_df.index if 'AUROC' in i[1] or 'AUPRC' in i[1]]]


# In[ ]:




