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
pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib.pyplot as plt

from scripts.utility import (initialize_folders, load_predictions,
                             get_nunique_entries, get_nmissing, 
                             data_splits_summary, feature_summary, subgroup_performance_summary)
from scripts.visualize import (importance_plot, subgroup_performance_plot)
from scripts.config import (root_path, acu_folder)
from scripts.prep_data import (PrepDataEDHD)
from scripts.train import (TrainML, TrainRNN, TrainENS)
from scripts.evaluate import (Evaluate)


# In[3]:


# config
processes = 64
days = 30 # predict event within this number of days since chemo visit (the look ahead window)
target_keyword = f'_within_{days}days'
main_dir = f'{root_path}/{acu_folder}'
output_path = f'{main_dir}/models/within_{days}_days'
initialize_folders(output_path, extra_folders=['figures/important_groups'])


# # Prepare Data for Model Training

# In[4]:


# Prepare Data for Model Input
prep = PrepDataEDHD(adverse_event='acu')


# In[5]:


model_data = prep.get_data(target_keyword, verbose=True)
model_data


# In[6]:


sorted(model_data.columns.tolist())


# In[7]:


get_nunique_entries(model_data)


# In[8]:


get_nmissing(model_data, verbose=True)


# In[9]:


model_data = prep.get_data(target_keyword, missing_thresh=80, verbose=True)
print(f"Size of model_data: {model_data.shape}")
print(f"Number of unique patients: {model_data['ikn'].nunique()}")
N = model_data.loc[model_data['ACU_within_30days'], 'ikn'].nunique()
print(f"Number of unique patients that had ACU within 30 days after a treatment session: {N}")


# In[10]:


model_data, clip_thresholds = prep.clip_outliers(model_data, lower_percentile=0.001, upper_percentile=0.999)
clip_thresholds.columns = clip_thresholds.columns.str.replace('baseline_', '').str.replace('_count', '')
clip_thresholds


# In[11]:


train, valid, test = prep.split_data(prep.dummify_data(model_data), target_keyword=target_keyword, convert_to_float=False)
(X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test) = train, valid, test


# In[12]:


prep.get_label_distribution(Y_train, Y_valid, Y_test)


# In[13]:


Y_train.columns = Y_train.columns.str.replace(target_keyword, '')
Y_valid.columns = Y_valid.columns.str.replace(target_keyword, '')
Y_test.columns = Y_test.columns.str.replace(target_keyword, '')


# # Train ML Models

# In[14]:


pd.set_option('display.max_columns', None)


# In[15]:


# Initialize Training class
dataset = (X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
train_ml = TrainML(dataset, output_path, n_jobs=processes)


# In[18]:


skip_alg = []
train_ml.tune_and_train(run_bayesopt=False, run_training=False, save_preds=True, skip_alg=skip_alg)


# # Train RNN Model

# In[17]:


X_train['ikn'] = model_data['ikn']
X_valid['ikn'] = model_data['ikn']
X_test['ikn'] = model_data['ikn']

# Initialize Training class 
dataset = (X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
train_rnn = TrainRNN(dataset, output_path)


# In[18]:


# Distrubution of the sequence lengths in the training set
dist_seq_lengths = X_train.groupby('ikn').apply(len)
fig = plt.figure(figsize=(15, 3))
plt.hist(dist_seq_lengths, bins=100)
plt.grid()
plt.show()


# In[19]:


# A closer look at the samples of sequences with length 1 to 21
fig = plt.figure(figsize=(15, 3))
plt.hist(dist_seq_lengths[dist_seq_lengths < 21], bins=20)
plt.grid()
plt.xticks(range(1, 21))
plt.show()


# In[50]:


train_rnn.tune_and_train(run_bayesopt=False, run_training=False, run_calibration=False, save_preds=True)


# # Train ENS Model 
# Find Optimal Ensemble Weights

# In[16]:


labels = {'Train': Y_train, 'Valid': Y_valid, 'Test': Y_test}
# combine rnn and ml predictions
preds = load_predictions(save_dir=f'{output_path}/predictions')
preds_rnn = load_predictions(save_dir=f'{output_path}/predictions', filename='rnn_predictions')
for split, pred in preds_rnn.items(): preds[split]['RNN'] = pred
del preds_rnn
# Initialize Training Class
train_ens = TrainENS(output_path, preds, labels)


# In[17]:


train_ens.tune_and_train(run_bayesopt=False, run_calibration=False)


# # Evaluate Models

# In[18]:


eval_models = Evaluate(output_path=output_path, preds=train_ens.preds, labels=labels, orig_data=model_data)


# In[28]:


splits = ['Test']
kwargs = {'get_baseline': True, 'display_ci': True, 'load_ci': True, 'save_ci': False, 'verbose': False}
eval_models.get_evaluation_scores(splits=splits, **kwargs)


# In[20]:


eval_models.plot_curves(curve_type='pr', legend_location='upper right', figsize=(12,12))
eval_models.plot_curves(curve_type='roc', legend_location='lower right', figsize=(12,12))
eval_models.plot_calibs(figsize=(12,12), n_bins=20, calib_strategy='quantile', include_pred_hist=True)
eval_models.plot_cdf_pred(figsize=(12,12)) # cumulative distribution function of model prediceval_models


# # Post-Training Analysis

# ## Study Characteristics

# In[30]:


data_splits_summary(eval_models, save_dir=f'{main_dir}/models')


# ## Feature Characteristics

# In[31]:


feature_summary(eval_models, prep, target_keyword, save_dir=f'{main_dir}/models').head(60)


# ## Threshold Op Points

# In[32]:


pred_thresholds = np.arange(0.05, 0.51, 0.05)
thresh_df = eval_models.threshold_op_points(algorithm='ENS', pred_thresholds=pred_thresholds, 
                                            include_outcome_recall=True, event_dates=prep.event_dates)
thresh_df


# ## Most Important Features/Feature Groups

# In[ ]:


get_ipython().system('python scripts/perm_importance.py --adverse-event ACU')


# In[31]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot('ENS', eval_models.target_types, output_path, figsize=(6,50), top=10, importance_by='feature', pad_x0=4.0)


# In[30]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot('ENS', eval_models.target_types, output_path, figsize=(6,50), top=10, importance_by='group', pad_x0=1.2)


# ## Performance on Subgroups

# In[19]:


df = subgroup_performance_summary('ENS', eval_models, pred_thresh=0.25, display_ci=False, load_ci=False, save_ci=False)
df


# ## ACU

# ### All the Plots

# In[139]:


eval_models.all_plots_for_single_target(algorithm='ENS', target_type='ACU')


# ### Subgroup Performance Plot

# In[20]:


subgroups = ['Entire Test Cohort', 'Age', 'Sex', 'Immigration', 'Regimen', 'Days Since Starting Regimen']
subgroup_performance_plot(df, subgroups=subgroups, figsize=(12,24), save=True, save_dir=f'{output_path}/figures')
# PPV = 0.3 means roughly for every 3 alarms, 2 are false alarms and 1 is true alarm
# Sesnsitivity = 0.5 means roughly for every 2 true alarms, the model predicts 1 of them correctly
# Event Rate = 0.15 means true alarms occur 15% of the time


# # Scratch Notes

# ## 2 variable model

# In[230]:


import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, average_precision_score, roc_curve, precision_recall_curve)
from sklearn.calibration import calibration_curve
from scripts.utility import twolevel, compute_bootstrap_scores


# In[232]:


def predict(df):
    x = 10.392 - 0.472*0.1*df['baseline_albumin_count'] - 0.075*df['baseline_sodium_count']
    return 1 / (1 + np.exp(-x))

def evaluate(labels, pred):
    result = pd.DataFrame()
    if isinstance(pred, pd.DataFrame): preds = pred
    for target_type, Y in labels.iteritems():
        if isinstance(pred, pd.DataFrame): pred = preds[target_type]
        auc_scores = compute_bootstrap_scores(Y, pred)
        auroc_scores, auprc_scores = np.array(auc_scores).T
        lower, upper = np.percentile(auroc_scores, [2.5, 97.5]).round(3)
        result.loc['AUROC Score', target_type] = f"{np.round(roc_auc_score(Y, pred), 3)} ({lower}-{upper})"
        lower, upper = np.percentile(auprc_scores, [2.5, 97.5]).round(3)
        result.loc['AUPRC Score', target_type] = f"{np.round(average_precision_score(Y, pred), 3)} ({lower}-{upper})"
    return result

def thresh_op(labels, pred, target_type='H'):
    result = pd.DataFrame()
    label = labels[target_type]
    if isinstance(pred, pd.DataFrame): pred = pred[target_type]
    for threshold in np.arange(0.05, 0.51, 0.05):
        threshold = np.round(threshold, 2)
        pred_bool = pred > threshold
        result.loc[threshold, 'Warning Rate'] = pred_bool.mean()
        result.loc[threshold, 'PPV'] = precision_score(label, pred_bool, zero_division=1)
        result.loc[threshold, 'Sensitivity'] = recall_score(label, pred_bool, zero_division=1)
    return result

def plot_curves(pred, labels, result, target_type='H'):
    label = labels[target_type]
    if isinstance(pred, pd.DataFrame): pred = pred[target_type]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,9))
    axes = axes.flatten()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    fpr, tpr, thresholds = roc_curve(label, pred)
    precision, recall, thresholds = precision_recall_curve(label, pred)
    prob_true, prob_pred = calibration_curve(label, pred, n_bins=20, strategy='quantile')
    axis_max_limit = max(prob_true.max(), prob_pred.max())
    max_calib_error = np.max(abs(prob_true - prob_pred)).round(3)
    mean_calib_error = np.mean(abs(prob_true - prob_pred)).round(3)
    N = len(label)
    x, y = np.sort(pred), np.arange(N) / float(N)
    axes[0].plot(recall, precision, label=f"AUPRC={result.loc['AUPRC Score', target_type]}")
    axes[1].plot(fpr, tpr, label=f"AUROC={result.loc['AUROC Score', target_type]}")
    axes[2].plot(prob_true, prob_pred)
    axes[2].text(axis_max_limit/2, 0.07, f'Mean Calibration Error {mean_calib_error}')
    axes[2].text(axis_max_limit/2, 0.1, f'Max Calibration Error {max_calib_error}')
    axes[2].plot([0,axis_max_limit],[0,axis_max_limit],'k:', label='Perfect Calibration')
    axes[3].plot(x, y)
    labels = [('Sensitivity', 'Positive Predictive Value', 'pr_curves', True),
              ('1 - Specificity', 'Sensitivity', 'roc_curves', True),
              ('Predicted Probability', 'Empirical Probability', 'calibration_curves', False),
              (f'Predicted Probability of {target_type}', 'Cumulative Proportion of Predictions', 'cdf_prediction_curves', False)]
    for idx, (xlabel, ylabel, filename, remove_legend_line) in enumerate(labels):
        axes[idx].set_xlabel(xlabel)
        axes[idx].set_ylabel(ylabel)
        leg = axes[idx].legend(loc='lower right', frameon=False)
        if remove_legend_line: leg.legendHandles[0].set_linewidth(0)


# In[233]:


df = prep.get_data(target_keyword)
print(f'Size of data = {len(df)}, Number of patients = {df["ikn"].nunique()}')
df = df[~df['baseline_sodium_count'].isnull() & ~df['baseline_albumin_count'].isnull()]
print(f'Size of data with both sodium and albumin count = {len(df)}, Number of patients = {df["ikn"].nunique()}')
df = df[df['days_since_starting_chemo'] == 0] # very first treatment
print(f'Size of data with only first day chemos = {len(df)}, Number of patients = {df["ikn"].nunique()}')
df = df.loc[df.index[df.index.isin(X_test.index)]]
print(f'Size of test data = {len(df)}, Number of patients = {df["ikn"].nunique()}')


# In[234]:


pred = predict(df)
labels = Y_test.loc[df.index]


# In[235]:


# label distribtuion
labels.apply(pd.value_counts)


# In[236]:


result = evaluate(labels, pred)
result


# In[237]:


plot_curves(pred, labels, result)


# In[238]:


thresh_op(labels, pred)


# In[239]:


compare_pred = eval_models.preds['Test']['ENS'].loc[df.index]
result = evaluate(labels, compare_pred)
result


# In[240]:


plot_curves(compare_pred, labels, result)


# In[241]:


thresh_op(labels, compare_pred)


# ### hyperparameters

# In[16]:


from scripts.utility import get_hyperparameters
get_hyperparameters(output_path, days=days)


# ### how to read multiindex csvs

# In[29]:


pd.read_csv(f'{output_path}/tables/subgroup_performance_summary_ENS.csv', header=[0,1], index_col=[0,1])


# ### over under sample

# In[ ]:


def over_under_sample(X, Y, undersample_factor=5, oversample_min_samples=5000, seed=42):
    # undersample
    n = Y.shape[0]
    nsamples = int(n / undersample_factor)
    mask = Y.sum(axis=1) == 0 # undersample the examples with none of the 9 positive targets ~700,000 rows
    Y = pd.concat([Y[~mask], Y[mask].sample(nsamples, replace=False, random_state=seed)])

    # oversample
    label_counts = prep.get_label_distribution(Y)
    label_counts = label_counts[('Train', 'True')] # Train is just the default name, ignore it, its not a bug
    for col, n_pos_samples in label_counts.iteritems():
        if n_pos_samples < oversample_min_samples:
            Y = pd.concat([Y, Y[Y[col]].sample(oversample_min_samples - n_pos_samples, replace=True, random_state=seed)])
    
    X = X.loc[Y.index]
    return X, Y
X_train, Y_train = over_under_sample(X_train, Y_train, undersample_factor=5, oversample_min_samples=5000)
X_valid, Y_valid = over_under_sample(X_valid, Y_valid, undersample_factor=5, oversample_min_samples=1000)
prep.get_label_distribution(Y_train, Y_valid, Y_test)
