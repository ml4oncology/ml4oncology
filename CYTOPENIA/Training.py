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


import os
import tqdm
import pickle
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import (recall_score)

from scripts.utility import (initialize_folders, load_predictions,
                             get_nunique_entries, get_nmissing, 
                             data_characteristic_summary, feature_summary, subgroup_performance_summary)
from scripts.visualize import (tree_plot, importance_plot, subgroup_performance_plot)
from scripts.config import (root_path, cyto_folder, split_date, blood_types)
from scripts.prep_data import (PrepDataCYTO)
from scripts.train import (TrainML, TrainRNN, TrainENS)
from scripts.evaluate import (Evaluate)


# In[3]:


# config
processes = 64
output_path = f'{root_path}/{cyto_folder}/models'
initialize_folders(output_path)


# # Prepare Data for Model Training

# In[4]:


# Preparing Data for Model Input
prep = PrepDataCYTO()


# In[5]:


model_data = prep.get_data(include_first_date=True, verbose=True)
model_data


# In[6]:


sorted(model_data.columns.tolist())


# In[7]:


model_data['first_visit_date'].dt.year.value_counts()


# In[8]:


get_nunique_entries(model_data)


# In[9]:


get_nmissing(model_data, verbose=True)


# In[10]:


model_data = prep.get_data(include_first_date=True, missing_thresh=75, verbose=True)
print(f"Size of model_data: {model_data.shape}")
print(f"Number of unique patients: {model_data['ikn'].nunique()}")
print(f'Non-missing entries: {model_data.notnull().sum().sum()}')
for blood_type, blood_info in blood_types.items():
    N = model_data.loc[model_data[f'target_{blood_type}_count'] < blood_info['cytopenia_threshold'], 'ikn'].nunique()
    print(f"Number of unique patients that had {blood_info['cytopenia_name']} before treatment session: {N}")


# In[11]:


model_data, clip_thresholds = prep.clip_outliers(model_data, lower_percentile=0.001, upper_percentile=0.999)
clip_thresholds


# In[12]:


# NOTE: any changes to X_train, X_valid, etc will also be seen in dataset
dataset = X_train, X_valid, X_test, Y_train, Y_valid, Y_test = prep.split_data(prep.dummify_data(model_data.copy()), split_date=split_date)


# In[13]:


prep.get_label_distribution(Y_train, Y_valid, Y_test)


# In[14]:


# number of blood tranfusion occurences between visit date and next visit date
chemo_df = prep.load_data()
cohorts = {'Development Cohort': pd.concat([Y_train, Y_valid]), 'Test Cohort': Y_test}
result = pd.DataFrame()
for name, cohort in cohorts.items():
    df = chemo_df.loc[cohort.index]
    for blood_type in ['hemoglobin', 'platelet']:
        occurence_masks = [df[f'{event}_{blood_type}_transfusion_date'].between(df['visit_date'], df['next_visit_date']) 
                           for event in ['ED', 'H']]
        result.loc[name, f'{blood_type}_transfusion'] = pd.concat(occurence_masks, axis=1).any(axis=1).sum()
result.astype(int)


# # Train ML Models

# In[16]:


pd.set_option('display.max_columns', None)


# In[16]:


# Initialize Training class
train_ml = TrainML(dataset, output_path, n_jobs=processes)


# In[19]:


skip_alg = []
train_ml.tune_and_train(run_bayesopt=False, run_training=True, save_preds=True, skip_alg=skip_alg)


# # Train RNN Model

# In[20]:


# Include ikn to the input data
X_train['ikn'] = model_data['ikn']
X_valid['ikn'] = model_data['ikn']
X_test['ikn'] = model_data['ikn']

# Initialize Training class 
train_rnn = TrainRNN(dataset, output_path)


# In[21]:


# Distrubution of the sequence lengths in the training set
dist_seq_lengths = X_train.groupby('ikn').apply(len)
fig = plt.figure(figsize=(15, 3))
plt.hist(dist_seq_lengths, bins=100)
plt.grid()
plt.show()


# In[22]:


# A closer look at the samples of sequences with length 1 to 21
fig = plt.figure(figsize=(15, 3))
plt.hist(dist_seq_lengths[dist_seq_lengths < 21], bins=20)
plt.grid()
plt.xticks(range(1, 21))
plt.show()


# In[23]:


train_rnn.tune_and_train(run_bayesopt=False, run_training=True, run_calibration=True, save_preds=True)


# # Train ENS Model 

# In[15]:


# combine rnn and ml predictions
preds = load_predictions(save_dir=f'{output_path}/predictions')
preds_rnn = load_predictions(save_dir=f'{output_path}/predictions', filename='rnn_predictions')
for split, pred in preds_rnn.items(): preds[split]['RNN'] = pred
del preds_rnn
# Initialize Training Class
train_ens = TrainENS(dataset, output_path, preds)


# In[16]:


train_ens.tune_and_train(run_bayesopt=False, run_calibration=False, calibrate_pred=True)


# # Evaluate Models

# In[17]:


eval_models = Evaluate(output_path=output_path, preds=train_ens.preds, labels=train_ens.labels, orig_data=model_data)


# In[36]:


baseline_cols = ['regimen'] + [f'baseline_{bt}_count' for bt in blood_types]
kwargs = {'get_baseline': True, 'baseline_cols': baseline_cols, 'display_ci': True, 'load_ci': True, 'save_ci': False, 'verbose': False}
eval_models.get_evaluation_scores(**kwargs)


# In[37]:


eval_models.plot_curves(curve_type='pr', legend_location='lower left', figsize=(12,18))
eval_models.plot_curves(curve_type='roc', legend_location='lower right', figsize=(12,18))
eval_models.plot_curves(curve_type='pred_cdf', figsize=(12,18)) # cumulative distribution function of model prediction
eval_models.plot_calibs(legend_location='upper left', figsize=(12,18)) 
# eval_models.plot_calibs(include_pred_hist=True, legend_location='upper left', figsize=(12,28), padding={'pad_y1': 0.3, 'pad_y0': 3.0})


# # Post-Training Analysis

# ## Study Characteristics

# In[33]:


data_characteristic_summary(eval_models, save_dir=f'{output_path}/tables', partition='cohort', include_gcsf=True)


# ## Feature Characteristics

# In[34]:


feature_summary(eval_models, prep, target_keyword='target_', save_dir=f'{output_path}/tables').head(60)


# ## Threshold Operating Points

# In[20]:


pred_thresholds = np.arange(0.05, 1.01, 0.05)
thresh_df = eval_models.operating_points(algorithm='ENS', points=pred_thresholds, metric='threshold')
thresh_df


# ## Precision Operating Points

# In[24]:


desired_precisions = [0.2, 0.25, 0.33, 0.4, 0.5, 0.6, 0.75]
eval_models.operating_points(algorithm='ENS', points=desired_precisions, metric='precision')


# ## Most Important Features

# In[ ]:


get_ipython().system('python scripts/perm_importance.py --adverse-event CYTO')


# In[18]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot('XGB', eval_models.target_types, output_path, figsize=(6,15), top=10, importance_by='feature', 
                padding={'pad_x0': 2.6}, colors=['#1f77b4', '#ff7f0e', '#2ca02c'])


# ## Performance on Subgroups

# In[26]:


df = subgroup_performance_summary('ENS', eval_models, pred_thresh=0.25, display_ci=False, load_ci=False, save_ci=False)
df


# ## All the Plots

# In[27]:


for blood_type, blood_info in blood_types.items():
    target_type = blood_info['cytopenia_name']
    print(f'Displaying all the plots for {target_type}')
    eval_models.all_plots_for_single_target(algorithm='ENS', target_type=target_type, save=False)


# ## Subgroup Performance Plot

# In[28]:


# PPV = 0.3 means roughly for every 3 alarms, 2 are false alarms and 1 is true alarm
# Sesnsitivity = 0.5 means roughly for every 2 true alarms, the model predicts 1 of them correctly
# Event Rate = 0.15 means true alarms occur 15% of the time
subgroups = ['Entire Test Cohort', 'Age', 'Sex', 'Immigration', 'Regimen', 'Days Since Starting Regimen']
padding = {'pad_y0': 1.2, 'pad_x1': 2.6, 'pad_y1': 0.2}
for blood_type, blood_info in blood_types.items():
    target_type = blood_info['cytopenia_name']
    print(f'Displaying subgroup performance for {target_type}')
    subgroup_performance_plot(df, target_type=target_type, subgroups=subgroups, padding=padding,
                              figsize=(12,24), save=True, save_dir=f'{output_path}/figures')


# ## Randomized Individual Patient Performance

# In[28]:


sex_mapping = {'M': 'male', 'F': 'female'}
# os.makedirs(f'{output_path}/figures/patients')


# In[49]:


def get_patient_info(orig_data):
    age = int(orig_data['age'].mean())
    sex = sex_mapping[orig_data['sex'].values[0]]
    regimen = orig_data['regimen'].values[0]
    patient_info = f"{age} years old {sex} patient under regimen {regimen}"
    return patient_info

def plot_patient_prediction(eval_models, X_test, algorithm='XGB', num_ikn=3, seed=0, save=False):
    """
    Args:
        num_ikn (int): the number of random patients to analyze
    """
    np.random.seed(seed)

    # get the original data corresponding with the testing set
    df = eval_models.orig_data.loc[X_test.index]

    # only consider patients who had more than 3 chemo cycles
    ikn_count = df['ikn'].value_counts()
    ikns = ikn_count[ikn_count > 3].index

    for _ in range(num_ikn):
        ikn = np.random.choice(ikns) # select a random patient from the consideration pool
        ikn_indices = df[df['ikn'] == ikn].index # get the indices corresponding with the selected patient
        pred_prob = eval_models.preds['Test'][algorithm].loc[ikn_indices]
        orig_data = df.loc[ikn_indices]
        patient_info = get_patient_info(orig_data)
        print(patient_info)
        if patient_info == '71 years old male patient under regimen gemcnpac(w)':
            raise

        fig = plt.figure(figsize=(15, 20))
        
        # days since admission at target date
        days_since_admission = orig_data['days_since_last_chemo'].shift(-1).cumsum().values
        # last target date will be "unknown", use default cycle length
        days_since_admission[-1] = days_since_admission[-2] + int(orig_data['cycle_length'].iloc[-1]) 
        
        for i, (blood_type, blood_info) in enumerate(blood_types.items()):
            true_count = orig_data[f'target_{blood_type}_count'].values
            thresh, name, unit = blood_info['cytopenia_threshold'], blood_info['cytopenia_name'], blood_info['unit']

            ax1 = fig.add_subplot(6, 3, i+1) # 3 blood types * 2 subplots each
            ax1.plot(days_since_admission, true_count, label=f'{blood_type}'.capitalize())
            ax1.axhline(y=thresh, color='r', alpha=0.5, label = f"{name} threshold ({thresh})".title())
            ax1.tick_params(labelbottom=False)
            ax1.set_ylabel(f"Blood count ({unit})")
            ax1.set_title(f"Patient {blood_type} measurements")
            ax1.legend()
            ax1.grid(axis='x')

            ax2 = fig.add_subplot(6, 3, i+1+3, sharex=ax1)
            ax2.plot(days_since_admission, pred_prob[name], label='XGB Model Prediction')
            ax2.axhline(y=0.5, color='r', alpha=0.5, label="Positive Prediction Threshold")
            ax2.set_xticks(days_since_admission)
            ax2.set_yticks(np.arange(0, 1.01, 0.2))
            ax2.set_xlabel('Days since admission')
            ax2.set_ylabel(f"Risk of {name}")
            ax2.set_title(f"Model Prediction for {name}")
            ax2.legend()
            ax2.grid(axis='x')
        if save:
            plt.savefig(f'{output_path}/figures/patients/{ikn}_performance.jpg', bbox_inches='tight') #dpi=300
        plt.show()


# In[50]:


plot_patient_prediction(eval_models, X_test, algorithm='XGB', num_ikn=8, seed=1, save=False)


# # SCRATCH NOTES

# ## XGB as txt file

# In[45]:


XGB_model = load_ml_model(output_path, 'XGB')
for idx, blood_type in enumerate(blood_types):
    estimator = XGB_model.estimators_[0].calibrated_classifiers_[0].base_estimator
    estimator.get_booster().dump_model(f'{output_path}/XGB_{blood_type}.txt')
    estimator.save_model(f'{output_path}/XGB_{blood_type}.model')


# ## Graph Visualization

# In[35]:


tree_plot(train_ml, target_type='Neutropenia', algorithm='RF')


# ## More Senstivity/Error Analysis

# In[29]:


df = subgroup_performance_summary('ENS', eval_models, subgroups=['all', 'cycle_length'], display_ci=False, save_ci=False)
subgroup_performance_plot(df, save=False, target_type='Neutropenia', figsize=(4,16))


# In[30]:


# analyze subgroups with the worst performance IN THE VALIDATION SET
from scripts.utility import get_worst_performing_subgroup
get_worst_performing_subgroup(eval_models, category='regimen', split='Valid')


# In[31]:


get_worst_performing_subgroup(eval_models, category='curr_topog_cd', split='Valid')


# ## Hyperparameters

# In[32]:


from scripts.utility import get_hyperparameters
get_hyperparameters(output_path)


# ## Data Summary of Filtered Data

# In[51]:


from scripts.utility import twolevel, DataPartitionSummary
summary_df = pd.DataFrame(index=twolevel, columns=twolevel)
dps = DataPartitionSummary(model_data, model_data, 'Included Data') # Both target and baseline blood count for hemoglobin, neutrophil, platelet required
dps.get_summary(summary_df, include_target=False)

mask = ~chemo_df.index.isin(model_data.index)
dps = DataPartitionSummary(chemo_df[mask], chemo_df[mask], 'Excluded Data')
dps.get_summary(summary_df, include_target=False)
summary_df


# ## Missingness By Splits

# In[53]:


from scripts.utility import get_nmissing_by_splits
missing = get_nmissing_by_splits(model_data, eval_models.labels)
missing.sort_values(by=(f'Test (N={len(Y_test)})', 'Missing (N)'), ascending=False)
