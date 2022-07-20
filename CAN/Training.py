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

from scripts.config import (root_path, can_folder, split_date, SCr_rise_threshold)
from scripts.utility import (initialize_folders, load_predictions, 
                             get_nunique_entries, get_nmissing, most_common_by_category,
                             data_characteristic_summary, feature_summary, subgroup_performance_summary,
                             get_hyperparameters)
from scripts.visualize import (importance_plot, subgroup_performance_plot)
from scripts.prep_data import (PrepData, PrepDataCAN)
from scripts.train import (TrainML, TrainRNN, TrainENS)
from scripts.evaluate import (Evaluate)


# In[3]:


processes = 64
target_keyword = 'SCr|dialysis|next'
main_dir = f'{root_path}/{can_folder}'


# # Acute Kidney Injury

# In[4]:


adverse_event = 'aki'
output_path = f'{main_dir}/models/{adverse_event.upper()}'
initialize_folders(output_path)


# ## Prepare Data for Model Training

# In[5]:


# Preparing Data for Model Input
prep = PrepDataCAN(adverse_event=adverse_event)


# In[6]:


model_data = prep.get_data(include_first_date=True, verbose=True)
model_data


# In[7]:


most_common_by_category(model_data, category='regimen', with_respect_to='patients', top=10)


# In[8]:


sorted(model_data.columns.tolist())


# In[9]:


get_nunique_entries(model_data)


# In[10]:


get_nmissing(model_data)


# In[11]:


model_data = prep.get_data(include_first_date=True, missing_thresh=80, verbose=True)
print(f"Size of model_data: {model_data.shape}")
print(f"Number of unique patients: {model_data['ikn'].nunique()}")
mask = (model_data['SCr_rise'] >= SCr_rise_threshold) | (model_data['SCr_fold_increase'] > 1.5)
N = model_data.loc[mask, 'ikn'].nunique()
print(f"Number of unique patients that had Acute Kidney Injury (AKI) " +      f"within 28 days or right before treatment session: {N}")


# In[12]:


model_data, clip_thresholds = prep.clip_outliers(model_data, lower_percentile=0.001, upper_percentile=0.999)
clip_thresholds


# In[13]:


kwargs = {'target_keyword': target_keyword, 'split_date': split_date}
# NOTE: any changes to X_train, X_valid, etc will also be seen in dataset
dataset = X_train, X_valid, X_test, Y_train, Y_valid, Y_test = prep.split_data(prep.dummify_data(model_data.copy()), **kwargs)


# In[14]:


prep.get_label_distribution(Y_train, Y_valid, Y_test)


# ## Train ML Models

# In[15]:


pd.set_option('display.max_columns', None)


# In[21]:


# Initialize Training class
train_ml = TrainML(dataset, output_path, n_jobs=processes)


# In[24]:


skip_alg = []
train_ml.tune_and_train(run_bayesopt=False, run_training=True, save_preds=True, skip_alg=skip_alg)


# ## Train RNN Model

# In[89]:


# Include ikn to the input data (recall that any changes to X_train, X_valid, etc will also be seen in dataset)
X_train['ikn'] = model_data['ikn']
X_valid['ikn'] = model_data['ikn']
X_test['ikn'] = model_data['ikn']

# Initialize Training class 
train_rnn = TrainRNN(dataset, output_path)


# In[90]:


train_rnn.tune_and_train(run_bayesopt=False, run_training=False, run_calibration=True, 
                         calibrate_pred=True, save_preds=True)


# ## Train ENS Model 

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


# ## Evaluate Models

# In[17]:


eval_models = Evaluate(output_path=output_path, preds=train_ens.preds, labels=train_ens.labels, orig_data=model_data)


# In[18]:


baseline_cols = ['regimen', 'baseline_eGFR']
kwargs = {'get_baseline': True, 'baseline_cols': baseline_cols, 'display_ci': True, 'load_ci': True, 'save_ci': False, 'verbose': False}
eval_models.get_evaluation_scores(**kwargs)


# In[21]:


eval_models.plot_curves(curve_type='pr', legend_location='lower left', figsize=(12,18), save=False)
eval_models.plot_curves(curve_type='roc', legend_location='lower right', figsize=(12,18), save=False)
eval_models.plot_curves(curve_type='pred_cdf', figsize=(12,18), save=False) # cumulative distribution function of model prediction
eval_models.plot_calibs(legend_location='upper left', figsize=(12,18), save=False) 
# eval_models.plot_calibs(include_pred_hist=True, legend_location='upper left', figsize=(12,28), padding={'pad_y1': 0.3, 'pad_y0': 3.0})


# ## Post-Training Analysis

# ### Study Population Characteristics

# In[25]:


data_characteristic_summary(eval_models, save_dir=f'{output_path}/tables', partition='cohort', 
                            include_combordity=True, include_ckd=True, include_dialysis=True)


# ### Feature Characteristic

# In[18]:


feature_summary(eval_models, prep, target_keyword=target_keyword, save_dir=f'{output_path}/tables').head(60)


# ### Threshold Op Points

# In[19]:


pred_thresholds = np.arange(0, 1.01, 0.05)
thresh_df = eval_models.operating_points(algorithm='ENS', points=pred_thresholds, metric='threshold')
thresh_df


# ### All the Plots

# In[24]:


eval_models.all_plots_for_single_target(algorithm='ENS', target_type='AKI', save=True)


# ### Most important features

# In[25]:


get_ipython().system('python scripts/perm_importance.py --adverse-event CAAKI')


# In[26]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot('ENS', eval_models.target_types, output_path, figsize=(6,5), top=10, importance_by='feature', padding={'pad_x0': 2.7})


# ### Performance on Subgroups

# In[27]:


df = subgroup_performance_summary('ENS', eval_models, pred_thresh=0.1, display_ci=False, load_ci=False, save_ci=False)
df # @ pred threshold = 0.1


# In[28]:


# PPV = 0.3 means roughly for every 3 alarms, 2 are false alarms and 1 is true alarm
# Sesnsitivity = 0.5 means roughly for every 2 true alarms, the model predicts 1 of them correctly
# Event Rate = 0.15 means true alarms occur 15% of the time
subgroups = ['Entire Test Cohort', 'Age', 'Sex', 'Immigration', 'Regimen', 'Days Since Starting Regimen']
padding = {'pad_y0': 1.2, 'pad_x1': 2.6, 'pad_y1': 0.2}
subgroup_performance_plot(df, target_type='AKI', subgroups=subgroups, padding=padding,
                          figsize=(12,24), save=True, save_dir=f'{output_path}/figures')


# ### Decision Curve Plot

# In[29]:


eval_models.plot_decision_curve_analysis('ENS')


# In[41]:


get_hyperparameters(output_path)


# # Chronic Kidney Disease

# In[4]:


adverse_event = 'ckd'
output_path = f'{main_dir}/models/{adverse_event.upper()}'
initialize_folders(output_path)


# ## Prepare Data for Model Training

# In[5]:


prep = PrepDataCAN(adverse_event=adverse_event)
model_data = prep.get_data(include_first_date=True, missing_thresh=80, verbose=True)
model_data['next_eGFR'].hist(bins=100)
print(f"Size of model_data: {model_data.shape}")
print(f"Number of unique patients: {model_data['ikn'].nunique()}")
mask = (model_data['next_eGFR'] < 60) | model_data['dialysis']
N = model_data.loc[mask, 'ikn'].nunique()
print(f"Number of unique patients that had Chronic Kidney Disease (CKD): {N}")
model_data, clip_thresholds = prep.clip_outliers(model_data, lower_percentile=0.001, upper_percentile=0.999)
kwargs = {'target_keyword': target_keyword, 'split_date': split_date}
# NOTE: any changes to X_train, X_valid, etc will also be seen in dataset
dataset = X_train, X_valid, X_test, Y_train, Y_valid, Y_test = prep.split_data(prep.dummify_data(model_data.copy()), **kwargs)


# In[6]:


prep.get_label_distribution(Y_train, Y_valid, Y_test)


# ## Train ML Models

# In[26]:


pd.set_option('display.max_columns', None)


# In[27]:


# Initialize Training class
train_ml = TrainML(dataset, output_path, n_jobs=processes)


# In[ ]:


skip_alg = []
train_ml.tune_and_train(run_bayesopt=False, run_training=True, save_preds=True, skip_alg=skip_alg)


# ## Train RNN Model

# In[ ]:


# Include ikn to the input data (recall that any changes to X_train, X_valid, etc will also be seen in dataset)
X_train['ikn'] = model_data['ikn']
X_valid['ikn'] = model_data['ikn']
X_test['ikn'] = model_data['ikn']

# Initialize Training class 
train_rnn = TrainRNN(dataset, output_path)


# In[50]:


train_rnn.tune_and_train(run_bayesopt=False, run_training=True, run_calibration=True, save_preds=True)


# ## Train ENS Model 

# In[7]:


# combine rnn and ml predictions
preds = load_predictions(save_dir=f'{output_path}/predictions')
preds_rnn = load_predictions(save_dir=f'{output_path}/predictions', filename='rnn_predictions')
for split, pred in preds_rnn.items(): preds[split]['RNN'] = pred
del preds_rnn
# Initialize Training Class
train_ens = TrainENS(dataset, output_path, preds)


# In[8]:


train_ens.tune_and_train(run_bayesopt=False, run_calibration=False, calibrate_pred=True)


# ## Evaluate Models

# In[9]:


eval_models = Evaluate(output_path=output_path, preds=train_ens.preds, labels=train_ens.labels, orig_data=model_data)


# In[36]:


baseline_cols = ['regimen', 'baseline_eGFR']
kwargs = {'get_baseline': True, 'baseline_cols': baseline_cols, 'display_ci': True, 'load_ci': True, 'save_ci': False, 'verbose': False}
eval_models.get_evaluation_scores(**kwargs)


# In[37]:


eval_models.plot_curves(curve_type='pr', legend_location='upper right', figsize=(12,18))
eval_models.plot_curves(curve_type='roc', legend_location='lower right', figsize=(12,18))
eval_models.plot_curves(curve_type='pred_cdf', figsize=(12,18)) # cumulative distribution function of model prediction
eval_models.plot_calibs(legend_location='upper left', figsize=(12,18)) 
# eval_models.plot_calibs(include_pred_hist=True, legend_location='upper left', figsize=(12,28), padding={'pad_y1': 0.3, 'pad_y0': 3.0})


# ## Post-Training Analysis

# ### Study Population Characteristics

# In[56]:


data_characteristic_summary(eval_models, save_dir=f'{output_path}/tables', partition='cohort', 
                            include_combordity=True, include_ckd=True, include_dialysis=True)


# ### Feature Characteristic

# In[25]:


feature_summary(eval_models, prep, target_keyword=target_keyword, save_dir=f'{output_path}/tables').head(60)


# ### Threshold Op Points

# In[38]:


pred_thresholds = np.arange(0, 1.01, 0.05)
thresh_df = eval_models.operating_points(algorithm='ENS', points=pred_thresholds, metric='threshold')
thresh_df


# ### All the Plots

# In[39]:


eval_models.all_plots_for_single_target(algorithm='ENS', target_type='CKD', save=True)


# ### Most important features

# In[40]:


get_ipython().system('python scripts/perm_importance.py --adverse-event CACKD')


# In[12]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot('ENS', eval_models.target_types, output_path, figsize=(6,15), top=10, importance_by='feature', padding={'pad_x0': 2.7})


# ### Performance on Subgroups

# In[13]:


subgroups = {'all', 'age', 'sex', 'immigrant', 'regimen', 'cancer_location', 'days_since_starting', 'ckd'}
df = subgroup_performance_summary('ENS', eval_models, pred_thresh=0.4, subgroups=subgroups, display_ci=False, load_ci=False, save_ci=False)
df # @ pred threshold = 0.4


# In[14]:


# PPV = 0.3 means roughly for every 3 alarms, 2 are false alarms and 1 is true alarm
# Sesnsitivity = 0.5 means roughly for every 2 true alarms, the model predicts 1 of them correctly
# Event Rate = 0.15 means true alarms occur 15% of the time
subgroups = ['Entire Test Cohort', 'Age', 'Sex', 'Immigration', 'Regimen', 'Days Since Starting Regimen']
padding = {'pad_y0': 1.2, 'pad_x1': 2.6, 'pad_y1': 0.2}
subgroup_performance_plot(df, target_type='CKD', subgroups=subgroups, padding=padding,
                          figsize=(12,24), save=True, save_dir=f'{output_path}/figures')


# ### Decision Curve Plot

# In[15]:


eval_models.plot_decision_curve_analysis('ENS')


# In[66]:


get_hyperparameters(output_path)


# # Scratch Notes

# ## CKD + AKI Summaries

# In[4]:


aki_prep = PrepDataCAN(adverse_event='aki')
ckd_prep = PrepDataCAN(adverse_event='ckd')

# get the union of ckd and aki dataset
ckd_data = ckd_prep.get_data(include_first_date=True, missing_thresh=80)
aki_data = aki_prep.get_data(include_first_date=True, missing_thresh=80)
# align the index and columns of both datasets, making them the same shape
# e.g. data['aki'].shape = (41171, 111)   -> data['aki'].shape = (43723, 112)
#      data['ckd'].shape = (24762, 106)   -> data['ckd'].shape = (43723, 112)
# the extra columns/indices for the respective datasets will be filled with nans
aki_data_aligned, ckd_data_aligned = aki_data.align(ckd_data, join='outer')
df = aki_data_aligned.fillna(ckd_data_aligned)

# sanity check
combined_idxs = aki_data.index.union(ckd_data.index)
assert df.shape[0] == len(combined_idxs)

# recompute missingness variables
cols = df.columns
cols = cols[cols.str.contains('_is_missing')]
df[cols] = df[cols.str.replace('_is_missing', '')].isnull()

# when aligning, many of the dtypes gets converted to object to account for np.nan
# fix the dtypes
dtypes = ckd_data.dtypes.combine_first(aki_data.dtypes)
df = df.astype(dtypes)

# aki and ckd may have different patient first visit dates in their dataset
# take the earlier date as the first visit date
patient_first_visit_date = df.groupby('ikn')['first_visit_date'].min()
df['first_visit_date'] = df['ikn'].map(patient_first_visit_date)

# set up the labels
create_labels = lambda target: pd.concat([aki_prep.convert_labels(target), ckd_prep.convert_labels(target)], axis=1)
kwargs = {'target_keyword': target_keyword, 'split_date': split_date, 'impute': False, 
          'normalize': False, 'verbose': False}
_, _, _, Y_train, Y_valid, Y_test = PrepData().split_data(df, **kwargs)
labels = {'Train': create_labels(Y_train), 'Valid': create_labels(Y_valid), 'Test': create_labels(Y_test)}

# set up the Evaluate object
pred_placeholder = {'Test': {}}
eval_models = Evaluate(output_path='placeholder', preds=pred_placeholder, labels=labels, orig_data=df)


# In[69]:


data_characteristic_summary(eval_models, save_dir=f'{main_dir}/models', partition='cohort', 
                            include_combordity=True, include_ckd=True, include_dialysis=True)


# In[7]:


feature_summary(eval_models, aki_prep, target_keyword=target_keyword, save_dir=f'{main_dir}/models').head(60)


# ## Spline Baseline Model

# In[31]:


from sklearn.preprocessing import StandardScaler
from scripts.train import TrainLOESSModel, TrainPolynomialModel
from scripts.evaluate import EvaluateBaselineModel
from scripts.visualize import get_bbox


# In[32]:


def end2end_pipeline(event='ckd', algorithm='SPLINE', split='Test'):
    if event not in {'ckd', 'aki', 'next_eGFR'}: 
        raise ValueError('event must be either ckd, aki, or next_eGFR')
    Trains = {'LOESS': TrainLOESSModel, 'SPLINE': TrainPolynomialModel, 'POLY': TrainPolynomialModel}
    Train = Trains[algorithm]
    base_col = 'baseline_eGFR'
    pred_next_eGFR = event == 'next_eGFR'
    if pred_next_eGFR: 
        cols = [event]
        name = 'Next eGFR'
        event = 'ckd'
        best_param_filename = f'{algorithm}_regressor_best_param'
        task_type = 'regression'
    else:
        name = event.upper()
        best_param_filename = ''
        task_type = 'classification'
    
    prep = PrepDataCAN(adverse_event=event)
    data = prep.get_data(include_first_date=True, missing_thresh=80)
    data, clip_thresholds = prep.clip_outliers(data, lower_percentile=0.001, upper_percentile=0.999)
    dataset = prep.split_data(prep.dummify_data(data.copy()), target_keyword=target_keyword, split_date=split_date, verbose=False)
    
    if pred_next_eGFR:
        X_train, X_valid, X_test, _, _, _ = dataset
        Y_train, Y_valid, Y_test =  data.loc[X_train.index, cols], data.loc[X_valid.index, cols], data.loc[X_test.index, cols]
        
        scaler = StandardScaler()
        Y_train[:] = scaler.fit_transform(Y_train)
        Y_valid[:] = scaler.transform(Y_valid)
        Y_test[:] = scaler.transform(Y_test)

        dataset = (X_train, X_valid, X_test, Y_train, Y_valid, Y_test)
        
    print(f'Training {algorithm} for {name}')
    output_path = f'{main_dir}/models/{event.upper()}'
    train = Train(dataset, output_path, base_col=base_col, algorithm=algorithm, task_type=task_type)
    best_param = train.bayesopt(filename=best_param_filename, verbose=0)
    model = train.train_model(**best_param)

    print(f'Evaluating {algorithm} for {name}')
    (preds, preds_min, preds_max), Y = train.predict(model, split=split)
    
    if pred_next_eGFR:
        preds[:] = scaler.inverse_transform(preds)
        preds_min[:] = scaler.inverse_transform(preds_min)
        preds_max[:] = scaler.inverse_transform(preds_max)
        
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,12))
        axes = axes.flatten()
        plt.subplots_adjust(hspace=0.2, wspace=0.2)
        
    data = data.loc[Y.index]
    for i, regimen in enumerate({'ALL', 'cisp(rt)', 'cisp(rt-w)'}):
        df = data if regimen == 'ALL' else data[data['regimen'] == regimen]
        idxs = df.index

        predictions, labels = {split: {algorithm: preds.loc[idxs]}},  {split: Y.loc[idxs]}
        eval_loess = EvaluateBaselineModel(base_col=train.col, preds_min=preds_min.loc[idxs], preds_max=preds_max.loc[idxs], 
                                           output_path=output_path, preds=predictions, labels=labels, orig_data=df)
        
        if pred_next_eGFR:
            eval_loess.plot_loess(axes[i], algorithm, cols[0], split=split)

            filename = f'{output_path}/figures/baseline/{cols[0]}_{regimen}_{algorithm}.jpg'
            fig.savefig(filename, bbox_inches=get_bbox(axes[i], fig), dpi=300) 
            axes[i].set_title(regimen)
        else:
            print(f'{algorithm} plot for regimen {regimen}')
            eval_loess.all_plots(algorithm=algorithm, filename=f'{regimen}_{algorithm}')
    
    if pred_next_eGFR:
        filename = f'{output_path}/figures/baseline/{cols[0]}_{algorithm}.jpg'
        plt.savefig(filename, bbox_inches='tight', dpi=300)


# In[23]:


end2end_pipeline(event='ckd')


# In[28]:


end2end_pipeline(event='aki')


# In[33]:


end2end_pipeline(event='next_eGFR')


# ## Motwani Score Based Model

# In[91]:


df = prep.get_data()
print(f'Size of data = {len(df)}, Number of patients = {df["ikn"].nunique()}')
df['cisplatin_dosage'] *= df['body_surface_area'] # convert from mg/m^2 to mg
df = df.loc[Y_test.index]
print(f'Size of test data = {len(df)}, Number of patients = {df["ikn"].nunique()}')
df = df[df['baseline_albumin_count'].notnull()]
print(f'Size of test data with albumin count = {len(df)}, Number of patients = {df["ikn"].nunique()}')
df = df[df['days_since_starting_chemo'] == 0] # very first treatment
print(f'Size of test data with only first day chemos = {len(df)}, Number of patients = {df["ikn"].nunique()}')


# In[92]:


def compute_score(data):
    data['score'] = 0
    data.loc[data['age'].between(61, 70), 'score'] += 1.5
    data.loc[data['age'] > 70, 'score'] += 2.5
    data.loc[data['baseline_albumin_count'] < 35, 'score'] += 2.0
    data.loc[data['cisplatin_dosage'].between(101, 150), 'score'] += 1.0
    data.loc[data['cisplatin_dosage'] > 150, 'score'] += 3.0
    data.loc[data['hypertension'], 'score'] += 2.0
    data['score'] /= data['score'].max()
    return data['score']


# In[98]:


split = 'Test'
score = compute_score(df)
labels = {split: Y_test.loc[df.index]}
preds = {split: {'ENS': train_ens.preds[split]['ENS'].loc[df.index],
                 'MSB': pd.DataFrame({col: score for col in Y_test.columns})}}
eval_motwani_model = Evaluate(output_path='', preds=preds, labels=labels, orig_data=df)


# In[99]:


# label distribtuion
labels[split].apply(pd.value_counts)


# In[100]:


kwargs = {'algorithms': ['ENS', 'MSB'], 'splits': ['Test'], 'display_ci': True, 'save_score': False}
result = eval_motwani_model.get_evaluation_scores(**kwargs)
result


# In[105]:


eval_motwani_model.all_plots_for_single_target(algorithm='MSB', target_type='AKI', split='Test',
                                               n_bins=20, calib_strategy='quantile', figsize=(12,12), save=False)


# In[106]:


points = np.arange(0, 8.6, 0.5)/8.5 # 8.5 is the highest score possible, 0 is lowest score possible
eval_motwani_model.operating_points('MSB', points, metric='threshold', target_types=['AKI'], split='Test', save=False)


# ### Compare with ENS

# In[107]:


eval_motwani_model.all_plots_for_single_target(algorithm='ENS', target_type='AKI', split='Test',
                                               n_bins=20, calib_strategy='quantile', figsize=(12,12), save=False)


# In[109]:


points = np.arange(0, 8.6, 0.5)/8.5 # 8.5 is the highest score possible, 0 is lowest score possible
eval_motwani_model.operating_points('ENS', points, metric='threshold', target_types=['AKI'], split='Test', save=False)


# ## Missingness By Splits

# In[110]:


from scripts.utility import get_nmissing_by_splits


# In[130]:


missing = get_nmissing_by_splits(model_data, train_ens.labels)
missing.sort_values(by=(f'Test (N={len(Y_test)})', 'Missing (N)'), ascending=False)


# In[221]:


# check CKD
missing = get_nmissing_by_splits(model_data, train_ens.labels)
missing.sort_values(by=(f'Test (N={len(Y_test)})', 'Missing (N)'), ascending=False)
