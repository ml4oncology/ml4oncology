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


# In[6]:


import os
import tqdm
import pandas as pd
import numpy as np
import warnings
import pickle
# warnings.filterwarnings('ignore')

from scripts.utilities import (read_partially_reviewed_csv, get_included_regimen,
                               most_common_by_category, pred_thresh_binary_search, data_splits_summary, plot_feat_importance)
from scripts.config import (root_path, blood_types, cytopenia_thresholds, 
                            cancer_location_mapping, cancer_type_mapping)
from scripts.prep_data import (PrepData)
from scripts.train import (Train)

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_score, recall_score, roc_auc_score, average_precision_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization


# In[38]:


# config
df = read_partially_reviewed_csv()
df = get_included_regimen(df)
cycle_lengths = df['cycle_length'].to_dict()
del df

cols = ['H_hemoglobin_transfusion_date', 'H_platelet_transfusion_date', 
        'ED_hemoglobin_transfusion_date', 'ED_platelet_transfusion_date']
dtype = {col: str for col in cols}
dtype.update({'curr_morph_cd': str, 'lhin_cd': str})
chemo_df = pd.read_csv(f'{root_path}/cytopenia/data/chemo_processed2.csv', dtype=dtype)

output_path = f'{root_path}/cytopenia/models'


# In[39]:


# TODO: Remove this temporary code
chemo_df['H_blood_tranfused'] = ~chemo_df[cols[:2]].isnull().all(axis=1)
chemo_df['ED_blood_tranfused'] = ~chemo_df[cols[-2:]].isnull().all(axis=1)
chemo_df = chemo_df.drop(columns=cols)


# In[40]:


# check out stats for blood transfusion and ODB growth factor
df = pd.DataFrame(index=['Number of Chemo Sessions Where Patient Had'])
cols = ['H_blood_tranfused', 'ED_blood_tranfused', 'ODBGF_given']
for col in cols: df[col] = sum(chemo_df[col])
df['Any'] = sum(chemo_df[cols[0]] | chemo_df[cols[1]] | chemo_df[cols[2]])
df['Total'] = len(chemo_df)
df


# # More Preprocessing

# In[91]:


def read_data(blood_type):
    df = pd.read_csv(f'{root_path}/cytopenia/data/{blood_type}.csv')

    # turn string of numbers columns into integer column 
    df.columns = df.columns.values.astype(int)
    
    # get baseline and target blood counts
    df['baseline_blood_count'] = chemo_df[f'baseline_{blood_type}_count']
    df['regimen'] = chemo_df['regimen']
    keep_indices = []
    for regimen, group in df.groupby('regimen'):
        # forward fill blood counts from a days before to the day after administration
        cycle_length = int(cycle_lengths[regimen])
        if cycle_length == 28:
            cycle_length_window = range(cycle_length-2, cycle_length+1)
            cycle_end = cycle_length
        else:  
            cycle_length_window = range(cycle_length-1,cycle_length+2)
            cycle_end = cycle_length+1
        df.loc[group.index, 'target_blood_count'] = df.loc[group.index, cycle_length_window].ffill(axis=1)[cycle_end]
    mask = (~df['baseline_blood_count'].isnull() & ~df['target_blood_count'].isnull())
    df = df[mask]
    
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
    # organize data for model input and target labels
    """
    NBC - neutrophil blood count
    HBC - hemoglobin blood count
    PBC - platelet blood count

    input:                               -->        MODEL         -->            target:
    regimen                                                                      NBC on next admin
    NBC on prev admin                                                            HBC on next admin
    HBC on prev admin                                                            PBC on next admin
    PBC on prev admin 
    days since prev observed NBC/HBC/PBC
    chemo cycle, immediate new regimen
    intent of systemic treatment, line of therapy
    lhin cd, curr morth cd, curr topog cd, age, sex
    body surface area, esas/ecog features
    blood work on prev admin
    """
    model_data = chemo_df.loc[data['neutrophil'].index] # indices are same for all blood types
    
    # remove chemo sessions where patient were administered blood transfusions and ODB growth factors during the session
    model_data = model_data[~(model_data['H_blood_tranfused'] | model_data['ED_blood_tranfused'] | model_data['ODBGF_given'])]
    
    drop_columns = ['visit_date', 'prev_visit', 'H_blood_tranfused', 'ED_blood_tranfused', 'ODBGF_given']
    model_data = model_data.drop(columns=drop_columns)
    
    # convert chemo interval (days between prev and next admin) into integer (e.g. '22 days' into 22)
    model_data['chemo_interval'] = model_data['chemo_interval'].str.split(' ').str[0].astype(int)
    
    # get column for corresponding cycle lengths
    model_data['cycle_lengths'] = model_data['regimen'].map(cycle_lengths).astype(float)
    
    # fill null values with 0
    model_data['line_of_therapy'] = model_data['line_of_therapy'].fillna(0) # the nth different chemotherapy taken

    for blood_type, df in data.items():
        model_data[f'target_{blood_type}_count'] = df['target_blood_count']
    
    return model_data


# In[92]:


model_data = organize_data(get_data())
model_data.to_csv(f'{output_path}/model_data.csv', index=False)


# In[93]:


model_data = pd.read_csv(f'{output_path}/model_data.csv', dtype={'curr_morph_cd': str, 'lhin_cd': str})
model_data


# In[94]:


model_data.columns


# In[95]:


num_missing = model_data.isnull().sum() # number of nans for each column
num_missing = num_missing[num_missing != 0] # remove columns without missing values
print("Missing values\n--------------")
print(num_missing)
print("\nDistribution of regimens\n------------------------")
print(model_data['regimen'].value_counts())


# In[96]:


# analyze the correlations
cols = ['baseline_neutrophil_count', 'target_neutrophil_count', 
        'baseline_hemoglobin_count', 'target_hemoglobin_count', 
        'baseline_platelet_count', 'target_platelet_count']
model_data[cols].corr(method='pearson').style.background_gradient(cmap='Greens')


# In[97]:


# analyze the distribution

# the min/max from ReferenceRanges column
# blood_ranges = utilities.get_blood_ranges()
blood_ranges = {'platelet': [0.0, 600.0], 'neutrophil': [0.0, 26.0], 'hemoglobin': [0.0, 256.0]}

def blood_count_value_distribution(model_data, bins=[50, 30, 60]):
    fig = plt.figure(figsize=(20,5))
    for idx, blood_type in enumerate(blood_types):
        min_range, max_range = blood_ranges[blood_type]
        blood_counts = np.clip(model_data[f'baseline_{blood_type}_count'], min_range, max_range) # remove/clip the outliers
        ax = fig.add_subplot(1,3,idx+1)
        plt.hist(blood_counts, bins=bins[idx])
        plt.xlabel('Blood Count Value')
        plt.title(blood_type)
    plt.show()
blood_count_value_distribution(model_data)


# In[98]:


# Preparing Data for Model Input
prep = PrepData()


# In[99]:


model_data = pd.read_csv(f'{output_path}/model_data.csv', dtype={'curr_morph_cd': str, 'lhin_cd': str})
print(f'Size of model_data: {model_data.shape}\nNumber of unique patients: {model_data["ikn"].nunique()}')


# In[100]:


model_data, clip_thresholds = prep.clip_outliers(model_data)
clip_thresholds


# In[101]:


model_data = prep.dummify_data(model_data)
print(f'Size of model_data: {model_data.shape}\nNumber of unique patients: {model_data["ikn"].nunique()}')


# In[102]:


train, valid, test = prep.split_data(model_data)


# # Model Training - Classification

# In[103]:


X_train, Y_train = train
X_valid, Y_valid = valid
X_test, Y_test = test


# In[104]:


Y_train = prep.regression_to_classification(Y_train, cytopenia_thresholds)
Y_valid = prep.regression_to_classification(Y_valid, cytopenia_thresholds)
Y_test = prep.regression_to_classification(Y_test, cytopenia_thresholds)


# In[105]:


prep.get_label_distribution(Y_train, Y_valid, Y_test)


# In[106]:


X_train, Y_train = prep.upsample(X_train, Y_train)
X_valid, Y_valid = prep.upsample(X_valid, Y_valid)


# In[107]:


prep.get_label_distribution(Y_train, Y_valid, Y_test)


# In[54]:


# Test speed of training
params = {'max_iter': 1000}
calib_param_logistic = {'method': 'sigmoid', 'cv': 3}
model = LogisticRegression(**params)
model = MultiOutputClassifier(CalibratedClassifierCV(model, **calib_param_logistic))
for i in tqdm.tqdm(range(1)):
    model.fit(X_test, Y_test)


# In[108]:


# Initialize Training class
dataset = (X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
train = Train(dataset, clip_thresholds=clip_thresholds)


# In[36]:


# Conduct Baysian Optimization
best_params = {}
for algorithm, model in train.ml_models.items():
    if algorithm in ['RF', 'LR', 'XGB']: continue # put the algorithms already trained and tuned in this list
    best_param = train.bayesopt(algorithm)
    best_params[algorithm] = best_param
    train.train_model_with_best_param(algorithm, model, best_param)


# In[110]:


# Optional - Retrain model with best hyperparam
for algorithm, model in tqdm.tqdm(train.ml_models.items()):
    filename = f'{output_path}/{algorithm}_classifier_best_param.pkl'
    with open(filename, 'rb') as file:
        best_param = pickle.load(file)
    train.train_model_with_best_param(algorithm, model, best_param, output_path)


# In[119]:


# Get Model Performance Scores
score_df = train.get_evaluation_scores(get_baseline=True, model_dir=output_path, splits=['Valid', 'Test'])
score_df


# In[120]:


score_df.loc[[i for i in score_df.index if 'AUROC' in i[1] or 'AUPRC' in i[1]]]


# In[122]:


train.plot_PR_curves(save_dir=output_path)


# In[124]:


train.plot_ROC_curves(save_dir=output_path)


# In[125]:


train.plot_calib_plots(save_dir=output_path)


# # Study Population Characteristics

# In[126]:


model_data = pd.read_csv(f'{output_path}/model_data.csv', dtype={'curr_morph_cd': str, 'lhin_cd': str})
model_data = prep.regression_to_classification(model_data, cytopenia_thresholds)


# In[ ]:


data_splits_summary(train, model_data, output_path)


# In[101]:


def cyto_pop_summary():
    # cytopenia population summary
    model = train.load_model(output_path, 'XGB')
    pred = model.predict(X_test)
    total_test_pop = len(model_data.loc[X_test.index, 'ikn'].unique())
    
    indices = pd.MultiIndex.from_product([[], []])
    summary_df = pd.DataFrame(index=indices)
    for idx, blood_type in enumerate(blood_types):
        col_name = f'{blood_type} < {cytopenia_thresholds[blood_type]}'
        df_cyto = model_data.loc[model_data[col_name]]
        patient_indices = df_cyto['ikn'].drop_duplicates(keep='last').index
        total_cyto_pop = len(patient_indices)

        num_patient_summary(summary_df, total_cyto_pop, col_name)
        avg_age_summary(summary_df, df_cyto, patient_indices, col_name)
        sex_summary(summary_df, df_cyto, patient_indices, total_cyto_pop, col_name)

        # Positive Predicted Prevalance (number of predicted positives in test set / total test population)
        pred_pos = sum(pred[:, idx])
        row_name = ('Positive Prediction Prevalence (Test Set)', '')
        summary_df.loc[row_name, col_name] = f"{pred_pos} ({np.round(pred_pos/total_test_pop, 1)})"
        
        regimen_summary(summary_df, df_cyto, total_cyto_pop, col_name, top=10)
        cancer_location_summary(summary_df, df_cyto, total_cyto_pop, col_name, top=5)
        cancer_type_summary(summary_df, df_cyto, total_cyto_pop, col_name, top=4)
    
    return summary_df.dropna()


# In[102]:


cyto_pop_summary()


# # XGB Post-Training Analysis

# In[19]:


model = train.load_model(output_path, 'XGB')


# ## XGB as txt file

# In[104]:


for idx, blood_type in enumerate(blood_types):
    model.estimators_[idx].get_booster().dump_model(f'{output_path}/XGB/{blood_type}.txt')
    model.estimators_[idx].save_model(f'{output_path}/XGB/{blood_type}.model')


# ## XGB most important features

# In[105]:


# use xgboost's default feature importance (Gini index)
cols = X_train.columns
fig = plt.figure(figsize=(15, 5))
plt.subplots_adjust(wspace=0.8)

for idx, blood_type in enumerate(blood_types):
    feature_importances = model.estimators_[idx].feature_importances_
    sorted_idx = (-feature_importances).argsort()
    sorted_idx = sorted_idx[0:20] # get the top 20 important features
    
    ax = fig.add_subplot(1,3,idx+1)
    ax.barh(cols[sorted_idx], feature_importances[sorted_idx])
    ax.invert_yaxis()
    ax.set_title(blood_type)
    ax.set_xlabel('Feature Importance Score')


# In[1]:


# use sklearn's permutation importance (no one-hot encoded data)
# WARNING!!!!! sighhhh, must run as a script in command line or else it fails for n_jobs > 1
get_ipython().system('python scripts/perm_importance.py')


# In[37]:


model_data = pd.read_csv(f'{output_path}/model_data.csv', dtype={'curr_morph_cd': str, 'lhin_cd': str})
cols = model_data.columns
target_cols = cols[cols.str.contains('target')]
cols = cols.drop(target_cols.tolist() + ['ikn'])


# In[42]:


save_dir = f'{output_path}/XGB'
figsize = (6, 15)
plot_feat_importance(cols, blood_types, save_dir, figsize)


# ## XGB performance for most common cancer regimens/cancer type

# In[16]:


def XGB_results_by_category(most_common, category='regimen', save=True):
    
    indices = pd.MultiIndex.from_product([[],[]])
    score_df = pd.DataFrame(index=indices)

    for entry in most_common: 
        mask = X_test[f'{category}_{entry}'] == 1
        X = X_test[mask]
        Y = Y_test[mask]
        pred_prob = model.predict_proba(X) 
        if category == 'curr_morph_cd': entry = cancer_type_mapping[entry]
        if category == 'curr_topog_cd': entry = cancer_location_mapping[entry]
        for idx, blood_type in enumerate(blood_types):
            col = Y.columns[Y.columns.str.contains(blood_type)]
            Y_true = Y[col]
            # if no positive examples exists, skip
            if not Y_true.any(axis=None):
                print(f'Found no positive examples in {category} {entry} for blood type {blood_type} - Skipping')
                continue
            Y_pred_prob = pred_prob[idx][:, 1]
            score_df.loc[(entry, 'AUROC Score'), blood_type] = roc_auc_score(Y_true, Y_pred_prob)
            score_df.loc[(entry, 'AUPRC Score'), blood_type] = average_precision_score(Y_true, Y_pred_prob)
    if save:
        name_map = {'regimen': 'regimen', 'curr_morph_cd': 'cancer_type', 'curr_topog_cd': 'cancer_location'}
        score_df.to_csv(f'{output_path}/XGB/common_{name_map[category]}_results.csv')
    return score_df


# In[17]:


model_data = pd.read_csv(f'{output_path}/model_data.csv', dtype={'curr_morph_cd': str, 'lhin_cd': str})

top_regimens = most_common_by_category(model_data, category='regimen', top=10)
most_common_regimens = list(top_regimens.index)

top_cancer_types = most_common_by_category(model_data, category='curr_morph_cd', top=10)
most_common_cancer_types = list(top_cancer_types.index)

top_cancer_locations = most_common_by_category(model_data, category='curr_topog_cd', top=10)
most_common_cancer_locations = list(top_cancer_locations.index)


# In[18]:


XGB_results_by_category(most_common_regimens, category='regimen')


# In[19]:


XGB_results_by_category(most_common_cancer_types, category='curr_morph_cd')


# In[20]:


XGB_results_by_category(most_common_cancer_locations, category='curr_topog_cd')


# ## XGB Randomized Individual Patient Performance

# In[31]:


sex_mapping = {'M': 'male', 'F': 'female'}
blood_info_mapping = {'neutrophil': {'low_count_name': 'neutropenia',
                                     'unit': '10^9/L'},
                      'hemoglobin': {'low_count_name': 'anemia',
                                     'unit': 'g/L'},
                      'platelet': {'low_count_name': 'thrombocytopenia',
                                   'unit': '10^9/L'}}


# In[50]:


def get_patient_info(orig_data):
    age = int(orig_data['age'].mean())
    sex = sex_mapping[orig_data['sex'].values[0]]
    regimen = orig_data['regimen'].values[0]
    patient_info = f"{age} years old {sex} patient under regimen {regimen}"
    return patient_info

def plot_patient_prediction(X_test, num_ikn=3, seed=0, save=False):
    """
    Args:
        num_ikn (int): the number of random patients to analyze
    """
    np.random.seed(seed)
    
    model_data = pd.read_csv(f'{output_path}/model_data.csv', dtype={'curr_morph_cd': str, 'lhin_cd': str})

    # get the original data corresponding with the testing set
    df = model_data.loc[X_test.index]

    # only consider patients who had more than 3 chemo cycles
    ikns = df.loc[df['chemo_cycle'] > 3, 'ikn'].unique()

    for _ in range(num_ikn):
        ikn = np.random.choice(ikns) # select a random patient from the consideration pool
        ikn_indices = df[df['ikn'] == ikn].index # get the indices corresponding with the selected patient
        X = X_test.loc[ikn_indices]
        pred_prob = model.predict_proba(X) # get model predictions
        orig_data = df.loc[ikn_indices]
        patient_info = get_patient_info(orig_data)
        print(patient_info)

        fig = plt.figure(figsize=(15, 20))
        days_since_admission = orig_data['chemo_interval'].cumsum().values
        for i, blood_type in enumerate(blood_types):
            true_count = orig_data[f'target_{blood_type}_count'].values

            ax1 = fig.add_subplot(6, 3, i+1) # 3 blood types * 2 subplots each
            ax1.plot(days_since_admission, true_count, label=f'{blood_type}'.capitalize())
            ax1.axhline(y=cytopenia_thresholds[blood_type], color='r', alpha=0.5, 
                       label = f"{blood_info_mapping[blood_type]['low_count_name']} threshold".title() + 
                               f" ({cytopenia_thresholds[blood_type]})")
            ax1.tick_params(labelbottom=False)
            ax1.set_ylabel(f"Blood count ({blood_info_mapping[blood_type]['unit']})")
            ax1.set_title(f"Patient {blood_type} measurements")
            ax1.legend()
            ax1.grid(axis='x')

            ax2 = fig.add_subplot(6, 3, i+1+3, sharex=ax1)
            ax2.plot(days_since_admission, pred_prob[i][:, 1], label='XGB Model Prediction')
            ax2.axhline(y=0.5, color='r', alpha=0.5, label="Positive Prediction Threshold")
            ax2.set_xticks(days_since_admission)
            ax2.set_yticks(np.arange(0, 1.01, 0.2))
            ax2.set_xlabel('Days since admission')
            ax2.set_ylabel(f"Risk of {blood_info_mapping[blood_type]['low_count_name']}")
            ax2.set_title(f"Model Prediction for {blood_info_mapping[blood_type]['low_count_name']}")
            ax2.legend()
            ax2.grid(axis='x')
        if save:
            plt.savefig(f'{output_path}/XGB/patient_prediction/{ikn}.jpg', bbox_inches='tight') #dpi=300
        plt.show()


# In[52]:


plot_patient_prediction(X_test, num_ikn=8, save=True)


# ## XGB Precision vs Senstivity Operating Points

# In[54]:


pred_thresholds = np.arange(0, 1.01, 0.1)
train.threshold_op_points(model, pred_thresholds, X_test, Y_test)


# In[16]:


def precision_op_points(precisions, X, Y):
    cols = pd.MultiIndex.from_product([blood_types, ['Prediction Threshold', 'Sensitivity']])
    df = pd.DataFrame(columns=cols)
    df.index.name = 'Precision'

    pred_prob = model.predict_proba(X) 
    for idx, blood_type in enumerate(blood_types):
        col = Y.columns[Y.columns.str.contains(blood_type)]
        Y_true = Y[col]
        Y_pred_prob = pred_prob[idx][:, 1]
        
        for desired_precision in precisions:
            # LOL binary search the threshold to get desired precision
            threshold = pred_thresh_binary_search(desired_precision, Y_pred_prob, Y_true)
            Y_pred_bool = Y_pred_prob > threshold
            df.loc[desired_precision, (blood_type, 'Prediction Threshold')] = threshold
            df.loc[desired_precision, (blood_type, 'Sensitivity')] = recall_score(Y_true, Y_pred_bool, zero_division=1)
    return df


# In[17]:


precisions = [0.2, 0.25, 0.33, 0.4, 0.5, 0.6, 0.75]
precision_op_points(precisions, X_test, Y_test)


# # SCRATCH NOTES

# ## Display Best PARAMS

# In[169]:


indices = [[], []]
values = []
for algorithm in ['LR', 'RF', 'XGB', 'NN']:
    filename = f'{output_path}/{algorithm}_classifier_best_param.pkl'
    with open(filename, 'rb') as file:
        best_param = pickle.load(file)
    for param, value in best_param.items():
        indices[0].append(algorithm)
        indices[1].append(param)
        values.append(value)


# In[170]:


pd.DataFrame(values, index=indices, columns=['Classification'])


# ## Results for each Grade

# In[190]:


thresholds = [(1.5, 100, 75), (1.0, 80, 50), (0.5, 80, 25)] # grade1 (neutrophil, hemoglobin, platelet), grade2, grade3

model_data = pd.read_csv(f'{output_path}/model_data.csv', dtype={'curr_morph_cd': str, 'lhin_cd': str})
model_data, clip_thresholds = clip_outliers(model_data)
model_data = dummify_data(model_data)
train, valid, test, scaler = split_data(model_data)

for threshold in thresholds:
    X_train, Y_train = train
    X_valid, Y_valid = valid
    X_test, Y_test = test

    threshold = {blood_type: threshold[i] for i, blood_type in enumerate(blood_types)}
    threshold = normalize_cytopenia_thresholds(clip_thresholds, threshold)
    Y_train = regression_to_classification(Y_train, threshold)
    Y_valid = regression_to_classification(Y_valid, threshold)
    Y_test = regression_to_classification(Y_test, threshold)
    dist = get_label_distribution(Y_train, Y_valid, Y_test, save=False)
    print(dist)

# WITHOUT POSITIVE EXAMPLES, RESULTS WILL BE MEANINGLESS


# # use eli5's permutation importance

# In[ ]:


import eli5
from eli5.sklearn import PermutationImportance
from IPython.display import display

cols = X_valid.columns
params = {'random_state': 42, 'n_iter': 15}
for idx, blood_type in tqdm.tqdm(enumerate(blood_types)):
    model.estimators_[idx].get_booster().feature_names = cols.tolist()
    perm = PermutationImportance(model.estimators_[idx]).fit(X_valid, Y_valid.iloc[:, idx])
    print(blood_type)
    display(eli5.show_weights(perm, feature_names=cols.tolist()))

