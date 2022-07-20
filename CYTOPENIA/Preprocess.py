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
from functools import partial

from scripts.config import (root_path, sas_folder, cyto_folder, blood_types, all_observations, event_map)
from scripts.spark import (preprocess_olis_data)
from scripts.preprocess import (split_and_parallelize, clean_string, group_observations,
                                load_chemo_df, load_included_regimens, 
                                filter_systemic_data, process_systemic_data, 
                                filter_y3_data, process_cancer_and_demographic_data,
                                filter_immigration_data, process_immigration_data,
                                olis_worker, postprocess_olis_data,
                                filter_esas_data, get_esas_responses, postprocess_esas_responses,
                                filter_body_functionality_data, body_functionality_worker,
                                filter_blood_transfusion_data, blood_transfusion_worker, 
                                extract_blood_transfusion_data, postprocess_blood_transfusion_data)


# In[3]:


get_ipython().system('ls $sas_folder')


# In[4]:


get_ipython().system('du -h $sas_folder/olis.sas7bdat $sas_folder/systemic.sas7bdat $sas_folder/y3.sas7bdat')


# In[5]:


# config
processes = 32
main_dir = f'{root_path}/{cyto_folder}'
if not os.path.exists(f'{main_dir}/data'):
    os.makedirs(f'{main_dir}/data')


# # Selected Regimens

# In[6]:


regimens = load_included_regimens(criteria='cytotoxic')
cycle_length_mapping = dict(regimens[['regimen', 'shortest_cycle_length']].values)
cycle_length_mapping['Other'] = 7.0 # make rare regimen cycle lengths default to 7
max_cycle_length = int(regimens['shortest_cycle_length'].max())
regimens_renamed = sorted(regimens['relabel'].fillna(regimens['regimen']).unique())
print(f'{len(regimens)} raw regimens -> {len(regimens_renamed)} relabeled total regimens')
regimens_renamed


# # Create my csvs

# ### Include features from systemic (chemo) dataset

# In[7]:


systemic = pd.read_csv(f'{root_path}/data/systemic.csv')
systemic = filter_systemic_data(systemic, regimens, remove_inpatients=False, verbose=True)
print(f"Size of data = {len(systemic)}")
print(f"Number of patients = {systemic['ikn'].nunique()}")
print(f"Number of unique regiments = {systemic['regimen'].nunique()}")
print(f"Chemotherapy Cohort from {systemic['visit_date'].min()} to {systemic['visit_date'].max()}")


# In[8]:


systemic = process_systemic_data(systemic, cycle_length_mapping)
systemic.to_csv(f'{main_dir}/data/systemic.csv', index=False)
print(f"Size of data = {len(systemic)}")
print(f"Number of patients = {systemic['ikn'].nunique()}")
print(f"Number of unique regiments = {systemic['regimen'].nunique()}")


# In[9]:


systemic = pd.read_csv(f'{main_dir}/data/systemic.csv', dtype={'ikn': str})
for col in ['visit_date', 'next_visit_date']: systemic[col] = pd.to_datetime(systemic[col])


# ### Include features from y3 (cancer and demographic) dataset

# In[10]:


col_arrangement = ['ikn', 'regimen', 'visit_date', 'next_visit_date', 'chemo_interval', 'days_since_starting_chemo', 'days_since_last_chemo', 
                   'cycle_length', 'chemo_cycle', 'immediate_new_regimen', 'intent_of_systemic_treatment', 'line_of_therapy', 'lhin_cd', 
                   'curr_morph_cd', 'curr_topog_cd', 'age', 'sex', 'body_surface_area']


# In[11]:


# Extract and Preprocess the Y3 Data
y3 = pd.read_csv(f'{root_path}/data/y3.csv')
y3 = filter_y3_data(y3)
print(f"Number of patients in y3 = {y3['ikn'].nunique()}")
print(f"Number of patients in y3 and systemic = {y3['ikn'].isin(systemic['ikn']).sum()}")


# In[12]:


# Process the Y3 and Systemic Data
chemo_df = process_cancer_and_demographic_data(y3, systemic, verbose=True)
chemo_df = chemo_df[col_arrangement]
print(f"Number of unique regiments = {chemo_df['regimen'].nunique()}")
print(f"Number of patients = {chemo_df['ikn'].nunique()}")
print(f"Number of female patients = {chemo_df.loc[chemo_df['sex'] == 'F', 'ikn'].nunique()}")
print(f"Number of male patients = {chemo_df.loc[chemo_df['sex'] == 'M', 'ikn'].nunique()}")


# ### Include features from immigration dataset

# In[13]:


# Extract and Preprocess the Immigration Data
immigration = pd.read_csv(f'{root_path}/data/immigration.csv')
immigration = filter_immigration_data(immigration)

# Process the Immigration Data
chemo_df = process_immigration_data(chemo_df, immigration)
chemo_df.to_csv(f'{main_dir}/data/chemo_processed.csv', index=False)


# ### Include features from olis (blood work/lab test observation count) dataset
# Note: I think they made a mistake. The variable <b>value_recommended_d</b> is the value of the test result (variable should be named "value")

# In[7]:


chemo_df = load_chemo_df(main_dir)
print(f"Number of rows now: {len(chemo_df)}")
print(f"Number of patients now: {chemo_df['ikn'].nunique()}")


# In[10]:


get_ipython().run_cell_magic('time', '', "# Preprocess the Raw OLIS Data using PySpark\npreprocess_olis_data(f'{main_dir}/data', set(chemo_df['ikn']))")


# In[8]:


# Extract the Olis Features
olis = pd.read_csv(f"{main_dir}/data/olis.csv", dtype=str) 
olis['ObservationDateTime'] = pd.to_datetime(olis['ObservationDateTime'])

# get results
filtered_chemo_df = chemo_df[chemo_df['ikn'].isin(olis['ikn'])] # filter out patients not in dataset
worker = partial(olis_worker, latest_limit=max_cycle_length)
result = split_and_parallelize((filtered_chemo_df, olis), worker, processes=processes)
result = pd.DataFrame(result, columns=['observation_code', 'chemo_idx', 'days_after_chemo', 'observation_count'])
result.to_csv(f'{main_dir}/data/olis2.csv', index=False)


# In[9]:


# Process the Olis Features
olis_df = pd.read_csv(f'{main_dir}/data/olis2.csv')
mapping, missing_df = postprocess_olis_data(chemo_df, olis_df, observations=all_observations, days_range=range(-5,max_cycle_length+1))
missing_df


# In[10]:


# group together obs codes with same obs name
freq_map = olis_df['observation_code'].value_counts()
grouped_observations = group_observations(all_observations, freq_map)

for blood_type in blood_types:
    obs_codes = grouped_observations[blood_type]
    for i, obs_code in enumerate(obs_codes):
        if i == 0: df = mapping[obs_code]
        else: df = df.fillna(mapping[obs_code])
    df.to_csv(f'{main_dir}/data/{blood_type}.csv', index=False)


# ### Include features from esas (symptom questionnaire) dataset

# In[19]:


# Preprocess the Questionnaire Data
esas = pd.read_csv(f"{root_path}/data/esas.csv")
esas = filter_esas_data(esas, chemo_df['ikn'])
esas.to_csv(f'{main_dir}/data/esas.csv', index=False)


# In[20]:


# Extract the Questionnaire Features
esas = pd.read_csv(f'{main_dir}/data/esas.csv', dtype=str)
result = get_esas_responses(chemo_df, esas, processes=processes)
result = pd.DataFrame(result, columns=['index', 'symptom', 'severity'])
result.to_csv(f'{main_dir}/data/esas2.csv', index=False)


# In[11]:


# Process the Questionnaire Responses
esas_df = pd.read_csv(f'{main_dir}/data/esas2.csv')
esas_df = postprocess_esas_responses(esas_df)

# put esas responses in chemo_df
chemo_df = chemo_df.join(esas_df, how='left') # ALT WAY: pd.merge(chemo_df, esas, left_index=True, right_index=True, how='left')


# ### Include features from ecog and prfs (body functionality grade) dataset

# In[22]:


for dataset in ['ecog', 'prfs']:
    # Extract and Preprocess the body functionality dataset
    bf = pd.read_csv(f'{root_path}/data/{dataset}.csv')
    bf = filter_body_functionality_data(bf, chemo_df['ikn'], dataset=dataset)
    
    # get results
    filtered_chemo_df = chemo_df[chemo_df['ikn'].isin(bf['ikn'])] # filter out patients not in dataset
    worker = partial(body_functionality_worker, dataset=dataset)
    result = split_and_parallelize((filtered_chemo_df, bf), worker, processes=processes)
    result = pd.DataFrame(result, columns=['index', f'{dataset}_grade'])
    result.to_csv(f'{main_dir}/data/{dataset}.csv', index=False)


# In[12]:


for dataset in ['ecog', 'prfs']:
    # Process the results
    bf = pd.read_csv(f'{main_dir}/data/{dataset}.csv')
    bf = bf.set_index('index')

    # put result in chemo_df
    chemo_df = chemo_df.join(bf, how='left') # ALT WAY: pd.merge(chemo_df, ecog, left_index=True, right_index=True, how='left')


# ### Include blood transfusion features from dad and nacrs dataset 
# (hospitalization and ED visits where blood tranfusion was administered)

# In[24]:


for event in ['H', 'ED']:
    # Preprocess the transfusion data
    database_name = event_map[event]['database_name']
    chunks = pd.read_csv(f'{root_path}/data/{database_name}_transfusion.csv', chunksize=10**6, dtype=str) 
    for i, chunk in tqdm.tqdm(enumerate(chunks), total=7):
        chunk = filter_blood_transfusion_data(chunk, chemo_df['ikn'], event=event)
        # write to csv
        header = True if i == 0 else False
        mode = 'w' if i == 0 else 'a'
        chunk.to_csv(f"{main_dir}/data/{database_name}_transfusion.csv", header=header, mode=mode, index=False)


# In[13]:


for event in ['H', 'ED']:
    # Extract Blood Transfusion Events During Chemotherapy
    extract_blood_transfusion_data(chemo_df, main_dir, event=event)
    
    # Process the Blood Transfusion Events
    chemo_df, h_bt_df = postprocess_blood_transfusion_data(chemo_df, main_dir, event=event)


# ### Include odb_growth factor features
# odb - ontario drug benefit

# In[14]:


def filter_odb_data(odb, chemo_ikns):
    # organize and format columns
    odb = clean_string(odb, ['ikn'])
    odb['servdate'] = pd.to_datetime(odb['servdate'])
    
    # filter patients not in chemo_df
    odb = odb[odb['ikn'].isin(chemo_ikns)]
    
    # sort by date
    odb = odb.sort_values(by='servdate')
    
    return odb

def odb_worker(partition):
    chemo_df, odb_df = partition
    result = set()
    for ikn, chemo_group in chemo_df.groupby('ikn'):
        odb_group = odb_df[odb_df['ikn'] == ikn]
        for i, odb_row in odb_group.iterrows():
            servdate = odb_row['servdate']
            mask = (servdate <= chemo_group['next_visit_date']) & (servdate >= chemo_group['visit_date'])
            result.update(chemo_group[mask].index)
    return result


# In[27]:


# Extract and Preprocess ODB Growth Factor Administration Events During Chemotherapy
odb = pd.read_csv(f'{root_path}/data/odb_growth_factors.csv')
odb = filter_odb_data(odb, chemo_df['ikn'])
filtered_chemo_df = chemo_df[chemo_df['ikn'].isin(odb['ikn'])]  # filter out patients not in dataset
result = split_and_parallelize((filtered_chemo_df, odb), odb_worker)
np.save(f'{main_dir}/data/odb_indices.npy', result)


# In[15]:


# Process the ODB Growth Factor Administration Events
indices = np.load(f'{main_dir}/data/odb_indices.npy')
chemo_df['ODBGF_given'] = False
chemo_df.loc[indices, 'ODBGF_given'] = True


# In[16]:


chemo_df.to_csv(f'{main_dir}/data/model_data.csv', index=False)


# # Scratch Notes

# ## Gaussian playground

# In[19]:


X = np.atleast_2d([1, 3, 5, 6, 7, 8]).T
f = lambda x: x*np.sin(x)
y = f(X).ravel()


# In[20]:


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from matplotlib import pyplot as plt


# In[21]:


kernel = C(1, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))


# In[22]:


gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)


# In[23]:


gp.fit(X, y)


# In[24]:


x = np.atleast_2d(np.linspace(0, 10, 1000)).T
y_pred, sigma = gp.predict(x, return_std=True)


# In[25]:


plt.figure()
plt.plot(x, f(x), 'r:', label='f(x) = x*sin(x)')
plt.plot(X, y, 'r.', markersize=10,label='Observations')
plt.plot(x, y_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.96*sigma, (y_pred + 1.96*sigma)[::-1]]),
         alpha=0.5, fc='b', ec='None', label="95% confidence interval"
         )
plt.legend()


# ## Include Fever from ED/H Visits
# For predicting febrile neutropenia
# 
# Arbitrary Time Window: If ED/H visit for Fever occured anytime between 2 days before to 2 days after 
# any neutrophil count measurement was taken between index date (aka prev_visit) and next chemo date (aka visit_date)<br>

# In[150]:


from scripts.config import (event_map, fever_codes, diag_cols)
from scripts.prep_data import (PrepDataCYTO)
from scripts.preprocess import (clean_string)


# In[141]:


def get_fever_visit_indices(partition, delta=2):
    """
    if patient needed to go to Emergency Department/Hospital for fever 
    anytime between 2 days after visit date to 2 days after next chemo visit date (visit_date)
    """
    chemo_df, E = partition
    result = []
    for ikn, chemo_group in tqdm.tqdm(chemo_df.groupby('ikn')):
        arrival_date = E.loc[E['ikn'] == ikn, 'arrival_date']
        for chemo_idx, chemo_row in chemo_group.iterrows():
            earliest_date = chemo_row['visit_date'] + pd.Timedelta(f'{delta} days')
            latest_date = chemo_row['next_visit_date'] + pd.Timedelta(f'{delta} days')
            if arrival_date.between(earliest_date, latest_date).any():
                result.append(chemo_idx)
    return result

def get_febrile_neutropenia_indices(partition, delta=2):
    """
    if patient needed to go to Emergency Department/Hospital for fever 
    anytime between between 2 days before to 2 days after any neutrophil count measurement was taken 
    between 4 days after visit date and next visit date
    AND neutrophil count was below 1.5
    """
    chemo_df, E = partition
    result = []
    for ikn, chemo_group in tqdm.tqdm(chemo_df.groupby('ikn')):
        arrival_date = E.loc[E['ikn'] == ikn, 'arrival_date']
        for chemo_idx, chemo_row in chemo_group.iterrows():
            chemo_interval = int(min(28, chemo_row['chemo_interval']))
            blood_counts = chemo_row.loc[range(4, chemo_interval+1)]
            blood_counts = blood_counts[blood_counts.notnull()]
            if blood_counts.empty: 
                continue
            for blood_day, blood_count in blood_counts.items():
                earliest_date = chemo_row['visit_date'] + pd.Timedelta(f'{blood_day-delta} days')
                latest_date = chemo_row['visit_date'] + pd.Timedelta(f'{blood_day+delta} days')
                if (blood_count < blood_types['neutrophil']['cytopenia_threshold'] and 
                    arrival_date.between(earliest_date, latest_date).any()):
                    result.append(chemo_idx)
    return result


# In[142]:


for event in ['H', 'ED']:
    database_name = event_map[event]['database_name']
    arr_date_col, dep_date_col = event_map[event]['date_col_name']
    df = pd.read_csv(f'{root_path}/data/{database_name}.csv', dtype=str) 
    df = clean_string(df, ['ikn'] + diag_cols)
    df['arrival_date'] = pd.to_datetime(df[arr_date_col])
    df = df[df['arrival_date'].notnull()]
    
    # only keep ED/H visits due to fever
    mask = False
    for diag_col in diag_cols:
        mask |= df[diag_col].isin(fever_codes)
    df = df[mask]
    
    df = df[df['ikn'].isin(chemo_df['ikn'])] # filter away patients not in chemo dataframe
    df = df[['ikn', 'arrival_date']] # keep only selected columns
    df = df.drop_duplicates() # drop duplicates
    df.to_csv(f'{main_dir}/data/{event}_fever.csv', index=False)


# In[143]:


neutrophil_df = pd.read_csv(f'{main_dir}/data/neutrophil.csv')
neutrophil_df.columns = neutrophil_df.columns.astype(int)
chemo_df = load_chemo_df(main_dir)
chemo_df = pd.concat([chemo_df, neutrophil_df], axis=1)
chemo_df['fever'] = False
chemo_df['febrile_neutropenia'] = False


# In[144]:


for event in ['H', 'ED']:
    df = pd.read_csv(f'{main_dir}/data/{event}_fever.csv', dtype=str)
    df['arrival_date'] = pd.to_datetime(df['arrival_date'])
    filtered_chemo_df = chemo_df[chemo_df['ikn'].isin(df['ikn'])]
    indices = split_and_parallelize((filtered_chemo_df, df), get_fever_visit_indices)
    chemo_df.loc[indices, 'fever'] = True
    indices = split_and_parallelize((filtered_chemo_df, df), get_febrile_neutropenia_indices)
    chemo_df.loc[indices, 'febrile_neutropenia'] = True


# In[145]:


pd.DataFrame([chemo_df['fever'].value_counts(), chemo_df['febrile_neutropenia'].value_counts()]).T


# In[151]:


prep = PrepDataCYTO()
model_data = prep.get_data(include_first_date=True, verbose=False)
model_data = prep.regression_to_classification(model_data)
model_data['febrile_neutropenia'] = chemo_df['febrile_neutropenia']
pd.DataFrame([model_data['febrile_neutropenia'].value_counts(), model_data['Neutropenia'].value_counts()]).T


# In[152]:


split_date = '2017-06-30'
train, valid, test = prep.split_data(prep.dummify_data(model_data), split_date=split_date, convert_to_float=False)
(X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test) = train, valid, test
pd.DataFrame([X_train['febrile_neutropenia'].value_counts(), 
              X_valid['febrile_neutropenia'].value_counts(), 
              X_test['febrile_neutropenia'].value_counts()], index=['Train', 'Valid', 'Test']).T
