#!/usr/bin/env python
# coding: utf-8

# use <b>./kevin_launch_jupyter-notebook_webserver.sh</b> instead of <b>launch_jupyter-notebook_webserver.sh</b> if you want to increase buffer memory size so we can load greater filesizes

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


import tqdm
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from functools import partial

from collections import Counter

import scripts.utilities as util
from scripts.config import (root_path, blood_types, all_observations, y3_cols, olis_cols, chemo_df_cols, event_map)
from scripts.preprocess import (shared_dict, split_and_parallelize, clean_string, group_observations,
                                filter_systemic_data, systemic_worker, clean_cancer_and_demographic_data, load_chemo_df, 
                                prefilter_olis_data, olis_worker, postprocess_olis_data,
                                preprocess_esas, get_esas_responses, postprocess_esas_responses,
                                filter_ecog_data, ecog_worker)


# In[7]:


# config
main_dir = f'{root_path}/cytopenia'
processes = 32


# # Selected Regimens

# In[8]:


df = util.read_partially_reviewed_csv()
df = util.get_included_regimen(df)


# In[9]:


# patient count plot
plt.plot(df['patient_count'].astype(int))
plt.xticks(rotation=90)
plt.title('Selected chemo regiments and their number of patients')
plt.show()


# In[10]:


regimen_name_mapping = df['mapping_to_proper_name'].to_dict()
regimen_name_mapping = {mapped_from: mapped_to if mapped_to != 'None' else mapped_from 
                        for mapped_from, mapped_to in regimen_name_mapping.items()}
cycle_lengths = df['cycle_length'].to_dict()


# In[11]:


regimens = df.index 
print(f'len(regiments) will show {len(regimens)} when it should be 33, the papaclicarbo will be renamed to crbppacl, thus we will end up with 33 regiments')


# # Create my csvs

# ### Include features from systemic (chemo) dataset

# In[12]:


systemic = pd.read_csv(f'{root_path}/data/systemic.csv')
systemic = filter_systemic_data(systemic, regimens, regimen_name_mapping)
print(f"Number of patients = {systemic['ikn'].nunique()}")
print(f"Chemotherapy Cohort from {systemic['visit_date'].min()} to {systemic['visit_date'].max()}")


# In[17]:


chemo = split_and_parallelize(systemic, systemic_worker, split_by_ikn=True, processes=16)
cols = systemic.columns.tolist() + ['prev_visit', 'chemo_interval', 'chemo_cycle', 'immediate_new_regimen']
chemo_df = pd.DataFrame(chemo, columns=cols)


# ### include features from y3 (cancer and demographic) dataset

# In[18]:


y3 = pd.read_csv(f'{root_path}/data/y3.csv')
y3 = y3[y3_cols]
y3 = clean_string(y3, ['ikn', 'lhin_cd', 'curr_morph_cd', 'curr_topog_cd', 'sex'])
chemo_df = pd.merge(chemo_df, y3, on='ikn', how='inner')
chemo_df['bdate'] = pd.to_datetime(chemo_df['bdate'])
chemo_df['age'] = chemo_df['prev_visit'].dt.year - chemo_df['bdate'].dt.year
chemo_df = clean_cancer_and_demographic_data(chemo_df, chemo_df_cols)


# In[19]:


chemo_df.to_csv(f'{main_dir}/data/chemo_processed.csv', index=False)


# ### Include features from olis (blood work/lab test observation count) dataset

# In[8]:


chemo_df = load_chemo_df(main_dir)
print(f"Number of rows now: {len(chemo_df)}")
print(f"Number of patients now: {chemo_df['ikn'].nunique()}")
print(f"Number of rows with chemo intervals less than 4 days: {sum(chemo_df['chemo_interval'] < pd.Timedelta('4 days'))}") # some still remained after merging of the intervals


# In[12]:


# Preprocess the Complete Olis Data
chunks = pd.read_csv(f'{root_path}/data/olis_complete.csv', chunksize=10**7, dtype=str) 
for i, chunk in tqdm.tqdm(enumerate(chunks), total=42):
    chunk = prefilter_olis_data(chunk, chemo_df['ikn'])
    # write to csv
    header = True if i == 0 else False
    chunk.to_csv(f"{main_dir}/data/olis_complete.csv", header=header, mode='a', index=False)


# In[44]:


# Extract the Olis Features
olis = pd.read_csv(f"{main_dir}/data/olis_complete.csv", dtype=str) 
olis['ObservationDateTime'] = pd.to_datetime(olis['ObservationDateTime'])
print('Completed Loading Olis CSV File')

# get results
worker = partial(olis_worker, main_dir=main_dir)
result = split_and_parallelize(olis, worker, processes=processes, split_by_ikn=True)

# save results
result = pd.DataFrame(result, columns=['observation_code', 'chemo_idx', 'days_after_chemo', 'observation_count'])
result.to_csv(f'{main_dir}/data/olis_complete2.csv', index=False)


# In[45]:


# Process the Olis Features
olis_df = pd.read_csv(f'{main_dir}/data/olis_complete2.csv')
mapping, missing_df = postprocess_olis_data(chemo_df, olis_df, observations=all_observations)
missing_df


# In[54]:


# group together obs codes with same obs name
freq_map = olis_df['observation_code'].value_counts()
grouped_observations = group_observations(all_observations, freq_map)

for blood_type in blood_types:
    obs_codes = grouped_observations[blood_type]
    for i, obs_code in enumerate(obs_codes):
        if i == 0: df = mapping[obs_code]
        else: df = df.fillna(mapping[obs_code])
    df.to_csv(f'{main_dir}/data/{blood_type}.csv', index=False)


# ### include features from esas (symptom questionnaire) dataset
# Interesting Observation: Chunking is MUCH faster than loading and operating on the whole Esas2 dataset, whereas for Olis it is the opposite: loading the whole Olis dataset and operating on it is much faster than chunking

# In[60]:


# Preprocess the Questionnaire Data
esas = preprocess_esas(chemo_df['ikn'])
esas.to_csv(f'{main_dir}/data/esas.csv', index=False)


# In[65]:


# Extract the Questionnaire Features
esas_chunks = pd.read_csv(f'{main_dir}/data/esas.csv', chunksize=10**6, dtype=str)
result = get_esas_responses(chemo_df, esas_chunks, len_chunks=16)

# save results
result = pd.DataFrame(result, columns=['index', 'symptom', 'severity'])
result.to_csv(f'{main_dir}/data/esas2.csv', index=False)


# In[66]:


# Process the Questionnaire Responses
esas_df = pd.read_csv(f'{main_dir}/data/esas2.csv')
esas_df = postprocess_esas_responses(esas_df)

# put esas responses in chemo_df
chemo_df = chemo_df.join(esas_df, how='left') # ALT WAY: pd.merge(chemo_df, esas, left_index=True, right_index=True, how='left')


# ### include features from ecog dataset

# In[71]:


# Extract the Ecog Grades
ecog = pd.read_csv(f'{root_path}/data/ecog.csv')
ecog = filter_ecog_data(ecog, chemo_df['ikn'])

# filter out patients not in ecog
filtered_chemo_df = chemo_df[chemo_df['ikn'].isin(ecog['ikn'])]

shared_dict['ecog'] = ecog
result = split_and_parallelize(filtered_chemo_df, ecog_worker, split_by_ikn=True)
result = pd.DataFrame(result, columns=['index', 'ecog_grade'])
result.to_csv(f'{main_dir}/data/ecog2.csv', index=False)


# In[72]:


# Process the Ecog Grades
ecog = pd.read_csv(f'{main_dir}/data/ecog2.csv')
ecog = ecog.set_index('index')

# put ecog grade in chemo_df
chemo_df = chemo_df.join(ecog, how='left') # ALT WAY: pd.merge(chemo_df, ecog, left_index=True, right_index=True, how='left')


# ### include blood transfusions

# In[83]:


def filter_blood_transfusion_data(chunk, chemo_ikns, event='H'):
    col, _ = event_map[event]['date_col_name']
    # organize and format columns
    chunk = clean_string(chunk, ['ikn', 'btplate', 'btredbc']) 
    chunk[col] = pd.to_datetime(chunk[col])
    
    # filter patients not in chemo_df
    chunk = chunk[chunk['ikn'].isin(chemo_ikns)]
    
    # filter rows where no transfusions occured
    chunk = chunk[((chunk['btplate'] == 'Y') | (chunk['btplate'] == '1')) | # btplate means blood transfusion - platelet
                  ((chunk['btredbc'] == 'Y') | (chunk['btredbc'] == '1'))]  # btredbc means blood transfusion - red blood cell
    
    # get only the select columns
    chunk = chunk[[col, 'ikn', 'btplate', 'btredbc']]
    
    # sort by date
    chunk = chunk.sort_values(by=col)
    
    return chunk

def blood_transfusion_worker(partition, event='H'):
    database_name = event_map[event]['database_name']
    date_col, _ = event_map[event]['date_col_name']
    bt_data = shared_dict[f'{database_name}_chunk']
    result = []
    for ikn, chemo_group in partition.groupby('ikn'):
        bt_data_specific_ikn = bt_data[bt_data['ikn'] == ikn]
        for i, bt_data_row in bt_data_specific_ikn.iterrows():
            admdate = bt_data_row[date_col]
            earliest_date = chemo_group['prev_visit'] - pd.Timedelta('5 days')
            latest_date = chemo_group['visit_date'] + pd.Timedelta('3 days')
            tmp = chemo_group[(earliest_date <= admdate) & (latest_date >= admdate)]
            for chemo_idx in tmp.index:
                if not pd.isnull(bt_data_row['btplate']): # can only be NaN, Y, or 1
                    result.append((chemo_idx, str(admdate.date()), f'{event}_platelet_transfusion_date'))
                if not pd.isnull(bt_data_row['btredbc']): # can only be NaN, Y, or 1
                    result.append((chemo_idx, str(admdate.date()), f'{event}_hemoglobin_transfusion_date'))
    return result

def extract_blood_transfusion_data(event='H'):
    database_name = event_map[event]['database_name']
    worker = partial(blood_transfusion_worker, event=event)
    chunks = pd.read_csv(f'{root_path}/data/{database_name}_transfusion.csv', chunksize=10**6, dtype=str)
    result = []
    for i, chunk in tqdm.tqdm(enumerate(chunks)):
        chunk = filter_blood_transfusion_data(chunk, chemo_df['ikn'], event=event)

        # filter out patients not in transfusion data
        filtered_chemo_df = chemo_df[chemo_df['ikn'].isin(chunk['ikn'])]

        # get results
        shared_dict[f'{database_name}_chunk'] = chunk
        chunk_result = split_and_parallelize(filtered_chemo_df, worker, split_by_ikn=True)
        result += chunk_result
    
    # save results
    result = pd.DataFrame(result, columns=['chemo_idx', 'transfusion_date', 'transfusion_type'])
    result.to_csv(f'{main_dir}/data/{database_name}_transfusion.csv', index=False)
    
def postprocess_blood_transfusion_data(chemo_df, event='h'):
    database_name = event_map[event]['database_name']
    df = pd.read_csv(f'{main_dir}/data/{database_name}_transfusion.csv')
    for transfusion_type, group in df.groupby('transfusion_type'):
        chemo_indices = group['chemo_idx'].values.astype(int)
        dates = group['transfusion_date'].values
        chemo_df.loc[chemo_indices, transfusion_type] = dates
    return chemo_df, df


# #### dad_transfusion 
# (hospitalization visit where blood transfusion was administered)

# In[84]:


# Extract Hospital Blood Transfusion Events During Chemotherapy
extract_blood_transfusion_data(event='H')


# In[85]:


# Process the Hospital Blood Transfusion Events
chemo_df, h_bt_df = postprocess_blood_transfusion_data(chemo_df, event='H')


# #### nacrs transfusion 
# (Emergency Room / Emergency Department Visit where Blood Tranfusion was Administered)

# In[86]:


# Extract ER Blood Transfusion Events During Chemotherapy
extract_blood_transfusion_data(event='ED')


# In[87]:


# Process the ER Blood Transfusion Events
chemo_df, ed_bt_df = postprocess_blood_transfusion_data(chemo_df, event='ED')


# ### exclude sessions with odb_growth factors

# In[88]:


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
    odb = shared_dict['odb']
    result = set()
    for ikn, group in partition.groupby('ikn'):
        odb_specific_ikn = odb[odb['ikn'] == ikn]
        for i, odb_row in odb_specific_ikn.iterrows():
            servdate = odb_row['servdate']
            mask = (servdate <= group['visit_date']) & (servdate >= group['prev_visit'])
            result.update(group[mask].index)
    return result


# In[89]:


# Extract ODB Growth Factor Administration Events During Chemotherapy
odb = pd.read_csv(f'{root_path}/data/odb_growth_factors.csv')
odb = filter_odb_data(odb, chemo_df['ikn'])

# filter out patients not in ecog
filtered_chemo_df = chemo_df[chemo_df['ikn'].isin(odb['ikn'])]

shared_dict['odb'] = odb
result = split_and_parallelize(filtered_chemo_df, odb_worker)
np.save(f'{main_dir}/data/odb_indices.npy', result)


# In[91]:


# Process the ODB Growth Factor Administration Events
indices = np.load(f'{main_dir}/data/odb_indices.npy')
chemo_df['ODBGF_given'] = False
chemo_df.loc[indices, 'ODBGF_given'] = True


# In[93]:


chemo_df.to_csv(f'{main_dir}/data/chemo_processed2.csv', index=False)


# Scratch Notes

# ### Gaussian playground

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


# ## systemic

# In[12]:


df = pd.read_csv(f'{root_path}/data/systemic.csv')
df = clean_string(df, ['ikn'])
    
# get occurence of regiments based on number of rows (inlcudes all the different drug counts)
regimen_count_by_rows = Counter(dict(df['regimen'].value_counts()))
    
# get occurence of regiments based on number of chemo regiments
df = df[['ikn', 'regimen', 'visit_date']]
df = df.drop_duplicates()
regimen_count_by_regimens = Counter(dict(df['regimen'].value_counts()))
    
# get occurence of regiments based on number of patients
regimen_count_by_patients = {regimen: len(group['ikn'].unique()) for regimen, group in df.groupby('regimen')}
regimen_count_by_patients = sorted(regimen_count_by_patients.items(), key=lambda x: x[1], reverse=True)
del df


# In[13]:


def plot_regimen_hist(regimens, count, fig, title, idx):
    ax = fig.add_subplot(1,3,idx)
    plt.title(title)
    plt.bar(regimens, count)
    plt.xticks(rotation=90)
    
fig = plt.figure(figsize=(15, 3))
regimens, count = zip(*regimen_count_by_rows.most_common(n=20))
plot_regimen_hist(regimens, count, fig, title='Top 20 chemo regimen occurence\n based on number of rows', idx=1)

regimens, count = zip(*regimen_count_by_regimens.most_common(n=20))
plot_regimen_hist(regimens, count, fig, title='Top 20 chemo regimen occurence\n based on number of chemo sessions', idx=2)

regimens, count = zip(*regimen_count_by_patients[0:20])
plot_regimen_hist(regimens, count, fig, title='Top 20 chemo regimen occurence\n based on number of patients', idx=3)

