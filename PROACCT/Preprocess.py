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


import tqdm
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from functools import partial

from scripts.config import (root_path, acu_folder, event_map)
from scripts.spark import (preprocess_olis_data)
from scripts.preprocess import (split_and_parallelize,
                                load_chemo_df, load_included_regimens, 
                                filter_systemic_data, process_systemic_data,
                                filter_y3_data, process_cancer_and_demographic_data, 
                                filter_immigration_data, process_immigration_data,
                                observation_worker, postprocess_olis_data,
                                filter_esas_data, get_esas_responses, postprocess_esas_responses,
                                filter_body_functionality_data, body_functionality_worker,
                                filter_event_data, extract_event_dates,
                                get_inpatient_indices)


# In[3]:


# config
output_path = f'{root_path}/{acu_folder}'
processes = 32


# # Selected Regimens

# In[4]:


regimens = load_included_regimens()
regimens_renamed = sorted(regimens['relabel'].fillna(regimens['regimen']).unique())
print(f'{len(regimens)} raw regimens -> {len(regimens_renamed)} relabeled total regimens')
regimens_renamed


# # Create my csvs

# ### Include features from systemic (chemo) dataset
# NOTE: ikn is the encoded ontario health insurance plan (OHIP) number of a patient. All ikns are valid (all patients have valid OHIP) in systemic.csv per the valikn column

# In[5]:


systemic = pd.read_csv(f'{root_path}/data/systemic.csv')
systemic = filter_systemic_data(systemic, regimens, exclude_dins=False, verbose=True)


# In[6]:


systemic = process_systemic_data(systemic, method='one-per-week')
systemic.to_csv(f'{output_path}/data/systemic.csv', index=False)
print(f"Number of patients = {systemic['ikn'].nunique()}")
print(f"Number of unique regiments = {systemic['regimen'].nunique()}")
print(f"Chemotherapy Cohort from {systemic['visit_date'].min()} to {systemic['visit_date'].max()}")


# In[7]:


systemic = pd.read_csv(f'{output_path}/data/systemic.csv', dtype={'ikn': str})
systemic['visit_date'] = pd.to_datetime(systemic['visit_date'])


# ### Include features from y3 (cancer and demographic) dataset

# In[8]:


col_arrangement = ['ikn', 'regimen', 'visit_date', 'days_since_starting_chemo', 'days_since_last_chemo', 'chemo_cycle', 
                   'immediate_new_regimen', 'intent_of_systemic_treatment', 'line_of_therapy', 'lhin_cd', 
                   'curr_morph_cd', 'curr_topog_cd', 'age', 'sex', 'body_surface_area']


# In[9]:


# Extract and Preprocess the Y3 Data
y3 = pd.read_csv(f'{root_path}/data/y3.csv')
y3 = filter_y3_data(y3)
print(f"Number of patients in y3 = {y3['ikn'].nunique()}")
print(f"Number of patients in y3 and systemic = {y3['ikn'].isin(systemic['ikn']).sum()}")


# In[10]:


# Process the Y3 and Systemic Data
chemo_df = process_cancer_and_demographic_data(y3, systemic, verbose=True)
chemo_df = chemo_df[col_arrangement]
print(f"Number of unique regiments = {chemo_df['regimen'].nunique()}")
print(f"Number of patients = {chemo_df['ikn'].nunique()}")
print(f"Number of female patients = {chemo_df.loc[chemo_df['sex'] == 'F', 'ikn'].nunique()}")
print(f"Number of male patients = {chemo_df.loc[chemo_df['sex'] == 'M', 'ikn'].nunique()}")


# ### Include features from immigration dataset

# In[11]:


# Extract and Preprocess the Immigration Data
immigration = pd.read_csv(f'{root_path}/data/immigration.csv')
immigration = filter_immigration_data(immigration)

# Process the Immigration Data
chemo_df = process_immigration_data(chemo_df, immigration)
chemo_df.to_csv(f'{output_path}/data/chemo_processed.csv', index=False)


# ### Include features from olis (lab test observations) dataset

# In[5]:


chemo_df = load_chemo_df(output_path, includes_next_visit=False)
print(f"Number of rows now: {len(chemo_df)}")
print(f"Number of patients now: {chemo_df['ikn'].nunique()}")


# In[28]:


get_ipython().run_cell_magic('time', '', "# Preprocess the Raw OLIS Data using PySpark\npreprocess_olis_data(f'{output_path}/data', set(chemo_df['ikn']))")


# In[6]:


# Extract the OLIS Features
olis = pd.read_csv(f"{output_path}/data/olis.csv", dtype=str) 
olis['ObservationDateTime'] = pd.to_datetime(olis['ObservationDateTime'])
filtered_chemo_df = chemo_df[chemo_df['ikn'].isin(olis['ikn'])] # filter out patients not in dataset
result = split_and_parallelize((filtered_chemo_df, olis), observation_worker, processes=processes)
result = pd.DataFrame(result, columns=['observation_code', 'chemo_idx', 'days_after_chemo', 'observation_count'])
result.to_csv(f'{output_path}/data/olis2.csv', index=False)


# In[7]:


# Process the OLIS Features
olis_df = pd.read_csv(f'{output_path}/data/olis2.csv')
mapping, missing_df = postprocess_olis_data(chemo_df, olis_df, days_range=range(-5,1))
missing_df


# ### Include features from esas (questionnaire) dataset
# Interesting Observation: Chunking is MUCH faster than loading and operating on the whole ESAS dataset

# In[64]:


# Preprocess the Questionnaire Data
esas = pd.read_csv(f"{root_path}/data/esas.csv")
esas = filter_esas_data(esas, chemo_df['ikn'])
esas.to_csv(f'{output_path}/data/esas.csv', index=False)


# In[65]:


# Extract the Questionnaire Features
esas = pd.read_csv(f'{output_path}/data/esas.csv', dtype=str)
result = get_esas_responses(chemo_df, esas, processes=processes)
result = pd.DataFrame(result, columns=['index', 'symptom', 'severity'])
result.to_csv(f'{output_path}/data/esas2.csv', index=False)


# In[8]:


# Process the Questionnaire responses
esas_df = pd.read_csv(f'{output_path}/data/esas2.csv')
esas_df = postprocess_esas_responses(esas_df)

# put esas responses in chemo_df
chemo_df = chemo_df.join(esas_df, how='left') # ALT WAY: pd.merge(chemo_df, esas, left_index=True, right_index=True, how='left')


# ### Include features from ecog and prfs (body functionality grade) dataset

# In[67]:


for dataset in ['ecog', 'prfs']:
    # Extract and Preprocess the body functionality dataset
    bf = pd.read_csv(f'{root_path}/data/{dataset}.csv')
    bf = filter_body_functionality_data(bf, chemo_df['ikn'], dataset=dataset)
    
    # get results
    filtered_chemo_df = chemo_df[chemo_df['ikn'].isin(bf['ikn'])] # filter out patients not in dataset
    worker = partial(body_functionality_worker, dataset=dataset)
    result = split_and_parallelize((filtered_chemo_df, bf), worker, processes=processes)
    result = pd.DataFrame(result, columns=['index', f'{dataset}_grade'])
    result.to_csv(f'{output_path}/data/{dataset}.csv', index=False)


# In[9]:


for dataset in ['ecog', 'prfs']:
    # Process the results
    bf = pd.read_csv(f'{output_path}/data/{dataset}.csv')
    bf = bf.set_index('index')

    # put result in chemo_df
    chemo_df = chemo_df.join(bf, how='left') # ALT WAY: pd.merge(chemo_df, ecog, left_index=True, right_index=True, how='left')


# In[10]:


chemo_df.to_csv(f'{output_path}/data/model_data.csv', index_label='index')


# ## Get ED/H Events from dad and nacrs dataset

# In[7]:


for event in ['H', 'ED']:
    # Preprocess the event data
    database_name = event_map[event]['database_name']
    event_df = pd.read_csv(f'{root_path}/data/{database_name}.csv', dtype=str)
    event_df = filter_event_data(event_df, chemo_df['ikn'], event=event)
    event_df.to_csv(f'{output_path}/data/{database_name}.csv', index=False)


# In[8]:


for event in ['H', 'ED']:
    # Extract the event dates
    database_name = event_map[event]['database_name']
    event_df = pd.read_csv(f'{output_path}/data/{database_name}.csv', dtype=str)
    for col in ['arrival_date', 'depart_date']: event_df[col] = pd.to_datetime(event_df[col])
    event_dates = extract_event_dates(chemo_df, event_df, output_path, event=event)


# ### Extra - Find errorneous inpatients

# In[74]:


dad = pd.read_csv(f'{output_path}/data/dad.csv', dtype=str)
for col in ['arrival_date', 'depart_date']: dad[col] = pd.to_datetime(dad[col])
filtered_chemo_df = chemo_df[chemo_df['ikn'].isin(dad['ikn'])] # filter out patients not in dataset
indices = split_and_parallelize((filtered_chemo_df, dad), get_inpatient_indices, processes=processes)
np.save(f'{output_path}/data/inpatient_indices.npy', indices)
