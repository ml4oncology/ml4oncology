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
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from functools import partial

from scripts.config import (root_path, regiments_folder, can_folder, all_observations)
from scripts.spark import (preprocess_olis_data)
from scripts.preprocess import (split_and_parallelize, 
                                load_included_regimens, load_chemo_df,
                                filter_systemic_data, filter_by_drugs, process_systemic_data,
                                filter_y3_data, process_cancer_and_demographic_data, 
                                filter_immigration_data, process_immigration_data, filter_combordity_data, 
                                filter_dialysis_data, process_dialysis_data,
                                olis_worker, closest_measurement_worker, postprocess_olis_data,
                                filter_esas_data, get_esas_responses, postprocess_esas_responses,
                                filter_body_functionality_data, body_functionality_worker)


# In[3]:


# config
processes = 32
main_dir = f'{root_path}/{can_folder}'
if not os.path.exists(f'{main_dir}/data'):
    os.makedirs(f'{main_dir}/data')


# # Selected Regimens

# In[4]:


regimens = load_included_regimens(criteria='cisplatin_containing')
only_first_day_regimens = regimens.loc[regimens['notes'] == 'cisplatin only on day 1', 'regimen']
regimens_renamed = sorted(regimens['relabel'].fillna(regimens['regimen']).unique())
print(f'{len(regimens)} raw regimens -> {len(regimens_renamed)} relabeled total regimens')
regimens_renamed


# # Create my csvs

# ### Include features from systemic (chemo) dataset

# In[8]:


systemic = pd.read_csv(f'{root_path}/data/systemic.csv')
systemic = filter_systemic_data(systemic, regimens, remove_inpatients=False, exclude_dins=False, 
                                include_drug_info=True, verbose=True)
print(f"Number of patients = {systemic['ikn'].nunique()}")
print(f"Chemotherapy Cohort from {systemic['visit_date'].min()} to {systemic['visit_date'].max()}")
print(f"Size of data = {len(systemic)}")


# In[9]:


systemic = filter_by_drugs(systemic, drug='cisplatin', verbose=True)
print(f"Number of patients = {systemic['ikn'].nunique()}")
print(f"Size of data = {len(systemic)}")


# In[10]:


systemic = process_systemic_data(systemic)
systemic.to_csv(f'{main_dir}/data/systemic.csv', index=False)
print(f"Number of patients = {systemic['ikn'].nunique()}")
print(f"Number of unique regiments = {systemic['regimen'].nunique()}")
print(f"Size of data = {len(systemic)}")


# In[7]:


systemic = pd.read_csv(f'{main_dir}/data/systemic.csv', dtype={'ikn': str})
for col in ['visit_date', 'next_visit_date']: systemic[col] = pd.to_datetime(systemic[col])

# for regimens where cisplatin is only provided only on the first sessions, remove the sessions thereafter
mask = systemic['regimen'].isin(only_first_day_regimens)
drop_indices = []
for ikn, group in tqdm.tqdm(systemic[mask].groupby('ikn')):
    drop_indices += group.index[group['regimen'].duplicated()].tolist()
print(f'Dropped {len(drop_indices)} where no cisplatin was administered after the first session')
systemic = systemic.drop(index=drop_indices)

# adjust columns
systemic = systemic.rename(columns={'dose_administered': 'cisplatin_dosage'})
systemic = systemic.drop(columns=['drug', 'measurement_unit'])

# convert dosage to dosage per body surface area
systemic['cisplatin_dosage'] = systemic['cisplatin_dosage'] / systemic['body_surface_area'] # mg / m^2
systemic = systemic[~systemic['cisplatin_dosage'].isin([np.nan, np.inf])] # remove nan and inf dosage values

print(f"Number of patients = {systemic['ikn'].nunique()}")
print(f"Size of data {len(systemic)}")


# ### Include features from y3 (cancer and demographic) dataset

# In[8]:


col_arrangement = ['ikn', 'regimen', 'visit_date', 'next_visit_date', 'chemo_interval', 'days_since_starting_chemo', 'days_since_last_chemo', 
                   'chemo_cycle', 'immediate_new_regimen', 'intent_of_systemic_treatment', 'line_of_therapy', 'lhin_cd', 
                   'curr_morph_cd', 'curr_topog_cd', 'age', 'sex', 'body_surface_area', 'cisplatin_dosage']


# In[9]:


# Extract and Preprocess the Y3 Data
y3 = pd.read_csv(f'{root_path}/data/y3.csv')
y3 = filter_y3_data(y3)
print(f"Number of patients in y3 = {y3['ikn'].nunique()}")
print(f"Number of patients in y3 and systemic = {y3['ikn'].isin(systemic['ikn']).sum()}")


# In[10]:


# Process the Y3 and Systemic Data
chemo_df = process_cancer_and_demographic_data(y3, systemic, exclude_blood_cancers=False, verbose=True)
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


# ### Include features from combordity dataset

# In[12]:


# Extract and Preprocess the Combordity Data
combordity = pd.read_csv(f'{root_path}/data/combordity.csv')
combordity = filter_combordity_data(combordity)

# Process the Combordity Data
chemo_df = pd.merge(chemo_df, combordity, how='left', on='ikn')
for col in ['hypertension', 'diabetes']:
    chemo_df[col] = chemo_df[col] < chemo_df['visit_date']


# ### Include features from dialysis dataset

# In[13]:


# Extract and Preprocess the Dialysis Data
dialysis = pd.read_csv(f'{root_path}/data/dialysis_ohip.csv')
dialysis = filter_dialysis_data(dialysis)

# Process the Dialysis Data
chemo_df = process_dialysis_data(chemo_df, dialysis)


# In[14]:


chemo_df.to_csv(f'{main_dir}/data/chemo_processed.csv', index=False)


# ### Include features from olis (blood work/lab test observation count) dataset

# In[5]:


scr_obs_code = '14682-9' # SCr: Serum Creatinine
earliest = -30 # days
latest = 28 # days


# In[6]:


chemo_df = load_chemo_df(main_dir)
print(f"Number of rows now: {len(chemo_df)}")
print(f"Number of patients now: {chemo_df['ikn'].nunique()}")


# In[17]:


get_ipython().run_cell_magic('time', '', "# Preprocess the Raw OLIS Data using PySpark\npreprocess_olis_data(f'{main_dir}/data', set(chemo_df['ikn']))")


# In[7]:


# Extract the Olis Features
olis = pd.read_csv(f"{main_dir}/data/olis.csv", dtype=str) 
olis['ObservationDateTime'] = pd.to_datetime(olis['ObservationDateTime'])

# save serum creatinine observations separately
olis_scr = olis[olis['ObservationCode'] == scr_obs_code]
olis_scr.to_csv(f'{main_dir}/data/olis_scr.csv', index=False)

# get results
filtered_chemo_df = chemo_df[chemo_df['ikn'].isin(olis['ikn'])] # filter out patients not in dataset
worker = partial(olis_worker, earliest_limit=earliest, latest_limit=latest)
result = split_and_parallelize((filtered_chemo_df, olis), worker, processes=processes)
result = pd.DataFrame(result, columns=['observation_code', 'chemo_idx', 'days_after_chemo', 'observation_count'])
result.to_csv(f'{main_dir}/data/olis2.csv', index=False)


# In[8]:


# Process the Olis Features
olis_df = pd.read_csv(f'{main_dir}/data/olis2.csv')
mapping, missing_df = postprocess_olis_data(chemo_df, olis_df, observations=all_observations, 
                                            days_range=range(earliest,latest+1))
missing_df


# In[9]:


mapping[scr_obs_code].to_csv(f'{main_dir}/data/serum_creatinine.csv', index=False)


# In[10]:


# Extract the closest Serum Creatinine measurement 9-24 months after treatment
olis_scr = pd.read_csv(f'{main_dir}/data/olis_scr.csv', dtype=str)
olis_scr['ObservationDateTime'] = pd.to_datetime(olis_scr['ObservationDateTime'])
olis_scr = olis_scr.sort_values(by='ObservationDateTime')

# get results
filtered_chemo_df = chemo_df[chemo_df['ikn'].isin(olis_scr['ikn'])] # filter out patients not in dataset
worker = partial(closest_measurement_worker, days_after=270)
result = split_and_parallelize((filtered_chemo_df, olis_scr), worker)
result = pd.DataFrame(result, columns=['index', 'days_after_chemo', 'next_SCr_count'])
result.to_csv(f'{main_dir}/data/olis_scr2.csv', index=False)


# In[11]:


# Process the serum creatinine measruements
olis_scr = pd.read_csv(f'{main_dir}/data/olis_scr2.csv')
olis_scr = olis_scr.set_index('index')
chemo_df = chemo_df.join(olis_scr['next_SCr_count'], how='left')


# ### Include features from esas (symptom questionnaire) dataset

# In[63]:


# Preprocess the Questionnaire Data
esas = pd.read_csv(f"{root_path}/data/esas.csv")
esas = filter_esas_data(esas, chemo_df['ikn'])
esas.to_csv(f'{main_dir}/data/esas.csv', index=False)


# In[64]:


# Extract the Questionnaire Features
esas = pd.read_csv(f'{main_dir}/data/esas.csv', dtype=str)
result = get_esas_responses(chemo_df, esas, processes=processes)
result = pd.DataFrame(result, columns=['index', 'symptom', 'severity'])
result.to_csv(f'{main_dir}/data/esas2.csv', index=False)


# In[12]:


# Process the Questionnaire Responses
esas_df = pd.read_csv(f'{main_dir}/data/esas2.csv')
esas_df = postprocess_esas_responses(esas_df)

# put esas responses in chemo_df
chemo_df = chemo_df.join(esas_df, how='left')


# ### Include features from ecog and prfs (body functionality grade) dataset

# In[66]:


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


# In[13]:


for dataset in ['ecog', 'prfs']:
    # Process the results
    bf = pd.read_csv(f'{main_dir}/data/{dataset}.csv')
    bf = bf.set_index('index')

    # put result in chemo_df
    chemo_df = chemo_df.join(bf, how='left') # ALT WAY: pd.merge(chemo_df, ecog, left_index=True, right_index=True, how='left')


# In[14]:


chemo_df.to_csv(f'{main_dir}/data/model_data.csv', index=False)


# # Scratch Notes

# ### Showcase multiple dosage on same day

# In[16]:


from scripts.config import cisplatin_dins as keep_dins, cisplatin_cco_drug_code as keep_cco_drug_code

df = systemic[systemic['din'].isin(keep_dins) | systemic['cco_drug_code'].isin(keep_cco_drug_code)]
df['drug'] = 'cisplatin'
df = df.drop(columns=['din', 'cco_drug_code'])
df = df[df['measurement_unit'].isin(['mg', 'MG'])] # remove rows with measurement unit of g, unit, mL, nan

cols = df.columns.drop(['dose_administered'])
df[df.duplicated(subset=cols, keep=False)].tail(n=60)


# ### Showcase fractionated dosage

# In[31]:


df = systemic.sort_values(by=['ikn', 'visit_date']).copy()
df['prev_visit'] = df['visit_date'].shift() # WARNING: Assumes visit date has already been sorted
df.loc[~df['regimen'].eq(df['regimen'].shift()), 'prev_visit'] = pd.NaT # break off when chemo regimen changes
df['chemo_interval'] = df['visit_date'] - df['prev_visit']
df[df['ikn'].isin(df.loc[df['chemo_interval'] < 5, 'ikn'])].tail(n=60)
