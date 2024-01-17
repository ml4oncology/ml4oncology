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


get_ipython().run_line_magic('cd', '../../')
# reloads all modules everytime before cell is executed (no need to restart kernel)
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[4]:


from tqdm import tqdm
import pandas as pd
pd.set_option('display.max_rows', 100)
import numpy as np

from src.config import (
    root_path, sas_folder, cyto_folder, acu_folder, 
    blood_types, neutrophil_dins, all_observations, event_map
)

from src.utility import load_included_regimens, group_observations
from src.preprocess import (
    Systemic, Demographic, Laboratory, Symptoms, BloodTransfusion,
    combine_demographic_data, combine_lab_data, combine_symptom_data,
    process_odb_data,
)


# In[5]:


# config
# NOTE: environment 64G memory, 32 cores
output_path = f'{root_path}/projects/{cyto_folder}'
processes = 32


# # Selected Regimens

# In[4]:


regimens = load_included_regimens(criteria='cytotoxic')
cycle_length_map = dict(regimens[['regimen', 'shortest_cycle_length']].values)
cycle_length_map['Other'] = 7.0 # make rare regimen cycle lengths default to 7
max_cycle_length = int(regimens['shortest_cycle_length'].max())
regimens_renamed = sorted(regimens['relabel'].fillna(regimens['regimen']).unique())
print(f'{len(regimens)} raw regimens -> {len(regimens_renamed)} relabeled total regimens')
regimens_renamed


# # Create my dataset

# In[5]:


def quick_summary(df):
    print(f"Number of treatment sessions = {len(df)}")
    print(f"Number of patients = {df['ikn'].nunique()}")
    print(f"Number of unique regiments = {df['regimen'].nunique()}")
    print(f"Cohort from {df['visit_date'].min().date()} to {df['visit_date'].max().date()}")


# ### Include features from systemic therapy treatment data
# NOTE: ikn is the encoded ontario health insurance plan (OHIP) number of a patient

# In[69]:


get_ipython().run_cell_magic('time', '', "syst = Systemic()\ndf = syst.run(\n    regimens, \n    filter_kwargs={'exclude_dins': neutrophil_dins, 'remove_inpatients': False, 'verbose': True}, \n    process_kwargs={'cycle_length_map': cycle_length_map}\n)\ndf.to_parquet(f'{output_path}/data/systemic.parquet.gzip', compression='gzip', index=False)")


# In[70]:


df = pd.read_parquet(f'{output_path}/data/systemic.parquet.gzip')
quick_summary(df)


# ### Include features from demographic data
# Includes cancer diagnosis, income, immigration, area of residence, etc

# In[73]:


demog = Demographic()
demo_df = demog.run()


# In[74]:


df = combine_demographic_data(df, demo_df)
drop_cols = ['inpatient_flag', 'ethnic', 'country_birth', 'official_language', 'nat_language']
df = df.drop(columns=drop_cols)
quick_summary(df)


# In[75]:


df['world_region_of_birth'].value_counts()


# ### Include features from lab test data

# In[76]:


labr = Laboratory(f'{output_path}/data', processes)


# In[30]:


get_ipython().run_cell_magic('time', '', "# NOTE: need to restart kerel after this point to avoid annoying IOStream.flush timed out messages\nlabr.preprocess(set(df['ikn']))")


# In[11]:


get_ipython().run_cell_magic('time', '', "lab_df = labr.run(df, time_window=(-5, max_cycle_length))\nlab_df.to_parquet(f'{output_path}/data/lab.parquet.gzip', compression='gzip', index=False)")


# In[77]:


get_ipython().run_cell_magic('time', '', "lab_df = pd.read_parquet(f'{output_path}/data/lab.parquet.gzip')\ndf, lab_map, count = combine_lab_data(df, lab_df)\ncount")


# In[84]:


# save all the main blood measurements taken within the time window of each treatment session
grouped_obs = group_observations(all_observations, lab_df['obs_code'].value_counts())
counts = {} 
for bt in tqdm(blood_types):
    for i, obs_code in enumerate(grouped_obs[bt]):
        obs = lab_map[obs_code] if i == 0 else obs.fillna(lab_map[obs_code])
    obs.columns = obs.columns.astype(str)
    obs.to_parquet(f'{output_path}/data/{bt}.parquet.gzip', compression='gzip', index=False)
    counts[bt] = obs.notnull().sum(axis=1).value_counts().sort_index()
# How many treatment sessions had x number of measurements in their time window
counts = pd.DataFrame(counts).fillna(0).astype(int)
counts.index.name = 'Number of Measurements'
counts


# ### Include features from symptom data

# In[14]:


get_ipython().run_cell_magic('time', '', "symp = Symptoms(processes=processes)\nsymp_df = symp.run(df)\nsymp_df.to_parquet(f'{output_path}/data/symptom.parquet.gzip', compression='gzip', index=False)")


# In[78]:


symp_df = pd.read_parquet(f'{output_path}/data/symptom.parquet.gzip')
df = combine_symptom_data(df, symp_df)


# ### Include features from ontario drug benefit
# use of growth factors

# In[79]:


get_ipython().run_cell_magic('time', '', 'df = process_odb_data(df)')


# In[80]:


df['GF_given'].value_counts()


# In[81]:


df.to_parquet(f'{output_path}/data/final_data.parquet.gzip', compression='gzip', index=False)


# ## Get blood transfusions during acute care use

# In[82]:


get_ipython().run_cell_magic('time', '', "bt = BloodTransfusion(f'{output_path}/data')\nbt.run(df.reset_index())")


# # Scratch Notes

# In[83]:


np.save(f'{output_path}/analysis/orig_chemo_idx.npy', df.index)
