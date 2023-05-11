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
import numpy as np

from src.config import root_path, can_folder, all_observations
from src.utility import load_included_regimens
from src.preprocess import (
    Systemic, Demographic, Laboratory, Symptoms, BloodTransfusion,
    combine_demographic_data, combine_lab_data, combine_symptom_data,
    process_dialysis_data
)


# In[3]:


# config
# NOTE: environment 64G memory, 32 cores
output_path = f'{root_path}/{can_folder}'
processes = 32


# # Selected Regimens

# In[4]:


regimens = load_included_regimens(criteria='cisplatin_containing')
cycle_length_map = dict(regimens[['regimen', 'shortest_cycle_length']].values)
cycle_length_map['Other'] = 7.0 # make rare regimen cycle lengths default to 7
day_one_regimens = regimens.loc[regimens['notes'] == 'cisplatin only on day 1', 'regimen']
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

# In[6]:


get_ipython().run_cell_magic('time', '', "syst = Systemic()\ndf = syst.run(\n    regimens, \n    drug='cisplatin',\n    filter_kwargs={'remove_inpatients': False, 'verbose': True}, \n    process_kwargs={'cycle_length_map': cycle_length_map, 'day_one_regimens': day_one_regimens}\n)\ndf.to_parquet(f'{output_path}/data/systemic.parquet.gzip', compression='gzip', index=False)")


# In[8]:


df = pd.read_parquet(f'{output_path}/data/systemic.parquet.gzip')
quick_summary(df)


# ### Include features from demographic data
# Includes cancer diagnosis, income, immigration, area of residence, etc

# In[9]:


demog = Demographic()
demo_df = demog.run(exclude_blood_cancer=False)


# In[10]:


df = combine_demographic_data(df, demo_df)
drop_cols = ['inpatient_flag', 'ethnic', 'country_birth', 'official_language', 'nat_language']
df = df.drop(columns=drop_cols)
quick_summary(df)


# In[11]:


df['world_region_of_birth'].value_counts()


# ### Include features from dialysis dataset

# In[12]:


get_ipython().run_cell_magic('time', '', 'df = process_dialysis_data(df)')


# ### Include features from lab test data

# In[13]:


labr = Laboratory(f'{output_path}/data', processes)


# In[14]:


get_ipython().run_cell_magic('time', '', "# NOTE: need to restart kerel after this point to avoid annoying IOStream.flush timed out messages\nlabr.preprocess(set(df['ikn']))")


# In[14]:


get_ipython().run_cell_magic('time', '', "lab_df = labr.run(df, time_window=(-30, 28))\nlab_df.to_parquet(f'{output_path}/data/lab.parquet.gzip', compression='gzip', index=False)")


# In[15]:


get_ipython().run_cell_magic('time', '', "lab_df = pd.read_parquet(f'{output_path}/data/lab.parquet.gzip')\ndf, lab_map, missing_df = combine_lab_data(df, lab_df)\nmissing_df")


# ####  Serum Creatinine

# In[16]:


scr_obs_code = '14682-9' # SCr: Serum Creatinine


# In[17]:


lab_map[scr_obs_code].columns = lab_map[scr_obs_code].columns.astype(str)
lab_map[scr_obs_code].to_parquet(f'{output_path}/data/creatinine.parquet.gzip', compression='gzip', index=False)


# In[18]:


get_ipython().run_cell_magic('time', '', "scr = labr.run(df, obs_codes=[scr_obs_code], get_closest_obs=True, days_after=270)\nscr = scr.rename(columns={'obs_value': 'next_SCr_value'}).set_index('chemo_idx')\ndf = df.join(scr['next_SCr_value'], how='left')")


# ### Include features from symptom data

# In[19]:


get_ipython().run_cell_magic('time', '', "symp = Symptoms(processes=processes)\nsymp_df = symp.run(df)\nsymp_df.to_parquet(f'{output_path}/data/symptom.parquet.gzip', compression='gzip', index=False)")


# In[20]:


symp_df = pd.read_parquet(f'{output_path}/data/symptom.parquet.gzip')
df = combine_symptom_data(df, symp_df)


# In[21]:


df.to_parquet(f'{output_path}/data/final_data.parquet.gzip', compression='gzip', index=False)


# In[ ]:
