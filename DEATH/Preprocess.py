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


import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option('display.max_rows', 150)

from src.spark import clean_string as spark_clean_string
from src.config import (root_path, acu_folder, death_folder, max_chemo_date, y3_cols)
from src.utility import (clean_string, get_mapping_from_textfile)
from src.preprocess import (filter_y3_data, get_world_region_of_birth, filter_ohip_data, process_ohip_data)


# In[3]:


# config
output_path = f'{root_path}/{death_folder}'
processes = 32


# #### <b>Reuse the data from Acute Care Use project so we don't perform repetetive preprocessing steps</b>

# In[4]:


get_ipython().system('cp $acu_folder/data/model_data.csv $death_folder/data/model_data.csv')
get_ipython().system('cp $acu_folder/data/H_dates.csv $death_folder/data/H_dates.csv')
get_ipython().system('cp $acu_folder/data/ED_dates.csv $death_folder/data/ED_dates.csv')


# # Create my csvs

# In[5]:


main_filepath = f'{output_path}/data/model_data.csv'
chemo_df = pd.read_csv(main_filepath, dtype=str)
chemo_df['visit_date'] = pd.to_datetime(chemo_df['visit_date'])
print(f"NSessions: {len(chemo_df)}. NPatients: {chemo_df['ikn'].nunique()}")


# ### Include urban vs rural feature

# In[6]:


rural = pd.read_csv(f'{root_path}/data/rural.csv')
rural = clean_string(rural, ['ikn', 'rural'])
rural['rural'].replace({'N': False, 'Y': True}, inplace=True)
chemo_df = pd.merge(chemo_df, rural, on='ikn', how='left')
chemo_df['rural'].fillna(False, inplace=True) # nans are negligible (0.7% prevalence), assume urban


# ### Include world region of birth feature

# In[7]:


immigration = pd.read_csv(f'{root_path}/data/immigration.csv')
cols = ['ikn', 'country_birth', 'nat_language']
immigration = clean_string(immigration, cols)
immigration = immigration[cols]


# In[8]:


# only use data from urban area
mask = immigration['ikn'].map(dict(rural.to_numpy()))
mask.fillna(True, inplace=True)
immigration = immigration[~mask]


# In[9]:


country_code_map = get_mapping_from_textfile(filepath=f'{root_path}/data/country_codes.txt')
immigration = get_world_region_of_birth(immigration, country_code_map)
chemo_df = pd.merge(chemo_df, immigration[['ikn', 'world_region_of_birth']], on='ikn', how='left')
chemo_df['world_region_of_birth'].fillna('Unknown', inplace=True)


# In[10]:


chemo_df['world_region_of_birth'].value_counts()


# ### Include death date features

# In[11]:


y3 = pd.read_csv(f'{root_path}/data/y3.csv')
y3 = filter_y3_data(y3, include_death=True)
y3 = y3[['ikn', 'D_date']]
chemo_df = chemo_df[chemo_df['ikn'].isin(y3['ikn'])]
chemo_df = pd.merge(chemo_df, y3, on='ikn', how='left')


# ### Include last seen date features

# In[12]:


chemo_df['last_seen_date'] = chemo_df['D_date']
chemo_df['last_seen_date'] = chemo_df['last_seen_date'].fillna(max_chemo_date)


# ### Include features from OHIP database
# Get date of palliative care consultation services (PCCS) via physician billing codes (A945 and C945)

# In[67]:


# Extract and Preprocess the OHIP Data
ohip = pd.read_csv(f'{root_path}/data/ohip.csv')
ohip = filter_ohip_data(ohip, billing_codes=['A945', 'C945'])

# Determine the date a patient received (PCCS) for the first time
initial_pccs_date = ohip.groupby('ikn')['servdate'].first()
chemo_df['first_PCCS_date'] = chemo_df['ikn'].map(initial_pccs_date)


# In[71]:


n = chemo_df.loc[chemo_df['first_PCCS_date'].notnull(), 'ikn'].nunique()
N = chemo_df['ikn'].nunique()
print(f"{n} patients out of {N} total patients have received at least one palliative care consultation services (PCCS)")


# In[68]:


chemo_df.to_csv(main_filepath, index=False)
