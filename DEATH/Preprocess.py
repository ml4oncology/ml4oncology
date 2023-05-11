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
pd.set_option('display.max_rows', 150)

from src.config import root_path, acu_folder, death_folder, max_chemo_date, event_map
from src.preprocess import filter_ohip_data


# In[3]:


# config
output_path = f'{root_path}/{death_folder}'
processes = 32


# #### <b>Reuse the data from PROACCT so we don't perform repetetive preprocessing steps</b>

# In[5]:


get_ipython().system('cp $acu_folder/data/final_data.parquet.gzip $death_folder/data/final_data.parquet.gzip')
get_ipython().system('cp $acu_folder/data/ED.parquet.gzip $death_folder/data/ED.parquet.gzip')
get_ipython().system('cp $acu_folder/data/H.parquet.gzip $death_folder/data/H.parquet.gzip')


# # Create my dataset

# In[4]:


main_filepath = f'{output_path}/data/final_data.parquet.gzip'
chemo_df = pd.read_parquet(main_filepath)
print(f"NSessions: {len(chemo_df)}. NPatients: {chemo_df['ikn'].nunique()}")


# ### Include last seen date features

# In[6]:


chemo_df['last_seen_date'] = chemo_df['death_date'].fillna(max_chemo_date)


# ### Include features from OHIP database
# Get date of palliative care consultation services (PCCS) via physician billing codes (A945 and C945)

# In[9]:


# Extract and Preprocess the OHIP Data
ohip = pd.read_parquet(f'{root_path}/data/ohip.parquet.gzip')
ohip = filter_ohip_data(ohip, billing_codes=['A945', 'C945'])

# Determine the date a patient received (PCCS) for the first time
initial_pccs_date = ohip.groupby('ikn')['servdate'].first()
chemo_df['first_PCCS_date'] = chemo_df['ikn'].map(initial_pccs_date)


# In[10]:


n = chemo_df.loc[chemo_df['first_PCCS_date'].notnull(), 'ikn'].nunique()
N = chemo_df['ikn'].nunique()
print(f"{n} patients out of {N} total patients have received at least one palliative care consultation services (PCCS)")


# In[11]:


chemo_df.to_parquet(main_filepath, compression='gzip', index=False)
