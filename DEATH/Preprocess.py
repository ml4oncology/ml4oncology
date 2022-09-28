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
from pyspark.sql import SparkSession

from src.spark import clean_string as spark_clean_string
from src.config import (root_path, acu_folder, death_folder, max_chemo_date, y3_cols)
from src.utility import (clean_string)
from src.preprocess import (filter_y3_data, filter_ohip_data, process_ohip_data)


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


# ### Include death date features

# In[6]:


y3 = pd.read_csv(f'{root_path}/data/y3.csv')
y3 = filter_y3_data(y3, include_death=True)
y3 = y3[['ikn', 'D_date']]
chemo_df = chemo_df[chemo_df['ikn'].isin(y3['ikn'])]
chemo_df = pd.merge(chemo_df, y3, on='ikn', how='left')
print(f"NSessions: {len(chemo_df)}. NPatients: {chemo_df['ikn'].nunique()}")


# ### Include last seen date features

# In[7]:


chemo_df['last_seen_date'] = chemo_df['D_date']
chemo_df['last_seen_date'] = chemo_df['last_seen_date'].fillna(max_chemo_date)


# ### Include features from OHIP database
# Get Physician Billing Codes for Palliative Consultation Service

# In[8]:


# Extract and Preprocess the OHIP Data
ohip = pd.read_csv(f'{root_path}/data/ohip.csv')
ohip = filter_ohip_data(ohip)

# Process the OHIP Data
chemo_df = process_ohip_data(chemo_df, ohip)


# In[9]:


chemo_df.to_csv(main_filepath, index=False)


# ### Update ED/H dates

# In[10]:


for event in ['ED', 'H']:
    filepath = f'{output_path}/data/{event}_dates.csv'
    event_dates = pd.read_csv(filepath, dtype=str)
    event_dates = event_dates[event_dates['chemo_idx'].isin(chemo_df['index'])]
    event_dates.to_csv(filepath, index=False)


# In[ ]:
