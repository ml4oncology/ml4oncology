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

from scripts.config import (root_path, acu_folder, death_folder, y3_cols)
from scripts.preprocess import (clean_string, filter_y3_data)


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


filepath = f'{output_path}/data/model_data.csv'
model_data = pd.read_csv(filepath, dtype=str)
model_data['visit_date'] = pd.to_datetime(model_data['visit_date'])


# In[6]:


y3 = pd.read_csv(f'{root_path}/data/y3.csv')
y3 = filter_y3_data(y3, include_death=True)
y3 = y3[['ikn', 'D_date']]
model_data = model_data[model_data['ikn'].isin(y3['ikn'])]
model_data = pd.merge(model_data, y3, on='ikn', how='left')
model_data = model_data.set_index('index')
model_data.to_csv(filepath, index_label='index')


# In[7]:


# Update ED/H dates
for event in ['ED', 'H']:
    filepath = f'{output_path}/data/{event}_dates.csv'
    event_dates = pd.read_csv(filepath, dtype=str)
    event_dates = event_dates[event_dates['chemo_idx'].isin(model_data.index)]
    event_dates.to_csv(filepath, index=False)


# In[ ]:
