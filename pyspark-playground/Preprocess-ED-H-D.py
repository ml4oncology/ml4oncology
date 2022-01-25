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
env = 'kevenv' # 'kevenv', 'myenv'
user_path = 'XXXXXXXX'
for i, p in enumerate(sys.path):
    sys.path[i] = sys.path[i].replace("/software/anaconda/3/", f"{user_path}/.conda/envs/{env}/")
sys.prefix = f'{user_path}/.conda/envs/{env}/'


# In[3]:


import tqdm
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import multiprocessing as mp
from functools import partial

from utilities import (get_y3)
from config import (root_path, blood_mapping, extra_blood_types, systemic_cols, y3_cols, diag_code_mapping)
from preprocess import (clean_string, split_and_parallelize,
                        clean_cancer_and_demographic_data,
                        prefilter_olis_blood_count_data, postprocess_olis_blood_count_data)


# In[4]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import substring, length, expr, split, to_date
from pyspark.sql.functions import min as ps_min, max as ps_max


# In[6]:


spark = SparkSession.builder.appName("Test").getOrCreate()


# In[7]:


# config
output_path = f'{root_path}/notebooks_pyspark'

processes = 32
manager = mp.Manager()
shared_dict = manager.dict()


# # Create my csvs

# In[5]:


def clean_string(df, cols):
    # remove first two characters "b'" and last character "'"
    for col in cols:
        # e.g "b'718-7'" start at the 3rd char (the 7), cut off after 8 (length of string) - 3 = 5 characters. We get "718-7"
        df = df.withColumn(col, expr(f"substring({col}, 3, length({col})-3)"))
    return df


# ### Include features from systemic (chemo) dataset

# In[8]:


regimen_exclude = []
regimen_name_mapping = {'paclicarbo': 'crbppacl'}


# In[27]:


def filter_systemic_data(df):
    df = clean_string(df, ['ikn', 'din', 'intent_of_systemic_treatment', 'inpatient_flag'])

    # convert string column into timestamp column
    df = df.withColumn('visit_date', to_date('visit_date'))
    
    # order the dataframe by date
    df = df.orderBy('visit_date')
    
    # remove chemo treatment recieved as an inpatient
    df = df.filter(df['inpatient_flag'] == 'N')
    
    # filter regimens
    col = 'regimen'
    df = df.na.drop(subset=[col]) # filter out rows with no regimen data 
    df = df.replace(to_replace=regimen_name_mapping, subset=[col]) # change regimen name to the correct mapping
    
    return df



    
    # filter regimens
    df = df[~df[col].isin(regimen_exclude)] # remove regimens in regimen_exclude
    
    # replace regimens with less than 6 patients to 'Other'
    counts = df.groupby('regimen').apply(lambda group: len(set(group['ikn'])))
    replace_regimens = counts.index[counts < 6]
    df.loc[df['regimen'].isin(replace_regimens), 'regimen'] = 'Other'
    
    df = df[systemic_cols] # keep only selected columns
    df = df.drop_duplicates()
    return df


# In[31]:


systemic = spark.read.csv(f'{root_path}/data/systemic.csv', header=True)
systemic = filter_systemic_data(systemic)


# In[38]:


df = systemic
df.show(n=2, vertical=True)


# ### Include features from y3 (cancer and demographic) dataset  - which includes death events

# In[7]:


def process_y3(y3):
    y3 = clean_string(y3, ['sex', 'vital_status_cd', 'ikn'])
    y3 = y3.withColumn('dthdate', to_date('dthdate'))
    
    # remove rows with conflicting information
    mask1 = (y3['vital_status_cd'] == 'A') & y3['dthdate'].isNotNull() # vital status says Alive but deathdate exists
    mask2 = (y3['vital_status_cd'] == 'D') & y3['dthdate'].isNull() # vital status says Dead but deathdate does not exist
    y3 = y3.filter(~(mask1 | mask2))
    return y3

