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


import scripts.utilities as util
from scripts.config import (root_path, blood_mapping, blood_types, olis_cols)
from scripts.preprocess import (clean_string)

import pandas as pd
import numpy as np
import tqdm
import time
import sklearn
import matplotlib.pyplot as plt


# In[4]:


import pyspark.pandas as ps
from pyspark.sql import SparkSession
from pyspark.sql.functions import substring, length, expr, split, to_timestamp, to_date
from pyspark.sql.functions import min as ps_min, max as ps_max


# # PySpark vs Pandas

# ## Comparison1: get the size of olis dataset (olis.csv is XXX GB)

# In[ ]:


trials = 3
trial_df = pd.DataFrame(index=[i for i in range(trials)])
trial_df.index.name = 'Trials'
for i in tqdm.tqdm(range(trials)):
    start = time.time()
    df = pd.read_csv("data/olis.csv", dtype={'SUBVALUE_RECOMMENDED_D':str})
    pandas_count = df.shape[0]
    end = time.time()
    trial_df.loc[i, 'Pandas'] = int(end-start)
    
    start = time.time()
    df = ps.read_csv('data/olis.csv')
    pyspark_count = df.shape[0]
    end = time.time()
    trial_df.loc[i, 'PySpark - Pandas API'] = int(end-start)
    
    assert(pandas_count == pyspark_count)
trial_df.loc['Average'] = trial_df.mean()
trial_df = trial_df.astype(int).astype(str) + 's'


# In[36]:


trial_df


# ## Comparison2: get the blood ranges in olis dataset

# In[4]:


def get_blood_ranges(df, framework='pandas'):
    blood_ranges = {blood_type: [np.inf, 0] for blood_type in blood_mapping.values()}
    cols = ['ReferenceRange', 'ObservationCode']
    # keep columns of interest
    df = df[cols]
    # remove rows where no reference range is given
    df = df[~df['ReferenceRange'].isnull()]
    # remove first two characters "b'" and last character "'"
    for col in cols:
        if framework == 'pandas':
            df[col] = df[col].str[2:-1]
        elif framework == 'pyspark':
            df[col] = df[col].str.slice(start=2, stop=-1)
    # map the blood type to blood code
    df['ObservationCode'] = df['ObservationCode'].map(blood_mapping)
    # get the min/max blood count values for this chunk and update the global min/max blood range
    if framework == 'pandas':
        iterator = df.groupby('ObservationCode')
    elif framework == 'pyspark':
        iterator = [(blood_type, df[df['ObservationCode'] == blood_type]) for blood_type in blood_types]
    for blood_type, group in iterator:
        ranges = group['ReferenceRange'].str.split('-')
        if framework == 'pandas':
            min_count = min(ranges.str[0].replace(r'^\s*$', np.nan, regex=True).fillna('inf').astype(float))
            max_count = max(ranges.str[1].replace(r'^\s*$', np.nan, regex=True).fillna('0').astype(float))
        elif framework == 'pyspark':
            min_count = ranges.str.get(0).replace('', None).fillna('inf').astype(float).min()
            max_count = ranges.str.get(1).replace('', None).fillna('0').astype(float).max()
        blood_ranges[blood_type][0] = min(min_count, blood_ranges[blood_type][0])
        blood_ranges[blood_type][1] = max(max_count, blood_ranges[blood_type][1])
    return blood_ranges


# In[10]:


# Pandas
start = time.time()
df = pd.read_csv("data/olis.csv", dtype={'SUBVALUE_RECOMMENDED_D':str})
blood_ranges = get_blood_ranges(df)
end = time.time()
print(f'Operation took {int(end - start)} seconds. Blood ranges: {blood_ranges}')


# In[5]:


# PySpark - Using Pandas API
start = time.time()
df = ps.read_csv('data/olis.csv')
blood_ranges = get_blood_ranges(df, framework='pyspark')
end = time.time()
print(f'Operation took {int(end - start)} seconds. Blood ranges: {blood_ranges}')


# ### Let's try PySpark - SQL API

# In[34]:


def get_blood_ranges(df):
    blood_ranges = {blood_type: [np.inf, 0] for blood_type in blood_mapping.values()}
    cols = ['ReferenceRange', 'ObservationCode']
    # keep columns of interest
    df = df[cols]
    # remove rows where no reference range is given
    df = df.na.drop(subset=['ReferenceRange'])
    # remove first two characters "b'" and last character "'"
    for col in cols:
        # e.g "b'718-7'" start at the 3rd char (the 7), cut off after 8 (length of string) - 3 = 5 characters. We get "718-7"
        df = df.withColumn(col, expr(f"substring({col}, 3, length({col})-3)"))
    # map the blood type to blood code
    df = df.replace(to_replace=blood_mapping, subset=['ObservationCode'])
    # get the min/max blood count values for this chunk and update the global min/max blood range
    iterator = [(blood_type, df.filter(df['ObservationCode'] == blood_type)) for blood_type in blood_types]
    for blood_type, group in iterator:
        ranges = split(group['ReferenceRange'], '-')
        group = group.withColumn('ReferenceRangeMin', ranges.getItem(0).cast('float'))
        group = group.withColumn('ReferenceRangeMax', ranges.getItem(1).cast('float'))
        min_count = float(group.select(ps_min("ReferenceRangeMin")).collect()[0][0])
        max_count = float(group.select(ps_max("ReferenceRangeMax")).collect()[0][0])
        blood_ranges[blood_type][0] = min(min_count, blood_ranges[blood_type][0])
        blood_ranges[blood_type][1] = max(max_count, blood_ranges[blood_type][1])
    return blood_ranges


# In[35]:


# PySpark - SQL
start = time.time()
spark = SparkSession.builder.appName("Test").getOrCreate()
df = spark.read.csv('data/olis.csv', header=True)
blood_ranges = get_blood_ranges(df)
end = time.time()
print(f'Operation took {int(end - start)} seconds. Blood ranges: {blood_ranges}')


# ## Comaprison 3 - Preprocess Olis dataaset

# ### PySpark Version

# In[15]:


def non_filter_preproc(df):
    df = clean_string(df, ['ikn', 'ObservationCode'])

    # keep only selected columns
    df = df[olis_cols]
    
    # map the blood type to blood code
    df['ObservationCode'] = df['ObservationCode'].map(blood_mapping)

    # convert string column into timestamp column
    df['ObservationDateTime'] = pd.to_datetime(df['ObservationDateTime'])
    df['ObservationDateTime'] = df['ObservationDateTime'].dt.floor('D') # keep only the date, not time
    df['ObservationReleaseTS'] = pd.to_datetime(df['ObservationReleaseTS'])
    
    # rename value recommended d to value
    df = df.rename(columns={'value_recommended_d': 'value'})
    
    # remove rows with blood count null values
    df = df[~df['value'].isnull()]
    return df

def ps_non_filter_preproc(df):
    # remove first two characters "b'" and last character "'"
    for col in ['ikn', 'ObservationCode']:
        df = df.withColumn(col, expr(f"substring({col}, 3, length({col})-3)"))

    # keep only selected columns
    df = df[olis_cols]
    
    # map the blood type to blood code
    df = df.replace(to_replace=blood_mapping, subset=['ObservationCode'])

    # convert string column into timestamp column
    df = df.withColumn('ObservationDateTime', to_date('ObservationDateTime'))
    df = df.withColumn('ObservationReleaseTS', to_timestamp('ObservationReleaseTS'))
    
    # rename value recommended d to value
    df = df.withColumnRenamed('value_recommended_d', 'value')
    
    # remove rows with blood count null values
    df = df.na.drop(subset=['value'])
    return df


# In[8]:


# PySpark - SQL
start = time.time()
spark = SparkSession.builder.appName("Test").getOrCreate()
df = spark.read.csv('data/olis.csv', header=True)
df = ps_non_filter_preproc(df)
end = time.time()
print(f'Operation took {int(end - start)} seconds.')


# In[10]:


# Pandas
start = time.time()
olis_df = pd.read_csv("data/olis.csv", dtype={'SUBVALUE_RECOMMENDED_D':str})
olis_df = non_filter_preproc(olis_df)
end = time.time()
print(f'Operation took {int(end - start)} seconds.')


# # Extra olis blood count

# In[5]:


spark = SparkSession.builder.appName("Test").getOrCreate()


# In[6]:


olis = spark.read.csv(f'{root_path}/data/olis.csv', header=True)
col = 'ObservationCode'
olis = olis.withColumn(col, expr(f"substring({col}, 3, length({col})-3)"))
observation_olis = olis.groupby('ObservationCode').count()
observation_olis = observation_olis.sort('ObservationCode')
observation_olis = observation_olis.filter(~(observation_olis['count'] < 10000)) # must have at least 10000 observations
size = observation_olis.count()
print(f"Number of unique observation codes: {size}")
observation_olis.show(n=size)


# In[7]:


olis_blood_count = spark.read.csv(f'{root_path}/data/olis_blood_count.csv', header=True)
col = 'ObservationCode'
olis_blood_count = olis_blood_count.withColumn(col, expr(f"substring({col}, 3, length({col})-3)"))
observation_olis_blood_count = olis_blood_count.groupby('ObservationCode').count()
observation_olis_blood_count = observation_olis_blood_count.sort('ObservationCode')
observation_olis_blood_count = observation_olis_blood_count.filter(
                                ~(observation_olis_blood_count['count'] < 10000)) # must have at least 10000 observations
size = observation_olis_blood_count.count()
print(f"Number of unique observation codes: {size}")
observation_olis_blood_count.show(n=size)

