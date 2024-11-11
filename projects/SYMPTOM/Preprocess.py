"""
========================================================================
© 2023 Institute for Clinical Evaluative Sciences. All rights reserved.

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

# In[2]:


get_ipython().run_line_magic('cd', '../../')
# reloads all modules everytime before cell is executed (no needto restart kernel)
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 150)

from src import logger
from src.config import root_path, symp_folder, DATE, symptom_cols
from src.preprocess import (
    Systemic, Demographic, Laboratory, Symptoms, 
    combine_demographic_data, combine_lab_data, combine_symptom_data,
)
from src.utility import load_included_regimens, make_log_msg


# In[4]:


# config
output_path = f'{root_path}/projects/{symp_folder}'
processes = 32


# # Create my dataset

# In[6]:


def quick_summary(df):
    print(f"Number of treatment sessions = {len(df)}")
    print(f"Number of patients = {df['ikn'].nunique()}")
    print(f"Number of unique regiments = {df['regimen'].nunique()}")
    print(f"Cohort from {df['visit_date'].min().date()} to {df['visit_date'].max().date()}")


# ### Include features from systemic therapy treatment data
# NOTE: ikn is the encoded ontario health insurance plan (OHIP) number of a patient

# In[13]:


get_ipython().run_cell_magic('time', '', "syst = Systemic()\ndf = syst.run(load_included_regimens(), filter_kwargs={'verbose': True, 'min_date': '2014-01-01', 'max_date': '2019-12-31'}, process_kwargs={'method': 'one-per-week'})\ndf.to_parquet(f'{output_path}/data/systemic.parquet.gzip', compression='gzip', index=False)")


# In[8]:


df = pd.read_parquet(f'{output_path}/data/systemic.parquet.gzip')
quick_summary(df)


# ### Include features from demographic data
# Includes cancer diagnosis, income, immigration, area of residence, etc

# In[39]:


demog = Demographic()
demo_df = demog.run()


# In[40]:


df = combine_demographic_data(df, demo_df)
drop_cols = ['inpatient_flag', 'ethnic', 'country_birth', 'official_language', 'nat_language']
df = df.drop(columns=drop_cols)
quick_summary(df)


# ### Include features from symptom data

# In[13]:


get_ipython().run_cell_magic('time', '', "symp = Symptoms(processes=processes)\nsymp_df = symp.run(df)\nsymp_df.to_parquet(f'{output_path}/data/symptom_features.parquet.gzip', compression='gzip', index=False)")


# In[44]:


symp_df = pd.read_parquet(f'{output_path}/data/symptom_features.parquet.gzip')
df = combine_symptom_data(df, symp_df)


# In[9]:


# save for future analysis
missingness = df[symptom_cols].isnull()
missingness = missingness.drop(columns=['ecog_grade', 'prfs_grade'])
missingness = missingness.groupby(df['visit_hospital_number']).mean()
missingness['nsessions'] = df.groupby('visit_hospital_number').apply(len)
missingness.to_csv(f'{output_path}/data/baseline_missingness_per_cancer_center.csv')


# In[10]:


mask = ~df[symptom_cols].isnull().all(axis=1)
logger.info(make_log_msg(df, mask, context=' without any baseline symptom scores'))
df = df[mask]
quick_summary(df)


# ### Include targets from symptom data

# In[13]:


def symptom_target_worker(partition, targets_within=30):
    """Get the symptom scores within X days after the treatment session date

    NOTE: The baseline score can include surveys taken on visit date. To make 
    sure the target score does not overlap with the baseline score, only take
    surveys AFTER the visit date
    """
    chemo_df, symps_df = partition
    result = []
    for ikn, chemo_group in chemo_df.groupby('ikn'):
        symps_group = symps_df.query('ikn == @ikn')
        for idx, visit_date in chemo_group[DATE].items():
            mask = symps_group['surveydate'].between(
                visit_date + pd.Timestamp(days=1),
                visit_date + pd.Timestamp(days=targets_within)
            )
            if not mask.any():
                continue
            # Note: Sx is a medical abbreviation for symptom
            for sx, sx_group in symps_group[mask].groupby('symptom'):
                # take the max (worst symptom) scores within the target timeframe
                row = sx_group.loc[sx_group['severity'].idxmax()]
                result.append([idx, sx, row['severity'], row['surveydate']])
    return result

symp = Symptoms(processes=processes)
symp_df = symp.run(df, worker=symptom_target_worker)
symp_df.to_parquet(f'{output_path}/data/symptom_targets.parquet.gzip', compression='gzip', index=False)


# In[11]:


symp_df = pd.read_parquet(f'{output_path}/data/symptom_targets.parquet.gzip')
symp_df['symptom'] = 'target_' + symp_df['symptom']
df = combine_symptom_data(df, symp_df)


# In[12]:


cols = [f'target_{col}' for col in symptom_cols]
mask = ~df[cols].isnull().all(axis=1)
logger.info(make_log_msg(df, mask, context=' without any target symptom scores'))
df = df[mask]
quick_summary(df)


# ### Include features from lab test data

# In[42]:


labr = Laboratory(f'{output_path}/data', processes)


# In[11]:


get_ipython().run_cell_magic('time', '', "# NOTE: need to restart kerel after this point to avoid annoying IOStream.flush timed out messages\nlabr.preprocess(set(df['ikn']))")


# In[11]:


get_ipython().run_cell_magic('time', '', "lab_df = labr.run(df)\nlab_df.to_parquet(f'{output_path}/data/lab.parquet.gzip', compression='gzip', index=False)")


# In[43]:


get_ipython().run_cell_magic('time', '', "lab_df = pd.read_parquet(f'{output_path}/data/lab.parquet.gzip')\ndf, lab_map, missing_df = combine_lab_data(df, lab_df)\nmissing_df")


# In[45]:


df.to_parquet(f'{output_path}/data/final_data.parquet.gzip', compression='gzip', index=False)
