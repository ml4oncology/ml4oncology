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


# In[3]:


import json
import pandas as pd
pd.options.mode.chained_assignment = None

from src.config import (root_path, can_folder, split_date, SCr_rise_threshold, SCr_rise_threshold2)
from src.preprocess import filter_y3_data
from src.utility import numpy_ffill
from src.prep_data import (PrepDataCAN)


# # Cohort Numbers

# In[3]:


y3 = pd.read_csv(f'{root_path}/data/y3.csv')
y3 = filter_y3_data(y3, include_death=True)
y3['ikn'] = y3['ikn'].astype(int)
ikn_death_date = y3.set_index('ikn')['D_date']


# In[4]:


scr = pd.read_csv(f'{root_path}/{can_folder}/data/serum_creatinine.csv')
scr.columns = scr.columns.astype(int)
base_scr, next_scr = 'baseline_creatinine_count', 'next_SCr_count'


# In[5]:


prep = PrepDataCAN(adverse_event='ckd')
# get data before filtering procedures
orig_df = prep.load_data()
# get data after filtering procedures
df = prep.get_data()
# get the finalized development and test cohort
dev_cohort, test_cohort = prep.create_cohort(df, split_date, verbose=False)
# get the mock development and test cohort
assert not set(dev_cohort['ikn']).intersection(set(test_cohort['ikn']))
mask = orig_df['ikn'].isin(dev_cohort['ikn'])
mock_dev_cohort, mock_test_cohort = orig_df[mask], orig_df[~mask]
assert not set(mock_dev_cohort['ikn']).intersection(set(mock_test_cohort['ikn']))


# In[6]:


cohorts = {'Development': mock_dev_cohort, 'Testing': mock_test_cohort}
for name, cohort in cohorts.items():
    # get serum creatinine measurement taken within the month before visit date, instead of 5 days before
    cohort[base_scr] = numpy_ffill(scr.loc[cohort.index, range(-30,1)])
    
    print(f'\n{name} cohort')
    print(f'Before any filtering: NSessions={len(cohort)}. NPatients={cohort["ikn"].nunique()}')
    
    cohort = cohort[cohort[base_scr].notnull()]
    print('After Excluding Samples with no Baseline SCr (measured 30 days before chemo visit): ' +\
          f'NSessions={len(cohort)}. NPatients={cohort["ikn"].nunique()}')
    
    mask = cohort[next_scr].notnull()
    print('After Excluding Samples with no Next SCr (measured 9-24 months after chemo visit): '+\
          f'NSessions={sum(mask)}. NPatients={cohort.loc[mask, "ikn"].nunique()}')
    
    no_mes = cohort[~mask]
    print(f'Samples with no Next SCr: NSessions={len(no_mes)}. NPatients={no_mes["ikn"].nunique()}')
    no_mes['d_date'] = no_mes['ikn'].map(ikn_death_date)
    died_before_mes = no_mes['d_date'] < no_mes['visit_date'] + pd.Timedelta('270 days')
    print('Died within 9 months after visit date: ' +\
          f'NSessions={sum(died_before_mes)}, NPatients={no_mes.loc[died_before_mes, "ikn"].nunique()}')


# # Create Alluvial Plot Numbers

# In[7]:


events = ['ckd', 'aki']
cohorts = ['Development', 'Testing']


# In[8]:


data = {cohort: {} for cohort in cohorts}
for event in events:
    prep = PrepDataCAN(adverse_event=event)
    orig_data = prep.get_data()
    dev_cohort, test_cohort = prep.create_cohort(orig_data, split_date, verbose=False)
    data['Development'][event] = dev_cohort
    data['Testing'][event] = test_cohort


# In[9]:


for event in events:
    df = pd.DataFrame({cohort: [len(event_data[event]), event_data[event]['ikn'].nunique()] for cohort, event_data in data.items()}, 
                      index=['NSessions', 'NPatients']).rename_axis(event.upper())
    print(f'{df}\n')


# In[10]:


for cohort in cohorts:
    for event in events:
        # keep only the first treatment session
        event_data = data[cohort][event]
        mask = event_data['days_since_starting_chemo'] == 0 # take first treatment session only
        total_ikns = set(event_data['ikn'].unique())
        dropped_ikns = total_ikns - set(event_data.loc[mask, 'ikn'].unique())
        print(f"Removing {(~mask).sum()} non-first treatment sessions out of " + \
              f"{len(mask)} total sessions from {event.upper()} {cohort} cohort")
        print(f"Removing {len(dropped_ikns)} patients out of " + \
              f"{len(total_ikns)} total patients from {event.upper()} {cohort} cohort")
        data[cohort][event] = event_data[mask]
    
    # get the union of aki and ckd data
    aki_data, ckd_data = data[cohort]['aki'].align(data[cohort]['ckd'], join='outer')
    df = aki_data.fillna(ckd_data)

    # sanity check
    combined_idxs = data[cohort]['aki'].index.union(data[cohort]['ckd'].index)
    assert df.shape[0] == len(combined_idxs)
    
    data[cohort] = df


# In[11]:


pd.DataFrame({cohort: df['ikn'].nunique() for cohort, df in data.items()}, index=['NPatients'])


# In[12]:


pretreatment_ckd_filters = {} # all sessions has baseline_eGFR (since all sessions has baseline_creatinine_count)
pretreatment_ckd_filters['Pre-Treatment CKD (stage1-2)'] = lambda df: df[df['baseline_eGFR'] > 60]
pretreatment_ckd_filters['Pre-Treatment CKD (stage3a)'] = lambda df: df[df['baseline_eGFR'].between(45, 60, inclusive='left')]
pretreatment_ckd_filters['Pre-Treatment CKD (stage3b)'] = lambda df: df[df['baseline_eGFR'].between(30, 45, inclusive='left')]
pretreatment_ckd_filters['Pre-Treatment CKD (stage4-5)'] = lambda df: df[df['baseline_eGFR'] < 30]

"""
Need to make each stage mutually exclusive

Reminder:
AKI Stage 1 and below: creatinine increase greater than or equal to 26.53umol/dl or creatinine level at least 1.5 times the baseline
AKI Stage 2 and below: creatinine level at least 2 times the baseline
AKI Stage 3 and below: creatinine increase greater than or equal to 353.58umol/dl or creatinine level at least 3 times the baseline
"""
aki_filters = {}
aki_filters['No Information'] = lambda df: df[df['SCr_peak'].isnull()]
# creatinine increase less than 26.53umol/dl and creatinine level less than 1.5 times the baseline
aki_filters['No AKI'] = lambda df: df[(df['SCr_rise'] < SCr_rise_threshold) & (df['SCr_fold_increase'] < 1.5)]
# creatinine increase between 26.53umol/dl and 353.68umol/dl and creatinine level equal or less than 2 times the baseline OR
# creatinine increase less than 353.68umol/dl and creatinine level between 1.5-2 times the baseline
aki_filters['Worst AKI (stage1)'] = lambda df: df[((df['SCr_rise'].between(SCr_rise_threshold, SCr_rise_threshold2) & (df['SCr_fold_increase'] <= 2)) | 
                                                   ((df['SCr_rise'] < SCr_rise_threshold2) & df['SCr_fold_increase'].between(1.5, 2)))]
# creatinine increas less than 353.68umol/dl and creatinine level between 2.001-3 times the baseline
aki_filters['Worst AKI (stage2)'] = lambda df: df[(df['SCr_rise'] < SCr_rise_threshold2) & df['SCr_fold_increase'].between(2.001, 3, inclusive=False)]
# creatinine increas greater than or equal to 353.68umol/dl and creatinine level at least 3 times the baseline
aki_filters['Worst AKI (stage3)'] = lambda df: df[(df['SCr_rise'] >= SCr_rise_threshold2) | (df['SCr_fold_increase'] >= 3)]

posttreatment_ckd_filters = {}
posttreatment_ckd_filters['No Information'] = lambda df: df[df['next_SCr_count'].isnull()]
posttreatment_ckd_filters['Post-Treatment CKD (stage1-2)'] = lambda df: df[df['next_eGFR'] > 60]
posttreatment_ckd_filters['Post-Treatment CKD (stage3a)'] = lambda df: df[df['next_eGFR'].between(45, 60, inclusive='left')]
posttreatment_ckd_filters['Post-Treatment CKD (stage3b)'] = lambda df: df[df['next_eGFR'].between(30, 45, inclusive='left')]
posttreatment_ckd_filters['Post-Treatment CKD (stage4-5)'] = lambda df: df[df['next_eGFR'] < 30]


# In[13]:


result = {cohort: {} for cohort in cohorts}
for cohort in cohorts:
    event_df = data[cohort]
    for pretreatment_ckd_stage, pretreatment_ckd_filter in pretreatment_ckd_filters.items():
        pretreatment_ckd_df = pretreatment_ckd_filter(event_df)
        pretreatment_ckd_name = f"{pretreatment_ckd_stage} [N={pretreatment_ckd_df['ikn'].nunique()}]"
        result[cohort][pretreatment_ckd_name] = {}
        
        for aki_stage, aki_filter in aki_filters.items():
            aki_df = aki_filter(pretreatment_ckd_df)
            aki_name = f"{aki_stage} [N={aki_df['ikn'].nunique()}]"
            result[cohort][pretreatment_ckd_name][aki_name] = {}
            
            for posttreatment_ckd_stage, posttreatment_ckd_filter in posttreatment_ckd_filters.items():
                posttreatment_ckd_df = posttreatment_ckd_filter(aki_df)
                result[cohort][pretreatment_ckd_name][aki_name][posttreatment_ckd_stage] = posttreatment_ckd_df['ikn'].nunique()


# In[14]:


print(json.dumps(result, indent=2))


# In[ ]:
