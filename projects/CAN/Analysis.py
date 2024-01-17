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


# In[2]:


import json
import pandas as pd

from src.config import (
    root_path, can_folder, split_date, 
    DATE,
    SCr_rise_threshold, SCr_rise_threshold2
)
from src.utility import numpy_ffill
from src.prep_data import PrepDataCAN


# In[3]:


main_dir = f'{root_path}/projects/{can_folder}'
prep = PrepDataCAN(adverse_event='ckd')
chemo_df = prep.load_data()


# # Cohort Numbers Before and After Exclusions

# In[4]:


# cohort after exclusions (of treatments without one or more of baseline/target blood values)
model_data = prep.get_data()
dev_cohort, test_cohort = prep.create_cohort(model_data, split_date, verbose=False)

# cohort before exclusions
first_visit_date = chemo_df.groupby('ikn')[DATE].min()
mask = chemo_df['ikn'].map(first_visit_date) <= split_date
dev_cohort2, test_cohort2 = chemo_df[mask], chemo_df[~mask]


# In[5]:


show = lambda x: f"NSessions={len(x)}. NPatients={x['ikn'].nunique()}"
cohorts = {'Development': (dev_cohort, dev_cohort2), 'Testing': (test_cohort, test_cohort2)}
for name, (post_exc_cohort, pre_exc_cohort) in cohorts.items():
    print(f'{name} cohort')
    print(f'Before exclusions: {show(pre_exc_cohort)}')
    print(f'After exclusions: {show(post_exc_cohort)}\n')


# # Create Alluvial Plot Numbers

# In[82]:


events = ['ckd', 'aki']
cohorts = ['Development', 'Testing']


# In[83]:


data = {cohort: {} for cohort in cohorts}
for event in events:
    prep = PrepDataCAN(adverse_event=event)
    orig_data = prep.get_data()
    dev_cohort, test_cohort = prep.create_cohort(orig_data, split_date, verbose=False)
    data['Development'][event] = dev_cohort
    data['Testing'][event] = test_cohort


# In[84]:


for event in events:
    df = pd.DataFrame({cohort: [len(event_data[event]), event_data[event]['ikn'].nunique()] for cohort, event_data in data.items()}, 
                      index=['NSessions', 'NPatients']).rename_axis(event.upper())
    print(f'{df}\n')


# In[85]:


for cohort in cohorts:
    for event in events:
        # keep only the first treatment session
        df = data[cohort][event]
        mask = df['days_since_starting_chemo'] == 0 # take first treatment session only
        all_ikns = set(df['ikn'])
        dropped_ikns = all_ikns - set(df.loc[mask, 'ikn'])
        print(f"Removing {(~mask).sum()} non-first treatment sessions out of {len(mask)} total sessions from "
              f"{event.upper()} {cohort} cohort")
        print(f"Removing {len(dropped_ikns)} patients out of {len(all_ikns)} total patients from {event.upper()} "
              f"{cohort} cohort")
        data[cohort][event] = df[mask]
    
    # get the union of aki and ckd data
    aki_data, ckd_data = data[cohort]['aki'].align(data[cohort]['ckd'], join='outer')
    df = aki_data.fillna(ckd_data).astype(df.dtypes)

    # sanity check
    combined_idxs = data[cohort]['aki'].index.union(data[cohort]['ckd'].index)
    assert df.shape[0] == len(combined_idxs)
    
    data[cohort] = df


# In[86]:


pd.DataFrame({cohort: df['ikn'].nunique() for cohort, df in data.items()}, index=['NPatients'])


# In[113]:


rise = 'SCr_rise'
growth = 'SCr_fold_increase'

pretreatment_ckd_filters = {} # all sessions has baseline_eGFR (since all sessions has baseline_creatinine_count)
pretreatment_ckd_filters['Pre-Treatment CKD (stage1-2)'] = lambda df: df.query('baseline_eGFR > 60')
pretreatment_ckd_filters['Pre-Treatment CKD (stage3a)'] = lambda df: df.query('45 <= baseline_eGFR < 60')
pretreatment_ckd_filters['Pre-Treatment CKD (stage3b)'] = lambda df: df.query('30 <= baseline_eGFR < 45')
pretreatment_ckd_filters['Pre-Treatment CKD (stage4-5)'] = lambda df: df.query('baseline_eGFR < 30')

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
aki_filters['No AKI'] = lambda df: df[(df[rise] < SCr_rise_threshold) & (df[growth] < 1.5)]
# creatinine increase between 26.53umol/dl and 353.68umol/dl and creatinine level equal or less than 2 times the baseline OR
# creatinine increase less than 353.68umol/dl and creatinine level between 1.5-2 times the baseline
aki_filters['Worst AKI (stage1)'] = lambda df: df[((df[rise].between(SCr_rise_threshold, SCr_rise_threshold2) & (df[growth] <= 2)) | 
                                                   ((df[rise] < SCr_rise_threshold2) & df[growth].between(1.5, 2)))]
# creatinine increas less than 353.68umol/dl and creatinine level between 2.001-3 times the baseline
aki_filters['Worst AKI (stage2)'] = lambda df: df[(df[rise] < SCr_rise_threshold2) & df[growth].between(2.001, 3, inclusive='neither')]
# creatinine increas greater than or equal to 353.68umol/dl and creatinine level at least 3 times the baseline
aki_filters['Worst AKI (stage3)'] = lambda df: df[(df[rise] >= SCr_rise_threshold2) | (df[growth] >= 3)]

posttreatment_ckd_filters = {}
posttreatment_ckd_filters['No Information'] = lambda df: df[df['next_SCr_value'].isnull()]
posttreatment_ckd_filters['Post-Treatment CKD (stage1-2)'] = lambda df: df.query('next_eGFR > 60')
posttreatment_ckd_filters['Post-Treatment CKD (stage3a)'] = lambda df: df.query('45 <= next_eGFR < 60')
posttreatment_ckd_filters['Post-Treatment CKD (stage3b)'] = lambda df: df.query('30 <= next_eGFR < 45')
posttreatment_ckd_filters['Post-Treatment CKD (stage4-5)'] = lambda df: df.query('next_eGFR < 30')


# In[114]:


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


# In[115]:


print(json.dumps(result, indent=2))
