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


import json
import pandas as pd
pd.options.mode.chained_assignment = None

from scripts.config import (root_path, can_folder, split_date, SCr_rise_threshold, SCr_rise_threshold2)
from scripts.prep_data import (PrepDataCAN)


# # Cohort Numbers Before Exclusions

# In[3]:


prep = PrepDataCAN(adverse_event='aki')
df = prep.load_data() # loads the same model_data for both aki and ckd
df = prep.get_visit_date_feature(df, include_first_date=True)
mask = df['first_visit_date'] <= split_date
dev_cohort, test_cohort = df[mask], df[~mask]
print(f'Number of patients in development cohort who recieved cisplatin treatments: {dev_cohort["ikn"].nunique()}')
print(f'Number of patients in test cohort who recieved cisplatin treatments: {test_cohort["ikn"].nunique()}')


# # Create Alluvial Plot Numbers

# In[4]:


split_date = '2017-06-30'
kwargs = {'target_keyword': 'SCr|dialysis|next', 'split_date': split_date, 'verbose': False}
events = ['ckd', 'aki']
cohorts = ['Development', 'Testing']


# In[5]:


data = {cohort: {} for cohort in cohorts}
for event in events:
    prep = PrepDataCAN(adverse_event=event)
    orig_data = prep.get_data(include_first_date=True)
    _, _, _, Y_train, Y_valid, Y_test = prep.split_data(orig_data, **kwargs)
    data['Development'][event] = orig_data.loc[(Y_train+Y_valid).index]
    data['Testing'][event] = orig_data.loc[(Y_test).index]


# In[6]:


pd.DataFrame({cohort: [len(event_data['ckd']), event_data['ckd']['ikn'].nunique()] for cohort, event_data in data.items()}, 
             index=['NSessions', 'NPatients']).rename_axis('CKD')


# In[7]:


pd.DataFrame({cohort: [len(event_data['aki']), event_data['aki']['ikn'].nunique()] for cohort, event_data in data.items()}, 
             index=['NSessions', 'NPatients']).rename_axis('AKI')


# In[8]:


for cohort in cohorts:
    for event in events:
        # keep only the first treatment session
        event_data = data[cohort][event]
        mask = event_data['days_since_starting_chemo'] == 0 # take first treatment session only
        total_ikns = set(event_data['ikn'].unique())
        dropped_ikns = total_ikns - set(event_data.loc[mask, 'ikn'].unique())
        print(f"Removing {(~mask).sum()} non-first treatment sessions out of " +               f"{len(mask)} total sessions from {event.upper()} {cohort} cohort")
        print(f"Removing {len(dropped_ikns)} patients out of " +               f"{len(total_ikns)} total patients from {event.upper()} {cohort} cohort")
        data[cohort][event] = event_data[mask]
    
    # get the union of aki and ckd data
    aki_data, ckd_data = data[cohort]['aki'].align(data[cohort]['ckd'], join='outer')
    df = aki_data.fillna(ckd_data)

    # sanity check
    combined_idxs = data[cohort]['aki'].index.union(data[cohort]['ckd'].index)
    assert df.shape[0] == len(combined_idxs)
    
    data[cohort] = df


# In[9]:


pd.DataFrame({cohort: df['ikn'].nunique() for cohort, df in data.items()}, index=['NPatients'])


# In[10]:


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


# In[11]:


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


# In[12]:


print(json.dumps(result, indent=2))


# In[ ]:
