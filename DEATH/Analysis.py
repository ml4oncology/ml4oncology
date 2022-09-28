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


import tqdm
import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option('display.max_rows', 150)
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from src.config import (root_path)
from src.preprocess import (filter_systemic_data, filter_y3_data)
from src.prep_data import (PrepDataEDHD)


# # Age at Death

# In[6]:


data = pd.read_csv(f'{root_path}/data/y3.csv')
data = filter_y3_data(data, include_death=True)
data = data[~(data['sex'] == 'O')]
born = data['bdate'].dt
died = data['D_date'].dt
data['age'] = died.year - born.year - ((died.month) < (born.month))
data = data[~data['age'].isnull()]


# In[7]:


fig = plt.figure(figsize=(15, 4))
for sex, group in data.groupby('sex'):
    plt.hist(group['age'], alpha=0.5, label=sex, bins=109)
plt.xticks(range(0,109,4))
plt.legend()
plt.show()


# # Days Survived Since Visit

# In[28]:


prep = PrepDataEDHD(adverse_event='death')
df = prep.load_data()
df['D_date'] = pd.to_datetime(df['D_date'])
df['visit_date'] = pd.to_datetime(df['visit_date'])
mask = df['visit_date'] > df['D_date']
print(f'Removing {sum(mask)} where chemo visit date was after the death date')
df = df[~mask]
df['days_survived'] = (df['D_date'] - df['visit_date']).dt.days
df = df[df['days_survived'].notnull()]


# In[45]:


fig = plt.figure(figsize=(15, 4))
for sex, group in df.groupby('sex'):
    plt.hist(group['days_survived'], alpha=0.5, label=sex, bins=100)
plt.legend()
plt.show()


# # Line of Therapy

# In[261]:


regimens = load_included_regimens()
systemic = pd.read_csv(f'{root_path}/data/systemic.csv')
systemic = filter_systemic_data(systemic, regimens, cols=['ikn', 'visit_date', 'regimen', 'line_of_therapy', 'intent_of_systemic_treatment'], 
                                exclude_dins=False, verbose=False)


# In[257]:


counts = defaultdict(int)
diff_ikns = []
for ikn, group in tqdm.tqdm(systemic.groupby('ikn')):
    if group['line_of_therapy'].isnull().all():
        # line of therapy was never filled out (intentionally or otherwise)
        counts['no_lot_sessions'] += len(group)
        counts['no_lot_ikns'] += 1
        continue
    
    # clean up original line of therapy
    if group['line_of_therapy'].isnull().any():
        group['line_of_therapy'] = group['line_of_therapy'].fillna(0) + 1
        
    new_regimen = (group['regimen'] != group['regimen'].shift())
    lot = new_regimen.cumsum()
    mask = group['line_of_therapy'] == lot
    
    counts['diff_lot_sessions'] += sum(~mask)
    if not mask.all():
        counts['diff_lot_ikns'] += 1
        diff_ikns.append(ikn)
    
print(f'Total sessions: {len(systemic)}. Total patients: {systemic["ikn"].nunique()}')
print(f'No LOT sessions: {counts["no_lot_sessions"]}. No LOT patients: {counts["no_lot_ikns"]}')
print(f'Differing LOT sessions: {counts["diff_lot_sessions"]}. Differing LOT patients: {counts["diff_lot_ikns"]}')


# In[276]:


mask = systemic['line_of_therapy'].isnull()
pd.DataFrame([systemic.loc[mask, 'intent_of_systemic_treatment'].value_counts(),
              systemic.loc[~mask, 'intent_of_systemic_treatment'].value_counts()], index=['No LOT', 'Has LOT']).T


# In[286]:


df = pd.DataFrame(systemic.groupby('intent_of_systemic_treatment')['line_of_therapy'].value_counts())
df.rename(columns={'line_of_therapy': 'Count'})
