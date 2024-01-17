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


from collections import defaultdict

import tqdm
import pandas as pd
pd.set_option('display.max_rows', 150)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import root_path
from src.preprocess import filter_systemic_data
from src.prep_data import PrepDataEDHD


# # Days Survived Since Visit

# In[3]:


prep = PrepDataEDHD(adverse_event='death')
df = prep.load_data()

mask = df['visit_date'] > df['death_date']
print(f'Removing {sum(mask)} where chemo visit date was after the death date')
df = df[~mask]

df['days_survived'] = (df['death_date'] - df['visit_date']).dt.days
df = df[df['days_survived'].notnull()]

fig, ax = plt.subplots(figsize=(15, 4))
sns.histplot(df, x='days_survived', hue='sex', bins=100, ax=ax, alpha=0.5)
plt.show()


# In[ ]:
