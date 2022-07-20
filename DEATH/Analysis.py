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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scripts.utility import (get_pearson_matrix)
from scripts.visualize import (pearson_plot)
from scripts.config import (root_path, death_folder)
from scripts.preprocess import (clean_string, filter_y3_data)
from scripts.prep_data import (PrepDataEDHD)


# # Age at Death

# In[3]:


data = pd.read_csv(f'{root_path}/data/y3.csv')
data = filter_y3_data(data, include_death=True)
data = data[~(data['sex'] == 'O')]
born = data['bdate'].dt
died = data['D_date'].dt
data['age'] = died.year - born.year - ((died.month) < (born.month))
data = data[~data['age'].isnull()]


# In[4]:


fig = plt.figure(figsize=(15, 4))
for sex, group in data.groupby('sex'):
    plt.hist(group['age'], alpha=0.5, label=sex, bins=109)
plt.xticks(range(0,109,4))
plt.legend()
plt.show()


# In[ ]:
