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


import pickle
import pandas as pd
from scipy.stats import mode
from difflib import SequenceMatcher
from pyspark.sql import SparkSession
from pyspark.sql.functions import (substring, length, expr, collect_set)
from scripts.config import (root_path, all_observations)


# In[3]:


spark = SparkSession.builder.appName("Main").getOrCreate()


# In[4]:


def clean_string(df, cols):
    # remove first two characters "b'" and last character "'"
    for col in cols:
        # e.g "b'718-7'" start at the 3rd char (the 7), cut off after 8 (length of string) - 3 = 5 characters. We get "718-7"
        df = df.withColumn(col, expr(f"substring({col}, 3, length({col})-3)"))
    return df


# In[5]:


get_ipython().run_cell_magic('time', '', "olis = spark.read.csv(f'{root_path}/data/olis_complete.csv', header=True)\nolis = clean_string(olis, ['ObservationCode', 'Units'])\nobservation_units = olis.groupBy('ObservationCode').agg(collect_set('Units'))")


# In[6]:


get_ipython().run_cell_magic('time', '', '# WOW its actually REALLY SLOW - the staging parts \n# Can be multiple reasons\n# - not using native hadoop?\n# - data too small?\n# - too many concurrent tasks (from other users)?\nobservation_units = observation_units.toPandas()\nobservation_units = dict(observation_units.values)')


# In[24]:


def clean_unit(unit):
    unit = unit.lower()
    
    splits = unit.split(' ')
    if splits[-1].startswith('cr'): # e.g. mg/mmol creat
        assert(len(splits) == 2)
        unit = splits[0] # remove the last text
    
    for c in ['"', ' ', '.']: unit = unit.replace(c, '')
    for c in ['-', '^', '*']: unit = unit.replace(c, 'e')
    if ((SequenceMatcher(None, unit, 'x10e9/l').ratio() > 0.5) or 
        (unit == 'bil/l')): 
        unit = 'x10e9/l'
    if unit in {'l/l', 'ratio', 'fract'}: 
        unit = '%'
    unit = unit.replace('u/', 'unit/')
    unit = unit.replace('/l', '/L')
    return unit

unit_map = {}
for obs_code, units in observation_units.items():
    units = [clean_unit(unit) for unit in units]
    # WARNING: there is a possibility the most frequent unit may be the wrong unit
    # Not enough manpower to check each one manually
    unit_map[obs_code] = mode(units)[0][0]


# In[28]:


filename = f'{root_path}/data/olis_units.pkl'
with open(filename, 'wb') as file:    
    pickle.dump(unit_map, file)


# In[6]:


spark.stop()
