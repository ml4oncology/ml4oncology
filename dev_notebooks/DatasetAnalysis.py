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


# In[16]:


import pandas as pd
pd.set_option('display.max_rows', 100)
from pyspark.sql import SparkSession
from pyspark.sql.functions import (min, max, count, to_date, to_timestamp, col, when, countDistinct)
from src.config import (root_path, all_observations)
from src.spark import (clean_string)


# In[3]:


# NOTE: default config at spark.apache.org/docs/latest/configuration.html
spark = SparkSession.builder.config("spark.driver.memory", "15G").appName("Main").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel('ERROR')


# In[4]:


sc


# In[5]:


sc.getConf().getAll()


# # OLIS Dataset

# In[6]:


date_col = 'ObservationDateTime'
code_col = 'ObservationCode'
mapping = {code: f'{code}={name}' for code, name in all_observations.items()}


# In[7]:


olis = spark.read.csv(f'{root_path}/data/olis.csv', header=True)
olis = olis.withColumn(date_col, to_date(date_col)) # convert string column into timestamp column
olis = clean_string(olis, ['ikn', code_col])
olis = olis.withColumnRenamed('value_recommended_d', 'value')


# In[8]:


get_ipython().run_cell_magic('time', '', 'olis.select(min(date_col), max(date_col), count(date_col)).show()\n')


# In[8]:


get_ipython().run_cell_magic('time', '', "# Get simple description of the data grouped by observation code\nobservation_counts = olis.groupBy(code_col).agg(min(date_col).alias('MinDate'),\n                                                max(date_col).alias('MaxDate'),\n                                                count(date_col).alias('TotalCount'), \n                                                count(when(col('value').isNull(),True)).alias('MissingValueCount'),\n                                                countDistinct('ikn').alias('TotalPatientCount')\n                                               ).orderBy(code_col)\nobservation_counts = observation_counts.replace(mapping)\nsize = observation_counts.count()\nprint(f'{size} unique observation codes')\nobservation_counts.show(size, truncate=False)\n")


# In[9]:


obs_counts = observation_counts.toPandas()
obs_counts.to_csv(f'{root_path}/data/olis_summary.csv', index=False)


# In[16]:


spark.stop()
