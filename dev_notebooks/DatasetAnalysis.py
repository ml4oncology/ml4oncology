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


from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    min, 
    max, 
    count, 
    to_date, 
    to_timestamp, 
    year,
    col, 
    when, 
    countDistinct
)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
pd.set_option('display.max_rows', 100)

from src.config import (root_path, all_observations, OBS_CODE, OBS_DATE)


# In[ ]:


# NOTE: default config at spark.apache.org/docs/latest/configuration.html
spark = SparkSession.builder.config("spark.driver.memory", "15G").appName("Main").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel('ERROR')


# In[ ]:


sc


# In[8]:


sc.getConf().getAll()


# # OLIS Dataset

# In[6]:


mapping = {code: f'{code}={name}' for code, name in all_observations.items()}


# In[7]:


olis = spark.read.parquet(f'{root_path}/data/olis', header=True)
olis = olis.withColumn(OBS_DATE, to_date(OBS_DATE)) # convert string column into timestamp column
olis = olis.withColumnRenamed('value_recommended_d', 'value')


# In[8]:


get_ipython().run_cell_magic('time', '', 'olis.select(min(OBS_DATE), max(OBS_DATE), count(OBS_DATE)).show()')


# In[12]:


get_ipython().run_cell_magic('time', '', "# Get simple description of the data grouped by observation code\nobservation_counts = olis.groupBy(OBS_CODE).agg(\n    min(OBS_DATE).alias('MinDate'),\n    max(OBS_DATE).alias('MaxDate'),\n    count(OBS_DATE).alias('TotalCount'), \n    count(when(col('value').isNull(),True)).alias('MissingValueCount'),\n    countDistinct('ikn').alias('TotalPatientCount')\n).orderBy(OBS_CODE)\nobservation_counts = observation_counts.replace(mapping)\nsize = observation_counts.count()\nprint(f'{size} unique observation codes')\nobservation_counts.show(size, truncate=False)")


# In[13]:


get_ipython().run_cell_magic('time', '', "obs_counts = observation_counts.toPandas()\nobs_counts.to_csv(f'{root_path}/data/olis_summary.csv', index=False)")


# In[61]:


obs_counts = pd.read_csv(f'{root_path}/data/olis_summary.csv', parse_dates=['MinDate', 'MaxDate'])
obs_counts = obs_counts.sort_values(by='TotalCount')
axes = obs_counts.plot(kind='bar', x='ObservationCode', subplots=True, figsize=(20,16), legend=False)
axes[0].set_ylim(pd.Timestamp('01-01-2000'), pd.Timestamp('01-01-2021'))
axes[1].set_ylim(pd.Timestamp('01-01-2000'), pd.Timestamp('01-01-2021'))
plt.show()


# ## Date Distribution

# In[50]:


get_ipython().run_cell_magic('time', '', "dist = olis.groupBy(OBS_CODE, year(OBS_DATE).alias('obs_year')).count()\ndist = dist.replace(mapping)\ndist = dist.toPandas()")


# In[66]:


col_order = dist.sort_values(by=count_col, ascending=False)[OBS_CODE].drop_duplicates()
g = sns.relplot(
    data=dist, x='obs_year', y=count_col, 
    col=OBS_CODE, col_wrap=3, col_order=col_order,
    kind='line', facet_kws={'sharex': False, 'sharey': False}
)
for ax in g.axes: ax.xaxis.get_major_locator().set_params(integer=True)


# # Systemic Dataset

# In[42]:


systemic = spark.read.parquet(f'{root_path}/data/systemic.parquet.gzip', header=True)


# In[43]:


# drop rows with missing regimen
total_count = systemic.count()
systemic = systemic.na.drop(subset=['regimen'])
missing_regimen_count = total_count - systemic.count()
print(f'Removed {missing_regimen_count} rows with missing regimens out of {total_count} total rows')


# In[44]:


systemic.select(
    min('visit_date').alias('MinDate'),
    max('visit_date').alias('MaxDate'),
    countDistinct('cco_drug_code').alias('CCOCount'),
    countDistinct('din').alias('DINCount'),
    countDistinct('regimen').alias('RegimenCount'),
    countDistinct('ikn', 'visit_date').alias('TotalSessionCount'),
    countDistinct('ikn').alias('TotalPatientCount')
).show()


# In[45]:


# Get simple description of the data grouped by regimen
regimen_counts = systemic.groupBy('regimen').agg(
    min('visit_date').alias('MinDate'),
    max('visit_date').alias('MaxDate'),
    countDistinct('ikn', 'visit_date').alias('TotalSessionCount'),
    countDistinct('ikn').alias('TotalPatientCount'),
    countDistinct('cco_drug_code').alias('CCOCount'),
    countDistinct('din').alias('DINCount'),
).orderBy(col('TotalPatientCount').desc())
regimen_counts.show(30, truncate=False)


# ## Date Distribution

# In[49]:


systemic = systemic.dropDuplicates(subset=['ikn', 'visit_date'])
systemic = systemic.toPandas()


# In[54]:


month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
season_mapping = {'Winter': [12,1,2], 'Spring': [3,4,5], 'Summer': [6,7,8], 'Fall': [9,10,11]}
season_mapping = {month: season for season, months in season_mapping.items() for month in months}
visit_date = pd.to_datetime(systemic['visit_date'])
visit_month = visit_date.dt.month
visit_season = visit_month.map(season_mapping)


# In[61]:


fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))
sns.countplot(x=visit_month, ax=axes[0])
axes[0].set(xlabel='Visit Month', ylabel='Count', xticklabels=month_labels)
sns.countplot(x=visit_season, ax=axes[1])
axes[1].set(xlabel='Visit Season', ylabel='Count')
sns.countplot(x=visit_date.dt.year, ax=axes[2])
axes[2].set(xlabel='Visit Year', ylabel='Count')
plt.show()


# In[76]:


spark.stop()
