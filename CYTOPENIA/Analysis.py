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


import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scripts.utility import (get_pearson_matrix, top_cancer_regimen_summary, nadir_summary, most_common_by_category)
from scripts.visualize import (blood_count_dist_plot, regimen_dist_plot, day_dist_plot,
                               scatter_plot, below_threshold_bar_plot, iqr_plot, violin_plot, mean_cycle_plot, 
                               pearson_plot, event_rate_stacked_bar_plot)
from scripts.config import (root_path, cyto_folder, split_date, blood_types)
from scripts.prep_data import (PrepDataCYTO)


# In[3]:


main_dir = f'{root_path}/{cyto_folder}'
output_path = f'{main_dir}/models'
prep = PrepDataCYTO()
chemo_df = prep.load_data()
top_cancer_regimen_summary(chemo_df)


# In[4]:


# regimens of interest
top_regimens = list(most_common_by_category(chemo_df, category='regimen', top=3))
regimens_of_interest = ['mfolfirinox', 'folfirinox', 'gemcnpac(w)']
test_regimens = ['pcv', 'fulcvr', 'ac-doce', 'vino', 'vnbl'] # each has different cycle lengths


# In[5]:


def load_data(blood_type='neutrophil'):
    df = pd.read_csv(f'{main_dir}/data/{blood_type}.csv')
    day_cols = df.columns = df.columns.astype(int)
    df = pd.concat([df, chemo_df], axis=1)
    # remove rows with no blood count measurements
    mask = df[day_cols].notnull().sum(axis=1) > 0
    print("Number of rows before filtering =", len(df))
    df = df[mask]
    print("Number of rows after filtering =", len(df))
    small_df = df[df['regimen'].isin(top_regimens+regimens_of_interest)]
    return df, small_df


# # Cohort Numbers Before Exclusions

# In[6]:


df = prep.get_visit_date_feature(chemo_df, include_first_date=True)
mask = df['first_visit_date'] <= split_date
dev_cohort, test_cohort = df[mask], df[~mask]
print(f'Number of patients in development cohort who recieved cytotoxic chemotherapy: {dev_cohort["ikn"].nunique()}')
print(f'Number of patients in test cohort who recieved cytotoxic chemotherapy: {test_cohort["ikn"].nunique()}')


# # Model Data - Exploratory Data Analysis

# In[37]:


target_keyword = 'target_'
model_data = prep.get_data(missing_thresh=75, verbose=False)


# In[8]:


# analyze the distribution
blood_count_dist_plot(model_data, include_sex=True)


# In[35]:


# analyze relationship between features and targets
# NOTE: for sessions where heoglobin/platelet transfusion occurs from day 4 to 3 days after next chemo administration, 
#       target hemoglobin/platelet count is set to 0
sns.pairplot(model_data, 
             x_vars=[f'target_{bt}_count' for bt in blood_types], 
             y_vars=[f'baseline_{bt}_count' for bt in blood_types],
             plot_kws={'alpha': 0.1},
             height=3.0)


# In[36]:


sns.pairplot(prep.convert_labels(model_data), 
             vars=[f'baseline_{bt}_count' for bt in blood_types]+['age'],
             hue='Neutropenia',
             plot_kws={'alpha': 0.1})


# In[9]:


# analyse the pearson correlation
model_data, clip_thresholds = prep.clip_outliers(model_data, lower_percentile=0.05, upper_percentile=0.95)
pearson_matrix = get_pearson_matrix(model_data, target_keyword, output_path)
pearson_plot(pearson_matrix, output_path, main_target='neutrophil_count')


# # Neutrophil

# In[12]:


neutrophil_df, small_neutrophil_df = load_data(blood_type='neutrophil')


# In[43]:


# nadir summary (peak day, cytopenia rate) per regimen
summary_df = nadir_summary(neutrophil_df, main_dir, cytopenia='Neutropenia', load_ci=False)
summary_df.loc[top_regimens+regimens_of_interest]


# In[14]:


# Proportion of cytopenia events vs day since administration
event_rate_stacked_bar_plot(neutrophil_df, top_regimens, main_dir, cytopenia='Neutropenia')
event_rate_stacked_bar_plot(neutrophil_df, regimens_of_interest, main_dir, cytopenia='Neutropenia')


# In[15]:


day_dist_plot(neutrophil_df, top_regimens+regimens_of_interest)


# In[16]:


regimen_dist_plot(neutrophil_df, by='patients') # number of patients per cancer regiment
regimen_dist_plot(neutrophil_df, by='blood_counts') # number of blood counts per regimen
regimen_dist_plot(neutrophil_df, by='sessions') # number of sessions per regimen


# In[17]:


scatter_plot(small_neutrophil_df)


# In[18]:


fig = plt.figure(figsize=(15,75))
x = list(range(-5,29))
mask = neutrophil_df[x].notnull().sum(axis=1) > 25 # keep rows with more than 25 blood measurements
df = neutrophil_df[mask]
for i, (idx, row) in enumerate(df.iterrows()):
    y = row[x]
    ax = fig.add_subplot(20,2,i+1)
    plt.subplots_adjust(hspace=0.3)
    plt.scatter(x, y)
    plt.title(f"Patient: {int(row['ikn'])}, Regimen: {row['regimen']}, Chemo cycle: {row['chemo_cycle']}")
    plt.ylabel('Blood Count (10^9/L)')
    plt.xlabel('Day')


# In[19]:


below_threshold_bar_plot(small_neutrophil_df, threshold=blood_types['neutrophil']['cytopenia_threshold'])


# In[20]:


iqr_plot(small_neutrophil_df, show_outliers=False)
             # save=True, filename='neutrophil-plot2')


# In[21]:


violin_plot(small_neutrophil_df)


# In[22]:


mean_cycle_plot(small_neutrophil_df)


# # Hemoglobin

# In[23]:


hemoglobin_df, small_hemoglobin_df = load_data(blood_type='hemoglobin')


# In[44]:


# nadir summary (peak day, cytopenia rate) per regimen
summary_df = nadir_summary(hemoglobin_df, main_dir, cytopenia='Anemia', load_ci=False)
summary_df.loc[top_regimens+regimens_of_interest]


# In[25]:


# Proportion of cytopenia events vs day since administration
event_rate_stacked_bar_plot(hemoglobin_df, top_regimens, main_dir, cytopenia='Anemia')
event_rate_stacked_bar_plot(hemoglobin_df, regimens_of_interest, main_dir, cytopenia='Anemia')


# # Platelet

# In[26]:


platelet_df, small_platelet_df = load_data(blood_type='platelet')


# In[45]:


# nadir summary (peak day, cytopenia rate) per regimen
summary_df = nadir_summary(platelet_df, main_dir, cytopenia='Thrombocytopenia', load_ci=False)
summary_df.loc[top_regimens+regimens_of_interest]


# In[28]:


# Proportion of cytopenia events vs day since administration
event_rate_stacked_bar_plot(platelet_df, top_regimens, main_dir, cytopenia='Thrombocytopenia')
event_rate_stacked_bar_plot(platelet_df, regimens_of_interest, main_dir, cytopenia='Thrombocytopenia')


# # Sctach Notes

# ## Excluding Blood Count Measurements Taken During H/ED Visits

# In[21]:


from scripts.config import (all_observations, event_map)
from scripts.preprocess import (split_and_parallelize, group_observations, postprocess_olis_data, get_inpatient_indices)


# In[26]:


obs_codes = [obs_code for obs_code, obs_name in all_observations.items() if obs_name in blood_types]
olis_df = pd.read_csv(f'{main_dir}/data/olis_complete2.csv')
olis_df = olis_df[olis_df['observation_code'].isin(obs_codes)]
olis_df = pd.merge(olis_df, chemo_df[['visit_date', 'ikn']], left_on='chemo_idx', right_index=True, how='left')
olis_df['days_after_chemo'] = pd.to_timedelta(olis_df['days_after_chemo'], unit='D')
# visit date now refers to blood count measurement visit date instead of chemo visit date
olis_df['visit_date'] = olis_df['visit_date'] + olis_df['days_after_chemo']


# In[55]:


"""
ED visits only have registration date. The arrival date and depart date is the same. 
So for ED visits, we are essentially removing blood count measurement taken on THAT day.
For H visits, the arrival date and depart date are different. 
We remove blood count measurment taken while hospitalized between those dates.
"""
for event in ['H', 'ED']:
    database_name = event_map[event]['database_name']
    event_df = pd.read_csv(f'{root_path}/PROACCT/data/{database_name}.csv', dtype=str)
    for col in ['arrival_date', 'depart_date']: event_df[col] = pd.to_datetime(event_df[col])
    event_df['ikn'] = event_df['ikn'].astype(int)
    filtered_olis_df = olis_df[olis_df['ikn'].isin(event_df['ikn'])] # filter out patients not in dataset
    indices = split_and_parallelize((filtered_olis_df[['ikn','visit_date']], event_df), get_inpatient_indices, processes=16)
    np.save(f'{main_dir}/data/analysis/{event}_indices.npy', indices)


# In[27]:


for event in ['H', 'ED']:
    indices = np.load(f'{main_dir}/data/analysis/{event}_indices.npy')
    print(f'Number of blood count measurements taken during {event} visits: {len(indices)}')
    olis_df = olis_df[~olis_df.index.isin(indices)]


# In[28]:


max_cycle_length = 42
days_range = range(-5,max_cycle_length+1)
olis_df['days_after_chemo'] = olis_df['days_after_chemo'].dt.days
observations = {obs_code: obs_name for obs_code, obs_name in all_observations.items() if obs_name in blood_types}
mapping, _ = postprocess_olis_data(chemo_df, olis_df, observations, days_range=days_range)
freq_map = olis_df['observation_code'].value_counts()
grouped_observations = group_observations(observations, freq_map)

for blood_type, blood_info in blood_types.items():
    obs_codes = grouped_observations[blood_type]
    for i, obs_code in enumerate(obs_codes):
        if i == 0: 
            df = mapping[obs_code]
        else: 
            df = df.fillna(mapping[obs_code])
            
    df = pd.concat([df, chemo_df], axis=1)
    mask = df[days_range].notnull().sum(axis=1) > 0 # remove rows with no blood count measurements
    df = df[mask]
    
    event_rate_stacked_bar_plot(df, top_regimens, main_dir, cytopenia=blood_info['cytopenia_name'], save=False)
    event_rate_stacked_bar_plot(df, regimens_of_interest, main_dir, cytopenia=blood_info['cytopenia_name'], save=False)


# ## Top Regimens for Sex, Age

# In[24]:


top_regimens_by_category = {}
for sex, group in chemo_df.groupby('sex'):
    top_regimens_by_sessions = most_common_by_category(group, category='regimen', top=3)
    top_regimens_by_category[sex] = top_regimens_by_sessions
    
mask = chemo_df['age'] >= 65
top_regimens_by_category['65 and over'] = most_common_by_category(chemo_df[mask], category='regimen', top=3)
top_regimens_by_category['under 65'] = most_common_by_category(chemo_df[~mask], category='regimen', top=3)


# In[31]:


for category, regimens in top_regimens_by_category.items():
    print(f'######################### {category.upper()} #########################')
    if category in {'F', 'M'}: 
        mask = neutrophil_df['sex'] == category
    elif category == '65 and over': 
        mask = neutrophil_df['age'] >= 65
    if category == 'under 65': 
        mask = neutrophil_df['age'] < 65
    event_rate_stacked_bar_plot(neutrophil_df[mask], regimens, main_dir, cytopenia='Neutropenia', save=False)


# In[ ]:
