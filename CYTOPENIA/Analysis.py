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


# In[36]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.utility import get_pearson_matrix, most_common_categories
from src.summarize import top_cancer_regimen_summary, nadir_summary
from src.visualize import (
    below_threshold_bar_plot, iqr_plot, event_rate_stacked_bar_plot, 
    blood_count_dist_plot, regimen_dist_plot, day_dist_plot,
    scatter_plot, violin_plot, mean_cycle_plot, 
    pearson_plot
)
from src.config import (
    root_path, cyto_folder, split_date, 
    blood_types, cytopenia_grades,
    DATE
)
from src.prep_data import PrepDataCYTO


# In[37]:


main_dir = f'{root_path}/{cyto_folder}'
prep = PrepDataCYTO()
chemo_df = prep.load_data()


# In[38]:


def load_data(chemo_df, blood_type='neutrophil'):
    df = pd.read_parquet(f'{main_dir}/data/{blood_type}.parquet.gzip')
    df.columns = day_cols = df.columns.astype(int)
    
    # add other important info
    cols = ['ikn', 'regimen', 'cycle_length', 'chemo_cycle']
    df[cols] = chemo_df[cols]
    
    # remove rare regimens / regimen sets (10 patients or less)
    patient_count = df.groupby('regimen').apply(lambda g: g['ikn'].nunique())
    drop_regimens = patient_count.index[patient_count <= 10]
    mask = df['regimen'].isin(drop_regimens)
    print(f'Removing {len(drop_regimens)} rare regimens/regimen sets out of {df["regimen"].nunique()} total regimens')
    print(f'Resulting in {sum(mask)} excluded sessions out of {len(mask)} total sessions')
    df = df[~mask]
    
    # remove rows with no blood count measurements
    mask = df[day_cols].isnull().all(axis=1)
    print(f'Removing {sum(mask)} sessions without any {blood_type} measurements in the time window')
    df = df[~mask]
        
    return df


# # Regimen Analysis

# In[39]:


top_cancer_regimen_summary(chemo_df)


# In[40]:


# regimens of interest
top_regimens = list(most_common_categories(chemo_df, catcol='regimen', top=3))
regimens_of_interest = ['mfolfirinox', 'folfirinox', 'gemcnpac(w)']
test_regimens = ['pcv', 'fulcvr', 'ac-doce', 'vino', 'vnbl'] # each has different cycle lengths


# # Cohort Numbers Before and After Exclusions

# In[7]:


# cohort after exclusions (of treatments without one or more of baseline/target blood values)
model_data = prep.get_data()
dev_cohort, test_cohort = prep.create_cohort(model_data, split_date, verbose=False)

# cohort before exclusions
first_visit_date = chemo_df.groupby('ikn')[DATE].min()
mask = chemo_df['ikn'].map(first_visit_date) <= split_date
dev_cohort2, test_cohort2 = chemo_df[mask], chemo_df[~mask]


# In[8]:


show = lambda x: f"NSessions={len(x)}. NPatients={x['ikn'].nunique()}"
cohorts = {'Development': (dev_cohort, dev_cohort2), 'Testing': (test_cohort, test_cohort2)}
for name, (post_exc_cohort, pre_exc_cohort) in cohorts.items():
    print(f'{name} cohort')
    print(f'Before exclusions: {show(pre_exc_cohort)}')
    print(f'After exclusions: {show(post_exc_cohort)}\n')


# # Blood Count Exploratory Data Analysis

# In[9]:


target_cols = [f'target_{bt}_value' for bt in blood_types]
baseline_cols = [f'baseline_{bt}_value' for bt in blood_types]
model_data = prep.get_data(missing_thresh=75, verbose=False)


# In[104]:


# analyze the distribution
blood_count_dist_plot(model_data, include_sex=True)


# In[105]:


get_ipython().run_cell_magic('time', '', "# analyze relationship between features and targets\n# NOTE: for sessions where heoglobin/platelet transfusion occurs from day 4 to 3 days after next chemo administration, \n#       target hemoglobin/platelet count is set to 0\nsns.pairplot(model_data, x_vars=target_cols, y_vars=baseline_cols, plot_kws={'alpha': 0.1}, height=3.0)")


# In[107]:


get_ipython().run_cell_magic('time', '', "sns.pairplot(\n    prep.convert_labels(model_data.copy()), \n    vars=baseline_cols+['age'],\n    hue='Neutropenia',\n    plot_kws={'alpha': 0.1}\n)")


# In[110]:


# analyse the pearson correlation
model_data = prep.clip_outliers(model_data, lower_percentile=0.05, upper_percentile=0.95)
pearson_matrix = get_pearson_matrix(model_data, target_keyword='target_')
pearson_plot(pearson_matrix, main_target='neutrophil_value')


# # Neutrophil

# In[52]:


neutrophil_df = load_data(chemo_df, blood_type='neutrophil')
mask = neutrophil_df['regimen'].isin(top_regimens+regimens_of_interest)
small_neutrophil_df = neutrophil_df[mask]


# In[188]:


# nadir summary (peak day, cytopenia rate) per regimen
summary_df = nadir_summary(neutrophil_df, main_dir, cyto='Neutropenia', load_ci=True)
summary_df.loc[top_regimens+regimens_of_interest]


# In[191]:


# Proportion of cytopenia events vs day since administration
event_rate_stacked_bar_plot(neutrophil_df, top_regimens, main_dir, cytopenia='Neutropenia')
event_rate_stacked_bar_plot(neutrophil_df, regimens_of_interest, main_dir, cytopenia='Neutropenia')


# In[11]:


day_dist_plot(neutrophil_df, top_regimens+regimens_of_interest)


# In[13]:


regimen_dist_plot(neutrophil_df, by='patients') # number of patients per cancer regiment
regimen_dist_plot(neutrophil_df, by='blood_counts') # number of blood counts per regimen
regimen_dist_plot(neutrophil_df, by='sessions') # number of sessions per regimen


# In[14]:


scatter_plot(small_neutrophil_df)


# In[95]:


for cycle_length, group in neutrophil_df.groupby('cycle_length'):
    if cycle_length in [7, 42]: continue
    days = range(-5, cycle_length+1)
    mes_counts = group[days].notnull().sum(axis=1)
    # keep rows where almost all days in the time window had blood measurements
    thresh = {14: 2, 21: 2, 28: 13}
    mask = mes_counts >= len(days) - thresh[cycle_length]
    df = group[mask]
    samples = df.sample(n=10, random_state=42)
    
    fig, axes = plt.subplots(5, 2, figsize=(20,25))
    axes = axes.flatten()
    plt.subplots_adjust(hspace=0.3)
    for i, (idx, row) in enumerate(samples.iterrows()):
        y = row[days]
        axes[i].scatter(days, y)
        axes[i].set(
            title=f"Patient={row['ikn']}. Regimen={row['regimen']}. Chemo cycle={row['chemo_cycle']}",
            ylabel='Neutrophil Count (10^9/L)',
            xlabel='Day',
            xticks=days
        )


# In[97]:


below_threshold_bar_plot(small_neutrophil_df, threshold=cytopenia_grades['Grade 2']['Neutropenia'])


# In[98]:


iqr_plot(small_neutrophil_df, show_outliers=False)


# In[99]:


violin_plot(small_neutrophil_df)


# In[100]:


mean_cycle_plot(small_neutrophil_df)


# # Hemoglobin

# In[101]:


hemoglobin_df = load_data(chemo_df, blood_type='hemoglobin')
mask = hemoglobin_df['regimen'].isin(top_regimens+regimens_of_interest)
small_hemoglobin_df = hemoglobin_df[mask]


# In[104]:


# nadir summary (peak day, cytopenia rate) per regimen
summary_df = nadir_summary(hemoglobin_df, main_dir, cyto='Anemia', load_ci=True)
summary_df.loc[top_regimens+regimens_of_interest]


# In[105]:


# Proportion of cytopenia events vs day since administration
event_rate_stacked_bar_plot(hemoglobin_df, top_regimens, main_dir, cytopenia='Anemia')
event_rate_stacked_bar_plot(hemoglobin_df, regimens_of_interest, main_dir, cytopenia='Anemia')


# # Platelet

# In[7]:


platelet_df = load_data(chemo_df, blood_type='platelet')
mask = platelet_df['regimen'].isin(top_regimens+regimens_of_interest)
small_platelet_df = platelet_df[mask]


# In[107]:


# nadir summary (peak day, cytopenia rate) per regimen
summary_df = nadir_summary(platelet_df, main_dir, cyto='Thrombocytopenia', load_ci=False)
summary_df.loc[top_regimens+regimens_of_interest]


# In[8]:


# Proportion of cytopenia events vs day since administration
event_rate_stacked_bar_plot(platelet_df, top_regimens, main_dir, cytopenia='Thrombocytopenia')
event_rate_stacked_bar_plot(platelet_df, regimens_of_interest, main_dir, cytopenia='Thrombocytopenia')


# # Excluding Lab Measurements Taken During H/ED Visits

# In[27]:


from src.config import DATE, all_observations, event_map
from src.utility import split_and_parallelize, group_observations
from src.preprocess import process_lab_data, filter_event_data, get_inpatient_idxs


# In[45]:


lab = pd.read_parquet(f'{main_dir}/data/lab.parquet.gzip')
orig_chemo_idx = np.load(f'{main_dir}/analysis/orig_chemo_idx.npy')
tmp = chemo_df.set_index(orig_chemo_idx)[[DATE, 'ikn', 'regimen', 'cycle_length']]
lab = pd.merge(lab, tmp, left_on='chemo_idx', right_index=True, how='left')
lab['days_after_chemo'] = pd.to_timedelta(lab['days_after_chemo'], unit='d')
lab[DATE] = lab[DATE] + lab['days_after_chemo'] # let DATE refer to the lab test date instead of treatment date


# In[30]:


get_ipython().run_cell_magic('time', '', '"""\nED visits only have registration date. The arrival date and depart date is the same. \nSo for ED visits, we are essentially removing blood count measurement taken on THAT day.\n\nH visits, have different arrival and depart date. \nSo for H vistis, we remove blood count measurments taken while hospitalized between those dates.\n"""\nfor event, database in {\'H\': \'dad\', \'ED\': \'nacrs\'}.items():\n    event_df = pd.read_parquet(f\'{root_path}/data/{database}.parquet.gzip\')\n    event_df[\'ikn\'] = event_df[\'ikn\'].astype(int)\n    event_df = filter_event_data(event_df, lab[\'ikn\'], event=event)\n    mask = lab[\'ikn\'].isin(event_df[\'ikn\']) \n    idxs = split_and_parallelize((lab.loc[mask, [\'ikn\', DATE]], event_df), get_inpatient_idxs, processes=16)\n    np.save(f\'{main_dir}/analysis/{event}_idxs.npy\', idxs)')


# In[46]:


for event in ['H', 'ED']:
    idxs = np.load(f'{main_dir}/analysis/{event}_idxs.npy')
    print(f'Number of blood count measurements taken during {event} visits: {len(idxs)}')
    lab = lab[~lab.index.isin(idxs)]
print(f'Number of blood count measurements left: {len(lab)}')


# In[47]:


get_ipython().run_cell_magic('time', '', "days = range(-5,43)\nlab['days_after_chemo'] = lab['days_after_chemo'].dt.days\nmask = lab['obs_code'].isin(all_observations)\nlab = lab[mask]\nlab_map = process_lab_data(lab, tmp)\n        \ngrouped_observations = group_observations(all_observations, lab['obs_code'].value_counts())\n\nfor blood_type, blood_info in blood_types.items():\n    obs_codes = grouped_observations[blood_type]\n    for i, obs_code in enumerate(obs_codes):\n        df = lab_map[obs_code] if i == 0 else df.fillna(lab_map[obs_code])\n            \n    df = pd.concat([df, tmp], axis=1)\n    mask = df[days].notnull().sum(axis=1) > 0 # remove rows with no blood count measurements\n    df = df[mask]\n    \n    event_rate_stacked_bar_plot(df, top_regimens, main_dir, cytopenia=blood_info['cytopenia_name'], save=False)\n    event_rate_stacked_bar_plot(df, regimens_of_interest, main_dir, cytopenia=blood_info['cytopenia_name'], save=False)")


# # Growth Factor

# In[61]:


over_65 = chemo_df.query('age >= 65')
regimen_with_gf_count = over_65.query('GF_given')['regimen'].value_counts()
regimen_count = over_65['regimen'].value_counts()
regimen_count = regimen_count.loc[regimen_with_gf_count.index]
(regimen_with_gf_count / regimen_count).sort_values(ascending=False).to_dict()


# In[ ]:
