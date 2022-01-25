#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('cd', '../')
# reloads all modules everytime before cell is executed (no need to restart kernel)
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[1]:


import sys


# In[2]:


env = 'myenv'
user_path = 'XXXXX'
for i, p in enumerate(sys.path):
    sys.path[i] = sys.path[i].replace("/software/anaconda/3/", f"{user_path}/.conda/envs/{env}/")
sys.prefix = f'{user_path}/.conda/envs/{env}/'


# In[3]:


import tqdm
import pandas as pd
import numpy as np
import utilities as util

import matplotlib.pyplot as plt

from scripts.config import (root_path)


# In[5]:


output_path = f'{root_path}/cytopenia'

df = util.read_partially_reviewed_csv()
df = util.get_included_regimen(df)
cycle_lengths = df['cycle_length'].to_dict()

chemo_df = pd.read_csv(f'{output_path}/data/chemo_processed.csv')


# # Neutrophil

# In[20]:


neutrophil_df = pd.read_csv(f'{output_path}/data/neutrophil.csv')
neutrophil_df.columns = neutrophil_df.columns.astype(int)
neutrophil_df = pd.concat([neutrophil_df, chemo_df], axis=1)
print("number of rows =", len(neutrophil_df))

# keep rows that have at least 2 blood count measures
mask = (~neutrophil_df[range(-5,29)].isnull()).sum(axis=1) >= 2
neutrophil_df = neutrophil_df[mask]
print("number of rows after filtering =", len(neutrophil_df))


# In[22]:


# number of patients per cancer regiment
util.num_patients_per_regimen(neutrophil_df)


# In[23]:


# number of blood counts per regimen
util.num_blood_counts_per_regimen(neutrophil_df)


# In[24]:


# histogram of numbers of blood counts measured (for a single row)
util.hist_blood_counts(neutrophil_df)


# In[17]:


util.scatter_plot(neutrophil_df, cycle_lengths)


# In[25]:


fig = plt.figure(figsize=(15,75))
i = 1
for idx, row in tqdm.tqdm(neutrophil_df.iterrows(), total=len(neutrophil_df)):
    y = row[range(-5,29)].values
    x = list(range(-5,29))

    # if more than 25 blood measurements
    if (~pd.isnull(y)).sum() > 25:
        ax = fig.add_subplot(20,2,i)
        plt.subplots_adjust(hspace=0.3)
        i += 1
        plt.scatter(x, y)
        plt.title(f"Patient: {int(row['ikn'])}, Regimen: {row['regimen']}, Chemo cycle: {row['chemo_cycle']}")
        plt.ylabel('Blood Count (10^9/L)')
        plt.xlabel('Day')

plt.show()


# In[26]:


neutrophil_threshold = 1.5
util.below_threshold_bar_plot(neutrophil_df, cycle_lengths, neutrophil_threshold)


# In[27]:


util.iqr_plot(neutrophil_df, cycle_lengths)
             # show_outliers=False, save=True, filename='neutrophil-plot2')


# In[28]:


util.mean_cycle_plot(neutrophil_df, cycle_lengths)


# # Hemoglobin

# In[29]:


hemoglobin_df = pd.read_csv(f'{output_path}/data/hemoglobin.csv')
hemoglobin_df.columns = hemoglobin_df.columns.astype(int)
hemoglobin_df = pd.concat([hemoglobin_df, chemo_df], axis=1)
print("number of rows =", len(hemoglobin_df))

# keep rows that have at least 2 blood count measures
mask = (~hemoglobin_df[range(-5,29)].isnull()).sum(axis=1) >= 2
hemoglobin_df = hemoglobin_df[mask]
print("number of rows after filtering =", len(hemoglobin_df))


# In[30]:


util.scatter_plot(hemoglobin_df, cycle_lengths, unit='g/L')


# In[31]:


hemoglobin_threshold = 100
util.below_threshold_bar_plot(hemoglobin_df, cycle_lengths, hemoglobin_threshold)


# In[32]:


util.iqr_plot(hemoglobin_df, cycle_lengths, unit='g/L')
             # show_outliers=False, save=True, filename='hemoglobin-plot2')


# In[33]:


util.mean_cycle_plot(hemoglobin_df, cycle_lengths, unit='g/L')


# # Platelet

# In[34]:


platelet_df = pd.read_csv(f'{output_path}/data/platelet.csv')
platelet_df.columns = platelet_df.columns.astype(int)
platelet_df = pd.concat([platelet_df, chemo_df], axis=1)
print("number of rows =", len(platelet_df))

# keep rows that have at least 2 blood count measures
mask = (~platelet_df[range(-5,29)].isnull()).sum(axis=1) >= 2
platelet_df = platelet_df[mask]
print("number of rows after filtering =", len(platelet_df))


# In[35]:


util.scatter_plot(platelet_df, cycle_lengths)


# In[36]:


platelet_threshold = 75
util.below_threshold_bar_plot(platelet_df, cycle_lengths, platelet_threshold)


# In[37]:


util.iqr_plot(platelet_df, cycle_lengths)
             # show_outliers=False, save=True, filename='platelet-plot2')


# In[38]:


util.mean_cycle_plot(platelet_df, cycle_lengths)


# ## IQR playground

# In[22]:


plt.boxplot([[0, 0.1, 0.7],[0.9,0.8,1.5]], labels=['y','x'])
plt.show()


# # Different Thresholds

# In[39]:


for (neutrophil_threshold, color) in [(1.5, None), (1.0, 'blue'), (0.5, 'green')]:
    print('###################################################################################################################')
    print(f'############################################ NEUTROPHIL THRESHOLD {neutrophil_threshold} #############################################')
    print('###################################################################################################################')
    util.below_threshold_bar_plot(neutrophil_df, cycle_lengths, neutrophil_threshold, color=color) 
                                  # save=True, filename=f"neutrophil-plot1-threshold{neutrophil_threshold}")


# In[41]:


for (hemoglobin_threshold, color) in [(100, None), (80, 'green')]:
    print('###################################################################################################################')
    print(f'############################################ HEMOGLOBIN THRESHOLD {hemoglobin_threshold} #############################################')
    print('###################################################################################################################')
    util.below_threshold_bar_plot(hemoglobin_df, cycle_lengths, hemoglobin_threshold, color=color)
                                 # save=True, filename=f"hemoglobin-plot1-threshold{hemoglobin_threshold}")


# In[40]:


for (platelet_threshold, color) in [(75, None), (50, 'blue'), (25, 'green')]:
    print('###################################################################################################################')
    print(f'############################################# PLATELET THRESHOLD {platelet_threshold} ###############################################')
    print('###################################################################################################################')
    util.below_threshold_bar_plot(platelet_df, cycle_lengths, platelet_threshold, color=color)
                                 # save=True, filename=f"platelet-plot1-threshold{platelet_threshold}")


# # How to Display Saved Images

# In[34]:


# how to display the saved images
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import Image
Image(f'{output_path}/plots/neutrophil-plot1-threshold1.5.jpg')


# # IQR plots based on sex/age

# In[44]:


top5_regimens = neutrophil_df['regimen'].value_counts().index.tolist()[0:5]
blood_types = ['neutrophil', 'hemoglobin', 'platelet']


# In[45]:


for idx, blood_df in enumerate([neutrophil_df, hemoglobin_df, platelet_df]):
    df = blood_df.copy()
    df = df[df['regimen'].isin(top5_regimens)]
    print('###################################################################################################################')
    print(f'#################################################### {blood_types[idx]} ###################################################')
    print('###################################################################################################################')
    iqr_plot_by_sex(df, cycle_lengths, show_outliers=False, figsize=(15,20))


# In[46]:


for idx, blood_df in enumerate([neutrophil_df, hemoglobin_df, platelet_df]):
    df = blood_df.copy()
    df = df[df['regimen'].isin(top5_regimens)]
    df['over60'] = df['age'] >= 60
    print('###################################################################################################################')
    print(f'#################################################### {blood_types[idx]} ###################################################')
    print('###################################################################################################################')
    iqr_plot_by_age(df, cycle_lengths, show_outliers=False, figsize=(15,20))


# In[42]:


def iqr_plot_by_sex(df, cycle_lengths, unit='10^9/L', show_outliers=True, save=False, 
                    filename='NEUTROPHIL_PLOT3', figsize=(15,150)):
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(hspace=0.3)
    num_regimen = len(set(df['regimen']))
    counter = 1
    for regimen, group_reg in df.groupby('regimen'):
        cycle_length = int(cycle_lengths[regimen])
        for sex, group in group_reg.groupby('sex'):
            data = np.array([group[day].dropna().values for day in range(0,cycle_length+1)], dtype=object)
            ax = fig.add_subplot(num_regimen,2,counter)
            bp = plt.boxplot(data, labels=range(0,cycle_length+1), showfliers=show_outliers)
            plt.title(regimen+' - Sex'+sex)
            plt.ylabel(f'Blood Count ({unit})')
            plt.xlabel('Day')
            medians = [median.get_ydata()[0] for median in bp['medians']]
            plt.plot(range(1,cycle_length+2), medians, color='red')
            counter += 1
    if save:
        plt.savefig(f'{output_path}/plots/{filename}.jpg', bbox_inches='tight')    
    plt.show()


# In[43]:


def iqr_plot_by_age(df, cycle_lengths, unit='10^9/L', show_outliers=True, save=False, 
                    filename='NEUTROPHIL_PLOT3', figsize=(15,150)):
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(hspace=0.3)
    num_regimen = len(set(df['regimen']))
    counter = 1
    for regimen, group_reg in df.groupby('regimen'):
        cycle_length = int(cycle_lengths[regimen])
        for over60, group in group_reg.groupby('over60'):
            data = np.array([group[day].dropna().values for day in range(0,cycle_length+1)], dtype=object)
            ax = fig.add_subplot(num_regimen,2,counter)
            bp = plt.boxplot(data, labels=range(0,cycle_length+1), showfliers=show_outliers)
            plt.title(regimen+' - Over60'+str(over60))
            plt.ylabel(f'Blood Count ({unit})')
            plt.xlabel('Day')
            medians = [median.get_ydata()[0] for median in bp['medians']]
            plt.plot(range(1,cycle_length+2), medians, color='red')
            counter += 1
    if save:
        plt.savefig(f'{output_path}/plots/{filename}.jpg', bbox_inches='tight')    
    plt.show()


# In[ ]:




