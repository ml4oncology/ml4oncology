#!/usr/bin/env python
# coding: utf-8

# use <b>./kevin_launch_jupyter-notebook_webserver.sh</b> instead of <b>launch_jupyter-notebook_webserver.sh</b> if you want to increase buffer memory size so we can load greater filesizes

# In[1]:


get_ipython().run_line_magic('cd', '../')
# reloads all modules everytime before cell is executed (no need to restart kernel)
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import sys


# In[3]:


env = 'myenv'
user_path = 'XXXXXX'
for i, p in enumerate(sys.path):
    sys.path[i] = sys.path[i].replace("/software/anaconda/3/", f"{user_path}/.conda/envs/{env}/")
sys.prefix = f'{user_path}/.conda/envs/{env}/'


# In[80]:


import tqdm
import pandas as pd
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from scripts.utilities import (get_clean_variable_names)
from scripts.config import (root_path, regiments_folder)
from scripts.preprocess import (clean_string, get_y3, get_systemic)
from scripts.prep_data import (PrepData, PrepDataEDHD)
from scripts.train import (Train)


# In[5]:


y3 = get_y3()
y3 = clean_string(y3, ['sex'])
y3 = y3[~(y3['sex'] == 'O')]
y3['bdate'] = pd.to_datetime(y3['bdate'])
y3['dthdate'] = pd.to_datetime(y3['dthdate'])


# # Age Demographic

# In[6]:


data = y3.copy()
today = pd.to_datetime('today')
data['dthdate'] = data['dthdate'].fillna(today) # to get age, fill missing death date using today's date
born = data['bdate'].dt
died = data['dthdate'].dt
data['age'] = died.year - born.year - ((died.month) < (born.month))
data = data[~((data['age'] > 110) & (data['dthdate'] == today))] # do not include patients over 110 years old today


# In[7]:


fig = plt.figure(figsize=(15, 4))
for sex, group in data.groupby('sex'):
    plt.hist(group['age'], alpha=0.5, label=sex, bins=110)
plt.xticks(range(0,110,4))
plt.legend()
plt.show()


# # Death vs Age

# In[8]:


data = y3.copy()
born = data['bdate'].dt
died = data['dthdate'].dt
data['age'] = died.year - born.year - ((died.month) < (born.month))
data = data[~data['age'].isnull()]


# In[9]:


fig = plt.figure(figsize=(15, 4))
for sex, group in data.groupby('sex'):
    plt.hist(group['age'], alpha=0.5, label=sex, bins=109)
plt.xticks(range(0,109,4))
plt.legend()
plt.show()


# # Target within 14, 30, 180, 365 days - Label Distribution

# In[10]:


def get_distribution():
    df = pd.read_csv(f'{root_path}/ED-H-D/data/model_data.csv', dtype={'curr_morph_cd': str, 'lhin_cd': str})
    cols = df.columns
    target_cols = cols[cols.str.contains('within')]
    prep = PrepData()
    
    results = []
    for days in [14, 30, 180, 365]:
        keep_cols = target_cols[target_cols.str.contains(str(days))]
        result = prep.get_label_distribution(df[keep_cols])
        result.columns = pd.MultiIndex.from_arrays([(f'{days} days', f'{days} days'), ('False', 'True')])
        result.index = result.index.str.replace(f'_within_{days}days', '')
        results.append(result)
    return pd.concat(results, axis=1)


# In[11]:


get_distribution()


# # Month Distribution

# In[16]:


df = pd.read_csv(f'{root_path}/{regiments_folder}/regiments_EDHD.csv', dtype=str)
regimens_keep = df.loc[df['include'] == '1', 'regimens'].values
del df
systemic = get_systemic(regimens_keep)


# In[37]:


fig = plt.figure(figsize=(15, 4))
labels, counts = np.unique(systemic['visit_date'].dt.month, return_counts=True)
all(labels == np.arange(1, 13))
labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.bar(labels, counts, align='center')
plt.show()


# In[51]:


mapping = {'Winter': [12,1,2], 'Spring': [3,4,5], 'Summer': [6,7,8], 'Fall': [9,10,11]}
mapping = {month: season for season, months in mapping.items() for month in months}
labels, counts = np.unique(systemic['visit_date'].dt.month.map(mapping), return_counts=True)
plt.bar(labels[[1, 2, 0, 3]], counts[[1, 2, 0, 3]], align='center')
plt.show()


# # All Lab Tests/Blood Work

# In[55]:


olis = pd.read_csv(f'{root_path}/data/olis.csv', dtype=str) 
observation_olis = olis['ObservationCode'].value_counts()
del olis
observation_olis.index = observation_olis.index.str[2:-1] # clean observation code string
observation_olis = observation_olis[~(observation_olis < 10000)] # must have at least 10000 observations
exclude = ['XON10382-0', 'XON10383-8', 'XON12394-3', 'XON12400-8'] # exclude these codes
observation_olis = observation_olis.drop(index=exclude)
observations = sorted(list(observation_olis.items()))
print(f"Number of unique observation codes: {len(observations)}")
observations


# In[6]:


olis_blood_count = pd.read_csv(f'{root_path}/data/olis_blood_count.csv', dtype=str) 
observation_olis_blood_count = olis_blood_count['ObservationCode'].value_counts()
del olis_blood_count
observation_olis_blood_count.index = observation_olis_blood_count.index.str[2:-1]
observation_olis_blood_count = observation_olis_blood_count[~(observation_olis_blood_count < 10000)]
observations = sorted(list(observation_olis_blood_count.items()))
print(f"Number of unique observation codes: {len(observations)}")
observations


# In[56]:


# for observation codes that exist in both datasets, check both datasets contain the same information
code_in_both_datasets = observation_olis_blood_count.index.intersection(observation_olis.index)
assert(all(observation_olis_blood_count[code_in_both_datasets] == observation_olis[code_in_both_datasets]))


# In[6]:


data = [# Alanine Aminotransferase
        ['1742-6', 'Alanine Aminotransferase (Enzymatic Activity/Volume)'],
        ['1744-2', 'Alanine Aminotransferase without Vitamin B6'],
        ['1743-4', 'Alanine Aminotransferase with Vitamin B6'],
    
        # Albumin
        ['1751-7', 'Albumin in Serum/Plasma'], # Does not exists in either data
        ['1754-1', 'Albumin in Urine'],
        ['32294-1', 'Albumin/Creatinine in Urine (Ratio)'],
    
        # Basophils
        ['704-7', 'Basophils in Blood (#/Volume): automated count'],
        ['705-4', 'Basophils in Blood (#/Volume): manual count'],
        ['26444-0', 'Basophils in Blood (#/Volume)'],
        
        # Bilirubin
        ['14629-0', 'Bilirubin - Direct'], # Direct bilirubin = glucoronidated bilirubin + albumin bound bilirubin
        ['29760-6', 'Bilirubin - Glucoronidated'],
        ['14630-8', 'Bilirubin - Indirect'],
        ['14631-6', 'Bilirubin - Total'], # Total bilirubin = direct bilirubin + indirect bilirubin
    
        # Calcium
        ['2000-8', 'Calcium'],
        ['29265-6', 'Calcium corrected for albumin'],
        ['18281-6', 'Calcium corrected for total protein'],
    
        # Cholestrol - None of these exists in either data
        ['14647-2', 'Cholestrol'],
        ['14646-4', 'Cholestrol in HDL (high density lipoprotein) (Moles/Volume)'],
        ['39469-2', 'Cholestrol in LDL (low density lipoprotein) (Moles/Volume): calculated'],
        ['32309-7', 'Cholestrol in HDL (Molar Ratio)'],
        ['70204-3', 'Cholestrol not in HDL (Moles/Volume)'],
    
        # Creatinine - None of these exists in either data
        ['14682-9', 'Creatinine in Serum/Plasma'],
        ['14683-7', 'Creatinine in Urine'],
    
        # Eosinophils
        ['711-2', 'Eosinophils in Blood (#/Volume): automated count'],
        ['712-0', 'Eosinophils in Blood (#/Volume): manual count'],
        ['26449-9', 'Eosinophils in Blood (#/Volume)'],
    
        # Erythrocyte
        ['788-0', 'Erythrocyte distribution width (Ratio): automated count'],
        ['30385-9', 'Erythrocyte distribution width (Ratio)'],
        ['21000-5', 'Erythrocyte distribution width (Entitic Volume): automated count'],
        ['30384-2', 'Erythrocyte distribution width (Entitic Volume)'],
        ['789-8', 'Erythrocytes in Blood (#/Volume): automated count'],
        ['790-6', 'Erythrocytes in Blood (#/Volume): manual count'],
        ['26453-1', 'Erythrocytes in Blood (#/Volume)'],
        ['787-2', 'Erythrocyte MCV (mean corpuscular volume) (Entitic Volume): automated count'],
        ['30428-7', 'Erythrocyte MCV (mean corpuscular volume) (Entitic Volume)'],
        ['786-4', 'Erythrocyte MCHC (mean corpuscular hemoglobin concentration) (Mass/Volume): automated count'],
        ['28540-3', 'Erythrocyte MCHC (mean corpuscular hemoglobin concentration) (Mass/Volume)'],
        ['785-6', 'Erythrocyte MCH (mean corpuscular hemoglobin) (Entitic Mass): automated count'],
        ['28539-5', 'Erythrocyte MCH (mean corpuscular hemoglobin) (Entitic Mass)'],
    
        # Prostate specific antigen
        ['12841-3', 'Free PSA (Prostate Specific Antigen)/ Total PSA (Mass Fraction)'],
        ['10886-0', 'Free PSA (Prostate Specific Antigen) (Mass/Volume)'],
        ['2857-1', 'Total Prostate specific antigen (Mass/Volume)'], # Total = PSA&alpha1 + free PSA (+ PSA&alpha2?)
        ['35741-8', 'Total Prostate specific antigen (Mass/Volume): detection limit <= 0.01 ng/mL'],
        ['19197-3', 'Total Prostate specific antigen (Moles/Volume)'],
    
        # Glucose - None of these exists in either data
        ['14749-6', 'Glucose'],
        ['14771-0', 'Glucose^Post CFst (Fasting Glucose)'],
    
        # Hematocrit
        ['4544-3', 'Hematocrit of Blood (Volume Fraction): automated count'],
        ['71833-8', 'Hematocrit of Blood (Pure Volume Fraction): automated count'], # same as 4544-3 but no units
        ['20570-8', 'Hematocrit of Blood (Volume Fraction)'],
    
        # Hemoglobin
        ['718-7', 'Hemoglobin in Blood (Mass/Volume)'],
        ['20509-6', 'Hemoglobin in Blood (Mass/Volume): calculation'],
        ['4548-4', 'Hemoglobin A1c / Total Hemoglobin in Blood'],
        ['71875-9', 'Hemoglobin A1c / Total Hemoglobin in Blood (Pure Mass Fraction)'], # same as 4548-4 but no units?
        ['17855-8', 'Hemoglobin A1c / Total Hemoglobin in Blood: calculation'], 
        ['17856-6', 'Hemoglobin A1c / Total Hemoglobin in Blood: HPLC'],
        ['59261-8', 'Hemoglobin A1c / Total Hemoglobin in Blood: IFCC'],
    
        # Ionized Calcium
        ['1995-0', 'Ionized Calcium in Serum/Plasma (Moles/Volume)'],
        ['19072-8', 'Ionized Calcium in Serum/Plasma adjusted to pH 7.4 (Moles/Volume)'],
        ['12180-6', 'Ionized Calcium in Serum/Plasma (Moles/Volume): ISE'],
        ['1994-3', 'Ionized Calcium in Blood (Moles/Volume)'],
        ['47598-8', 'Ionized Calcium in Blood adjusted to pH 7.4 (Moles/Volume)'],
        ['34581-9', 'Ionized Calcium in Arterial Blood (Moles/Volume)'],
        ['41645-3', 'Ionized Calcium in Venous Blood (Moles/Volume)'],
        ['59473-9', 'Ionized Calcium in Venous Blood adjusted to pH 7.4 (Mass/Volume): ISE'],
    
        # Leukocytes
        ['6690-2', 'Leukocytes in Blood (#/Volume): automated count'],
        ['804-5', 'Leukocytes in Blood (#/Volume): manual count'],
        ['26464-8', 'Leukocytes in Blood (#/Volume)'],
    
        # Lymphocytes
        ['731-0', 'Lymphocytes in Blood (#/Volume): automated count'],
        ['732-8', 'Lymphocytes in Blood (#/Volume): manual count'],
        ['26474-7', 'Lymphocytes in Blood (#/Volume)'],
    
        # Microalbumin
        ['14957-5', 'Microalbumin in Urine'],
        ['14959-1', 'Microalbumin/Creatinine in Urine (Mass Ratio)'],
        ['30000-4', 'Microalbumin/Creatinine in Urine (Ratio)'],
    
        # Monocytes
        ['742-7', 'Monocytes in Blood (#/Volume): automated count'],
        ['743-5', 'Monocytes in Blood (#/Volume): manual count'],
        ['26484-6', 'Monocytes in Blood (#/Volume)'],
    
        # Neutrophil
        ['751-8', 'Neutrophils in Blood (#/Volume): automated count'],
        ['753-4', 'Neutrophils in Blood (#/Volume): manual count'],
        ['26499-4', 'Neutrophil in Blood (#/Volume)'], 
    
        # Platelets
        ['32623-1', 'Platelet mean volume in Blood (Entitic Volume): automated count'],
        ['28542-9', 'Platelets mean volume in Blood (Entitic Volume)'], 
        ['777-3', 'Platelets in Blood: automated count'],
        ['26515-7', 'Platelets in Blood'],
        ['13056-7', 'Platelets in Plasma: automated count'],
    
        # Potassium
        ['6298-4', 'Potassium in Blood (Moles/Volume)'],
        ['2823-3', 'Potassium in Serum/Plasma (Moles/Volume)'],
        ['39789-3', 'Potassium in Venous Blood (Moles/Volume)'],
    
        ['1920-8', 'Asparate'],
        ['6301-6', 'Coagulation tissue factor induced'],
        ['2157-6', 'Creatinine kinase'],
        ['14196-0', 'Reticulocytes'],
        ['2951-2', 'Sodium'],
    
        ['14685-2', 'Cobalamins'], # Does not exists in either data
        ['1988-5', 'C reactive protein'], # Does not exists in either data
        ['2276-4', 'Ferritin'], # Does not exists in either data
        ['2601-3', 'Magnesium'], # Does not exists in either data
        ['14879-1', 'Phosphate'], # Does not exists in either data
        ['5804-0', 'Protein'], # Does not exists in either data
        ['14927-8', 'Triglyceride'], # Does not exists in either data
        ['3016-3', 'Thyrotropin'], # Does not exists in either data
        ['14920-3', 'Thyroxine (T4) free (Moles/Volume)'], # Does not exists in either data
        ['14933-6', 'Urate'], # Does not exists in either data
       ]
pd.set_option('display.max_colwidth', None)
df = pd.DataFrame(data, columns=['ObservationCode', 'ObservationCodeName'])


# In[ ]:


df['In olis.csv'] = df['ObservationCode'].isin(observation_olis.index)
df['In olis_blood_count.csv'] = df['ObservationCode'].isin(observation_olis_blood_count.index)
df['NumObservations'] = df['ObservationCode'].map(observation_olis)
df['NumObservations'] = df['NumObservations'].fillna(df['ObservationCode'].map(observation_olis_blood_count))


# In[133]:


# observations only in both datasets
df[df['In olis.csv'] & df['In olis_blood_count.csv']]


# In[134]:


# observations only in olis dataset
df[df['In olis.csv'] & ~df['In olis_blood_count.csv']]


# In[135]:


# observations only in olis_blood_count dataset
df[~df['In olis.csv'] & df['In olis_blood_count.csv']]


# In[136]:


# missing observations from both datasets
df[~df['In olis.csv'] & ~df['In olis_blood_count.csv']]


# # Analyaze groupings of lab tests/blood work 
# (e.g. differences in count values are too large for the various bilirubins, can't group them together)

# In[13]:


from scripts.config import all_observations


# In[5]:


olis = pd.read_csv(f'{root_path}/ED-H-D/data/olis_complete.csv', dtype=str) 
olis['value'] = olis['value'].astype(float)


# In[219]:


obs_a = '2823-3'
obs_b = '39789-3'
tmp = olis[olis['ObservationCode'].isin([obs_a, obs_b])]
tmp = tmp[~tmp.duplicated(subset=['ikn', 'ObservationCode', 'ObservationDateTime'])]
tmp = tmp[tmp.duplicated(subset=['ikn', 'ObservationDateTime'], keep=False)]
tmp = tmp.sort_values(by=['ikn', 'ObservationDateTime'])


# In[220]:


df_a = tmp[tmp['ObservationCode'] == obs_a]
df_b = tmp[tmp['ObservationCode'] == obs_b]
df_a = df_a.reset_index()
df_b = df_b.reset_index()
assert(df_a[['ikn', 'ObservationDateTime']].equals(df_b[['ikn', 'ObservationDateTime']]))


# In[221]:


diff = abs(df_a['value'] - df_b['value'])
df = pd.concat([diff.describe(), df_a['value'].describe(), df_b['value'].describe()], axis=1)
df.columns = ['diff', obs_a, obs_b]
df = df.T
df


# # Analyze Correlations

# In[54]:


output_path = f'{root_path}/ED-H-D'
target_keyword = '_within_30days'


# In[55]:


prep = PrepDataEDHD()
model_data = prep.get_data(output_path, target_keyword)
model_data, clip_thresholds = prep.clip_outliers(model_data, lower_percentile=0.05, upper_percentile=0.95)

dtypes = model_data.dtypes
cols = dtypes[~(dtypes == object)].index
cols = cols.drop('ikn')
target_cols = cols[cols.str.contains(target_keyword)]
feature_cols = cols[~cols.str.contains(target_keyword)]


# In[56]:


pearson_matrix = pd.DataFrame(columns=feature_cols, index=target_cols)
for target in target_cols:
    for feature in tqdm.tqdm(feature_cols):
        data = model_data[~model_data[feature].isnull()]
        corr, prob = pearsonr(data[target], data[feature])
        pearson_matrix.loc[target, feature] = np.round(corr, 3)
pearson_matrix = pearson_matrix.T
pearson_matrix.index = get_clean_variable_names(pearson_matrix.index)
pearson_matrix.columns = pearson_matrix.columns.str.replace(target_keyword, '')
pearson_matrix.to_csv(f'{output_path}/data/pearson_matrix.csv', index_label='index')


# In[92]:


pearson_matrix = pd.read_csv(f'{output_path}/data/pearson_matrix.csv')
pearson_matrix = pearson_matrix.set_index('index')
pearson_matrix.style.background_gradient(cmap='Greens')


# In[95]:


fig, ax = plt.subplots(figsize=(25,9))
indices = pearson_matrix['ACU'].sort_values().index
ax.plot(pearson_matrix.loc[indices], marker='o')
ax.set_ylabel('Pearson Correlation Coefficient', fontsize=16)
ax.set_xlabel('Feature Columns', fontsize=16)
plt.legend(pearson_matrix.columns, fontsize=14)
plt.xticks(rotation='90')
plt.savefig(f'{output_path}/data/pearson_coefficient.jpg', bbox_inches='tight', dpi=300)


# In[96]:


print("Variables with high pearson correlation to any of the targets")
pearson_matrix.max(axis=1).sort_values(ascending=False).head(n=20)


# In[ ]:




