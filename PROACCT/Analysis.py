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

# In[2]:


get_ipython().run_line_magic('cd', '../')
# reloads all modules everytime before cell is executed (no need to restart kernel)
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.utility import get_pearson_matrix
from src.visualize import time_to_event_plot, pearson_plot
from src.config import (
    root_path, regiments_folder, acu_folder, 
    max_chemo_date, 
    diag_cols, 
    diag_code_mapping, 
    event_map
)
from src.prep_data import PrepDataEDHD


# In[20]:


def show_year_dist(df, col='date', title=''):
    df[col] = pd.to_datetime(df[col])
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.countplot(x=df[col].dt.year, ax=ax)
    ax.set(xlabel='Year', ylabel='Count')
    if title: ax.set_title(title)
    
    print(f"Earliest date: {df[col].min()}")
    print(f"Latest date: {df[col].max()}")
    
    plt.show()


# In[5]:


chemo_df = pd.read_parquet(f'{root_path}/{acu_folder}/data/final_data.parquet.gzip')


# # Age Distribution

# In[6]:


fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))
plt.subplots_adjust(hspace=0.3)

sns.histplot(chemo_df, x='age', hue='sex', bins=chemo_df['age'].max() - chemo_df['age'].min() + 1, ax=axes[0])
axes[0].set_title('By Treatment')
axes[0].grid()

df = chemo_df.drop_duplicates(subset=['ikn'])
sns.histplot(df, x='age', hue='sex', bins=df['age'].max() - df['age'].min() + 1, ax=axes[1])
axes[1].set_title('By Patient (Last Treatment)')
axes[1].grid()

df = chemo_df.drop_duplicates(subset=['ikn']).copy()
df['age'] = df['death_date'].fillna(max_chemo_date).dt.year - df['birth_date'].dt.year
sns.histplot(df, x='age', hue='sex', bins=df['age'].max() - df['age'].min() + 1, ax=axes[2])
axes[2].set_title('By Patient (Death / Max Chemo Date)')
axes[2].grid()


# # Label Distribution
# Target within 14, 30, 90, 180, 365 days

# In[41]:


def get_distribution():
    results = []
    for days in tqdm([14, 30, 90, 180, 365]):
        prep = PrepDataEDHD(adverse_event='acu', target_keyword=f'within_{days}_days')
        df = prep.load_data()
        prep.event_dates['visit_date'] = df['visit_date']
        df = prep.get_event_data(df, event='H', create_targets=True)
        df = prep.get_event_data(df, event='ED', create_targets=True)
        df[prep.add_tk('ACU')] = df[prep.add_tk('ED')] | df[prep.add_tk('H')] 
        df[prep.add_tk('TR_ACU')] = df[prep.add_tk('TR_ED')] | df[prep.add_tk('TR_H')] 
        target_cols = df.columns[df.columns.str.contains(prep.target_keyword)]
        result = df[target_cols].apply(pd.value_counts).T
        result.columns = pd.MultiIndex.from_arrays([(f'{days} days', f'{days} days'), ('False', 'True')])
        result.index = result.index.str.replace(prep.target_keyword, '')
        results.append(result)
    return pd.concat(results, axis=1)


# In[42]:


dist = get_distribution()
dist


# # ED/H Occurence to Treatment Session Distribution

# In[43]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
for i, event in enumerate(['H', 'ED']):
    event_df = pd.read_parquet(f'{root_path}/{acu_folder}/data/{event}.parquet.gzip')
    event_df = event_df.query('feat_or_targ == "feature"')
    event_df = event_df.set_index('chemo_idx')
    diff = chemo_df.loc[event_df.index, 'visit_date'] - event_df['date']
    time_to_event_plot(
        diff.dt.days, axes[i], plot_type='cdf', 
        xlabel='Years', ylabel="Cumulative Proportion of Events", 
        xticks=range(0, 365*5+1, 365), xticklabels=range(6), 
        title=f"Time from {event_map[event]['event_name']} occurence\n to treatment session"
    )


# # ED/H Date Distribution

# In[8]:


H_dates = pd.read_parquet(f'{root_path}/{acu_folder}/data/H.parquet.gzip')
show_year_dist(H_dates)


# In[9]:


# ED Visit - Filtered (only within past 5 years of treatment session)
ED_dates = pd.read_parquet(f'{root_path}/{acu_folder}/data/ED.parquet.gzip')
show_year_dist(ED_dates)


# # Number of ED/H Occurence Past 5 Years Distribution

# In[16]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
for i, event in enumerate(['H', 'ED']):
    col = f'num_prior_{event}s_within_5_years'
    event_df = pd.read_parquet(f'{root_path}/{acu_folder}/data/{event}.parquet.gzip')
    event_df = event_df.query('feat_or_targ == "feature"')
    event_df = event_df.set_index('chemo_idx')
    chemo_df[col] = 0
    chemo_df.loc[event_df.index, col] = event_df[col].astype(int)
    time_to_event_plot(
        chemo_df[f'num_prior_{event}s_within_5_years'], axes[i], plot_type='cdf',
        xlim=(-1,21), xticks=range(21), ylabel="Cumulative proportion",
        xlabel=f"Number of {event_map[event]['event_name']} occurences\nwithin past 5 years of treatment sessions"
    )


# # Survey Date Distribution

# In[21]:


# Questionnaires filtered (only within past 30 days of treatment session)
cols = chemo_df.columns
for col in cols[cols.str.contains('survey_date')]:
    show_year_dist(chemo_df, col=col, title=col.split('_survey_date')[0].title())


# # Survey Date to Treatment Session Distribution

# In[22]:


cols = chemo_df.columns
cols = cols[cols.str.contains('survey_date')]

fig, axes = plt.subplots(nrows=len(cols), ncols=1, figsize=(6, 6*len(cols)))
plt.subplots_adjust(hspace=0.3)
for i, col in enumerate(cols):
    name = col.split('_')[0]
    mask = chemo_df[col].notnull()
    visit_date = chemo_df.loc[mask, 'visit_date']
    survey_date = pd.to_datetime(chemo_df.loc[mask, col])
    
    time_from_survey_to_treatment = (visit_date - survey_date).dt.days
    time_to_event_plot(
        time_from_survey_to_treatment[2:], axes[i], plot_type='cdf', 
        xlabel='Days', ylabel=f"Cumulative Proportion of {name} Surveys",
        title=f"Time from closest {name} survey date prior to treatment\n to treatment session"
    )


# # All Lab Tests

# In[278]:


data = {
    # Alanine Aminotransferase
    '1742-6': 'Alanine Aminotransferase (Enzymatic Activity/Volume)',
    '1744-2': 'Alanine Aminotransferase without Vitamin B6',
    '1743-4': 'Alanine Aminotransferase with Vitamin B6',
    
    # Albumin
    '1751-7': 'Albumin in Serum/Plasma', # Does not exists in either data
    '1754-1': 'Albumin in Urine',
    '32294-1': 'Albumin/Creatinine in Urine (Ratio)',

    # Basophils
    '704-7': 'Basophils in Blood (#/Volume): automated count',
    '705-4': 'Basophils in Blood (#/Volume): manual count',
    '26444-0': 'Basophils in Blood (#/Volume)',

    # Bilirubin
    '14629-0': 'Bilirubin - Direct', # Direct bilirubin = glucoronidated bilirubin + albumin bound bilirubin
    '29760-6': 'Bilirubin - Glucoronidated',
    '14630-8': 'Bilirubin - Indirect',
    '14631-6': 'Bilirubin - Total', # Total bilirubin = direct bilirubin + indirect bilirubin

    # Calcium
    '2000-8': 'Calcium',
    '29265-6': 'Calcium corrected for albumin',
    '18281-6': 'Calcium corrected for total protein',

    # Cholestrol - None of these exists in either data
    '14647-2': 'Cholestrol',
    '14646-4': 'Cholestrol in HDL (high density lipoprotein) (Moles/Volume)',
    '39469-2': 'Cholestrol in LDL (low density lipoprotein) (Moles/Volume): calculated',
    '32309-7': 'Cholestrol in HDL (Molar Ratio)',
    '70204-3': 'Cholestrol not in HDL (Moles/Volume)',

    # Creatinine - None of these exists in either data
    '14682-9': 'Creatinine in Serum/Plasma',
    '14683-7': 'Creatinine in Urine',

    # Eosinophils
    '711-2': 'Eosinophils in Blood (#/Volume): automated count',
    '712-0': 'Eosinophils in Blood (#/Volume): manual count',
    '26449-9': 'Eosinophils in Blood (#/Volume)',

    # Erythrocyte
    '788-0': 'Erythrocyte distribution width (Ratio): automated count',
    '30385-9': 'Erythrocyte distribution width (Ratio)',
    '21000-5': 'Erythrocyte distribution width (Entitic Volume): automated count',
    '30384-2': 'Erythrocyte distribution width (Entitic Volume)',
    '789-8': 'Erythrocytes in Blood (#/Volume): automated count',
    '790-6': 'Erythrocytes in Blood (#/Volume): manual count',
    '26453-1': 'Erythrocytes in Blood (#/Volume)',
    '787-2': 'Erythrocyte MCV (mean corpuscular volume) (Entitic Volume): automated count',
    '30428-7': 'Erythrocyte MCV (mean corpuscular volume) (Entitic Volume)',
    '786-4': 'Erythrocyte MCHC (mean corpuscular hemoglobin concentration) (Mass/Volume): automated count',
    '28540-3': 'Erythrocyte MCHC (mean corpuscular hemoglobin concentration) (Mass/Volume)',
    '785-6': 'Erythrocyte MCH (mean corpuscular hemoglobin) (Entitic Mass): automated count',
    '28539-5': 'Erythrocyte MCH (mean corpuscular hemoglobin) (Entitic Mass)',

    # Prostate specific antigen
    '12841-3': 'Free PSA (Prostate Specific Antigen)/ Total PSA (Mass Fraction)',
    '10886-0': 'Free PSA (Prostate Specific Antigen) (Mass/Volume)',
    '2857-1': 'Total Prostate specific antigen (Mass/Volume)', # Total = PSA&alpha1 + free PSA (+ PSA&alpha2?)
    '35741-8': 'Total Prostate specific antigen (Mass/Volume): detection limit <= 0.01 ng/mL',
    '19197-3': 'Total Prostate specific antigen (Moles/Volume)',

    # Glucose - None of these exists in either data
    '14749-6': 'Glucose',
    '14771-0': 'Glucose^Post CFst (Fasting Glucose)',

    # Hematocrit
    '4544-3': 'Hematocrit of Blood (Volume Fraction): automated count',
    '71833-8': 'Hematocrit of Blood (Pure Volume Fraction): automated count', # same as 4544-3 but no units
    '20570-8': 'Hematocrit of Blood (Volume Fraction)',

    # Hemoglobin
    '718-7': 'Hemoglobin in Blood (Mass/Volume)',
    '20509-6': 'Hemoglobin in Blood (Mass/Volume): calculation',
    '4548-4': 'Hemoglobin A1c / Total Hemoglobin in Blood',
    '71875-9': 'Hemoglobin A1c / Total Hemoglobin in Blood (Pure Mass Fraction)', # same as 4548-4 but no units?
    '17855-8': 'Hemoglobin A1c / Total Hemoglobin in Blood: calculation', 
    '17856-6': 'Hemoglobin A1c / Total Hemoglobin in Blood: HPLC',
    '59261-8': 'Hemoglobin A1c / Total Hemoglobin in Blood: IFCC',

    # Ionized Calcium
    '1995-0': 'Ionized Calcium in Serum/Plasma (Moles/Volume)',
    '19072-8': 'Ionized Calcium in Serum/Plasma adjusted to pH 7.4 (Moles/Volume)',
    '12180-6': 'Ionized Calcium in Serum/Plasma (Moles/Volume): ISE',
    '1994-3': 'Ionized Calcium in Blood (Moles/Volume)',
    '47598-8': 'Ionized Calcium in Blood adjusted to pH 7.4 (Moles/Volume)',
    '34581-9': 'Ionized Calcium in Arterial Blood (Moles/Volume)',
    '41645-3': 'Ionized Calcium in Venous Blood (Moles/Volume)',
    '59473-9': 'Ionized Calcium in Venous Blood adjusted to pH 7.4 (Mass/Volume): ISE',

    # Leukocytes
    '6690-2': 'Leukocytes in Blood (#/Volume): automated count',
    '804-5': 'Leukocytes in Blood (#/Volume): manual count',
    '26464-8': 'Leukocytes in Blood (#/Volume)',

    # Lymphocytes
    '731-0': 'Lymphocytes in Blood (#/Volume): automated count',
    '732-8': 'Lymphocytes in Blood (#/Volume): manual count',
    '26474-7': 'Lymphocytes in Blood (#/Volume)',

    # Microalbumin
    '14957-5': 'Microalbumin in Urine',
    '14959-1': 'Microalbumin/Creatinine in Urine (Mass Ratio)',
    '30000-4': 'Microalbumin/Creatinine in Urine (Ratio)',

    # Monocytes
    '742-7': 'Monocytes in Blood (#/Volume): automated count',
    '743-5': 'Monocytes in Blood (#/Volume): manual count',
    '26484-6': 'Monocytes in Blood (#/Volume)',

    # Neutrophil
    '751-8': 'Neutrophils in Blood (#/Volume): automated count',
    '753-4': 'Neutrophils in Blood (#/Volume): manual count',
    '26499-4': 'Neutrophil in Blood (#/Volume)',

    # Platelets
    '32623-1': 'Platelet mean volume in Blood (Entitic Volume): automated count',
    '28542-9': 'Platelets mean volume in Blood (Entitic Volume)',
    '777-3': 'Platelets in Blood: automated count',
    '26515-7': 'Platelets in Blood',
    '13056-7': 'Platelets in Plasma: automated count',

    # Potassium
    '6298-4':'Potassium in Blood (Moles/Volume)',
    '2823-3': 'Potassium in Serum/Plasma (Moles/Volume)',
    '39789-3': 'Potassium in Venous Blood (Moles/Volume)',

    '1920-8': 'Asparate',
    '6301-6': 'Coagulation tissue factor induced',
    '2157-6': 'Creatinine kinase',
    '14196-0': 'Reticulocytes',
    '2951-2': 'Sodium',

    '14685-2': 'Cobalamins', # Does not exists in either data
    '1988-5': 'C reactive protein', # Does not exists in either data
    '2276-4': 'Ferritin', # Does not exists in either data
    '2601-3': 'Magnesium', # Does not exists in either data
    '14879-1': 'Phosphate', # Does not exists in either data
    '5804-0': 'Protein', # Does not exists in either data
    '14927-8': 'Triglyceride', # Does not exists in either data
    '3016-3': 'Thyrotropin', # Does not exists in either data
    '14920-3': 'Thyroxine (T4) free (Moles/Volume)', # Does not exists in either data
    '14933-6': 'Urate', # Does not exists in either data
}


# # Diagnostic Codes

# In[23]:


df = pd.read_parquet(f'{root_path}/data/dad.parquet.gzip')
df[diag_cols].head(n=10)


# In[24]:


all_diag_codes = df[diag_cols].values.flatten()
all_diag_codes = set(all_diag_codes[pd.notnull(all_diag_codes)])
print(f"Total number of unique diag codes: {len(all_diag_codes)}")


# In[25]:


for cause, cause_specific_diag_codes in diag_code_mapping.items():
    codes = all_diag_codes.intersection(cause_specific_diag_codes)
    print(f"Total number of {cause} specific diagnostic codes\n" +          f"- in diag_code_mapping (that I curated and made): {len(cause_specific_diag_codes)}\n" +          f"- in dad.csv that also matches with the codes in diag_code_mapping: {len(codes)}")


# In[26]:


# Checking Lengths of Diagnostic Codes
ALL = pd.Series(list(all_diag_codes))
freq_df = pd.DataFrame(ALL.str.len().value_counts(), columns=['Count of Unique Diagnostic Codes']).sort_index()
freq_df.index.name = 'Length of Diagnostic Code'
freq_df


# In[27]:


# Checking Lengths of Diagnostic Codes with Length Greater than 5
example_diag_code_prefix= 'A05'
ALL[ALL.str.contains(example_diag_code_prefix)]


# # Line of Therapy

# In[21]:


systemic = pd.read_parquet(f'{root_path}/{acu_folder}/data/systemic.parquet.gzip')
systemic = systemic[['ikn', 'visit_date', 'regimen', 'line_of_therapy', 'intent_of_systemic_treatment']]
df = pd.DataFrame(systemic.groupby('intent_of_systemic_treatment')['line_of_therapy'].value_counts())
df.rename(columns={'line_of_therapy': 'Count'})


# # Analyze Correlations

# In[13]:


target_keyword = 'within_30_days'
output_path = f'{root_path}/{acu_folder}/models/{target_keyword}'


# In[16]:


prep = PrepDataEDHD(adverse_event='acu', target_keyword=target_keyword)
model_data = prep.get_data(target_keyword, missing_thresh=80)
model_data = prep.clip_outliers(model_data, lower_percentile=0.05, upper_percentile=0.95)
pearson_matrix = get_pearson_matrix(model_data, target_keyword, save_path=output_path)


# In[17]:


pearson_matrix = pd.read_csv(f'{output_path}/tables/pearson_matrix.csv')
pearson_matrix = pearson_matrix.set_index('index')
pearson_matrix.style.background_gradient(cmap='Greens')


# In[18]:


pearson_plot(pearson_matrix, save_path=output_path)


# In[19]:


print("Variables with high pearson correlation to any of the targets")
pearson_matrix.max(axis=1).sort_values(ascending=False).head(n=20)


# In[ ]:
