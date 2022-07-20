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
pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib.pyplot as plt

from scripts.utility import (get_pearson_matrix)
from scripts.visualize import (pearson_plot)
from scripts.config import (root_path, regiments_folder, acu_folder, y3_cols)
from scripts.preprocess import (clean_string)
from scripts.prep_data import (PrepDataEDHD)


# # Age Demographic

# In[8]:


y3 = pd.read_csv(f'{root_path}/data/y3.csv')
y3 = y3[y3_cols+['dthdate']]
y3 = clean_string(y3, ['sex'])
y3 = y3[~(y3['sex'] == 'O')]
y3['bdate'] = pd.to_datetime(y3['bdate'])
y3['dthdate'] = pd.to_datetime(y3['dthdate'])


# In[9]:


data = y3.copy()
today = pd.to_datetime('today')
data['dthdate'] = data['dthdate'].fillna(today) # to get age, fill missing death date using today's date
born = data['bdate'].dt
died = data['dthdate'].dt
data['age'] = died.year - born.year - ((died.month) < (born.month))
data = data[~((data['age'] > 110) & (data['dthdate'] == today))] # do not include patients over 110 years old today


# In[10]:


fig = plt.figure(figsize=(15, 4))
for sex, group in data.groupby('sex'):
    plt.hist(group['age'], alpha=0.5, label=sex, bins=110)
plt.xticks(range(0,110,4))
plt.legend()
plt.show()


# # Target within 14, 30, 90, 180, 365 days - Label Distribution

# In[42]:


def get_distribution():
    prep = PrepDataEDHD(adverse_event='acu')
    df = prep.load_data()
    prep.event_dates['visit_date'] = df['visit_date']
    results = []
    for days in tqdm.tqdm([14, 30, 90, 180, 365]):
        target_keyword = f'_within_{days}days'
        df = prep.get_event_data(df, target_keyword, event='H', create_targets=True)
        df = prep.get_event_data(df, target_keyword, event='ED', create_targets=True)
        df['ACU'+target_keyword] = df['ED'+target_keyword] | df['H'+target_keyword] 
        df['TR_ACU'+target_keyword] = df['TR_ED'+target_keyword] | df['TR_H'+target_keyword] 
        target_cols = df.columns[df.columns.str.contains(target_keyword)]
        result = prep.get_label_distribution(df[target_cols])
        result.columns = pd.MultiIndex.from_arrays([(f'{days} days', f'{days} days'), ('False', 'True')])
        result.index = result.index.str.replace(target_keyword, '')
        results.append(result)
    return pd.concat(results, axis=1)


# In[43]:


get_distribution()


# # Month/Year Distribution

# In[58]:


systemic = pd.read_csv(f'{root_path}/data/systemic.csv')
systemic['visit_date'] = pd.to_datetime(systemic['visit_date'])


# In[59]:


fig = plt.figure(figsize=(15, 4))
labels, counts = np.unique(systemic['visit_date'].dt.month, return_counts=True)
assert all(labels == np.arange(1, 13))
labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.bar(labels, counts, align='center')
plt.show()


# In[60]:


mapping = {'Winter': [12,1,2], 'Spring': [3,4,5], 'Summer': [6,7,8], 'Fall': [9,10,11]}
mapping = {month: season for season, months in mapping.items() for month in months}
labels, counts = np.unique(systemic['visit_date'].dt.month.map(mapping), return_counts=True)
plt.bar(labels[[1, 2, 0, 3]], counts[[1, 2, 0, 3]], align='center')
plt.show()


# In[61]:


fig = plt.figure(figsize=(15, 4))
labels, counts = np.unique(systemic['visit_date'].dt.year, return_counts=True)
plt.bar(labels, counts, align='center')
plt.show()


# # All Lab Tests

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


# # Diagnostic Codes

# In[98]:


from scripts.preprocess import clean_string


# In[99]:


df = pd.read_csv(f'{root_path}/data/dad.csv', dtype=str)
print(f'Completed Loading dad Dataset')
df = clean_string(df, diag_cols)
raw = pd.Series(df[diag_cols].values.flatten())
raw = raw[~raw.isnull()]
all_diag_codes = pd.Series(raw.unique())
print(f"Total number of unique diag codes: {len(all_diag_codes)}")


# In[100]:


for cause, mapping in diag_code_mapping.items():
    X = pd.Series(mapping)
    codes = X[X.isin(all_diag_codes)].values.tolist()
    print(f"Total number of {cause} codes: {len(X)}")
    print(f"Total number of {cause} codes in dad.csv: {len(codes)}")
    print(f'{cause} codes in dad.csv: {codes}')


# In[101]:


ALL = all_diag_codes
ALL[ALL.str.contains('K52')]


# In[102]:


freq_df = pd.DataFrame(ALL.str.len().value_counts(), columns=['Frequency of code lengths']).sort_index()
freq_df


# In[103]:


for code in X: 
    tmp = ALL[ALL.str.contains(code)]
    if any(tmp.str.len() > 4):
        break
tmp


# In[104]:


complete_diag_code_mapping = {}
for cause, diag_codes in diag_code_mapping.items():
    complete_diag_codes = []
    for code in diag_codes:
        complete_diag_codes += all_diag_codes[all_diag_codes.str.contains(code)].values.tolist()
    complete_diag_code_mapping[cause] = complete_diag_codes


# In[105]:


for cause, diag_codes in complete_diag_code_mapping.items():
    mask = False
    for diag_col in diag_cols:
        mask |= df[diag_col].isin(diag_codes)
    df[f'{cause}_H'] = mask


# In[106]:


pd.DataFrame([df['INFX_H'].value_counts(), 
              df['GI_H'].value_counts(), 
              df['TR_H'].value_counts()])


# In[107]:


df[diag_cols]


# # Analyze Correlations

# In[76]:


target_keyword = '_within_30days'
output_path = f'{root_path}/{acu_folder}/models/within_30_days'


# In[73]:


prep = PrepDataEDHD(adverse_event='acu')
model_data = prep.get_data(target_keyword, missing_thresh=80)
model_data, clip_thresholds = prep.clip_outliers(model_data, lower_percentile=0.05, upper_percentile=0.95)
pearson_matrix = get_pearson_matrix(df, target_keyword, output_path)


# In[77]:


pearson_matrix = pd.read_csv(f'{output_path}/tables/pearson_matrix.csv')
pearson_matrix = pearson_matrix.set_index('index')
pearson_matrix.style.background_gradient(cmap='Greens')


# In[78]:


pearson_plot(pearson_matrix, output_path)


# In[79]:


print("Variables with high pearson correlation to any of the targets")
pearson_matrix.max(axis=1).sort_values(ascending=False).head(n=20)


# In[ ]:
