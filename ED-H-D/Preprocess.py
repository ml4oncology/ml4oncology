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


# In[4]:


import tqdm
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import multiprocessing as mp
from functools import partial

from scripts.config import (root_path, regiments_folder,
                            systemic_cols, y3_cols, diag_cols, event_main_cols,
                            diag_code_mapping, event_map)
from scripts.preprocess import (shared_dict,
                                clean_string, replace_rare_col_entries, split_and_parallelize,
                                clean_cancer_and_demographic_data,
                                prefilter_olis_data, postprocess_olis_data,
                                preprocess_esas, get_esas_responses, postprocess_esas_responses,
                                filter_ecog_data, ecog_worker,
                                filter_immigration_data,
                                get_y3, get_systemic, observation_worker)


# In[5]:


# config
output_path = f'{root_path}/ED-H-D'
processes = 32


# # Create my csvs

# ### Include features from systemic (chemo) dataset
# NOTE: ikn is the encoded ontario health insurance plan (OHIP) number of a patient. All ikns are valid (all patients have valid OHIP) in systemic.csv per the valikn column

# In[6]:


df = pd.read_csv(f'{root_path}/{regiments_folder}/regiments_EDHD.csv', dtype=str)
regimens_keep = df.loc[df['include'] == '1', 'regimens'].values
del df


# In[7]:


systemic = get_systemic(regimens_keep)
systemic.to_csv(f'{output_path}/data/systemic.csv', index=False)
print(f"Number of patients = {systemic['ikn'].nunique()}")
print(f"Number of unique regiments = {systemic['regimen'].nunique()}")


# In[9]:


systemic = pd.read_csv(f'{output_path}/data/systemic.csv', dtype={'ikn': str})
systemic['visit_date'] = pd.to_datetime(systemic['visit_date'])


# ### Include features from y3 (cancer and demographic) dataset  - which includes death dates

# In[10]:


chemo_df_cols = ['ikn', 'regimen', 'visit_date', 'd_date', 'age', 'sex', 'intent_of_systemic_treatment', 'line_of_therapy', 
                 'lhin_cd', 'curr_morph_cd', 'curr_topog_cd', 'body_surface_area', 'days_since_starting_chemo', 
                 'days_since_true_prev_chemo', 'days_since_prev_chemo']


# In[17]:


y3 = get_y3()
y3 = y3[y3_cols + ['dthdate']]
y3 = clean_string(y3, ['lhin_cd', 'curr_morph_cd', 'curr_topog_cd', 'sex'])
chemo_df = pd.merge(systemic, y3, on='ikn', how='inner')

print(f"Number of patients in y3 = {y3['ikn'].nunique()}")
print(f"Number of patients in y3 and systemic = {y3['ikn'].isin(systemic['ikn']).sum()}")

chemo_df['bdate'] = pd.to_datetime(chemo_df['bdate'])
chemo_df['d_date'] = pd.to_datetime(chemo_df['dthdate'])
chemo_df['age'] = chemo_df['visit_date'].dt.year - chemo_df['bdate'].dt.year
chemo_df = clean_cancer_and_demographic_data(chemo_df, chemo_df_cols, verbose=True)

print(f"Number of unique regiments = {chemo_df['regimen'].nunique()}")
print(f"Number of patients = {chemo_df['ikn'].nunique()}")
print(f"Number of female patients = {chemo_df.loc[chemo_df['sex'] == 'F', 'ikn'].nunique()}")
print(f"Number of male patients = {chemo_df.loc[chemo_df['sex'] == 'M', 'ikn'].nunique()}")


# ### Include features from immigration dataset

# In[33]:


immigration = pd.read_csv(f'{root_path}/data/immigration.csv')
immigration = filter_immigration_data(immigration)
indices = chemo_df.index # TMP CODE: REMOVE WHEN YOU WANT TO RERUN EVERYTHING
chemo_df = pd.merge(chemo_df, immigration, on='ikn', how='left')
chemo_df.index = indices # TMP CODE: REMOVE WHEN YOU WANT TO RERUN EVERYTHING
chemo_df['speaks_english'] = chemo_df['speaks_english'].fillna(True)
chemo_df['is_immigrant'] = chemo_df['is_immigrant'].fillna(False)
chemo_df.to_csv(f'{output_path}/data/chemo_processed.csv', index_label='index')


# ### Include features from olis complete (blood work and lab test observations) dataset

# In[10]:


# Preprocess the Complete Blood Work Data
chunks = pd.read_csv(f'{root_path}/data/olis_complete.csv', chunksize=10**7, dtype=str) 
for i, chunk in tqdm.tqdm(enumerate(chunks), total=44):
    chunk = prefilter_olis_data(chunk, chemo_df['ikn'])
    # write to csv
    header = True if i == 0 else False
    chunk.to_csv(f"{output_path}/data/olis_complete.csv", header=header, mode='a', index=False)


# In[11]:


# Extract the Extra Blood Count Features
olis = pd.read_csv(f"{output_path}/data/olis_complete.csv", dtype=str) 
olis['ObservationDateTime'] = pd.to_datetime(olis['ObservationDateTime'])
print('Completed Loading Olis CSV File')

# get results
worker = partial(observation_worker, main_dir=output_path)
result = split_and_parallelize(olis, worker, processes=processes, split_by_ikn=True)

# save results
result = pd.DataFrame(result, columns=['observation_code', 'chemo_idx', 'days_after_chemo', 'observation_count'])
result.to_csv(f'{output_path}/data/olis_complete2.csv', index=False)


# In[38]:


# Process the Extra Blood Work Features
olis_df = pd.read_csv(f'{output_path}/data/olis_complete2.csv')
mapping, missing_df = postprocess_olis_data(chemo_df, olis_df, days_range=range(-5,0))
missing_df


# ### Include features from esas (questionnaire) dataset
# Interesting Observation: Chunking is MUCH faster than loading and operating on the whole ESAS dataset

# In[22]:


# Preprocess the Questionnaire Data
esas = preprocess_esas(chemo_df['ikn'])
esas.to_csv(f'{output_path}/data/esas.csv', index=False)


# In[23]:


# Extract the Questionnaire responses
esas_chunks = pd.read_csv(f'{output_path}/data/esas.csv', chunksize=10**6, dtype=str)
result = get_esas_responses(chemo_df[['ikn', 'visit_date']], esas_chunks, len_chunks=17)

# save results
result = pd.DataFrame(result, columns=['index', 'symptom', 'severity'])
result.to_csv(f'{output_path}/data/esas2.csv', index=False)


# In[39]:


# Process the Questionnaire responses
esas_df = pd.read_csv(f'{output_path}/data/esas2.csv')
esas_df = postprocess_esas_responses(esas_df)

# put esas responses in chemo_df
chemo_df = chemo_df.join(esas_df, how='left') # ALT WAY: pd.merge(chemo_df, esas, left_index=True, right_index=True, how='left')


# ### Include features from ecog (body functionality grade) dataset

# In[25]:


# Extract the Ecog Grades
ecog = pd.read_csv(f'{root_path}/data/ecog.csv')
ecog = filter_ecog_data(ecog, chemo_df['ikn'])

# filter out patients not in ecog
filtered_chemo_df = chemo_df[chemo_df['ikn'].isin(ecog['ikn'])]

shared_dict['ecog'] = ecog
result = split_and_parallelize(filtered_chemo_df, ecog_worker, split_by_ikn=True)

# save results
result = pd.DataFrame(result, columns=['index', 'ecog_grade'])
result.to_csv(f'{output_path}/data/ecog.csv', index=False)


# In[40]:


# Process the Ecog Grades
ecog = pd.read_csv(f'{output_path}/data/ecog.csv')
ecog = ecog.set_index('index')

# put ecog grade in chemo_df
chemo_df = chemo_df.join(ecog, how='left') # ALT WAY: pd.merge(chemo_df, ecog, left_index=True, right_index=True, how='left')


# In[41]:


chemo_df.to_csv(f'{output_path}/data/chemo_processed.csv', index=False)


# ## Get ED/H/D Events

# In[42]:


chemo_df = pd.read_csv(f'{output_path}/data/chemo_processed.csv', dtype=str)
chemo_df['visit_date'] = pd.to_datetime(chemo_df['visit_date'])


# In[43]:


def get_event_reason(df, event):
    raw_diag_codes = pd.Series(df[diag_cols].values.flatten())
    raw_diag_codes = raw_diag_codes[~raw_diag_codes.isnull()]
    all_diag_codes = pd.Series(raw_diag_codes.unique())
    """
    diag_code_mapping does not contain the complete codes (e.g. R10 can refer to codes R1012, R104, etc)
    extract the complete codes
    """
    complete_diag_code_mapping = {}
    for cause, diag_codes in diag_code_mapping.items():
        complete_diag_codes = []
        for code in diag_codes:
            complete_diag_codes += all_diag_codes[all_diag_codes.str.contains(code)].values.tolist()
        complete_diag_code_mapping[cause] = complete_diag_codes
    
    for cause, diag_codes in complete_diag_code_mapping.items():
        mask = False
        for diag_col in diag_cols:
            mask |= df[diag_col].isin(diag_codes)
        df[f'{cause}_{event}'] = mask
    
    return df

def preprocess_event_database(chemo_ikns, event='H'):
    database_name = event_map[event]['database_name']
    arr_date_col, dep_date_col = event_map[event]['date_col_name']
    event_cause_cols = event_map[event]['event_cause_cols']

    df = pd.read_csv(f'{root_path}/data/{database_name}.csv', dtype=str)
    print(f'Completed Loading {database_name} Dataset')

    df = clean_string(df, ['ikn'] + diag_cols)
    df['arrival_date'] = pd.to_datetime(df[arr_date_col])
    df['depart_date'] = pd.to_datetime(df[dep_date_col])
    
    # remove rows with null date values
    df = df[~df['arrival_date'].isnull()]
    df = df[~df['depart_date'].isnull()]
    
    # remove ED visits that resulted in hospitalizations
    if event == 'ED':
        df = clean_string(df, ['to_type'])
        df = df[~df['to_type'].isin(['I', 'P'])]

    # filter away patients not in chemo dataframe
    df = df[df['ikn'].isin(chemo_ikns)]

    # get reason for event visit (Treatment Related, Fever and Infection, GI Toxicity)
    # NOTE: Treatment Related means all treatments which INCLUDES Fever and Infection, GI Toxicity
    df = get_event_reason(df, event)

    # keep only selected columns
    df = df[event_main_cols + event_cause_cols]

    # REMOVE DUPLICATES
    df = df.drop_duplicates()
    # sort by departure date
    df = df.sort_values(by=['depart_date'])
    # for duplicates with different departure dates, keep the row with the later departure date
    df = df[~df.duplicated(subset=['ikn', 'arrival_date'], keep='last')]
    
    # sort by arrival date
    df = df.sort_values(by=['arrival_date'])
    
    return df

def event_worker(partition, event='H'):          
    event = shared_dict['event']
    cols = event.columns.drop(['ikn']).tolist()
    placeholder = ''
    result = []
    for ikn, chemo_group in tqdm.tqdm(partition.groupby('ikn')):
        event_group = event[event['ikn'] == ikn]
        event_arrival_dates = event_group['arrival_date'] 
        event_depart_dates = event_group['depart_date'] # keep in mind, for ED, depart date and arrival date are the same
        for chemo_idx, visit_date in chemo_group['visit_date'].iteritems():
            # get feature - closest event before and on chemo visit, and number of prev events prior to visit
            mask = event_depart_dates <= visit_date
            num_prior_events = mask.sum()
            closest_event_date = event_depart_dates[mask].max()
            if not pd.isnull(closest_event_date):
                closest_event = event_group.loc[event_group['depart_date'] == closest_event_date, cols]
                # if there are more than one row with same depart dates, get the first row (the earlier arrival date)
                result.append(['feature', chemo_idx, num_prior_events] + closest_event.values.tolist()[0]) 

            # get potential target - closest event after chemo visit
            closest_event_date = event_arrival_dates[event_arrival_dates > visit_date].min()
            if not pd.isnull(closest_event_date):
                closest_event = event_group.loc[event_group['arrival_date'] == closest_event_date, cols]
                result.append(['target', chemo_idx, placeholder] + closest_event.values.tolist()[0])
                
    return result

def get_event_dates(chemo_df, event='H'):
    event_name = event_map[event]['event_name']
    print(f"Beginning mutiprocess to extract {event_name} dates using {processes} processes")
    worker = partial(event_worker, event=event)
    event_dates = split_and_parallelize(chemo_df, worker, split_by_ikn=True, processes=processes)
    event_dates = np.array(event_dates).astype(str)
    
    # save results
    event_cause_cols = event_map[event]['event_cause_cols']
    event_dates = pd.DataFrame(event_dates, columns=['feature_or_target', 'chemo_idx', f'num_prior_{event}s', 
                                                     'arrival_date', 'depart_date'] + event_cause_cols)
    event_dates.to_csv(f'{output_path}/data/{event}_dates.csv', index=False)
    
    return event_dates

def load_event_dates(event='H'):
    df = pd.read_csv(f'{output_path}/data/{event}_dates.csv')
    df['arrival_date'] = pd.to_datetime(df['arrival_date'])
    df['depart_date'] = pd.to_datetime(df['depart_date'])
    df = df.set_index('chemo_idx')
    return df

def postprocess_event_dates(df, event='H', days_within=[14, 30, 180, 365]):
    event_dates = load_event_dates(event=event)
    event_cause_cols = event_map[event]['event_cause_cols']

    # create the features - number of days since previous event occured, their causes, 
    #                       length of prev hospitalization event, and number of events prior to visit
    features = event_dates[event_dates['feature_or_target'] == 'feature']
    col = f'days_since_prev_{event}'
    df.loc[features.index, col] = (df.loc[features.index, 'visit_date'] - features['arrival_date']).dt.days
    df[col] = df[col].fillna(df[col].max()) # fill rows where patients had no prev event with the max value
    col = f'num_prior_{event}s'
    df.loc[features.index, col] = features[col]
    for cause in event_cause_cols:
        df['prev_'+cause] = False # initialize
        df.loc[features.index, 'prev_'+cause] = features[cause]
            
    for days in days_within:
        # create the targets - event within x days after visit date
        targets = event_dates[event_dates['feature_or_target'] == 'target']
        targets['days_since_chemo_visit'] = targets['arrival_date'] - df.loc[targets.index, 'visit_date']
        # e.g. (within 14 days) if chemo visit is on Nov 1, a positive example is when event occurs between 
        #                       Nov 3rd to Nov 14th. We do not include the day of chemo visit and the day after
        targets[event] = (targets['days_since_chemo_visit'] < pd.Timedelta(f'{days} days')) &                          (targets['days_since_chemo_visit'] > pd.Timedelta('1 days'))
        targets = targets[targets[event]]
        for col in event_cause_cols+[event]:
            df[col+f'_within_{days}days'] = False # initialize
            df.loc[targets.index, col+f'_within_{days}days'] = targets[col]


# ### Get H visits (dad dataset)

# In[30]:


# Preprocess the H data
dad = preprocess_event_database(chemo_df['ikn'], event='H')
dad.to_csv(f'{output_path}/data/dad.csv', index=False)


# In[31]:


# Extract the H dates
dad = pd.read_csv(f'{output_path}/data/dad.csv', dtype=str)
dad['arrival_date'] = pd.to_datetime(dad['arrival_date'])
dad['depart_date'] = pd.to_datetime(dad['depart_date'])

shared_dict['event'] = dad
event_dates = get_event_dates(chemo_df, event='H')


# In[44]:


# Process the H dates
postprocess_event_dates(chemo_df, event='H')


# ### Get ED visits (nacrs dataset)

# In[33]:


# Preprocess the ED data
nacrs = preprocess_event_database(chemo_df['ikn'], event='ED')
nacrs.to_csv(f'{output_path}/data/nacrs.csv', index=False)


# In[34]:


# Extract the ED dates
nacrs = pd.read_csv(f'{output_path}/data/nacrs.csv', dtype=str)
nacrs['arrival_date'] = pd.to_datetime(nacrs['arrival_date'])
nacrs['depart_date'] = pd.to_datetime(nacrs['depart_date'])

shared_dict['event'] = nacrs
event_dates = get_event_dates(chemo_df, event='ED')


# In[45]:


# Process the ED dates
postprocess_event_dates(chemo_df, event='ED')


# ### Get D Events

# In[14]:


def get_death_dates(chemo_df, days_within=[14, 30, 180, 365]):
    chemo_df['d_date'] = pd.to_datetime(chemo_df['d_date'])
    days_until_d = chemo_df['d_date'] - chemo_df['visit_date']

    # remove rows with negative days until death
    chemo_df = chemo_df[~(days_until_d < pd.Timedelta('0 days'))]

    for days in days_within:
        chemo_df[f'D_within_{days}days'] = ((pd.Timedelta('1 days') < days_until_d) & 
                                                (days_until_d < pd.Timedelta(f'{days} days')))
    chemo_df = chemo_df.drop(columns=['d_date'])
    return chemo_df


# In[46]:


chemo_df = get_death_dates(chemo_df)


# ### Extra - remove errorneous inpatients from chemo_df

# In[16]:


def get_inpatient_indices(partition):
    H = shared_dict['H']
    result = set()
    for ikn, chemo_group in tqdm.tqdm(partition.groupby('ikn')):
        H_group = H[H['ikn'] == ikn]
        visit_date = chemo_group['visit_date']
        for H_idx, H_row in H_group.iterrows():
            arrival_date = H_row['arrival_date']
            depart_date = H_row['depart_date']
            mask = (arrival_date < visit_date) & (visit_date < depart_date)
            result.update(chemo_group[mask].index)
    return result


# In[39]:


dad = pd.read_csv(f'{output_path}/data/dad.csv', dtype=str)
dad['arrival_date'] = pd.to_datetime(dad['arrival_date'])
dad['depart_date'] = pd.to_datetime(dad['depart_date'])

shared_dict['H'] = dad
indices = split_and_parallelize(chemo_df, get_inpatient_indices, split_by_ikn=True, processes=processes)

# save results
np.save(f'{output_path}/data/inpatient_indices.npy', indices)


# In[47]:


inpatient_indices = np.load(f'{output_path}/data/inpatient_indices.npy')
chemo_df = chemo_df.drop(index=inpatient_indices)


# In[48]:


chemo_df.to_csv(f'{output_path}/data/model_data.csv', index_label='index')


# # Scratch Notes

# ### Diagnostic Codes

# In[6]:


df = pd.read_csv(f'{root_path}/data/dad.csv', dtype=str)
print(f'Completed Loading dad Dataset')


# In[42]:


raw = pd.Series(df[diag_cols].values.flatten())
raw = raw[~raw.isnull()]
all_diag_codes = pd.Series(raw.unique())
print(f"Total number of unique diag codes: {len(all_diag_codes)}")


# In[43]:


INFX = pd.Series(diag_code_mapping['INFX'])
print(f"Total number of INFX codes: {len(INFX)}")
codes = INFX[INFX.isin(all_diag_codes)].values.tolist()
print(f"Total number of INFX codes in dad.csv: {len(codes)}")
print(f'INFX codes in dad.csv: {codes}')


# In[44]:


GI = pd.Series(diag_code_mapping['GI'])
print(f"Total number of INFX codes: {len(GI)}")
codes = GI[GI.isin(all_diag_codes)].values.tolist()
print(f"Total number of INFX codes in dad.csv: {len(codes)}")
print(f'GI codes in dad.csv: {codes}')


# In[45]:


TR = pd.Series(diag_code_mapping['TR'])
print(f"Total number of INFX codes: {len(TR)}")
codes = TR[TR.isin(all_diag_codes)].values.tolist()
print(f"Total number of INFX codes in dad.csv: {len(codes)}")
print(f'GI codes in dad.csv: {codes}')


# In[46]:


ALL = all_diag_codes
ALL[ALL.str.contains('K52')]


# In[47]:


freq_df = pd.DataFrame(ALL.str.len().value_counts(), columns=['Frequency of code lengths']).sort_index()
freq_df


# In[48]:


for code in GI: 
    tmp = ALL[ALL.str.contains(code)]
    if any(tmp.str.len() > 4):
        break
tmp


# In[49]:


complete_diag_code_mapping = {}
for cause, diag_codes in diag_code_mapping.items():
    complete_diag_codes = []
    for code in diag_codes:
        complete_diag_codes += all_diag_codes[all_diag_codes.str.contains(code)].values.tolist()
    complete_diag_code_mapping[cause] = complete_diag_codes


# In[50]:


for cause, diag_codes in complete_diag_code_mapping.items():
    mask = False
    for diag_col in diag_cols:
        mask |= df[diag_col].isin(diag_codes)
    df[f'{cause}_H'] = mask


# In[51]:


pd.DataFrame([df['INFX_H'].value_counts(), 
              df['GI_H'].value_counts(), 
              df['TR_H'].value_counts()])


# In[19]:


df[diag_cols]


# ### write regimen csv file

# In[ ]:


from scripts.config import share_path


# In[7]:


systemic = get_systemic()


# In[22]:


y3 = get_y3()
df = pd.merge(systemic, y3, on='ikn', how='inner')
df2 = replace_rare_col_entries(chemo_df.copy(), ['regimen'])

regimens = df2['regimen'].unique()
rare_regimens = df.loc[~df['regimen'].isin(df2['regimen']), 'regimen'].unique()


# In[23]:


x = pd.DataFrame(regimens, columns=['regimens'])
x = x.sort_values(by='regimens')
x['include'] = 0
x.to_csv(f'{share_path}/regimens.csv', index=False)

x = pd.DataFrame(rare_regimens, columns=['regimens'])
x = x.sort_values(by='regimens')
x['include'] = 0
x.to_csv(f'{share_path}/rare_regimens_replaced_with_OTHER.csv', index=False)


# ### Experiment with Longitudinal Lab Data

# In[53]:


from scripts.config import all_observations
from scripts.preprocess import group_observations


# In[54]:


chemo_df = pd.read_csv(f'{output_path}/data/model_data.csv', dtype={'curr_morph_cd': str, 'lhin_cd': str})
chemo_df['visit_date'] = pd.to_datetime(chemo_df['visit_date'])
chemo_df = chemo_df.set_index('index')


# In[62]:


# Extract the Extra Blood Count Features
olis = pd.read_csv(f"{output_path}/data/olis_complete.csv", dtype=str) 
olis['ObservationDateTime'] = pd.to_datetime(olis['ObservationDateTime'])
print('Completed Loading Olis CSV File')

# get results
worker = partial(olis_worker, days_ago=28, filename='model_data')
result = split_and_parallelize(olis, worker, processes=processes, split_by_ikn=True)

# save results
result = pd.DataFrame(result, columns=['observation_code', 'chemo_idx', 'days_after_chemo', 'observation_count'])
result.to_csv(f'{output_path}/data/experiment/olis_complete3.csv', index=False)


# In[63]:


# Process the Extra Blood Work Features
olis_df = pd.read_csv(f'{output_path}/data/experiment/olis_complete3.csv')

# group together obs codes with same obs name
freq_map = olis_df['observation_code'].value_counts()
grouped_observations = group_observations(all_observations, freq_map)


# In[64]:


for obs_name, obs_codes in tqdm.tqdm(grouped_observations.items()):
    df = pd.DataFrame(index=chemo_df.index, columns=range(-28, 1))
    for i, obs_code in enumerate(obs_codes):
        obs_group = olis_df[olis_df['observation_code'] == obs_code]
        for day, day_group in obs_group.groupby('days_after_chemo'):
            day_group = day_group.set_index('chemo_idx')
            chemo_indices = day_group.index
            obs_count_values = day_group['observation_count']
            if i == 0:
                df.loc[chemo_indices, int(day)] = obs_count_values
            else:
                df.loc[chemo_indices, int(day)].fillna(obs_count_values)
    df.to_csv(f'{output_path}/data/experiment/{obs_name}.csv', index_label='index')


# In[ ]:




