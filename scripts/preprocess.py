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
import tqdm
import itertools
import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial

from scripts.config import (root_path, sas_folder, regiments_folder, 
                            all_observations, english_lang_codes,
                            din_exclude, cisplatin_dins, cisplatin_cco_drug_code,
                            cancer_location_exclude,
                            systemic_cols, drug_cols, y3_cols, olis_cols, immigration_cols, diag_cols, event_main_cols,
                            diag_code_mapping, event_map)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s', datefmt='%I:%M:%S')
    
# Multiprocessing
def parallelize(generator, worker, processes=16):
    pool = mp.Pool(processes=processes)
    result = pool.map(worker, generator)
    pool.close()
    pool.join() # wait for all threads
    result = list(itertools.chain(*result))
    return result

def split_and_parallelize(data, worker, split_by_ikns=True, processes=16):
    """
    Split up the data and parallelize processing of data
    
    Args:
        data: array-like, DataFrame or tuple of DataFrames sharing the same patient ids
        split_by_ikns: split up the data by patient ids
    """
    generator = []
    if split_by_ikns:
        ikns = data[0]['ikn'] if isinstance(data, tuple) else data['ikn']
        ikn_groupings = np.array_split(ikns.unique(), processes)
        if isinstance(data, tuple):
            for ikn_grouping in ikn_groupings:
                items = tuple(df[df['ikn'].isin(ikn_grouping)] for df in data)
                generator.append(items)
        else:
            for ikn_grouping in ikn_groupings:
                item = data[ikns.isin(ikn_grouping)]
                generator.append(item)
    else:
        # splits df into x number of partitions, where x is number of processes
        generator = np.array_split(data, processes)
    return parallelize(generator, worker, processes=processes)

# sas7bdat files
def clean_sas(df, name):
    # remove empty rows
    df = df[~df.iloc[:,1:].isnull().all(axis=1)]
    if name == 'systemic':
        # create cleaned up regimen column
        df = create_clean_regimen(df.copy())
    return df

def create_clean_regimen(df):
    df['regimen'] = df['cco_regimen'].astype(str)
    df['regimen'] = df['regimen'].str[2:-1]
    df.loc[df['regimen'].str.contains('NCT'), 'regimen'] = 'TRIALS'
    df['regimen'] = df['regimen'].str.replace("*", "")
    df['regimen'] = df['regimen'].str.replace(" ", "")
    df['regimen'] = df['regimen'].str.lower()
    return df

def sas_to_csv(name, new_name='', folder=sas_folder, transfer_date=None, chunk_load=False, chunksize=10**6):
    filename = name if transfer_date is None else f'transfer{transfer_date}/{name}'
    name = new_name if new_name else name
    if not chunk_load:
        df = pd.read_sas(f'{root_path}/{folder}/{filename}.sas7bdat')
        df = clean_sas(df, name)
        df.to_csv(f"{root_path}/data/{name}.csv", index=False)
    else:
        chunks = pd.read_sas(f'{root_path}/{folder}/{filename}.sas7bdat', chunksize=chunksize)
        for i, chunk in tqdm.tqdm(enumerate(chunks)):
            chunk = clean_sas(chunk, name)
            header = True if i == 0 else False
            chunk.to_csv(f"{root_path}/data/{name}.csv", header=header, mode='a', index=False)

# Helper functions
def clean_string(df, cols):
    # remove first two characters "b'" and last character "'"
    for col in cols:
        mask = df[col].str.startswith("b'") & df[col].str.endswith("'")
        df.loc[mask, col] = df.loc[mask, col].str[2:-1].values
    return df

def replace_rare_col_entries(df, cols, with_respect_to='patients', n=6, verbose=False):
    for col in cols:
        if with_respect_to == 'patients':
            # replace unique values with less than n patients to 'Other'
            counts = df.groupby(col).apply(lambda group: len(set(group['ikn'])))
        elif with_respect_to == 'rows':
            # replace unique values that appears less than n number of rows in the dataset to 'Other'
            counts = df[col].value_counts()
        replace_values = counts.index[counts < n]
        if verbose: 
            logging.info(f'The following {col} entries have less than {n} {with_respect_to} ' + \
                         f'and will be replaced with "Other": {replace_values.tolist()}')
        df.loc[df[col].isin(replace_values), col] = 'Other'
    return df

def pandas_ffill(df):
    # uh...unfortunately pandas ffill is super slow when scaled up
    return df.ffill(axis=1)[0].values

def numpy_ffill(df):
    # courtesy of stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
    arr = df.values.astype(float)
    num_rows, num_cols = arr.shape
    mask = np.isnan(arr)
    indices = np.where(~mask, np.arange(num_cols), 0)
    np.maximum.accumulate(indices, axis=1, out=indices)
    arr[mask] = arr[np.nonzero(mask)[0], indices[mask]]
    return arr[:, -1]

def group_observations(observations, freq_map):
    grouped_observations = {}
    for obs_code, obs_name in observations.items():
        if obs_name in grouped_observations:
            grouped_observations[obs_name].append(obs_code)
        else:
            grouped_observations[obs_name] = [obs_code]

    for obs_name, obs_codes in grouped_observations.items():
        # sort observation codes based on their number of occurences / observation frequencies
        grouped_observations[obs_name] = freq_map[obs_codes].sort_values(ascending=False).index.tolist()
   
    return grouped_observations

def compute_chemo_cycle(mask):
    """
    Args:
        mask (pd.Series): indicates where regimens changed and where there were significant gaps between same regimens
    
    e.g. fec-d True
         fec-d False
         fec-d False
         tras  True
         tras  False
    """
    """
    Essentially assigns a number to each regimen timeframe
    
    e.g. fec-d 1
         fec-d 1
         fec-d 1
         tras  2
         tras  2
    """
    regimen_numbering = mask.cumsum()
    """
    Group the anti-mask together based on regimen timeframe
    
    e.g. Group1        Group2
            
         fec-d False   tras False
         fec-d True    tras True
         fec-d True
         
    """
    antimask_groupings = (~mask).groupby(regimen_numbering)
    """
    Use cumulative sum + 1 to compute chemo cycle
    
    e.g. fec-d 0+1 = 1
         fec-d 1+1 = 2
         fec-d 2+1 = 3
         tras  0+1 = 1
         tras  1+1 = 2
    """
    chemo_cycle = antimask_groupings.cumsum()+1
    return chemo_cycle

def merge_with_row(df, curr_idx, merge_cols, which='above'):
    if which not in {'above', 'below'}: 
        raise ValueError('which must be either above or below')
        
    if which =='above':
        if curr_idx == 0: # if its the very first entry of the whole dataframe, leave it as is
            return
        merge_idx = curr_idx-1
        df.loc[merge_idx, 'next_visit_date'] = df.loc[curr_idx, 'next_visit_date']
    elif which == 'below':
        merge_idx = curr_idx+1
        df.loc[merge_idx, 'visit_date'] = df.loc[curr_idx, 'visit_date']
            
    for col in merge_cols: 
        df.loc[merge_idx, col] += df.loc[curr_idx, col]

def merge_intervals(df, merge_days=4):
    """
    Merges small intervals into a x day cycle, or to the row below/above that has interval greater than x days
    If after merging and the final interval is still less than x days, an empty dataframe is essentially returned
    (this includes single row (patient only had two chemo sessions) that has interval less than x days)
    """
    df = df.reset_index(drop=True)
    duration, dosage = 'chemo_interval', 'dose_administered'
    merge_cols = [duration]
    merge_dosage = dosage in df.columns # WARNING: Assumes dataframe only contains one type of drug if drug info is included
    if merge_dosage:
        if df['drug'].nunique() > 1: raise NotImplementedError('Merging dosage for more than 1 drug is not supported yet')
        merge_cols.append(dosage)
    
    remove_indices = []
    for i in range(len(df)):
        if df.loc[i, duration] >= merge_days or pd.isnull(df.loc[i, duration]):
            continue
        if i == len(df)-1 or df.loc[i, 'regimen'] != df.loc[i+1, 'regimen']:
            # merge with the row above if last entry or last entry of an old regimen
            merge_with_row(df, i, merge_cols, which='above')
        else:
            # merge with the row below
            merge_with_row(df, i, merge_cols, which='below')
        remove_indices.append(i)
    df = df.drop(index=remove_indices)
    return df

# Loaders
def load_chemo_df(main_dir, includes_next_visit=True):
    dtype = {'ikn': str, 'lhin_cd': str, 'curr_morph_cd': str}
    chemo_df = pd.read_csv(f'{main_dir}/data/chemo_processed.csv', dtype=dtype)
    chemo_df['visit_date'] = pd.to_datetime(chemo_df['visit_date'])
    if includes_next_visit:
        chemo_df['next_visit_date'] = pd.to_datetime(chemo_df['next_visit_date'])
    return chemo_df

def load_reviewed_regimens():
    """
    Get the annotated regimens with their cycle lengths, name relabelings, splits, etc
    """
    df = pd.read_csv(f'{root_path}/{regiments_folder}/regimens.csv', dtype=str)
    df = clean_string(df, df.columns)

    # convert select columns to floats
    df['shortest_interval'] = df['shortest_interval'].replace('unclear', -1)
    float_cols = df.columns.drop(['regimen', 'relabel', 'reason', 'notes'])
    df[float_cols] = df[float_cols].astype(float)
    
    # ensure regimen names are all lowercase
    df['regimen'] = df['regimen'].str.lower()
    df['relabel'] = df['relabel'].str.lower()
    
    df = df.rename(columns={'split_number_first_dose_second_c': 'split_at_this_cycle',
                            'split_second_component_shortest_': 'cycle_length_after_split',
                            'shortest_interval': 'cycle_length'})
    df['shortest_cycle_length'] = df[['cycle_length', 'cycle_length_after_split']].min(axis=1)
    return df

def load_included_regimens(criteria=None):
    """
    Args:
        criteria (str or None): inclusion criteria for regimens based on different projects, 
                                either None (no critera), 'cytotoxic' for CYTOPENIA, or 'cisplatin_containing' for CAN
    """
    if criteria not in {None, 'cytotoxic', 'cisplatin_containing'}:
        raise ValueError('criteria must be either None, "cytotoxic", or "cisplatin_containing"')
    df = load_reviewed_regimens()

    # filter outpatient regimens, hematological regimens, non-IV administered regimens for all projects
    mask = df['iv_non_hematological_outpatient'] == 0
    df = df[~mask]
    
    if criteria is not None:
        # keep only selected reigmens for the project
        df = df[df[criteria] == 1]
        
        if criteria == 'cytotoxic':
            # remove regimens with unclear cycle lengths
            df = df[df['shortest_cycle_length'] != -1]

    return df
    
# Systemic (chemo) data
def clean_up_systemic(df):
    df = clean_string(df, ['ikn', 'din', 'cco_drug_code', 'intent_of_systemic_treatment', 'inpatient_flag', 'measurement_unit'])
    df['visit_date'] = pd.to_datetime(df['visit_date'])
    df = df.sort_values(by='visit_date') # order the data dataframe by date
    return df

def filter_regimens(df, regimens, verbose=False):
    col = 'regimen'
    df = df[~df[col].isnull()] # filter out rows with no regimen data 
    df = df[df[col].isin(regimens[col])] # keep only selected reigmens
    
    # change regimen name to the correct mapping
    mask = regimens['relabel'].notnull()
    old_name, new_name = regimens.loc[mask, [col, 'relabel']].T.values
    df[col] = df[col].replace(old_name, new_name)
    
    return df

def filter_date(df, min_date='2014-07-01', max_date='2020-06-30', verbose=False):
    if verbose: logging.info(f"Treatments occured between {df['visit_date'].min()} to {df['visit_date'].max()}")
    mask = df['visit_date'].between(min_date, max_date)
    if verbose: logging.info(f"Removing {sum(~mask)} chemo treatments that occured before {min_date} and after {max_date}.")
    df = df[mask] # remove chemo recieved before July 1st 2014 - before the ALR system was set up
    return df

def filter_inpatients(df, verbose=False):
    if verbose:
        logging.info(f"There are {sum(df['inpatient_flag'] == 'Y')} inpatient chemo treatment and " + \
                     f"{sum(df['inpatient_flag'].isnull())} unknown inpatient status out of {len(df)} total chemo treatments")
    df = df[df['inpatient_flag'] == 'N'] # remove chemo treatment recieved as an inpatient
    return df

def filter_systemic_data(df, regimens, remove_inpatients=True, exclude_dins=True, 
                          replace_rare_regimens=True, include_drug_info=False, verbose=False):
    """
    Args:
        regimens: pd.DataFrame of select annotated regimens, see load_reviewed_regimens/load_included_regimens
    """
    df = clean_up_systemic(df)
    df = filter_regimens(df, regimens, verbose=verbose)
    df = filter_date(df, verbose=verbose)
    
    if remove_inpatients: 
        df = filter_inpatients(df, verbose=verbose)
        
    if exclude_dins: # DIN: Drug Identification Number
        df = df[~df['din'].isin(din_exclude)] # remove dins in din_exclude
        
    if replace_rare_regimens:
        # replace regimens with less than 6 patients to 'Other'
        df = replace_rare_col_entries(df, ['regimen'], verbose=verbose)
    
    cols = systemic_cols.copy()
    if include_drug_info: cols += drug_cols
    df = df[cols]
    df = df.drop_duplicates()
    return df

def filter_by_drugs(df, drug='cisplatin', verbose=False):
    if drug == 'cisplatin':
        keep_dins = cisplatin_dins
        keep_cco_drug_code = cisplatin_cco_drug_code
    
    mask = df['din'].isin(keep_dins) | df['cco_drug_code'].isin(keep_cco_drug_code)
    if verbose: 
        logging.info(f"Removing {sum(~mask)} sessions with no {drug} administrated")
    df = df[mask]
    df['drug'] = drug
    df = df.drop(columns=['din', 'cco_drug_code'])
    df = df[df['measurement_unit'].isin(['mg', 'MG'])] # remove rows with measurement unit of g, unit, mL, nan

    # multiple dose administration on the same day and same drug - combine them together
    dosage = 'dose_administered'
    cols = df.columns.drop(dosage)
    mask = df.duplicated(subset=cols, keep=False)
    if verbose: 
        logging.info(f"Collapsing {sum(~df[mask].duplicated())} same day sessions with multiple doses administered")
    df.loc[mask, dosage] = df[mask].groupby(['ikn', 'visit_date'])[dosage].transform('sum')
    df = pd.concat([df[~mask], df[mask].drop_duplicates()])
    df = df.sort_values(by='visit_date') # order the dataframe by date
        
    return df

def systemic_worker(partition, method='merge', merge_days=4):
    """
    Creates engineered features such as days_since_starting_chemo, days_since_last_chemo, chemo_cycle, immediate_new_regimen
    Handles small interval chemo sessions 2 ways
    1. Merge them together
       e.g. May 1  10gm     ->    May 1  12gm
            May 2  1gm            May 16 10gm
            May 4  1gm
            May 16 10gm
    2. Include only the first chemo session in a given week
       e.g. May 1  10gm     ->    May 1  10gm
            May 2  1gm            May 16 5gm
            May 4  1gm
            May 16 5gm
    NOTE: use 2nd method when you don't care about the chemo dosages and just care about the rough dates
    
    Args:
        method (str): either 'merge' or 'one-per-week'
        merge_days (int): max number of intervals/days you can merge
    """
    if method not in {'merge', 'one-per-week'}:
        raise ValueError('method must be either merge or one-per-week')
        
    values = []
    for ikn, df in tqdm.tqdm(partition.groupby('ikn')):
        df = df.sort_values(by='visit_date') # might be redundant but just in case

        if method == 'one-per-week':
            # keep only the first chemo visit of a given week
            keep_indices = []
            tmp = df
            while not tmp.empty:
                first_chemo_idx = tmp.index[0]
                keep_indices.append(first_chemo_idx)
                first_visit_date = df['visit_date'].loc[first_chemo_idx]
                tmp = tmp[tmp['visit_date'] >= first_visit_date + pd.Timedelta('7 days')]
            df = df.loc[keep_indices]
        elif method == 'merge':
            # merges small intervals into a 4 day cycle, or to the row below/above that has interval greater than 4 days
            df['next_visit_date'] = df['visit_date'].shift(-1)
            df['chemo_interval'] = (df['next_visit_date'] - df['visit_date']).dt.days
            df = merge_intervals(df, merge_days=4)
            if df.empty:
                # most likely patient had consecutive 1 day interval chemo sessions that totaled less than 4 days
                # OR patient has only two chemo sessions (one row) in which interval was less than 4 days
                continue

        start_date = df['visit_date'].iloc[0]
        df['days_since_starting_chemo'] = (df['visit_date'] - start_date).dt.days
        df['days_since_last_chemo'] = (df['visit_date'] - df['visit_date'].shift()).dt.days

        # identify chemo cycle number (resets when patient undergoes new chemo regimen or there is a 60 day gap)
        new_regimen = (df['regimen'] != df['regimen'].shift())
        mask = new_regimen | (df['days_since_last_chemo'] >= 60)
        df['chemo_cycle'] = compute_chemo_cycle(mask)

        # identify if this is the first chemo cycle of a new regimen immediately after the old one
        mask = new_regimen & (df['days_since_last_chemo'] < 30)
        df['immediate_new_regimen'] = mask

        # convert to list and combine for faster performance
        values.extend(df.values.tolist())
        
    return values

def process_systemic_data(df, cycle_length_mapping=None, method='merge', merge_days=4):
    """
    Args:
        cycle_length_mapping (dict or None): mapping between regimens and their shortest cycle lengths
    """
    if cycle_length_mapping is not None:
        df['cycle_length'] = df['regimen'].map(cycle_length_mapping)
        
    worker = partial(systemic_worker, method=method, merge_days=merge_days)
    values = split_and_parallelize(df, worker)
    extra_cols = ['days_since_starting_chemo', 'days_since_last_chemo', 'chemo_cycle', 'immediate_new_regimen']
    if method == 'merge': extra_cols = ['next_visit_date', 'chemo_interval'] + extra_cols
    cols = df.columns.tolist() + extra_cols
    df = pd.DataFrame(values, columns=cols)
    return df

# y3 (cancer diagnosis and demographic) data
def filter_y3_data(y3, include_death=False):
    cols = y3_cols.copy()
    y3 = clean_string(y3, ['ikn', 'lhin_cd', 'curr_morph_cd', 'curr_topog_cd', 'sex'])
    y3['bdate'] = pd.to_datetime(y3['bdate'])
    
    if include_death:
        cols += ['vital_status_cd', 'D_date']
        
        # organize and format columns
        y3 = clean_string(y3, ['vital_status_cd'])
        y3['dthdate'] = pd.to_datetime(y3['dthdate'])
        y3 = y3.rename(columns={'dthdate': 'D_date'})

        # remove rows with conflicting information
        mask1 = (y3['vital_status_cd'] == 'A') & ~y3['D_date'].isnull() # vital status says Alive but deathdate exists
        mask2 = (y3['vital_status_cd'] == 'D') & y3['D_date'].isnull() # vital status says Dead but deathdate does not exist
        y3 = y3[~(mask1 | mask2)]
        
    y3 = y3[cols]
    return y3

def clean_cancer_and_demographic_data(chemo_df):
    chemo_df.loc[:, 'body_surface_area'] = chemo_df['body_surface_area'].replace(0, np.nan)
    chemo_df.loc[:, 'body_surface_area'] = chemo_df['body_surface_area'].replace(-99, np.nan)
    chemo_df.loc[:, 'intent_of_systemic_treatment'] = chemo_df['intent_of_systemic_treatment'].replace('U', np.nan) # only 1 sample exists
    cols = ['curr_morph_cd', 'curr_topog_cd']
    chemo_df.loc[:, cols] = chemo_df[cols].replace('*U*', np.nan)
    for col in cols: chemo_df.loc[:, col] = chemo_df[col].str[:3] # only keep first three characters - the rest are for specifics
                                                                  # e.g. C50 Breast: C501 Central portion, C504 Upper-outer quadrant, etc
    return chemo_df

def filter_cancer_and_demographic_data(chemo_df, exclude_blood_cancers=True, verbose=False):
    # do not include patients under 18
    mask = chemo_df['age'] < 18
    if verbose:
        logging.info(f"Removing {chemo_df.loc[mask, 'ikn'].nunique()} patients under 18")
    chemo_df = chemo_df[~mask]

    if exclude_blood_cancers:
        mask = chemo_df['curr_topog_cd'].isin(cancer_location_exclude) # remove entries in cancer_location_exclude
        if verbose:
            logging.info(f"Removing {chemo_df.loc[mask, 'ikn'].nunique()} patients with blood cancers")
        chemo_df = chemo_df[~mask]
        
    # remove morphology codes >= 959
    mask = chemo_df['curr_morph_cd'] >= '959'
    if verbose:
        logging.info(f"Removing {chemo_df.loc[mask, 'ikn'].nunique()} patients and " + \
                     f"cancer types {chemo_df.loc[mask, 'curr_morph_cd'].unique()}")
    chemo_df = chemo_df[~mask]
    
    cols = ['curr_morph_cd', 'curr_topog_cd']
    chemo_df = replace_rare_col_entries(chemo_df, cols, verbose=verbose)
                                                    
    return chemo_df

def process_cancer_and_demographic_data(y3, systemic, exclude_blood_cancers=True, verbose=True):
    """Combine systemic (chemotherapy) and y3 (cancer diagnosis and demographic) data
    """
    chemo_df = pd.merge(systemic, y3, on='ikn', how='inner')
    if verbose: logging.info(f"Size of data = {len(chemo_df)}")
    chemo_df['age'] = chemo_df['visit_date'].dt.year - chemo_df['bdate'].dt.year
    chemo_df = clean_cancer_and_demographic_data(chemo_df)
    chemo_df = filter_cancer_and_demographic_data(chemo_df, exclude_blood_cancers=exclude_blood_cancers, verbose=verbose)
    if verbose: logging.info(f"Size of data after cleaning = {len(chemo_df)}")
    return chemo_df

# Immigration
def filter_immigration_data(immigration):
    immigration = clean_string(immigration, ['ikn', 'official_language'])
    immigration['speaks_english'] = immigration['official_language'].isin(english_lang_codes) # able to speak english
    immigration['is_immigrant'] = True
    immigration = immigration[immigration_cols]
    return immigration

def process_immigration_data(chemo_df, immigration):
    """Combine immigration data into chemo_df
    """
    chemo_df = pd.merge(chemo_df, immigration, on='ikn', how='left')
    chemo_df['speaks_english'] = chemo_df['speaks_english'].fillna(True)
    chemo_df['is_immigrant'] = chemo_df['is_immigrant'].fillna(False)
    return chemo_df

# Combordity
def filter_combordity_data(combordity):
    combordity = clean_string(combordity, ['ikn'])
    for col in ['diabetes_diag_date', 'hypertension_diag_date']:
        combordity[col] = pd.to_datetime(combordity[col])
    combordity = combordity.drop(columns=['dxdate'])
    combordity.columns = combordity.columns.str.split('_').str[0]
    return combordity

# Dialysis
def filter_dialysis_data(dialysis):
    dialysis = clean_string(dialysis, ['ikn'])
    dialysis['servdate'] = pd.to_datetime(dialysis['servdate'])
    dialysis = dialysis.sort_values(by='servdate')
    dialysis = dialysis.drop(columns=['feecode'])
    dialysis = dialysis.drop_duplicates()
    dialysis = dialysis.rename(columns={'servdate': 'dialysis_date'})
    return dialysis

def dialysis_worker(partition, days_after=90):
    # determine if patient recieved dialysis x to x+180 days (e.g. 3-6 months) after treatment
    chemo_df, dialysis = partition
    result = []
    for ikn, chemo_group in tqdm.tqdm(chemo_df.groupby('ikn')):
        dialysis_group = dialysis[dialysis['ikn'] == ikn]
        for chemo_idx, chemo_row in chemo_group.iterrows():
            earliest_date = chemo_row['visit_date'] + pd.Timedelta(f'{days_after} days')
            latest_date = earliest_date + pd.Timedelta('180 days')
            mask = dialysis_group['dialysis_date'].between(earliest_date, latest_date)
            if mask.any(): result.append(chemo_idx)
    return result

def process_dialysis_data(chemo_df, dialysis, days_after=90):
    """
    If patient require dialysis between 90 days after chemo visit and before 6 months, patient will be cosidered having CKD
    """
    # filter out patients not in dataset
    filtered_chemo_df = chemo_df[chemo_df['ikn'].isin(dialysis['ikn'])]
    logging.info(f"Only {filtered_chemo_df['ikn'].nunique()} patients have recorded administration of dialysis")
    chemo_df['dialysis'] = False
    result = split_and_parallelize((filtered_chemo_df, dialysis), dialysis_worker, processes=1)
    chemo_df.loc[result, 'dialysis'] = True
    return chemo_df

# Olis (blood work/lab test) data
def filter_olis_data(olis, chemo_ikns, observations=None):
    """
    Args:
        observations (sequence or None): sequeunce of observation codes associated with specific lab tests we want to keep
    """
    logging.warning('For much faster performance, consider using spark.filter_olis_data')
    # organize and format columns
    olis = olis[olis_cols]
    olis = clean_string(olis, ['ikn', 'ObservationCode', 'ReferenceRange', 'Units'])
    olis['ObservationDateTime'] = pd.to_datetime(olis['ObservationDateTime']).dt.floor('D').values # keep only the date, not time
    olis['ObservationReleaseTS'] = pd.to_datetime(olis['ObservationReleaseTS'], errors='coerce').values # there are out of bound timestamp errors
    
    # filter patients not in chemo_df
    olis = olis[olis['ikn'].isin(chemo_ikns)]
    
    if observations:
        # filter rows with excluded observations
        olis = olis[olis['ObservationCode'].isin(observations)]
    
    # remove rows with blood count null or neg values
    olis['value'] = olis.pop('value_recommended_d').astype(float)
    olis = olis[olis['value'].notnull() & (olis['value'] >= 0)]
    
    # remove duplicate rows
    subset = ['ikn','ObservationCode', 'ObservationDateTime', 'value']
    olis = olis.drop_duplicates(subset=subset) 
    
    # if only the patient id, blood, and observation timestamp are duplicated (NOT the blood count value), 
    # keep the most recently RELEASED row
    olis = olis.sort_values(by='ObservationReleaseTS')
    subset = ['ikn','ObservationCode', 'ObservationDateTime']
    olis = olis.drop_duplicates(subset=subset, keep='last')
    
    return olis

def olis_worker(partition, earliest_limit=-5, latest_limit=28):
    chemo_df, olis_df = partition
    result = []
    for ikn, chemo_group in tqdm.tqdm(chemo_df.groupby('ikn')):
        olis_group = olis_df[olis_df['ikn'] == ikn]
        for chemo_idx, chemo_row in chemo_group.iterrows():
            # see if there any blood count data within the dates of interest
            earliest_date = chemo_row['visit_date'] + pd.Timedelta(f'{earliest_limit} days')
            # e.g. set limit to 28 days after chemo administration or the day of next chemo administration, whichever comes first
            latest_date = min(chemo_row['next_visit_date'], chemo_row['visit_date'] + pd.Timedelta(f'{latest_limit} days'))
            tmp = olis_group[olis_group['ObservationDateTime'].between(earliest_date, latest_date)]
            tmp['chemo_idx'] = chemo_idx
            tmp['days_after_chemo'] = (tmp['ObservationDateTime'] - chemo_row['visit_date']).dt.days
            result.extend(tmp[['ObservationCode', 'chemo_idx', 'days_after_chemo', 'value']].values.tolist()) 
    return result

def observation_worker(partition, days_ago=5):
    """A different version of olis worker, where chemo_df does not have next_visit_date column 
    (only has visit_date, usually only one chemo per given week)
    """
    chemo_df, olis_df = partition
    result = []
    for ikn, chemo_group in tqdm.tqdm(chemo_df.groupby('ikn')):
        olis_group = olis_df[olis_df['ikn'] == ikn]
        for chemo_idx, chemo_row in chemo_group.iterrows():
            # see if there any blood count data since last chemo up to x days ago 
            days_since_last_chemo = float(chemo_row['days_since_last_chemo'])
            if not pd.isnull(days_since_last_chemo):
                days_ago = min(days_ago, days_since_last_chemo)
            latest_date = chemo_row['visit_date']
            earliest_date = latest_date - pd.Timedelta(days=days_ago)
            tmp = olis_group[olis_group['ObservationDateTime'].between(earliest_date, latest_date)]
            tmp['chemo_idx'] = chemo_idx
            tmp['days_after_chemo'] = (tmp['ObservationDateTime'] - latest_date).dt.days
            result.extend(tmp[['ObservationCode', 'chemo_idx', 'days_after_chemo', 'value']].values.tolist())
    return result

def closest_measurement_worker(partition, days_after=90):
    """
    Finds the closest measurement x number of days after treatment to 2 years after treatment
    partition must contain only one type of observation (e.g. serume creatinine observations only)
    """
    chemo_df, obs_df = partition
    obs_df = obs_df.sort_values(by='ObservationDateTime') # might be redundant but just in case
    result = []
    for ikn, chemo_group in tqdm.tqdm(chemo_df.groupby('ikn')):
        obs_group = obs_df[obs_df['ikn'] == ikn]
        if obs_group.empty:
            continue

        for chemo_idx, chemo_row in chemo_group.iterrows():
            earliest_date = chemo_row['visit_date'] + pd.Timedelta(f'{days_after} days')
            latest_date = earliest_date + pd.Timedelta('730 days')
            closest_obs = obs_group[obs_group['ObservationDateTime'].between(earliest_date, latest_date)]
            if closest_obs.empty:
                continue
                
            closest_obs = closest_obs.iloc[0]
            days_after_chemo = (closest_obs['ObservationDateTime'] - chemo_row['visit_date']).days
            result.append((chemo_idx, days_after_chemo, closest_obs['value']))
    return result
                          
def postprocess_olis_data(chemo_df, olis_df, observations=all_observations, days_range=range(-5,29)):
    # fill up the blood count dataframes for each blood type
    mapping = {obs_code: pd.DataFrame(index=chemo_df.index, columns=days_range) for obs_code in observations}
    olis_df = olis_df[olis_df['observation_code'].isin(observations)] # exclude obs codes not in selected observations
    for obs_code, obs_group in tqdm.tqdm(olis_df.groupby('observation_code')):
        for day, day_group in obs_group.groupby('days_after_chemo'):
            chemo_indices = day_group['chemo_idx'].values.astype(int)
            obs_count_values = day_group['observation_count'].values.astype(float)
            mapping[obs_code].loc[chemo_indices, int(day)] = obs_count_values
            
    # group together obs codes with same obs name
    freq_map = olis_df['observation_code'].value_counts()
    grouped_observations = group_observations(observations, freq_map)
    
    # get baseline observation counts, combine it to chemo_df 
    num_rows = {'chemo_df': len(chemo_df)} # Number of non-missing rows
    for obs_name, obs_codes in tqdm.tqdm(grouped_observations.items()):
        col = f'baseline_{obs_name}_count'
        for i, obs_code in enumerate(obs_codes):
            # forward fill observation counts from day -5 to day 0
            values = numpy_ffill(mapping[obs_code][range(-5,1)])
            values = pd.Series(values, index=chemo_df.index)
            """
            Initialize using observation counts with the most frequent observations
            This takes priority when there is conflicting count values among the grouped observations
            Fill any missing counts using the observations counts with the next most frequent observations, and so on

            e.g. an example of a group of similar observations
            'alanine_aminotransferase': ['1742-6',  #                    (n=BIG) <- initalize
                                         '1744-2',  # without vitamin B6 (n=MED) <- fill na, does not overwrite '1742-6'
                                         '1743-4'], # with vitamin B6    (n=SMALL) <- fill na, does not overwrite '1742-6', '1744-2'
            """
            chemo_df[col] = values if i == 0 else chemo_df[col].fillna(values)
        num_rows[obs_name] = sum(~chemo_df[col].isnull())
    missing_df = pd.DataFrame(num_rows, index=['num_rows_not_null'])
    missing_df = missing_df.T.sort_values(by='num_rows_not_null')
       
    return mapping, missing_df

# Esas (questionnaire) data
def filter_esas_data(esas, chemo_ikns):
    esas = esas.rename(columns={'esas_value': 'severity', 'esas_resourcevalue': 'symptom'})
    esas = clean_string(esas, ['ikn', 'severity', 'symptom'])
    
    # filter out patients not in chemo_df
    esas = esas[esas['ikn'].isin(chemo_ikns)]
    
    # remove duplicate rows
    subset = ['ikn', 'surveydate', 'symptom', 'severity']
    esas = esas.drop_duplicates(subset=subset) 
    
    # remove Activities & Function as they only have 4 samples
    esas = esas[~(esas['symptom'] == 'Activities & Function:')]
    
    return esas

def esas_worker(partition):
    chemo_df, esas_df = partition
    result = []
    for ikn, chemo_group in tqdm.tqdm(chemo_df.groupby('ikn')):
        esas_group = esas_df[esas_df['ikn'] == ikn]
        for idx, chemo_row in chemo_group.iterrows():
            visit_date = chemo_row['visit_date']
            # even if survey date is like a month ago, we will be forward filling the entries to the visit date
            esas_most_recent = esas_group[esas_group['surveydate'] < visit_date]
            if not esas_most_recent.empty:
                symptoms = list(esas_most_recent['symptom'].unique())
                for symptom in symptoms:
                    esas_specific_symptom = esas_most_recent[esas_most_recent['symptom'] == symptom]
                    # last item is always the last observed grade (we've already sorted by date)
                    result.append((idx, symptom, esas_specific_symptom['severity'].iloc[-1]))
    return result

def get_esas_responses(chemo_df, esas, processes=16):
    # organize and format columns
    esas['surveydate'] = pd.to_datetime(esas['surveydate'])
    esas['severity'] = esas['severity'].astype(int)
    esas = esas.sort_values(by='surveydate') # sort by date
    
    # filter out patients not in dataset
    filtered_chemo_df = chemo_df[chemo_df['ikn'].isin(esas['ikn'])]
    
    # get results
    result = split_and_parallelize((filtered_chemo_df, esas), esas_worker, processes=processes)
    return result

def postprocess_esas_responses(esas_df):
    esas_df = esas_df.sort_values(by=['index', 'symptom', 'severity'])

    # remove duplicates - keep the more severe entry
    # TODO: remove duplicates by using surveydate
    mask = esas_df[['index', 'symptom']].duplicated(keep='last')
    logging.warning(f'{sum(mask)} duplicate entries among total {len(esas_df)} entries')
    esas_df = esas_df[~mask]

    # make each symptom its own column, with severity as its entry value
    esas_df = esas_df.pivot(index='index', columns='symptom')['severity']
    
    return esas_df

# ECOG (perforamnce status)/PRFS (functionality status) data (aka body functionality grade)
def filter_body_functionality_data(df, chemo_ikns, dataset='ecog'):
    if dataset not in {'ecog', 'prfs'}: raise ValueError('dataset must be either ecog or prfs')
        
    # organize and format columns
    df = df.drop(columns=[f'{dataset}_resourcevalue'])
    df = df.rename(columns={f'{dataset}_value': f'{dataset}_grade'})
    df = clean_string(df, ['ikn', f'{dataset}_grade'])
    df[f'{dataset}_grade'] = df[f'{dataset}_grade'].astype(int)
    df['surveydate'] = pd.to_datetime(df['surveydate'])
    
    # filter patients not in chemo_df
    df = df[df['ikn'].isin(chemo_ikns)]
    
    # sort by date
    df = df.sort_values(by='surveydate')
    
    return df

def body_functionality_worker(partition, dataset='ecog'):
    chemo_df, bf_df = partition
    result = []
    for ikn, chemo_group in tqdm.tqdm(chemo_df.groupby('ikn'), position=0):
        bf_group = bf_df[bf_df['ikn'] == ikn]
        for idx, chemo_row in chemo_group.iterrows():
            visit_date = chemo_row['visit_date']
            bf_most_recent = bf_group[bf_group['surveydate'] < visit_date]
            if not bf_most_recent.empty:
                # last item is always the last observed grade (we've already sorted by date)
                result.append((idx, bf_most_recent[f'{dataset}_grade'].iloc[-1]))
    return result

# Blood Transfusions
def filter_blood_transfusion_data(chunk, chemo_ikns, event='H'):
    col, _ = event_map[event]['date_col_name']
    # organize and format columns
    chunk = clean_string(chunk, ['ikn', 'btplate', 'btredbc']) 
    chunk[col] = pd.to_datetime(chunk[col])
    
    # filter patients not in chemo_df
    chunk = chunk[chunk['ikn'].isin(chemo_ikns)]
    
    # filter rows where no transfusions occured
    chunk = chunk[((chunk['btplate'] == 'Y') | (chunk['btplate'] == '1')) | # btplate means blood transfusion - platelet
                  ((chunk['btredbc'] == 'Y') | (chunk['btredbc'] == '1'))]  # btredbc means blood transfusion - red blood cell
    
    # get only the select columns
    chunk = chunk[[col, 'ikn', 'btplate', 'btredbc']]
    
    # sort by date
    chunk = chunk.sort_values(by=col)
    
    return chunk

def blood_transfusion_worker(partition, event='H'):
    date_col, _ = event_map[event]['date_col_name']
    chemo_df, bt_df = partition
    result = []
    for ikn, chemo_group in tqdm.tqdm(chemo_df.groupby('ikn')):
        earliest_date = chemo_group['visit_date'] - pd.Timedelta('5 days')
        latest_date = chemo_group['next_visit_date'] + pd.Timedelta('3 days') 
        bt_group = bt_df[bt_df['ikn'] == ikn]
        for i, bt_row in bt_group.iterrows():
            admdate = bt_row[date_col]
            tmp = chemo_group[(earliest_date <= admdate) & (latest_date >= admdate)]
            for chemo_idx in tmp.index:
                if not pd.isnull(bt_row['btplate']): # can only be NaN, Y, or 1
                    result.append((chemo_idx, str(admdate.date()), f'{event}_platelet_transfusion_date'))
                if not pd.isnull(bt_row['btredbc']): # can only be NaN, Y, or 1
                    result.append((chemo_idx, str(admdate.date()), f'{event}_hemoglobin_transfusion_date'))
    return result

def extract_blood_transfusion_data(chemo_df, main_dir, event='H'):
    database_name = event_map[event]['database_name']
    col, _ = event_map[event]['date_col_name']
    bt_data = pd.read_csv(f'{main_dir}/data/{database_name}_transfusion.csv', dtype=str)
    bt_data[col] = pd.to_datetime(bt_data[col])
    
    # filter out patients not in transfusion data
    filtered_chemo_df = chemo_df[chemo_df['ikn'].isin(bt_data['ikn'])]

    # get results
    worker = partial(blood_transfusion_worker, event=event)
    result = split_and_parallelize((filtered_chemo_df, bt_data), worker)
    
    # save results
    result = pd.DataFrame(result, columns=['chemo_idx', 'transfusion_date', 'transfusion_type'])
    result.to_csv(f'{main_dir}/data/{database_name}_transfusion2.csv', index=False)
    
def postprocess_blood_transfusion_data(chemo_df, main_dir, event='H'):
    database_name = event_map[event]['database_name']
    df = pd.read_csv(f'{main_dir}/data/{database_name}_transfusion2.csv')
    for transfusion_type, group in df.groupby('transfusion_type'):
        chemo_indices = group['chemo_idx'].values.astype(int)
        dates = group['transfusion_date'].values
        chemo_df.loc[chemo_indices, transfusion_type] = dates
    return chemo_df, df

# ED/H Events
def get_event_reason(df, event):
    raw_diag_codes = pd.Series(df[diag_cols].values.flatten())
    raw_diag_codes = raw_diag_codes[raw_diag_codes.notnull()]
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

def filter_event_data(df, chemo_ikns, event='H', remove_ED_causing_H=True):
    arr_date_col, dep_date_col = event_map[event]['date_col_name']
    
    # organize and format columns
    df = clean_string(df, ['ikn'] + diag_cols)
    df['arrival_date'] = pd.to_datetime(df[arr_date_col])
    df['depart_date'] = pd.to_datetime(df[dep_date_col])
    
    # remove rows with null date values
    df = df[df['arrival_date'].notnull()]
    df = df[df['depart_date'].notnull()]
    
    # remove ED visits that resulted in hospitalizations
    if remove_ED_causing_H and event == 'ED':
        df = clean_string(df, ['to_type'])
        df = df[~df['to_type'].isin(['I', 'P'])]

    # filter away patients not in chemo dataframe
    df = df[df['ikn'].isin(chemo_ikns)]

    # get reason for event visit (Treatment Related, Fever and Infection, GI Toxicity)
    # NOTE: Treatment Related means all treatments which INCLUDES Fever and Infection, GI Toxicity
    df = get_event_reason(df, event)

    # keep only selected columns
    event_cause_cols = event_map[event]['event_cause_cols']
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
    chemo_df, event_df = partition
    cols = event_df.columns.drop(['ikn']).tolist()
    placeholder = ''
    result = []
    for ikn, chemo_group in tqdm.tqdm(chemo_df.groupby('ikn')):
        event_group = event_df[event_df['ikn'] == ikn]
        event_arrival_dates = event_group['arrival_date'] 
        event_depart_dates = event_group['depart_date'] # keep in mind, for ED, depart date and arrival date are the same
        for chemo_idx, visit_date in chemo_group['visit_date'].iteritems():
            # get feature - closest event before and on chemo visit, and number of prev events prior to visit
            mask = event_depart_dates <= visit_date
            num_prior_events = mask.sum()
            closest_event_date = event_depart_dates[mask].max()
            if pd.notnull(closest_event_date):
                closest_event = event_group.loc[event_group['depart_date'] == closest_event_date, cols]
                # if there are more than one row with same depart dates, get the first row (the earlier arrival date)
                result.append(['feature', chemo_idx, num_prior_events] + closest_event.values.tolist()[0]) 

            # get potential target - closest event after chemo visit
            closest_event_date = event_arrival_dates[event_arrival_dates > visit_date].min()
            if pd.notnull(closest_event_date):
                closest_event = event_group.loc[event_group['arrival_date'] == closest_event_date, cols]
                result.append(['target', chemo_idx, placeholder] + closest_event.values.tolist()[0])
                
    return result

def extract_event_dates(chemo_df, event_df, output_path, event='H', processes=16):
    event_name = event_map[event]['event_name']
    logging.info(f"Beginning mutiprocess to extract {event_name} dates using {processes} processes")
    worker = partial(event_worker, event=event)
    chemo_df = chemo_df[chemo_df['ikn'].isin(event_df['ikn'])] # filter out patients not in dataset
    event_dates = split_and_parallelize((chemo_df, event_df), worker, processes=processes)
    event_dates = np.array(event_dates).astype(str)
    
    # save results
    event_cause_cols = event_map[event]['event_cause_cols']
    event_dates = pd.DataFrame(event_dates, columns=['feature_or_target', 'chemo_idx', f'num_prior_{event}s', 
                                                     'arrival_date', 'depart_date'] + event_cause_cols)
    event_dates.to_csv(f'{output_path}/data/{event}_dates.csv', index=False)
    
    return event_dates

# Inpatients
def get_inpatient_indices(partition):
    chemo_df, H_df = partition
    result = set()
    for ikn, chemo_group in tqdm.tqdm(chemo_df.groupby('ikn')):
        H_group = H_df[H_df['ikn'] == ikn]
        for H_idx, H_row in H_group.iterrows():
            arrival_date = H_row['arrival_date']
            depart_date = H_row['depart_date']
            mask = chemo_group['visit_date'].between(arrival_date, depart_date)
            result.update(chemo_group[mask].index)
    return result
