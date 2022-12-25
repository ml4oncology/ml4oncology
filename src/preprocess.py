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
from functools import partial

from tqdm import tqdm
import pandas as pd
import numpy as np

from src.config import (
    root_path, sas_folder,  
    min_chemo_date, max_chemo_date, 
    all_observations, 
    english_lang_codes,
    din_exclude, cisplatin_dins, cisplatin_cco_drug_code,
    cancer_location_exclude,
    world_region_country_map, world_region_language_map,
    INTENT, OBS_CODE, OBS_VALUE, OBS_DATE, OBS_RDATE,
    systemic_cols, drug_cols, y3_cols, immigration_cols, diag_cols, 
    olis_cols, observation_cols, observation_change_cols, event_main_cols,
    event_map, diag_code_mapping)
from src.utility import (
    clean_string, 
    get_years_diff, 
    group_observations,
    numpy_ffill, 
    replace_rare_categories,   
    split_and_parallelize, 
)

import logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s', 
    datefmt='%I:%M:%S'
)

###############################################################################
# sas7bdat Files
###############################################################################
def sas_to_csv(
    name, 
    new_name='', 
    folder=sas_folder, 
    transfer_date=None, 
    chunk_load=False, 
    chunksize=10**6
):
    filename = (name if transfer_date is None 
                else f'transfer{transfer_date}/{name}')
    name = new_name if new_name else name
    if not chunk_load:
        df = pd.read_sas(f'{root_path}/{folder}/{filename}.sas7bdat')
        df = clean_sas(df, name)
        df.to_csv(f"{root_path}/data/{name}.csv", index=False)
    else:
        chunks = pd.read_sas(
            f'{root_path}/{folder}/{filename}.sas7bdat', 
            chunksize=chunksize
        )
        for i, chunk in tqdm(enumerate(chunks), desc='Converting to csv'):
            chunk = clean_sas(chunk, name)
            header = True if i == 0 else False
            chunk.to_csv(
                f"{root_path}/data/{name}.csv", 
                header=header, 
                mode='a', 
                index=False
            )
            
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

###############################################################################
# Systemic (Chemo) Data
###############################################################################
def filter_systemic_data(
    df, 
    regimens, 
    cols=None, 
    remove_inpatients=True, 
    exclude_dins=True,            
    replace_rare_regimens=True, 
    include_drug_info=False, 
    verbose=False
):
    """
    Args:
        regimens (pd.DataFrame): Table of select annotated regimens. 
            See load_reviewed_regimens/load_included_regimens
    """
    if cols is None: cols = systemic_cols.copy()
    df = clean_up_systemic(df)
    df = filter_regimens(df, regimens, verbose=verbose)
    df = filter_date(df, verbose=verbose)
    
    if remove_inpatients: 
        df = filter_inpatients(df, verbose=verbose)
        
    if exclude_dins: # DIN: Drug Identification Number
        df = df[~df['din'].isin(din_exclude)] # remove dins in din_exclude
        
    if replace_rare_regimens:
        # replace regimens with less than 6 patients to 'Other'
        df = replace_rare_categories(df, ['regimen'], verbose=verbose)
    
    if include_drug_info: cols += drug_cols
    df = df[cols]
    df = df.drop_duplicates()
    return df

def filter_by_drugs(df, drug='cisplatin', verbose=False):
    if drug == 'cisplatin':
        keep_dins = cisplatin_dins
        keep_ccos = cisplatin_cco_drug_code
    
    mask = df['din'].isin(keep_dins) | df['cco_drug_code'].isin(keep_ccos)
    if verbose: 
        logging.info(f"Removing {sum(~mask)} sessions with no {drug} "
                     "administrated")
    df = df[mask]
    df['drug'] = drug
    df = df.drop(columns=['din', 'cco_drug_code'])
    # remove rows with measurement unit of g, unit, mL, nan
    df = df[df['measurement_unit'].isin(['mg', 'MG'])]

    # combine together multiple administration of the same drug on the same day
    dosage = 'dose_administered'
    cols = df.columns.drop(dosage)
    mask = df.duplicated(subset=cols, keep=False)
    if verbose: 
        logging.info(f"Collapsing {sum(~df[mask].duplicated())} same day "
                     "sessions with multiple doses administered")
    same_day_doses = df[mask].groupby(['ikn', 'visit_date'])[dosage]
    df.loc[mask, dosage] = same_day_doses.transform('sum')
    df = pd.concat([df[~mask], df[mask].drop_duplicates()])
    
    # order the dataframe by date
    df = df.sort_values(by='visit_date')
        
    return df

def systemic_worker(partition, method='merge', merge_days=4):
    """Creates engineered features such as:
    1. days_since_starting_chemo
    2. days_since_last_chemo
    3. chemo_cycle
    4. immediate_new_regimen
    
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
    NOTE: use 2nd method when you don't care about the chemo dosages and just 
    care about the rough dates
    
    Args:
        method (str): either 'merge' or 'one-per-week'
        merge_days (int): max number of intervals/days you can merge
    """
    if method not in {'merge', 'one-per-week'}:
        raise ValueError('method must be either merge or one-per-week')
        
    values = []
    date = 'visit_date'
    for ikn, df in tqdm(partition.groupby('ikn')):
        df = df.sort_values(by=date) # might be redundant but just in case

        if method == 'one-per-week':
            # keep only the first chemo visit of a given week
            keep_indices = []
            tmp = df
            while not tmp.empty:
                first_chemo_idx = tmp.index[0]
                keep_indices.append(first_chemo_idx)
                first_visit_date = df[date].loc[first_chemo_idx]
                next_week_date = first_visit_date + pd.Timedelta('7 days')
                tmp = tmp[tmp[date] >= next_week_date]
            df = df.loc[keep_indices]
        elif method == 'merge':
            # merges small intervals into a 4 day cycle, or to the row 
            # below/above that has interval greater than 4 days
            df[f'next_{date}'] = df[date].shift(-1)
            df['chemo_interval'] = (df[f'next_{date}'] - df[date]).dt.days
            df = merge_intervals(df, merge_days=4)
            if df.empty:
                # most likely patient had consecutive 1 day interval chemo 
                # sessions that totaled less than 4 days OR patient has only 
                # two chemo sessions (one row) in which interval was less than 
                # 4 days
                continue

        start_date = df[date].iloc[0]
        df['days_since_starting_chemo'] = (df[date] - start_date).dt.days
        df['days_since_last_chemo'] = (df[date] - df[date].shift()).dt.days
        
        # identify line of therapy (the nth different palliative intent 
        # chemotherapy regimen taken)
        # NOTE: all other intent treatment are given line of therapy of 0. 
        # Usually (not always but oh well) once the first palliative treatment 
        # appears, the rest of the treatments remain palliative
        new_regimen = (df['regimen'] != df['regimen'].shift())
        palliative_intent = df[INTENT] == 'P'
        df['line_of_therapy'] = (new_regimen & palliative_intent).cumsum()

        # identify chemo cycle number (resets when patient undergoes new chemo 
        # regimen or there is a 60 day gap)
        mask = new_regimen | (df['days_since_last_chemo'] >= 60)
        df['chemo_cycle'] = compute_chemo_cycle(mask)

        # identify if this is the first chemo cycle of a new regimen 
        # immediately after the old one
        mask = new_regimen & (df['days_since_last_chemo'] < 30)
        df['immediate_new_regimen'] = mask

        # convert to list and combine for faster performance
        values.extend(df.values.tolist())
        
    return values

def process_systemic_data(
    df, 
    cycle_length_mapping=None, 
    method='merge', 
    merge_days=4
):
    """
    Args:
        cycle_length_mapping (dict): A mapping between regimens (str) and their
            shortest cycle lengths in days (int). If None, no cycle length 
            feature is created
    """
    if cycle_length_mapping is not None:
        df['cycle_length'] = df['regimen'].map(cycle_length_mapping)
        
    worker = partial(systemic_worker, method=method, merge_days=merge_days)
    values = split_and_parallelize(df, worker)
    extra_cols = [
        'days_since_starting_chemo', 'days_since_last_chemo', 
        'line_of_therapy', 'chemo_cycle', 'immediate_new_regimen'
    ]
    if method == 'merge': 
        extra_cols = ['next_visit_date', 'chemo_interval'] + extra_cols
    cols = df.columns.tolist() + extra_cols
    df = pd.DataFrame(values, columns=cols)
    return df

###############################################################################
# Systemic (Chemo) Data - Helper Functions
###############################################################################
def clean_up_systemic(df):
    cols = [
        'ikn', 'din', 'cco_drug_code', INTENT, 'inpatient_flag', 
        'measurement_unit'
    ]
    df = clean_string(df, cols)
    df['visit_date'] = pd.to_datetime(df['visit_date'])
    df = df.sort_values(by='visit_date') # order the data dataframe by date
    return df

def filter_regimens(df, regimens, verbose=False):
    col = 'regimen'
    df = df[df[col].notnull()] # filter out rows with no regimen data 
    df = df[df[col].isin(regimens[col])] # keep only selected reigmens
    
    # change regimen name to the correct mapping
    mask = regimens['relabel'].notnull()
    old_name, new_name = regimens.loc[mask, [col, 'relabel']].T.values
    df[col] = df[col].replace(old_name, new_name)
    
    return df

def filter_date(
    df, 
    min_date=min_chemo_date, 
    max_date=max_chemo_date, 
    verbose=False
):
    """Remove chemo recieved before the new ALR system was set up"""
    if verbose: 
        logging.info(f"{len(df)} chemo treatments occured between "
                     f"{df['visit_date'].min()} to {df['visit_date'].max()}")
    mask = df['visit_date'].between(min_date, max_date)
    if verbose: 
        logging.info(f"Removing {sum(~mask)} chemo treatments that occured "
                     f"before {min_date} and after {max_date}.")
    df = df[mask]
    return df

def filter_inpatients(df, verbose=False):
    """Remove chemo treatment recieved as an inpatient"""
    outpatient_mask = df['inpatient_flag'] == 'N'
    inpatient_mask = df['inpatient_flag'] == 'Y'
    missing_mask = df['inpatient_flag'].isnull()
    if verbose:
        logging.info(f"Removing {sum(inpatient_mask)} inpatient chemo "
                     f"treatments and {sum(missing_mask)} unknown inpatient "
                     "status chemo treatements")
    df = df[outpatient_mask]
    return df

def compute_chemo_cycle(mask):
    """
    Args:
        mask (pd.Series): An alignable boolean series indicating where regimens
            changed and where there were significant gaps between same regimens
    
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
        if curr_idx == 0:
            # leave the very first entry of the whole dataframe as is
            return
        merge_idx = curr_idx-1
        df.loc[merge_idx, 'next_visit_date'] = df.loc[curr_idx, 'next_visit_date']
    elif which == 'below':
        merge_idx = curr_idx+1
        df.loc[merge_idx, 'visit_date'] = df.loc[curr_idx, 'visit_date']
            
    for col in merge_cols: 
        df.loc[merge_idx, col] += df.loc[curr_idx, col]

def merge_intervals(df, merge_days=4):
    """Merges small intervals into a x day cycle, or to the row below/above 
    that has interval greater than x days. 
    
    If after merging and the final interval is still less than x days, an 
    empty dataframe is essentially returned (this includes single row (patient 
    only had two chemo sessions) that has interval less than x days)
    """
    df = df.reset_index(drop=True)
    duration, dosage = 'chemo_interval', 'dose_administered'
    merge_cols = [duration]
    # WARNING: Assumes dataframe only contains one type of drug if drug info 
    # is included
    merge_dosage = dosage in df.columns
    if merge_dosage:
        if df['drug'].nunique() > 1: 
            raise NotImplementedError('Merging dosage for more than 1 drug is '
                                      'not supported yet')
        merge_cols.append(dosage)
    
    remove_indices = []
    for i in range(len(df)):
        if df.loc[i, duration] >= merge_days or pd.isnull(df.loc[i, duration]):
            continue
        if i == len(df)-1 or df.loc[i, 'regimen'] != df.loc[i+1, 'regimen']:
            # merge with the row above if last entry or last entry of an old 
            # regimen
            merge_with_row(df, i, merge_cols, which='above')
        else:
            # merge with the row below
            merge_with_row(df, i, merge_cols, which='below')
        remove_indices.append(i)
    df = df.drop(index=remove_indices)
    return df

###############################################################################
# y3 (Cancer Diagnosis and Demographic) Data
###############################################################################
def filter_y3_data(y3, include_death=False):
    clean_cols = ['ikn', 'lhin_cd', 'curr_morph_cd', 'curr_topog_cd', 'sex']
    y3 = clean_string(y3, clean_cols)
    y3['bdate'] = pd.to_datetime(y3['bdate'])
    
    keep_cols = y3_cols.copy()
    if include_death:
        # organize and format columns
        y3['dthdate'] = pd.to_datetime(y3['dthdate'])
        y3 = y3.rename(columns={'dthdate': 'D_date'})
        keep_cols += ['D_date']
        
    y3 = y3[keep_cols]
    return y3

def clean_cancer_and_demographic_data(chemo_df):
    chemo_df[BSA] = chemo_df[BSA].replace(0, np.nan).replace(-99, np.nan)
    chemo_df[INTENT] = chemo_df[INTENT].replace('U', np.nan)
    for col in ['curr_morph_cd', 'curr_topog_cd']: 
        chemo_df[col] = chemo_df[col].replace('*U*', np.nan)
        # only keep first three characters - the rest are for specifics
        # e.g. C50 Breast: C501 Central portion, C504 Upper-outer quadrant, etc
        chemo_df[col] = chemo_df[col].str[:3]
    return chemo_df

def filter_cancer_and_demographic_data(
    chemo_df, 
    exclude_blood_cancers=True, 
    verbose=False
):
    # do not include patients under 18
    mask = chemo_df['age'] < 18
    if verbose:
        N = chemo_df.loc[mask, 'ikn'].nunique()
        logging.info(f"Removing {N} patients under 18")
    chemo_df = chemo_df[~mask]

    if exclude_blood_cancers:
        # remove codes in cancer_location_exclude
        mask = chemo_df['curr_topog_cd'].isin(cancer_location_exclude)
        if verbose:
            N = chemo_df.loc[mask, 'ikn'].nunique()
            logging.info(f"Removing {N} patients with blood cancers")
        chemo_df = chemo_df[~mask]
        
    # remove morphology codes >= 959
    mask = chemo_df['curr_morph_cd'] >= '959'
    if verbose:
        N = chemo_df.loc[mask, 'ikn'].nunique()
        removed_cancers = chemo_df.loc[mask, 'curr_morph_cd'].unique()
        logging.info(f"Removing {N} patients and cancer types {removed_cancers}")
    chemo_df = chemo_df[~mask]
    
    cols = ['curr_morph_cd', 'curr_topog_cd']
    chemo_df = replace_rare_categories(chemo_df, cols, verbose=verbose)
                                                    
    return chemo_df

def process_cancer_and_demographic_data(
    y3, 
    systemic, 
    exclude_blood_cancers=True, 
    verbose=True
):
    """Combine systemic (chemotherapy) and y3 (cancer diagnosis and 
    demographic) data
    """
    chemo_df = pd.merge(systemic, y3, on='ikn', how='inner')
    if verbose: logging.info(f"Size of data = {len(chemo_df)}")
    chemo_df['age'] = get_years_diff(chemo_df, 'visit_date', 'bdate')
    chemo_df = clean_cancer_and_demographic_data(chemo_df)
    chemo_df = filter_cancer_and_demographic_data(
        chemo_df, exclude_blood_cancers=exclude_blood_cancers, verbose=verbose
    )
    if verbose: logging.info(f"Size of data after cleaning = {len(chemo_df)}")
    return chemo_df

###############################################################################
# Immigration
###############################################################################
def filter_immigration_data(df):
    df = clean_string(df, ['ikn', 'official_language'])
    # able to speak english
    df['speaks_english'] = df['official_language'].isin(english_lang_codes)
    df['is_immigrant'] = True
    df['landing_date'] = pd.to_datetime(df['landing_date'])
    df = df[immigration_cols]
    return df

def process_immigration_data(chemo_df, immigration):
    """Combine immigration data into chemo_df"""
    chemo_df = pd.merge(chemo_df, immigration, on='ikn', how='left')
    chemo_df['speaks_english'].fillna(True, inplace=True)
    chemo_df['is_immigrant'].fillna(False, inplace=True)
    
    # get years since immigrating to Canada 
    col = 'years_since_immigration'
    chemo_df[col] = get_years_diff(chemo_df, 'visit_date', 'landing_date')
    # for non-immigrants / long-term residents, we will say they "immigrated"
    # the year they were born
    chemo_df[col].fillna(chemo_df['age'], inplace=True)
    
    # remove landing date
    chemo_df.drop(columns=['landing_date'], inplace=True)
    
    return chemo_df

def get_world_region_of_birth(df, ctry_code_map):
    """Get patient's birthplace by world region
    """
    reformat = lambda mapping: {v: k for k, vs in mapping.items() for v in vs}
    col = 'country_birth'
    # map the country code to country name
    df[col] = df[col].map(reformat(ctry_code_map))
    # map the country name to world region
    df['world_region_of_birth'] = df[col].map(reformat(world_region_country_map))
    df['world_region_of_birth'].fillna('Other', inplace=True)
    return df

def get_years_diff(df, col1, col2):
    return df[col1].dt.year - df[col2].dt.year

###############################################################################
# Combordity
###############################################################################
def filter_combordity_data(combordity):
    combordity = clean_string(combordity, ['ikn'])
    for col in ['diabetes_diag_date', 'hypertension_diag_date']:
        combordity[col] = pd.to_datetime(combordity[col])
    combordity = combordity.drop(columns=['dxdate'])
    combordity.columns = combordity.columns.str.split('_').str[0]
    return combordity

###############################################################################
# Dialysis
###############################################################################
def filter_dialysis_data(dialysis):
    dialysis = clean_string(dialysis, ['ikn'])
    dialysis['servdate'] = pd.to_datetime(dialysis['servdate'])
    dialysis = dialysis.sort_values(by='servdate')
    dialysis = dialysis.drop(columns=['feecode'])
    dialysis = dialysis.drop_duplicates()
    dialysis = dialysis.rename(columns={'servdate': 'dialysis_date'})
    return dialysis

def dialysis_worker(partition, days_after=90):
    """Determine if patient recieved dialysis x to x+180 days (e.g. 3-6 months)
    after treatment
    """
    days_after_treatment = pd.Timedelta(f'{days_after} days')
    
    chemo_df, dialysis = partition
    result = []
    for ikn, chemo_group in tqdm(chemo_df.groupby('ikn')):
        dialysis_group = dialysis[dialysis['ikn'] == ikn]
        for chemo_idx, chemo_row in chemo_group.iterrows():
            earliest_date = chemo_row['visit_date'] + days_after_treatment
            latest_date = earliest_date + pd.Timedelta('180 days')
            mask = dialysis_group['dialysis_date'].between(earliest_date, latest_date)
            if mask.any(): result.append(chemo_idx)
    return result

def process_dialysis_data(chemo_df, dialysis, days_after=90):
    """Determine if patients had dialysis
    
    If patient require dialysis between 90 days after chemo visit and before
    6 months, patient will be cosidered having CKD
    """
    # filter out patients not in dataset
    filtered_chemo_df = chemo_df[chemo_df['ikn'].isin(dialysis['ikn'])]
    logging.info(f"Only {filtered_chemo_df['ikn'].nunique()} patients have "
                 "recorded administration of dialysis")
    chemo_df['dialysis'] = False
    result = split_and_parallelize(
        (filtered_chemo_df, dialysis), 
        dialysis_worker, 
        processes=1
    )
    chemo_df.loc[result, 'dialysis'] = True
    return chemo_df

###############################################################################
# OHIP (Palliative Care Consultation Service (PCCS) Billing) Data
###############################################################################
def filter_ohip_data(ohip, billing_codes=None):
    """
    Args:
        billing_codes (list): A sequeunce of billing codes (str) associated 
            with specific services we want to keep
    """
    ohip = clean_string(ohip, ['ikn', 'feecode'])
    if billing_codes is not None:
        ohip = ohip[ohip['feecode'].isin(billing_codes)]
    ohip['servdate'] = pd.to_datetime(ohip['servdate'])
    ohip = ohip.sort_values(by='servdate')
    return ohip

def ohip_worker(partition):
    """Determine if patient received any palliative care consultation service 
    (PCCS) prior to treatment
    """
    chemo_df, ohip = partition
    result = []
    for ikn, chemo_group in tqdm(chemo_df.groupby('ikn')):
        ohip_group = ohip[ohip['ikn'] == ikn]
        for chemo_idx, chemo_row in chemo_group.iterrows():
            mask = ohip_group['servdate'] <= chemo_row['visit_date']
            ohip_before_chemo = ohip_group[mask]
            if ohip_before_chemo.empty:
                continue
            # last item is always the last observed service date 
            # (we've already sorted by date)
            ohip_before_chemo = ohip_before_chemo.iloc[-1]
            result.append((chemo_idx, ohip_before_chemo['servdate']))
    return result

def process_ohip_data(chemo_df, ohip):
    # filter out patients not in dataset
    filtered_chemo_df = chemo_df[chemo_df['ikn'].isin(ohip['ikn'])]
    logging.info(f"{filtered_chemo_df['ikn'].nunique()} patients have received "
                 "palliative care consultation services (PCCS)")
    result = split_and_parallelize((filtered_chemo_df, ohip), ohip_worker)
    result = pd.DataFrame(
        result, columns=['index', 'PCCS_date']
    )
    result = result.set_index('index')
    chemo_df = chemo_df.join(result, how='left')
    return chemo_df

###############################################################################
# OLIS (Blood Work/Lab Test) Data
###############################################################################
def filter_olis_data(olis, chemo_ikns, observations=None):
    """
    Args:
        observations (list): A sequeunce of observation codes (str) associated 
            with specific lab tests we want to keep
    """
    logging.warning('For much faster performance, consider using '
                    'spark.filter_olis_data')
    # organize and format columns
    clean_cols = ['ikn', OBS_CODE, 'ReferenceRange', 'Units']
    olis = clean_string(olis, clean_cols)
    # get the observation date
    olis[OBS_DATE] = pd.to_datetime(olis[OBS_DATE]).dt.floor('D').values
    # get the observation release date
    olis[OBS_RDATE] = pd.to_datetime(olis[OBS_RDATE], errors='coerce').values
    olis = olis[olis_cols]
    
    # filter patients not in chemo_df
    olis = olis[olis['ikn'].isin(chemo_ikns)]
    
    if observations:
        # filter rows with excluded observations
        olis = olis[olis[OBS_CODE].isin(observations)]
    
    # remove rows with blood count null or neg values
    olis['value'] = olis.pop(OBS_VALUE).astype(float)
    olis = olis[olis['value'].notnull() & (olis['value'] >= 0)]
    
    # remove duplicate rows
    subset = ['ikn', OBS_CODE, OBS_DATE, 'value']
    olis = olis.drop_duplicates(subset=subset) 
    
    # if only the patient id, blood, and observation timestamp are duplicated 
    # (NOT the blood count value), keep the most recently RELEASED row
    olis = olis.sort_values(by=OBS_RDATE)
    subset = ['ikn', OBS_CODE, OBS_DATE]
    olis = olis.drop_duplicates(subset=subset, keep='last')
    
    return olis

def olis_worker(partition, earliest_limit=-5, latest_limit=28):
    earliest_days = pd.Timedelta(f'{earliest_limit} days')
    latest_days = pd.Timedelta(f'{latest_limit} days')
    
    chemo_df, olis_df = partition
    keep_cols = [OBS_CODE, 'chemo_idx', 'days_after_chemo', 'value']
    result = []
    for ikn, chemo_group in tqdm(chemo_df.groupby('ikn')):
        olis_group = olis_df[olis_df['ikn'] == ikn]
        for chemo_idx, chemo_row in chemo_group.iterrows():
            # See if there any blood count data within the dates of interest.
            # Set latest date to x days after chemo administration or the day 
            # of next chemo administration, whichever comes first
            earliest_date = chemo_row['visit_date'] + earliest_days
            latest_date = chemo_row['visit_date'] + latest_days
            latest_date = min(latest_date, chemo_row['next_visit_date'])
            mask = olis_group[OBS_DATE].between(earliest_date, latest_date)
            
            tmp = olis_group[mask]
            date_diff = tmp[OBS_DATE] - chemo_row['visit_date']
            tmp['days_after_chemo'] = date_diff.dt.days
            tmp['chemo_idx'] = chemo_idx
            result.extend(tmp[keep_cols].values.tolist()) 
    return result

def observation_worker(partition, days_ago=5):
    """A different version of olis worker, where chemo_df does not have 
    next_visit_date column (only has visit_date, usually only one chemo per 
    given week)
    """
    chemo_df, olis_df = partition
    keep_cols = [OBS_CODE, 'chemo_idx', 'days_after_chemo', 'value']
    result = []
    for ikn, chemo_group in tqdm(chemo_df.groupby('ikn')):
        olis_group = olis_df[olis_df['ikn'] == ikn]
        for chemo_idx, chemo_row in chemo_group.iterrows():
            # see if there any blood count data since last chemo up to x days 
            # ago 
            days_since_last_chemo = float(chemo_row['days_since_last_chemo'])
            if not pd.isnull(days_since_last_chemo):
                days_ago = min(days_ago, days_since_last_chemo)
            latest_date = chemo_row['visit_date']
            earliest_date = latest_date - pd.Timedelta(days=days_ago)
            mask = olis_group[OBS_DATE].between(earliest_date, latest_date)
            tmp = olis_group[mask]
            tmp['chemo_idx'] = chemo_idx
            tmp['days_after_chemo'] = (tmp[OBS_DATE] - latest_date).dt.days
            result.extend(tmp[keep_cols].values.tolist())
    return result

def closest_measurement_worker(partition, days_after=90):
    """Finds the closest measurement x number of days after treatment to 2 
    years after treatment. Partition must contain only one type of observation 
    (e.g. serume creatinine observations only)
    """
    days_after_treatment = pd.Timedelta(f'{days_after} days')
    
    chemo_df, obs_df = partition
    obs_df = obs_df.sort_values(by=OBS_DATE) # might be redundant but just in case
    result = []
    for ikn, chemo_group in tqdm(chemo_df.groupby('ikn')):
        obs_group = obs_df[obs_df['ikn'] == ikn]
        if obs_group.empty:
            continue

        for chemo_idx, chemo_row in chemo_group.iterrows():
            earliest_date = chemo_row['visit_date'] + days_after_treatment
            latest_date = earliest_date + pd.Timedelta('730 days')
            mask = obs_group[OBS_DATE].between(earliest_date, latest_date)
            closest_obs = obs_group[mask]
            if closest_obs.empty:
                continue
                
            closest_obs = closest_obs.iloc[0]
            diff_date = closest_obs[OBS_DATE] - chemo_row['visit_date']
            days_after_chemo = diff_date.days
            result.append((chemo_idx, days_after_chemo, closest_obs['value']))
    return result

def observation_change_worker(partition):
    # finds the change since last measurement observation
    result = []
    for ikn, group in tqdm(partition.groupby('ikn')):
        change = group[observation_cols] - group[observation_cols].shift()
        result.append(change.reset_index().to_numpy())
    return np.concatenate(result)
                          
def postprocess_olis_data(
    chemo_df, 
    olis_df, 
    observations=all_observations, 
    days_range=range(-5,29)
):
    # for each observation type, create a time series of observation counts 
    # during day X to day Y after treatment (X and Y from days_range) for each 
    # session
    mapping = {obs_code: pd.DataFrame(index=chemo_df.index, columns=days_range) 
               for obs_code in observations}
    # exclude obs codes not in selected observations
    olis_df = olis_df[olis_df['observation_code'].isin(observations)]
    for obs_code, obs_group in tqdm(olis_df.groupby('observation_code')):
        for day, day_group in obs_group.groupby('days_after_chemo'):
            chemo_indices = day_group['chemo_idx'].values.astype(int)
            obs_values = day_group['observation_count'].values.astype(float)
            mapping[obs_code].loc[chemo_indices, int(day)] = obs_values
            
    # group together obs codes with same obs name
    freq_map = olis_df['observation_code'].value_counts()
    grouped_observations = group_observations(observations, freq_map)
    
    # get baseline (pre-treatment) observation values, combine it to chemo_df 
    num_rows = {'chemo_df': len(chemo_df)} # Number of non-missing rows
    for obs_name, obs_codes in tqdm(grouped_observations.items()):
        col = f'baseline_{obs_name}_count'
        # Note: observation codes are ordered by their prevalence
        for i, obs_code in enumerate(obs_codes):
            # forward fill observation counts from day -5 to day 0
            values = numpy_ffill(mapping[obs_code][range(-5,1)])
            values = pd.Series(values, index=chemo_df.index)
            # When there are conflicting observations values among the grouped
            # observations, take the value of the most frequent observation code
            # E.g. alanine aminotransferase includes observation codes 
            # 1742-6, 1744-2, 1743-4, ordered by their prevalence. If we see
            # two different baseline observation value from 1742-6 and 1744-2, 
            # which value do we use for alanine aminotransferase baseline value?
            # We use the value from 1742-6 as that is the more prevalent code
            chemo_df[col] = values if i == 0 else chemo_df[col].fillna(values)
        num_rows[obs_name] = sum(~chemo_df[col].isnull())
    missing_df = pd.DataFrame(num_rows, index=['num_rows_not_null'])
    missing_df = missing_df.T.sort_values(by='num_rows_not_null')
    
    # get changes since last baseline measurements
    result = split_and_parallelize(chemo_df, observation_change_worker)
    result = pd.DataFrame(result, columns=['index']+observation_change_cols)
    result = result.set_index('index')
    chemo_df = chemo_df.join(result)
       
    return chemo_df, mapping, missing_df

###############################################################################
# ESAS (Questionnaire) Data
###############################################################################
def filter_esas_data(esas, chemo_ikns):
    name_map = {'esas_value': 'severity', 'esas_resourcevalue': 'symptom'}
    esas = esas.rename(columns=name_map)
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
    for ikn, chemo_group in tqdm(chemo_df.groupby('ikn')):
        esas_group = esas_df[esas_df['ikn'] == ikn]
        for idx, chemo_row in chemo_group.iterrows():
            # even if survey date is like a month ago, we will be forward 
            # filling the entries to the visit date
            mask = esas_group['surveydate'] <= chemo_row['visit_date']
            esas_before_chemo = esas_group[mask]
            if esas_before_chemo.empty: 
                continue
            # Note: Sx is a medical abbreviation for symptom
            for sx, esas_sx in esas_before_chemo.groupby('symptom'):
                # last item is always the last observed response (we've already
                # sorted by date)
                esas_sx = esas_sx.iloc[-1]
                result.append([idx, sx, esas_sx['severity'], esas_sx['surveydate']])
    return result

def get_esas_responses(chemo_df, esas, processes=16):
    # organize and format columns
    esas['surveydate'] = pd.to_datetime(esas['surveydate'])
    esas['severity'] = esas['severity'].astype(int)
    esas = esas.sort_values(by='surveydate') # sort by date
    
    # filter out patients not in dataset
    filtered_chemo_df = chemo_df[chemo_df['ikn'].isin(esas['ikn'])]
    
    # get results
    result = split_and_parallelize(
        (filtered_chemo_df, esas), 
        esas_worker, 
        processes=processes
    )
    return result

def postprocess_esas_responses(esas_df):
    esas_df = esas_df.sort_values(by=['index', 'symptom', 'severity'])
    
    # assert no duplicates
    mask = esas_df[['index', 'symptom']].duplicated()
    assert(not any(mask))
    
    # make each symptom its own column (will be a mutillevel column with 
    # severity and survey date at the 1st level, and symptoms at the 2nd level
    esas_df = esas_df.pivot(index='index', columns='symptom')
    # flatten the multilevel to one level
    severity = esas_df['severity']
    survey_date = esas_df['survey_date']
    survey_date.columns += '_survey_date'
    esas_df = pd.concat([severity, survey_date], axis=1)
    
    return esas_df

###############################################################################
# ECOG (Body Perforamnce Status) / PRFS (Body Functionality Status) Data
###############################################################################
def filter_body_functionality_data(df, chemo_ikns, dataset='ecog'):
    if dataset not in {'ecog', 'prfs'}: 
        raise ValueError('dataset must be either ecog or prfs')
        
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
    for ikn, chemo_group in tqdm(chemo_df.groupby('ikn'), position=0):
        bf_group = bf_df[bf_df['ikn'] == ikn]
        for idx, chemo_row in chemo_group.iterrows():
            mask = bf_group['surveydate'] <= chemo_row['visit_date']
            bf_before_chemo = bf_group[mask]
            if bf_before_chemo.empty:
                continue
                
            # last item is always the last observed grade (we've already sorted
            # by date)
            bf_row = bf_before_chemo.iloc[-1]
            result.append((idx, bf_row[f'{dataset}_grade'], bf_row['surveydate']))
    return result

###############################################################################
# Blood Transfusions
###############################################################################
def filter_blood_transfusion_data(df, chemo_ikns, event='H'):
    col, _ = event_map[event]['date_col_name']
    # organize and format columns
    df = clean_string(df, ['ikn', 'btplate', 'btredbc']) 
    df[col] = pd.to_datetime(df[col])
    
    # filter patients not in chemo_df
    df = df[df['ikn'].isin(chemo_ikns)]
    
    # filter rows where no transfusions occured
    # btplate: blood transfusion - platelet
    # btredbc: blood transfusion - red blood cell
    btplate_mask = (df['btplate'] == 'Y') | (df['btplate'] == '1')
    btredbc_mask = (df['btredbc'] == 'Y') | (df['btredbc'] == '1')
    df = df[btplate_mask | btredbc_mask] 
    
    # get only the select columns
    df = df[[col, 'ikn', 'btplate', 'btredbc']]
    
    # sort by date
    df = df.sort_values(by=col)
    
    return df

def blood_transfusion_worker(partition, event='H'):
    date_col, _ = event_map[event]['date_col_name']
    chemo_df, bt_df = partition
    result = []
    for ikn, chemo_group in tqdm(chemo_df.groupby('ikn')):
        earliest_date = chemo_group['visit_date'] - pd.Timedelta('5 days')
        latest_date = chemo_group['next_visit_date'] + pd.Timedelta('3 days') 
        bt_group = bt_df[bt_df['ikn'] == ikn]
        for i, bt_row in bt_group.iterrows():
            admdate = bt_row[date_col]
            mask = (earliest_date <= admdate) & (latest_date >= admdate)
            tmp = chemo_group[mask]
            for chemo_idx in tmp.index:
                if not pd.isnull(bt_row['btplate']): # can only be NaN, Y, or 1
                    name = f'{event}_platelet_transfusion_date'
                    result.append((chemo_idx, str(admdate.date()), name))
                if not pd.isnull(bt_row['btredbc']): # can only be NaN, Y, or 1
                    name = f'{event}_hemoglobin_transfusion_date'
                    result.append((chemo_idx, str(admdate.date()), name))
    return result

def extract_blood_transfusion_data(chemo_df, main_dir, event='H'):
    database_name = event_map[event]['database_name']
    col, _ = event_map[event]['date_col_name']
    filepath = f'{main_dir}/data/{database_name}_transfusion.csv'
    bt_data = pd.read_csv(filepath, dtype=str)
    bt_data[col] = pd.to_datetime(bt_data[col])
    
    # filter out patients not in transfusion data
    filtered_chemo_df = chemo_df[chemo_df['ikn'].isin(bt_data['ikn'])]

    # get results
    worker = partial(blood_transfusion_worker, event=event)
    result = split_and_parallelize((filtered_chemo_df, bt_data), worker)
    
    # save results
    cols = ['chemo_idx', 'transfusion_date', 'transfusion_type']
    result = pd.DataFrame(result, columns=cols)
    result.to_csv(filepath.replace('.csv', '2.csv'), index=False)
    
def postprocess_blood_transfusion_data(chemo_df, main_dir, event='H'):
    database_name = event_map[event]['database_name']
    df = pd.read_csv(f'{main_dir}/data/{database_name}_transfusion2.csv')
    for transfusion_type, group in df.groupby('transfusion_type'):
        chemo_indices = group['chemo_idx'].values.astype(int)
        dates = group['transfusion_date'].values
        chemo_df.loc[chemo_indices, transfusion_type] = dates
    return chemo_df, df

###############################################################################
# ED/H Events
###############################################################################
def get_event_reason(df, event):
    raw_diag_codes = pd.Series(df[diag_cols].values.flatten())
    raw_diag_codes = raw_diag_codes[raw_diag_codes.notnull()]
    all_diag_codes = pd.Series(raw_diag_codes.unique())
    
    # diag_code_mapping does not contain the complete codes (e.g. R10 can refer
    # to codes R1012, R104, etc). Extract the complete codes
    complete_diag_code_mapping = {}
    for cause, diag_codes in diag_code_mapping.items():
        complete_diag_codes = []
        for code in diag_codes:
            mask = all_diag_codes.str.contains(code)
            complete_diag_codes += all_diag_codes[mask].values.tolist()
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

    # get reason for event visit (either TR: Treatment Related, FI: Fever and 
    # Infection, or GI: Gastrointestinal Toxicity)
    # NOTE: Treatment Related means all treatments which INCLUDES FI and GI
    df = get_event_reason(df, event)

    # keep only selected columns
    event_cause_cols = event_map[event]['event_cause_cols']
    df = df[event_main_cols + event_cause_cols]

    # REMOVE DUPLICATES
    df = df.drop_duplicates()
    # sort by departure date
    df = df.sort_values(by=['depart_date'])
    # for duplicates with different departure dates, keep the row with the 
    # later departure date
    df = df[~df.duplicated(subset=['ikn', 'arrival_date'], keep='last')]
    
    # sort by arrival date
    df = df.sort_values(by=['arrival_date'])
    
    return df

def event_worker(partition, event='H'):          
    chemo_df, event_df = partition
    cols = event_df.columns.drop(['ikn']).tolist()
    placeholder = ''
    result = []
    for ikn, chemo_group in tqdm(chemo_df.groupby('ikn')):
        event_group = event_df[event_df['ikn'] == ikn]
        
        # NOTE: for ED, depart date and arrival date are the same
        arrival_date = event_group['arrival_date'] 
        depart_date = event_group['depart_date']
        
        for chemo_idx, visit_date in chemo_group['visit_date'].iteritems():
            # get feature - closest event before and on chemo visit, and number
            # of prev events prior to visit
            mask = depart_dates <= visit_date
            N_prior_events = mask.sum()
            most_recent_date = depart_dates[mask].max()
            if pd.notnull(most_recent_date):
                mask = event_group['depart_date'] == most_recent_date
                most_recent_event = event_group.loc[mask, cols]
                # if there are more than one row with same depart dates, get the
                # first row (the earlier arrival date)
                event_row = most_recent_event.values.tolist()[0]
                result.append(['feature', chemo_idx, N_prior_events] + event_row)

            # get target - closest event after chemo visit
            closest_date = arrival_dates[arrival_date > visit_date].min()
            if pd.notnull(closest_date):
                mask = event_group['arrival_date'] == closest_date
                closest_event = event_group.loc[mask, cols]
                event_row = closest_event.values.tolist()[0]
                result.append(['target', chemo_idx, placeholder] + event_row)
                
    return result

def extract_event_dates(
    chemo_df, 
    event_df, 
    output_path, 
    event='H', 
    processes=16
):
    event_name = event_map[event]['event_name']
    logging.info(f"Beginning mutiprocess to extract {event_name} dates using "
                 f"{processes} processes")
    # filter out patients not in dataset
    chemo_df = chemo_df[chemo_df['ikn'].isin(event_df['ikn'])]
    event_dates = split_and_parallelize(
        (chemo_df, event_df), 
        partial(event_worker, event=event), 
        processes=processes
    )
    event_dates = np.array(event_dates).astype(str)
    
    # save results
    cols = [
        'feature_or_target', 'chemo_idx', f'num_prior_{event}s', 
        'arrival_date', 'depart_date'
    ] + event_map[event]['event_cause_cols']
    event_dates = pd.DataFrame(event_dates, columns=cols)
    event_dates.to_csv(f'{output_path}/data/{event}_dates.csv', index=False)
    
    return event_dates

###############################################################################
# Inpatients
###############################################################################
def get_inpatient_indices(partition):
    chemo_df, H_df = partition
    result = set()
    for ikn, chemo_group in tqdm(chemo_df.groupby('ikn')):
        H_group = H_df[H_df['ikn'] == ikn]
        for H_idx, H_row in H_group.iterrows():
            arrival_date = H_row['arrival_date']
            depart_date = H_row['depart_date']
            mask = chemo_group['visit_date'].between(arrival_date, depart_date)
            result.update(chemo_group[mask].index)
    return result
