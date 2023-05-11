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
import os
import shutil

import pandas as pd
import numpy as np

from src import logging
from src.config import (
    root_path, min_chemo_date, max_chemo_date, all_observations, eng_lang_codes,
    cisplatin_dins, cisplatin_cco_drug_codes, blood_cancer_code,
    world_region_country_map, world_region_language_map,
    DATE, BSA, INTENT, OBS_CODE, OBS_VALUE, OBS_DATE, OBS_RDATE, DOSE,
    systemic_cols, drug_cols, y3_cols, diag_cols, 
    observation_cols, observation_change_cols, event_main_cols,
    event_map, diag_code_mapping
)
from src.utility import (
    get_mapping_from_textfile,
    get_years_diff, 
    group_observations,
    numpy_ffill,  
    split_and_parallelize, 
)
from src.spark import start_spark, filter_lab_data

###############################################################################
# Systemic Therapy Treatment Data
###############################################################################
class Systemic:
    def load(self):
        return pd.read_parquet(f'{root_path}/data/systemic.parquet.gzip')
    
    def run(self, regimens, drug=None, filter_kwargs=None, process_kwargs=None):
        if filter_kwargs is None: filter_kwargs = {}
        if process_kwargs is None: process_kwargs = {}
        df = self.load()        
        df = filter_systemic_data(df, regimens, drug=drug, **filter_kwargs)
        df = process_systemic_data(df, drug=drug, **process_kwargs)
        return df
        
def filter_systemic_data(
    df,
    regimens,
    drug=None,
    remove_inpatients=True,
    exclude_dins=None,
    verbose=False
):
    """
    Args:
        regimens (pd.DataFrame): Table of select annotated regimens. 
            See load_reviewed_regimens/load_included_regimens
        drug (str): name of drug to keep, removing all treatment sessions that 
            did not receive it. Additional columns for drug info are added
        exclude_dins (list): sequence of DIN (Drug Identification Number) (str)
            to exclude. If None, all drugs are included
    """
    cols = systemic_cols.copy()
    
    df = clean_up_systemic(df)
    df = filter_regimens(df, regimens, verbose=verbose)
    df = filter_date(df, verbose=verbose)
    if drug is not None:
        df = filter_drug(df, keep_drug=drug, verbose=verbose)
        cols += drug_cols
    if remove_inpatients: 
        df = filter_inpatients(df, verbose=verbose)
    if exclude_dins is not None: 
        df = df[~df['din'].isin(exclude_dins)]
        
    df = df[cols]
    df = df.drop_duplicates()
    return df

def systemic_worker(partition, method='merge', min_interval=4):
    """Creates engineered features such as:
    1. days_since_starting_chemo
    2. days_since_last_chemo
    3. chemo_cycle
    4. immediate_new_regimen
    
    Handles small interval treatment sessions 2 ways
    1. Merge them together
       e.g. May 1  10gm     ->    May 1  12gm
            May 2  1gm            May 16 10gm
            May 4  1gm
            May 16 10gm
    2. Include only the first treatment session in a given week
       e.g. May 1  10gm     ->    May 1  10gm
            May 2  1gm            May 16 5gm
            May 4  1gm
            May 16 5gm
    NOTE: use 2nd method when you don't care about the dosages and just care 
    about the rough dates
    
    Args:
        method (str): either 'merge' or 'one-per-week'
        min_interval (int): the minimum number of days between treatment 
            sessions
    """
    if method not in ['merge', 'one-per-week']:
        raise ValueError('method must be either merge or one-per-week')
        
    result = []
    for ikn, df in partition.groupby('ikn'):
        df = df.sort_values(by=[DATE, 'regimen'])
        
        if method == 'one-per-week':
            # combine concurrent regimens by merging same day treatments
            df = merge_intervals(df, min_interval=0)
            # keep only the first treatment session of a given week
            keep_idxs = []
            previous_date = pd.Timestamp.min
            for i, visit_date in df[DATE].items():
                if visit_date >= previous_date + pd.Timedelta(days=7):
                    keep_idxs.append(i)
                    previous_date = visit_date
            df = df.loc[keep_idxs]
        elif method == 'merge':
            # merges small interval treatment sessions into an x day cycle
            df[f'next_{DATE}'] = df[DATE].shift(-1)
            df = merge_intervals(df, min_interval=min_interval)
            if df.empty:
                # most likely patient had consecutive 1 day interval treatment 
                # sessions that totaled less than x days OR patient has only 
                # two chemo sessions (one row) in which interval was less than 
                # x days
                continue

        start_date = df[DATE].iloc[0]
        df['days_since_starting_chemo'] = (df[DATE] - start_date).dt.days
        df['days_since_last_chemo'] = (df[DATE] - df[DATE].shift()).dt.days
        
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
        result += df.to_numpy().tolist()
        
    return result

def process_systemic_data(
    df, 
    cycle_length_map=None, 
    method='merge', 
    min_interval=4,
    drug=None,
    day_one_regimens=None,
    verbose=True
):
    """
    Args:
        cycle_length_mapping (dict): A mapping between regimens (str) and their
            shortest cycle lengths in days (int). If None, no cycle length 
            feature is created
        drug (str): name of drug administered for all treatment sessions. 
            If None, no additional processing steps are done.
        day_one_regimens (list): A list of regimens (str) in which the drug 
            is (intended to be) only administered on the first day of the 
            treatment regimen
    """
    if cycle_length_map is not None:
        df['cycle_length'] = df['regimen'].map(cycle_length_map).astype(int)
    
    # forward fill body surface area 
    # WARNING: DO NOT BACKWARD FILL! THAT IS DATA LEAKAGE!
    df[BSA] = df.groupby('ikn')[BSA].ffill()
    
    worker = partial(systemic_worker, method=method, min_interval=min_interval)
    result = split_and_parallelize(df, worker)
    extra_cols = [
        'days_since_starting_chemo', 'days_since_last_chemo', 
        'line_of_therapy', 'chemo_cycle', 'immediate_new_regimen'
    ]
    if method == 'merge': 
        extra_cols = [f'next_{DATE}'] + extra_cols
    cols = df.columns.tolist() + extra_cols
    df = pd.DataFrame(result, columns=cols)
    
    if drug is not None:
        # convert dosage to dosage per body surface area
        df[DOSE] = df[DOSE] / df[BSA].fillna(2) # mg / m^2
        
        df[f'{drug}_dosage'] = df[DOSE]
        df = df.drop(columns=drug_cols)
        
        # for regimens where drug is only provided on the first session, remove
        # the sessions thereafter
        if day_one_regimens is not None:
            mask = df['regimen'].isin(day_one_regimens)
            drop_indices = []
            for ikn, group in df[mask].groupby('ikn'):
                duplicate_mask = group['regimen'].duplicated(keep='first')
                drop_indices += group[duplicate_mask].index.tolist()
            if verbose:
                logging.info(f'Dropped {len(drop_indices)} where no {drug} was '
                             'intended to be administered after the first session')
            df = df.drop(index=drop_indices)
    
    return df

###############################################################################
# Systemic Threapy Treatment Data - Helper Functions
###############################################################################
def clean_up_systemic(df):
    df = df.rename(columns={'cco_regimen': 'regimen'})
    
    # filter out rows with no regimen data 
    mask = df['regimen'].notnull()
    df = df[mask].copy()
    
    # clean regimen name
    mask = df['regimen'].str.contains('NCT')
    df.loc[mask, 'regimen'] = 'TRIALS'
    df['regimen'] = df['regimen'] \
        .str.replace("*", "", regex=False) \
        .str.replace(" ", "", regex=False) \
        .str.lower()
    
    # clean other features
    df[BSA] = df[BSA].replace(0, np.nan).replace(-99, np.nan)
    df[INTENT] = df[INTENT].replace('U', np.nan)
    
    df[DATE] = pd.to_datetime(df[DATE])
    df = df.sort_values(by=DATE) # order the data dataframe by date
    return df

def filter_regimens(df, regimens, verbose=False):
    # keep only selected reigmens
    mask = df['regimen'].isin(regimens['regimen'])
    df = df[mask].copy()
    
    # change regimen name to the correct mapping
    mask = regimens['relabel'].notnull()
    old_name, new_name = regimens.loc[mask, ['regimen', 'relabel']].T.values
    df['regimen'] = df['regimen'].replace(old_name, new_name)
    
    return df

def filter_date(
    df, 
    min_date=min_chemo_date, 
    max_date=max_chemo_date, 
    verbose=False
):
    """Remove treatment sessions recieved before the new ALR system was set up
    """
    if verbose: 
        logging.info(f"{len(df)} chemo treatments occured between "
                     f"{df[DATE].min().date()} to {df[DATE].max().date()}")
    mask = df[DATE].between(min_date, max_date)
    if verbose: 
        logging.info(f"Removing {sum(~mask)} chemo treatments that occured "
                     f"before {min_date} and after {max_date}.")
    df = df[mask]
    return df

def filter_inpatients(df, verbose=False):
    """Remove treatment sessions recieved as an inpatient"""
    outpatient_mask = df['inpatient_flag'] == 'N'
    inpatient_mask = df['inpatient_flag'] == 'Y'
    missing_mask = df['inpatient_flag'].isnull()
    if verbose:
        logging.info(f"Removing {sum(inpatient_mask)} inpatient chemo "
                     f"treatments and {sum(missing_mask)} unknown inpatient "
                     "status chemo treatements")
    df = df[outpatient_mask]
    return df

def filter_drug(df, keep_drug='cisplatin', verbose=False):
    """Remove treatment sessions that did not receive `keep_drug`"""
    if keep_drug == 'cisplatin':
        keep_dins = cisplatin_dins
        keep_ccos = cisplatin_cco_drug_codes
    else:
        raise NotImplementedError(f'Drug {keep_drug} is not supported yet')
    
    mask = df['din'].isin(keep_dins) | df['cco_drug_code'].isin(keep_ccos)
    if verbose: 
        logging.info(f"Removing {sum(~mask)} sessions with no {keep_drug} "
                     "administrated")
    df = df[mask]
    
    # remove rows with measurement unit of g, unit, mL, nan
    mask = df['measurement_unit'].isin(['mg', 'MG'])
    if verbose: 
        logging.info(f"Removing {sum(~mask)} sessions in which measurement "
                     f"unit of {keep_drug} administered was not in mg")
    df = df[mask]
    
    # remove rows with missing dosage values
    mask = df[DOSE].isnull()
    if verbose:
        logging.info(f"Removing {sum(mask)} sessions with missing dosages")
    df = df[~mask].copy()
    
    df['drug'] = keep_drug
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

def merge_intervals(df, min_interval=4, merge_cols=None):
    """Merges treatment sessions with less than x days apart into one
    
    E.g.
    regimen  visit_date  interval    
    A        2018-02-02  1 days   
    B        2018-02-03  2 days
    A        2018-02-05  11 days
    D        2018-02-16  NaN
    
    regimen  visit_date  interval    
    ----- MERGED AND REMOVED ----
    A+B      2018-02-02  3 days
    A        2018-02-05  11 days
    D        2018-02-16  NaN
    
    regimen  visit_date  interval    
    ----- MERGED AND REMOVED ----
    ----- MERGED AND REMOVED ----
    A+B      2018-02-02  14 days
    D        2018-02-16  NaN
    """
    if merge_cols is None: merge_cols = []
    
    df = df.reset_index(drop=True)
    df['interval'] = (df[f'next_{DATE}'] - df[DATE]).dt.days
    merge_cols.append('interval')
    
    # WARNING: Assumes df only contains one drug type if merging dosages
    if DOSE in merge_cols:
        if df['drug'].nunique() > 1: 
            msg = 'Merging dosage for more than 1 drug is not supported yet'
            raise NotImplementedError(msg)
            
    # to keep regimen format consistent during merging, sort all regimen sets
    format_regimen = lambda regimen: '+'.join(sorted(set(regimen.split('+'))))
    df['regimen'] = df['regimen'].apply(format_regimen)
    
    remove_idxs = []
    for cur_idx, interval in df['interval'].items():
        if interval >= min_interval or np.isnan(interval): 
            continue
        
        # Merge current treatment session with the next treatment session
        next_idx = cur_idx + 1
        
        # set next treatment date to current treatment date
        df.loc[next_idx, DATE] = df.loc[cur_idx, DATE]
        
        # add the current and next values of merge_cols
        df.loc[next_idx, merge_cols] += df.loc[cur_idx, merge_cols]
        
        # merge different regimens and their related info
        # TODO: merge different intents
        cur_regimen = df.loc[cur_idx, 'regimen']
        next_regimen = df.loc[next_idx, 'regimen']
        if next_regimen != cur_regimen:
            # two different regimens or sets of regimens are taken within a 
            # short span of time, combine the regimens together
            regimen = f'{next_regimen}+{cur_regimen}'
            df.loc[next_idx, 'regimen'] = format_regimen(regimen)
                
            # take the minimum cycle length, if provided
            if 'cycle_length' in df.columns:
                df.loc[next_idx, 'cycle_length'] = min(
                    df.loc[cur_idx, 'cycle_length'],
                    df.loc[next_idx, 'cycle_length'],
                )
        
        remove_idxs.append(cur_idx)  
        
    df = df.drop(index=remove_idxs, columns=['interval'])
    return df

###############################################################################
# Demographic Data - includes cancer diagnosis, comorbidity
###############################################################################
class Demographic:
    def load(self):
        read = lambda x: pd.read_parquet(f'{root_path}/data/{x}.parquet.gzip')
        data = {}
        data['basic'] = read('y3') # basic patient and disease info
        data['income'] = read('income')
        data['rural'] = read('rural')
        data['immigration'] = read('immigration')
        data['comorbidity'] = read('comorbidity')
        return data
    
    def run(self, verbose=True, **kwargs):
        data = self.load()
        # Basic patient and disease info
        df = data['basic']
        if verbose: _get_n_patients(df, 'y3')
        df = filter_basic_demographic_data(df, verbose=verbose, **kwargs)
        
        # Income
        income = data['income']
        if verbose: _get_n_patients(income, 'income')
        income = income.astype({'incquint': float})
        income = income.rename(columns={'incquint': 'neighborhood_income_quintile'})
        df = pd.merge(df, income, on='ikn', how='left')
        
        # Area of Residence
        rural = data['rural']
        if verbose: _get_n_patients(rural, 'area')
        rural['rural'] = rural['rural'].replace({'N': False, 'Y': True})
        df = pd.merge(df, rural, on='ikn', how='left')

        # Immigration
        immig = data['immigration']
        if verbose: _get_n_patients(immig, 'immigration')
        country_code_map = get_mapping_from_textfile(f'{root_path}/data/country_codes.txt')
        immig = process_immigration_data(immig, rural, country_code_map)
        df = pd.merge(df, immig, on='ikn', how='left')
        
        # Comorbidity
        comorbidity = data['comorbidity']
        if verbose: _get_n_patients(comorbidity, 'comorbidity')
        df = pd.merge(df, comorbidity, on='ikn', how='left')

        return df
    
def filter_basic_demographic_data(df, exclude_blood_cancer=True, verbose=False):
    df = df[y3_cols]
    df = clean_basic_demographic_data(df)
    
    if exclude_blood_cancer:
        mask = df['cancer_topog_cd'].isin(blood_cancer_code)
        if verbose:
            N = df.loc[mask, 'ikn'].nunique()
            logging.info(f"Removing {N} patients with blood cancer")
        df = df[~mask]
        
    # remove morphology codes >= 959
    mask = df['cancer_morph_cd'] >= '959'
    if verbose:
        N = df.loc[mask, 'ikn'].nunique()
        removed_cancers = df.loc[mask, 'cancer_morph_cd'].unique()
        logging.info(f"Removing {N} patients and cancer types {removed_cancers}")
    df = df[~mask]
    
    return df

def clean_basic_demographic_data(df):
    df = df.rename(columns={
        'dthdate': 'death_date', 'bdate': 'birth_date', 
        'curr_morph_cd': 'cancer_morph_cd', 'curr_topog_cd': 'cancer_topog_cd'
    })
    df['birth_date'] = pd.to_datetime(df['birth_date'])
    df['death_date'] = pd.to_datetime(df['death_date'])
    for col in ['cancer_morph_cd', 'cancer_topog_cd']: 
        df[col] = df[col].replace('*U*', np.nan)
        # only keep first three characters - the rest are for specifics
        # e.g. C50 Breast: C501 Central portion, C504 Upper-outer quadrant, etc
        df[col] = df[col].str[:3]
    return df

def process_immigration_data(df, *args):
    df['speaks_english'] = df['official_language'].isin(eng_lang_codes)
    df['recent_immigrant'] = True
    df['landing_date'] = pd.to_datetime(df['landing_date'])
    df = get_world_region_of_birth(df, *args)
    return df

def get_world_region_of_birth(df, rural, ctry_code_map):
    """Get urban patient's birthplace by world region
    """
    # Only use data from urban area as rural area is not too reliable
    rural_mask = df['ikn'].map(dict(rural[['ikn', 'rural']].to_numpy()))
    rural_mask = rural_mask.fillna(True) # don't include data with missing area
    mask = ~rural_mask
    
    ctry_col, regi_col = 'country_birth', 'world_region_of_birth'
    reformat = lambda mapping: {v: k for k, vs in mapping.items() for v in vs}
    # map the country code to country name
    df[ctry_col] = df[ctry_col].map(reformat(ctry_code_map))
    # map the country name to world region
    df.loc[mask, regi_col] = df.loc[mask, ctry_col].map(reformat(world_region_country_map))
    df.loc[mask, regi_col] = df.loc[mask, regi_col].fillna('Other')
    return df

def combine_demographic_data(systemic, demographic, verbose=True):
    """Combine systemic therapy treatment data and demographic data
    """    
    df = pd.merge(systemic, demographic, on='ikn', how='inner')
    df['age'] = get_years_diff(df, DATE, 'birth_date')
    
    # do not include patients under 18
    mask = df['age'] < 18
    if verbose:
        N = df.loc[mask, 'ikn'].nunique()
        logging.info(f"Removing {N} patients under 18")
    df = df[~mask]
    
    # get years since immigrating to Canada 
    col = 'years_since_immigration'
    df[col] = get_years_diff(df, DATE, 'landing_date')
    # for non-immigrants / long-term residents, we will say they "immigrated"
    # the year they were born
    df[col] = df[col].fillna(df['age'])
    
    # clean up other features
    df['speaks_english'] = df['speaks_english'].fillna(True)
    df['recent_immigrant'] = df['recent_immigrant'].fillna(False)
    df['world_region_of_birth'] = df['world_region_of_birth'].fillna('Unknown')
    # assume missing area as urban since nans are negligible (~0.73% prevalence)
    assert df['rural'].isnull().mean() < 0.008
    df['rural'] = df['rural'].fillna(False)

    return df

###############################################################################
# Laboratory Test Data
###############################################################################
class Laboratory:
    def __init__(self, output_path, processes=32):
        self.output_path = output_path
        self.processes = processes
        
    def preprocess(self, chemo_ikns, **kwargs):
        spark = start_spark()
        lab = spark.read.parquet(f'{root_path}/data/olis', header=True)
        lab = filter_lab_data(lab, chemo_ikns, **kwargs)
        # Need to convert to pandas dataframe for next step. Easier to write to
        # disk and read using pd.read_parquet then directly converting from 
        # spark dataframe to panda dataframe
        if os.path.exists(f'{self.output_path}/olis'):
            shutil.rmtree(f'{self.output_path}/olis')
        lab.write.save(f'{self.output_path}/olis', compression='gzip')
        # TODO: Prevent annoying "IOSteam.flush timed out" messages. Occurs when 
        # multiprocessing and kernel is not restarted afterwards
        spark.stop()
        
    def run(self, df, obs_codes=None, get_closest_obs=False, **kwargs):
        lab = pd.read_parquet(f'{self.output_path}/olis')
        lab[OBS_DATE] = pd.to_datetime(lab[OBS_DATE])
        if obs_codes is not None:
            lab = lab[lab[OBS_CODE].isin(obs_codes)]
            
        worker = closest_measurement_worker if get_closest_obs else lab_worker
        worker = partial(worker, **kwargs)
        mask = df['ikn'].isin(lab['ikn']) 
        result = split_and_parallelize(
            (df[mask], lab), worker, processes=self.processes
        )
        
        cols = ['obs_code', 'chemo_idx', 'days_after_chemo', 'obs_value']
        lab = pd.DataFrame(result, columns=cols)
        return lab

def lab_worker(partition, time_window=(-5,0)):
    """Extract the laboratory test values measured within a time window of each
    treatment session.
    
    Args:
        time_window (tuple): the number of days (int) representing the start
            and end of the time window centered on the treatment date
    """
    chemo_df, lab_df = partition
    lower_limit, upper_limit = time_window
    lower_limit = pd.Timedelta(days=lower_limit)
    upper_limit = pd.Timedelta(days=upper_limit)
    keep_cols = [OBS_CODE, 'chemo_idx', 'days_after_chemo', 'value']
    
    result = []
    for ikn, chemo_group in chemo_df.groupby('ikn'):
        lab_group = lab_df.query('ikn == @ikn')
        for chemo_idx, chemo_row in chemo_group.iterrows():
            visit_date = chemo_row[DATE]
            # See if there any observation data within the dates of interest.
            # Set latest date to x days after treatment session or the day 
            # of next treatment session, whichever comes first
            earliest_date = visit_date + lower_limit
            latest_date = visit_date + upper_limit
            if f'next_{DATE}' in chemo_row:
                latest_date = min(latest_date, chemo_row[f'next_{DATE}'])
            mask = lab_group[OBS_DATE].between(earliest_date, latest_date)
            tmp = lab_group[mask].copy()
            tmp['days_after_chemo'] = (tmp[OBS_DATE] - visit_date).dt.days
            tmp['chemo_idx'] = chemo_idx
            result.extend(tmp[keep_cols].values.tolist()) 
    return result

def closest_measurement_worker(partition, days_after=90):
    """Finds the closest measurement x number of days after treatment to 2 
    years after treatment. 
    
    Partition must contain only one type of observation (e.g.creatinine only)
    """
    chemo_df, obs_df = partition
    obs_df = obs_df.sort_values(by=OBS_DATE) # might be redundant but just in case
    if obs_df[OBS_CODE].nunique() > 1:
        msg = ('Multiple observation codes for closest_measurement_worker '
               'supported yet')
        raise NotImplementedError(msg)
    
    results = []
    for ikn, chemo_group in chemo_df.groupby('ikn'):
        obs_group = obs_df.query('ikn == @ikn')
        for chemo_idx, visit_date in chemo_group[DATE].items():
            earliest_date = visit_date + pd.Timedelta(days=days_after)
            latest_date = earliest_date + pd.Timedelta('730 days')
            mask = obs_group[OBS_DATE].between(earliest_date, latest_date)
            if not mask.any(): continue
            tmp = obs_group[mask].iloc[0]
            days_after_chemo = (tmp[OBS_DATE] - visit_date).days
            result = (tmp[OBS_CODE], chemo_idx, days_after_chemo, tmp['value'])
            results.append(result)
    return results

def observation_change_worker(partition):
    """Finds the change since last measurement observation"""
    result = []
    for ikn, group in partition.groupby('ikn'):
        change = group[observation_cols] - group[observation_cols].shift()
        result.append(change.reset_index().to_numpy())
    return np.concatenate(result)
                          
def combine_lab_data(chemo_df, lab_df):
    """Combine systemic therapy treatment data and laboratory data
    """
    # exclude obs codes not in selected observations
    mask = lab_df['obs_code'].isin(all_observations)
    lab_df = lab_df[mask]
    
    # convert lab data format
    lab_map = process_lab_data(lab_df, chemo_df)
    
    # group together obs codes with same obs name
    freq_map = lab_df['obs_code'].value_counts()
    grouped_obs = group_observations(all_observations, freq_map)
    
    # get baseline (pre-treatment session) observation values and combine it to
    # chemo_df
    baseline_window = range(-5,1)
    for obs_name, obs_codes in grouped_obs.items():
        col = f'baseline_{obs_name}_value'
        # NOTE: observation codes are ordered by their prevalence
        for i, obs_code in enumerate(obs_codes):
            # forward fill observation counts from day -5 to day 0
            values = numpy_ffill(lab_map[obs_code][baseline_window])
            values = pd.Series(values, index=chemo_df.index)
            # When there are conflicting observations values among the grouped
            # observations, take the value of the most frequent observation code
            # E.g. alanine aminotransferase includes observation codes 
            # 1742-6, 1744-2, 1743-4, ordered by their prevalence. If we see
            # two different baseline observation value from 1742-6 and 1744-2, 
            # which value do we use for alanine aminotransferase baseline value?
            # We use the value from 1742-6 as that is the more prevalent code
            chemo_df[col] = values if i == 0 else chemo_df[col].fillna(values)
    
    # get changes since last baseline measurements
    result = split_and_parallelize(chemo_df, observation_change_worker)
    result = pd.DataFrame(result, columns=['index']+observation_change_cols)
    result = result.set_index('index')
    chemo_df = chemo_df.join(result)
    
    # get number of non-missing rows for baseline observation values
    count = {name: sum(chemo_df[f'baseline_{name}_value'].notnull()) 
             for name in grouped_obs}
    count = pd.DataFrame(count, index=['Number of non-missing rows'])
    count = count.T.sort_values(by='Number of non-missing rows')
       
    return chemo_df, lab_map, count

def process_lab_data(lab_df, chemo_df):
    """Convert lab data in long format to a mapping of observation codes and
    their measurement data in wide format
    
    Reference: https://stefvanbuuren.name/fimd/sec-longandwide.html
    """
    # get time window
    min_day = lab_df['days_after_chemo'].min()
    max_day = lab_df['days_after_chemo'].max()
    time_window = range(min_day, max_day+1)
    
    # for each observation code, get all the observation values taken within 
    # the time window of each treatment session in wide format (each day in the
    # time window is a column)
    lab_map = {obs_code: pd.DataFrame(index=chemo_df.index, columns=time_window)
               for obs_code in all_observations}
    for obs_code, obs_group in lab_df.groupby('obs_code'):
        for day, group in obs_group.groupby('days_after_chemo'):
            idxs = group['chemo_idx'].to_numpy().astype(int)
            obs_values = group['obs_value'].to_numpy().astype(float)
            lab_map[obs_code].loc[idxs, int(day)] = obs_values
    return lab_map

###############################################################################
# Symptoms Data - includes functional status
###############################################################################
class Symptoms:
    def __init__(self, processes=32):
        self.processes = processes
        
    def load(self):
        read = lambda x: pd.read_parquet(f'{root_path}/data/{x}.parquet.gzip')
        data = {}
        data['ESAS'] = read('esas')
        data['ECOG'] = read('ecog')
        data['PRFS'] = read('prfs')
        return data
    
    def combine(self, data, chemo_ikns, verbose=True):
        # ESAS - Edmonton Symptom Assessment System
        esas = data['ESAS']
        if verbose: _get_n_patients(esas, 'ESAS')
        esas = filter_symptom_data(esas, chemo_ikns, name='esas')
        
        # ECOG - Eastern Cooperative Oncology Group performance status
        ecog = data['ECOG']
        if verbose: _get_n_patients(ecog, 'ECOG')
        ecog = filter_symptom_data(ecog, chemo_ikns, name='ecog')
        
        # PRFS - Patient-Reported Functional Status
        prfs = data['PRFS']
        if verbose: _get_n_patients(prfs, 'PRFS')
        prfs = filter_symptom_data(prfs, chemo_ikns, name='prfs')
        
        symp = pd.concat([esas, ecog, prfs])
        symp = symp.drop_duplicates()
        symp = symp.sort_values(by='surveydate')
        return symp
    
    def run(self, df, verbose=True, **kwargs):
        data = self.load()
        symp = self.combine(data, df['ikn'], verbose=verbose)
        mask = df['ikn'].isin(symp['ikn'])
        worker = partial(symptom_worker, **kwargs)
        result = split_and_parallelize(
            (df[mask], symp), worker, processes=self.processes
        )
        cols = ['chemo_idx', 'symptom', 'severity', 'survey_date']
        symp = pd.DataFrame(result, columns=cols)
        if verbose: logging.info(f"\n{symp['symptom'].value_counts()}")

        return symp

def filter_symptom_data(df, chemo_ikns, name='esas'):
    df = clean_symptom_data(df, name=name)
    
    # filter out patients not in chemo_df
    mask = df['ikn'].isin(chemo_ikns)
    df = df[mask]
    
    # remove duplicate rows
    df = df.drop_duplicates()
    
    return df

def clean_symptom_data(df, name='esas'):
    col_map = {f'{name}_value': 'severity', f'{name}_resourcevalue': 'symptom'}
    df = df.rename(columns=col_map)
    df['surveydate'] = pd.to_datetime(df['surveydate'])
    df['severity'] = df['severity'].astype(int)
    df['symptom'] = df['symptom'].replace('Activities & Function:', 'PRFS Grade')
    df['symptom'] = df['symptom'].str.replace(' ', '_').str.lower()
    return df

def symptom_worker(partition, days_ago=30):
    chemo_df, symps_df = partition
    result = []
    for ikn, chemo_group in chemo_df.groupby('ikn'):
        symps_group = symps_df.query('ikn == @ikn')
        for idx, visit_date in chemo_group[DATE].items():
            # find closest survey date in the past x days (we are forward 
            # filling the entries to the treatment date, up to x days ago)
            earliest_date = visit_date - pd.Timedelta(days=days_ago)
            mask = symps_group['surveydate'].between(earliest_date, visit_date)
            if not mask.any(): 
                continue
            # Note: Sx is a medical abbreviation for symptom
            for sx, sx_group in symps_group[mask].groupby('symptom'):
                # last item is always the last observed response (we've already
                # sorted by date)
                row = sx_group.iloc[-1]
                result.append([idx, sx, row['severity'], row['surveydate']])
    return result

def combine_symptom_data(chemo_df, symp_df):
    """Combine systemic therapy treatment data and symptom data
    """
    symp_df = symp_df.sort_values(by=['chemo_idx', 'symptom', 'severity'])
    # assert no duplicates
    mask = symp_df[['chemo_idx', 'symptom']].duplicated()
    assert(not any(mask))
    
    # make each symptom its own column (will be a mutillevel column with 
    # severity and survey date at the 1st level, and symptoms at the 2nd level
    symp_df = symp_df.pivot(index='chemo_idx', columns='symptom')
    # flatten the multilevel to one level
    severity = symp_df['severity']
    survey_date = symp_df['survey_date']
    survey_date.columns += '_survey_date'
    symp_df = pd.concat([severity, survey_date], axis=1)
    
    chemo_df = chemo_df.join(symp_df, how='left')
    return chemo_df

###############################################################################
# Acute Care Use Data - emergency department visits and hospitalizations
###############################################################################
class AcuteCareUse:
    def __init__(self, output_path, processes=32):
        self.output_path = output_path
        self.processes = processes
        
    def load(self):
        read = lambda x: pd.read_parquet(f'{root_path}/data/{x}.parquet.gzip')
        data = {'ED': read('nacrs'), 'H': read('dad')}
        return data
    
    def run(self, df, verbose=True, **kwargs):
        data = self.load()
        
        for event in ['ED', 'H']:
            event_df = data[event]
            event_name = event_map[event]['event_name']
            if verbose: _get_n_patients(event_df, event_name)
            event_df = filter_event_data(event_df, df['ikn'], event=event)
            result = process_event_data(df, event_df, event=event, **kwargs)
            filepath = f'{self.output_path}/{event}.parquet.gzip'
            result.to_parquet(filepath, compression='gzip', index=False)
            
            if verbose:
                cols = event_map[event]['event_cause_cols']
                logging.info(f"\n{result[cols].apply(pd.value_counts)}")
        
        # inpatients
        mask = df['ikn'].isin(event_df['ikn'])
        idxs = split_and_parallelize(
            (df[mask], event_df), get_inpatient_idxs, processes=self.processes
        )
        if verbose:
            N = df.loc[idxs, 'ikn'].nunique()
            logging.info('Number of patients that received treatment while '
                         f'hospitalized: {N}')
        np.save(f'{self.output_path}/inpatient_idxs.npy', idxs)
        
def filter_event_data(df, chemo_ikns, event='H', remove_ED_causing_H=True):
    # remove rows with null date values
    arrival_col = event_map[event]['arrival_col']
    depart_col = event_map[event]['depart_col']
    df['arrival_date'] = pd.to_datetime(df[arrival_col])
    df['depart_date'] = pd.to_datetime(df[depart_col])
    df = df[df['arrival_date'].notnull()]
    df = df[df['depart_date'].notnull()]
    
    # remove ED visits that resulted in hospitalizations
    if remove_ED_causing_H and event == 'ED':
        df = df[~df['to_type'].isin(['I', 'P'])]

    # filter away patients not in chemo dataframe
    df = df[df['ikn'].isin(chemo_ikns)]

    # remove duplicates
    df = df.drop_duplicates()
    df = df.sort_values(by=['depart_date'])
    # for duplicates with different departure dates, keep the row with the 
    # later departure date
    mask = df.duplicated(subset=['ikn', 'arrival_date'], keep='last')
    df = df[~mask]
    
    df = df.sort_values(by=['arrival_date'])
    return df

def process_event_data(
    chemo_df, 
    event_df, 
    event='H',
    years_ago=5,
    processes=16,
):
    # get reason for event visit (either TR: Treatment Related, FI: Fever and 
    # Infection, or GI: Gastrointestinal Toxicity)
    # NOTE: Treatment Related means all treatments which INCLUDES FI and GI
    event_df = get_event_reason(event_df, event)
    cause_cols = event_map[event]['event_cause_cols']
    
    mask = chemo_df['ikn'].isin(event_df['ikn'])
    result = split_and_parallelize(
        (chemo_df[mask], event_df[event_main_cols+cause_cols]), 
        partial(event_worker, event=event, years_ago=years_ago), 
        processes=processes
    )
    n_prior_col = f'num_prior_{event}s_within_{years_ago}_years'
    cols = ['feat_or_targ', 'chemo_idx', n_prior_col, 'date'] + cause_cols
    result = pd.DataFrame(result, columns=cols)
    
    return result

def event_worker(partition, event='H', years_ago=5):          
    chemo_df, event_df = partition
    cols = event_df.columns.drop(['ikn', 'depart_date'])
    result = []
    for ikn, chemo_group in chemo_df.groupby('ikn'):
        event_group = event_df.query('ikn == @ikn')
        arrival_dates = event_group['arrival_date']
        
        for chemo_idx, visit_date in chemo_group[DATE].items():
            # get feature
            # 1. closest event prior to visit date
            # 2. number of events in the past x years 
            earliest_date = visit_date - pd.Timedelta(days=years_ago*365)
            mask = arrival_dates.between(earliest_date, visit_date)
            if mask.any():
                N_prior_events = mask.sum()
                # assert(sum(arrival_dates == arrival_dates[mask].max()) == 1)
                row = event_group.loc[mask, cols].iloc[-1].tolist()
                result.append(['feature', chemo_idx, N_prior_events] + row)

            # get target - closest event from treatment visit
            # NOTE: if event occured on treatment date, most likely the event 
            # occured right after patient received treatment. We will deal with
            # it in downstream pipeline
            mask = arrival_dates >= visit_date
            if mask.any():
                # assert(sum(arrival_dates == arrival_dates[mask].min()) == 1)
                row = event_group.loc[mask, cols].iloc[0].tolist()
                result.append(['target', chemo_idx, np.nan] + row)
                
    return result

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

def get_inpatient_idxs(partition):
    chemo_df, H_df = partition
    result = set()
    for ikn, chemo_group in chemo_df.groupby('ikn'):
        H_group = H_df.query('ikn == @ikn')
        for H_idx, H_row in H_group.iterrows():
            arrival_date = H_row['arrival_date']
            depart_date = H_row['depart_date']
            mask = chemo_group[DATE].between(arrival_date, depart_date)
            result.update(chemo_group[mask].index)
    return result

###############################################################################
# Blood Transfusions Data - during acute care use
###############################################################################
class BloodTransfusion:
    def __init__(self, output_path):
        self.output_path = output_path
        
    def load(self):
        read = lambda x: pd.read_parquet(f'{root_path}/data/{x}.parquet.gzip')
        data = {'ED': read('nacrs_transfusion'), 'H': read('dad_transfusion')}
        return data
    
    def run(self, df, verbose=True, **kwargs):
        data = self.load()
        
        result = []
        for event in ['ED', 'H']:
            bt = data[event]
            event_name = f"{event_map[event]['event_name']} blood transfusion"
            if verbose: _get_n_patients(bt, event_name)
            bt = filter_blood_transfusion_data(bt, df['ikn'], event=event)
            bt = process_blood_transfusion_data(df, bt, event=event, **kwargs)
            result.append(bt)
        result = pd.concat(result)
        
        filepath = f'{self.output_path}/blood_transfusion.parquet.gzip'
        result.to_parquet(filepath, compression='gzip', index=False)
        
        if verbose:
            cols = ['feat_or_targ', 'event', 'type']
            for col in cols: logging.info(result[col].value_counts().to_dict())
            
def filter_blood_transfusion_data(df, chemo_ikns, event='H'):
    # filter patients not in chemo_df
    mask = df['ikn'].isin(chemo_ikns)
    df = df[mask].copy()
    
    # filter rows where no transfusions occured
    # btplate: blood transfusion - platelet
    # btredbc: blood transfusion - red blood cell
    df['btplate'] = df['btplate'].isin(['Y', '1'])
    df['btredbc'] = df['btredbc'].isin(['Y', '1'])
    df = df[df['btplate'] | df['btredbc']] 
    
    # get only the select columns
    arrival_col = event_map[event]['arrival_col']
    df = df[[arrival_col, 'ikn', 'btplate', 'btredbc']]
    
    df = df.drop_duplicates()
    df = df.sort_values(by=arrival_col)
    return df

def process_blood_transfusion_data(chemo_df, bt_df, event='H', processes=8):
    bt_map = {'platelet': 'btplate', 'hemoglobin': 'btredbc'}
    output = []
    for bt_type, bt_col in bt_map.items():
        bt_mask = bt_df[bt_col]
        chemo_mask = chemo_df['ikn'].isin(bt_df.loc[bt_mask, 'ikn'])
        worker = partial(blood_transfusion_worker, event=event)
        result = split_and_parallelize(
            (chemo_df[chemo_mask], bt_df[bt_mask]), worker, processes=processes
        )
        cols = ['feat_or_targ', 'chemo_idx', 'transfusion_date']
        result = pd.DataFrame(result, columns=cols)
        result['event'] = event
        result['type'] = bt_type
        output.append(result)
    return pd.concat(output)

def blood_transfusion_worker(partition, event='H', days_after=4):
    chemo_df, bt_df = partition
    arrival_col = event_map[event]['arrival_col']

    # treatment data has more than 4 times as many rows per patient than bt data
    # loop thru rows of bt data to minimize unecessary computation
    func = lambda df: df.groupby('ikn').apply(len).mean()
    assert func(chemo_df) > func(bt_df) * 4

    feats, targs = {}, {}
    for ikn, chemo_group in chemo_df.groupby('ikn'):
        bt_group = bt_df.query('ikn == @ikn')
        for _, bt_date in bt_group[arrival_col].items():
            # get feature - closest transfusion prior to visit date
            mask = bt_date <= chemo_group[DATE]
            if mask.any(): 
                for chemo_idx in chemo_group.index[mask]:
                    feats[chemo_idx] = max(feats.get(chemo_idx, bt_date), bt_date)
            # get target - closest tranfusion x days after visit date
            mask = bt_date >= chemo_group[DATE] + pd.Timedelta(days=days_after)
            if mask.any(): 
                for chemo_idx in chemo_group.index[mask]:
                    targs[chemo_idx] = min(targs.get(chemo_idx, bt_date), bt_date)
    result = [('feature', *item) for item in feats.items()] + \
             [('target', *item) for item in targs.items()]
    return result

###############################################################################
# Ontario Drug Benefit Data - growth factor
###############################################################################
def odb_worker(partition):
    """Get the treatment indices in which patients recieved growth factor 
    within 5 days before to 5 days after treatment visit.
    
    This does not affect label leakage, as provision of growth factor is 
    planned beforehand.
    """
    chemo_df, odb_df = partition
    result = set()
    for ikn, chemo_group in chemo_df.groupby('ikn'):
        odb_group = odb_df.query('ikn == @ikn')
        earliest_date = chemo_group[DATE] - pd.Timedelta(days=5)
        latest_date = chemo_group[DATE] + pd.Timedelta(days=5)
        for _, servdate in odb_group['servdate'].items():
            mask = (earliest_date <= servdate) & (servdate <= latest_date)
            result.update(chemo_group.index[mask])
    return result

def process_odb_data(df, verbose=True):
    odb = pd.read_parquet(f'{root_path}/data/odb.parquet.gzip')
    odb = odb.sort_values(by='servdate')
    if verbose:
        _get_n_patients(odb, 'Ontario Drug Benefit')
        logging.info(f"\n{odb['din'].value_counts()}")
    mask = df['ikn'].isin(odb['ikn'])
    idxs = split_and_parallelize((df[mask], odb), odb_worker)
    df['GF_given'] = False
    df.loc[idxs, 'GF_given'] = True
    return df

###############################################################################
# Dialysis Data
###############################################################################
def dialysis_worker(partition, days_after=90):
    """Determine if patient recieved dialysis x to x+180 days (e.g. 3-6 months)
    after treatment
    """
    chemo_df, dialysis = partition
    
    # dialysis data has more than 4 times as many rows per patient than 
    # treatment data, loop thru rows of treatment data to minimize unecessary 
    # computation
    func = lambda df: df.groupby('ikn').apply(len).mean()
    assert func(dialysis) > func(chemo_df) * 4
    
    result = []
    for ikn, chemo_group in chemo_df.groupby('ikn'):
        dialysis_group = dialysis.query('ikn == @ikn')            
        for chemo_idx, visit_date in chemo_group[DATE].items():
            earliest_date = visit_date + pd.Timedelta(days=days_after)
            latest_date = earliest_date + pd.Timedelta('180 days')
            mask = dialysis_group['servdate'].between(earliest_date, latest_date)
            if mask.any(): result.append(chemo_idx)
    return result

def process_dialysis_data(df, verbose=True, **kwargs):
    """Determine if patients received dialysis within a time window"""
    dialysis = pd.read_parquet(f'{root_path}/data/dialysis_ohip.parquet.gzip')
    dialysis = dialysis.sort_values(by='servdate')
    mask = df['ikn'].isin(dialysis['ikn'])
    worker = partial(dialysis_worker, **kwargs)
    idxs = split_and_parallelize((df[mask], dialysis), worker, processes=2)
    df['dialysis'] = False
    df.loc[idxs, 'dialysis'] = True
    if verbose: 
        _get_n_patients(dialysis, 'Dialysis')
        logging.info(f"\n{df['dialysis'].value_counts()}")
    return df

###############################################################################
# OHIP Data - includes palliative care consultation service (PCCS) billing
###############################################################################
def filter_ohip_data(ohip, billing_codes=None):
    """
    Args:
        billing_codes (list): A sequeunce of billing codes (str) associated 
            with specific services we want to keep
    """
    ohip['servdate'] = pd.to_datetime(ohip['servdate'])
    if billing_codes is not None:
        mask = ohip['feecode'].isin(billing_codes)
        ohip = ohip[mask]
    ohip = ohip.query('feecode != "E083"') # remove billing code E083
    ohip = ohip.drop_duplicates()
    ohip = ohip.sort_values(by='servdate')
    return ohip

###############################################################################
# Helper Functions
###############################################################################
def _get_n_patients(df, name):
    N = df['ikn'].nunique()
    logging.info(f"Number of patients in {name} data = {N}")
