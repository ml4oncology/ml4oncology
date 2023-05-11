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
"""
Script run the entire data pipeline
"""
from functools import partial
from tqdm import tqdm
import argparse
import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np

from src.config import neutrophil_dins, all_observations, event_map
from src.preprocess import (
    Systemic, Demographic, Laboratory, Symptoms, AcuteCareUse, BloodTransfusion,
    combine_demographic_data, combine_lab_data, combine_symptom_data,
    filter_ohip_data, process_odb_data, process_dialysis_data,
)
from src.utility import (
    load_included_regimens,
    split_and_parallelize, 
    group_observations,
)

PROCESSES = 32
DROP_COLS = [
    'inpatient_flag', 'ethnic', 'country_birth', 'official_language', 
    'nat_language'
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='./')
    parser.add_argument('--regimen-criteria', type=str, choices=['cytotoxic', 'cisplatin_containing'])
    parser.add_argument('--include-ohip', action='store_true')
    parser.add_argument('--include-odb', action='store_true')
    parser.add_argument('--include-dialysis', action='store_true')
    parser.add_argument('--include-acute-care-use', action='store_true')
    parser.add_argument('--include-blood-transfusion', action='store_true')
    parser.add_argument('--remove-inpatients', action='store_true')
    parser.add_argument('--remove-blood-cancer-patients', action='store_true')
    parser.add_argument('--exclude-neutrophil-drugs', action='store_true')
    msg = ('name of drug to keep, removing all treatment sessions that did not receive it. e.g. --drug cisplatin')
    parser.add_argument('--drug', type=str, help=msg)
    msg = ('how to handle small interval treatment sessions. '
           '`merge` merges them together. '
           '`one-per-week` takes the first session of a given week')
    parser.add_argument('--systemic-method', type=str, default='merge', choices=['merge', 'one-per-week'], help=msg)
    msg = ('the lower and upper limit of the time window centered on the treatment date '
           'in which laboratory test values are extracted.')
    parser.add_argument('--time-window', type=int, default=[-5,0], nargs=2, metavar=('start', 'end'), help=msg)
    msg = ('special observations (lab tests) in which all their measurements are saved in separate files '
           'for later analysis. e.g. --special-obs neutrophil platelet hemoglobin')
    parser.add_argument('--special-obs', type=str, nargs='*', help=msg)
    args = parser.parse_args()
    return args
                                
def main():
    args = parse_args()
    output_path = args.output_path
    regimen_criteria = args.regimen_criteria
    include_ohip = args.include_ohip
    include_odb = args.include_odb
    include_dialysis = args.include_dialysis
    include_acute_care_use = args.include_acute_care_use
    include_blood_transfusion = args.include_blood_transfusion
    remove_inpatients = args.remove_inpatients
    remove_blood_cancer_patients = args.remove_blood_cancer_patients
    exclude_dins = None
    if args.exclude_neutrophil_drugs:
        exclude_dins = neutrophil_dins
    drug = args.drug
    systemic_method = args.systemic_method
    time_window = args.time_window
    
    
    regimens = load_included_regimens(criteria=regimen_criteria)
    day_one_regimens = None
    if regimen_criteria == 'cisplatin_containing':
        day_one_regimens = regimens.loc[regimens['notes'] == 'cisplatin only on day 1', 'regimen']
    cycle_length_map = dict(regimens[['regimen', 'shortest_cycle_length']].to_numpy())
    cycle_length_map['Other'] = 7.0 # make rare regimen cycle lengths default to 7
    
    # Systemic Therapy Treatment Data
    syst = Systemic()
    df = syst.run(
        regimens, 
        drug=drug,
        filter_kwargs={
            'remove_inpatients': remove_inpatients, 
            'exclude_dins': exclude_dins, 
            'verbose': True
        }, 
        process_kwargs={
            'method': systemic_method, 
            'cycle_length_map': cycle_length_map,
            'day_one_regimens': day_one_regimens
        }
    )
    
    # Demographic Data
    demog = Demographic()
    demo_df = demog.run(exclude_blood_cancer=remove_blood_cancer_patients)
    df = combine_demographic_data(df, demo_df)
    df = df.drop(columns=DROP_COLS)
    
    # Laboratory Data
    labr = Laboratory(f'{output_path}/data', PROCESSES)
    labr.preprocess(set(df['ikn']))
    lab_df = labr.run(df, time_window=time_window)
    df, lab_map, missing_df = combine_lab_data(df, lab_df)
    if special_obs is not None:
        # for each lab test in special_obs, save all measurements for later analysis
        grouped_obs = group_observations(all_observations, lab_df['obs_code'].value_counts())
        for obs_name in tqdm(special_obs):
            for i, obs_code in enumerate(grouped_obs[obs_name]):
                obs = lab_map[obs_code] if i == 0 else obs.fillna(lab_map[obs_code])
            obs.columns = obs.columns.astype(str)
            obs.to_parquet(f'{output_path}/data/{obs_name}.parquet.gzip', compression='gzip', index=False)
    
    # Symptom Data
    symp = Symptoms(processes=PROCESSES)
    symp_df = symp.run(df)
    df = combine_symptom_data(df, symp_df)
    
    if include_ohip:
        ohip = pd.read_parquet(f'{root_path}/data/ohip.parquet.gzip')
        ohip = filter_ohip_data(ohip, billing_codes=['A945', 'C945'])
        initial_pccs_date = ohip.groupby('ikn')['servdate'].first()
        df['first_PCCS_date'] = df['ikn'].map(initial_pccs_date)
        
    if include_odb:
        df = process_odb_data(df)
        
    if include_dialysis:
        df = process_dialysis_data(df)
    
    filepath = f'{output_path}/data/final_data.parquet.gzip'
    df.to_parquet(filepath, compression='gzip')
    
    if include_acute_care_use:
        # Acute Care Use Data
        acu = AcuteCareUse(f'{output_path}/data', PROCESSES)
        acu.run(df.reset_index())
    
    if include_blood_transfusion:
        # Blood Transfusion during acute care use Data
        bt = BloodTransfusion(f'{output_path}/data')
        bt.run(df.reset_index())

if __name__ == '__main__':
    main()
