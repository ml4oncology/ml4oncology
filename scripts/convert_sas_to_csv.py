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
import sys
import os
sys.path.append(os.getcwd())
import pandas as pd

from scripts.config import (root_path)
from scripts.preprocess import (sas_to_csv)
                                
def main():
    # Please Refer to datadictionary.ices.on.ca
    sas_to_csv('y3') # cancer and demographic database
    sas_to_csv('esas') # ESAS: edmonton symptom assessment system, symptom management database
    sas_to_csv('ecog') # ECOG: eastern cooperative oncology group performance status, symptom management database
    sas_to_csv('prfs') # PRFS: patient-reported functional status, symptom management database
    sas_to_csv('systemic') # chemotherapy database
    sas_to_csv('immigration') # immigration and language database
    sas_to_csv('dad', chunk_load=True) # DAD: discharge abstract database
    sas_to_csv('nacrs', chunk_load=True) # NACRS: National Ambulatory Care Reporting System
    sas_to_csv('olis', chunk_load=True) # OLIS: Ontario Laboratories Information System - 187it [59:42]
    sas_to_csv('olis_blood_count', chunk_load=True) # 331it [1:59:40]
    sas_to_csv('olis_update', chunk_load=True) # 28it [08:33]
    sas_to_csv('dad_transfusion', transfer_date='20210810') # 7it [08:50] 
    sas_to_csv('nacrs_transfusion', transfer_date='20210810') # 26it [34:20] 
    sas_to_csv('odb_growth_factors') # ODB - Ontario Drug Benefit
    sas_to_csv('cohort_upd_17jan2022', new_name='combordity') # combordity
    sas_to_csv('dialysis_ohip', transfer_date='20220509') # dialysis
    
    # Create olis_complete.csv by combining olis.csv, olis_blood_count.csv, and olis_update.csv
    # Blood work/lab test data in olis that do not overlap with olis_blood_count will be combined with all of olis_blood_count
    # olis_update.csv does not overlap with olis_blood_count.csv or olis_update.csv
    olis = pd.read_csv(f'{root_path}/data/olis.csv', dtype=str) 
    obc = pd.read_csv(f'{root_path}/data/olis_blood_count.csv', dtype=str)
    ou = pd.read_csv(f'{root_path}/data/olis_update.csv', dtype=str) # only contains two observation codes
    
    olis = olis.rename(columns={'value_recommended_d': 'value', 'SUBVALUE_RECOMMENDED_D': 'SubValue_recommended_d'})
    obc = obc.rename(columns={'Value_recommended_d': 'value'})
    ou = ou.rename(columns={'value_recommended_d': 'value', 'subvalue_recommended_d': 'SubValue_recommended_d'})
    
    olis_code_counts = olis['ObservationCode'].value_counts()
    obc_code_counts = obc['ObservationCode'].value_counts()
    olis_codes = olis_code_counts[olis_code_counts > 10000].index.tolist() # remove data with less than 10000 observations
    obc_codes = obc_code_counts[obc_code_counts > 10000].index.tolist()
    keep_olis_codes = [code for code in olis_codes if code not in obc_codes and 'XON' not in code] # XON is deprecated code, do not include
    
    olis_complete = pd.concat([olis[olis['ObservationCode'].isin(keep_olis_codes)], obc, ou])
    olis_complete.to_csv(f'{root_path}/data/olis_complete.csv', index=False)

if __name__ == '__main__':
    main()
