import sys
import os
sys.path.append(os.getcwd())
import pandas as pd

from scripts.config import (root_path)
from scripts.preprocess import (sas_to_csv)
                                
def main():
    sas_to_csv('y3')
    sas_to_csv('esas')
    sas_to_csv('ecog')
    sas_to_csv('systemic')
    sas_to_csv('immigration')
    sas_to_csv('dad', chunk_load=True) 
    sas_to_csv('nacrs', chunk_load=True)
    sas_to_csv('olis', chunk_load=True) # 187it [59:42, 19.16s/it]
    sas_to_csv('olis_blood_count', chunk_load=True) # 331it [1:59:40, 21.69s/it]
    sas_to_csv('olis_update', chunk_load=True) # 28it [08:33, 18.33s/it]
    sas_to_csv('dad_transfusion', transfer=True) # 7it [08:50, 75.80s/it] 
    sas_to_csv('nacrs_transfusion', transfer=True) # 26it [34:20, 79.24s/it] 
    sas_to_csv('odb_growth_factors') # 1it [00:00,  1.37it/s]
    
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