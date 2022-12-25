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

from src.config import (root_path, sas_folder2)
from src.preprocess import (sas_to_csv)
from src.spark import (extract_observation_units)
                                
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
    sas_to_csv('olis_new', new_name='olis', folder=sas_folder2, chunk_load=True) # OLIS: Ontario Laboratories Information System - 628it [3:09:12]
    sas_to_csv('dad_transfusion', transfer_date='20210810') # 7it [08:50] 
    sas_to_csv('nacrs_transfusion', transfer_date='20210810') # 26it [34:20] 
    sas_to_csv('odb_growth_factors') # ODB - Ontario Drug Benefit
    sas_to_csv('cohort_upd_17jan2022', new_name='combordity') # combordity
    sas_to_csv('dialysis_ohip', transfer_date='20220509') # dialysis
    sas_to_csv('cohort_ohip', new_name='ohip', folder=sas_folder2) # OHIP: Ontario Health Insurance Plan database
    sas_to_csv('cohort_incq', new_name='income', folder=sas_folder2) # neighborhood income quintile database
    sas_to_csv('cohort_rural_eth', new_name='rural', folder=sas_folder2) # urban area vs rural area
    
    # Extra - extract olis units
    extract_observation_units()

if __name__ == '__main__':
    main()
