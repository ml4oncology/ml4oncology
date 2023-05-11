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
Script to convert sas7bdat files to parquet files

Comparison of csv file format vs parquet file format
olis_new: 71.1GB vs 4.0GB
systemic: 2.2GB vs 62.5MB
esas: 1.8GB vs 25.1MB
"""
import sys
import os
sys.path.append(os.getcwd())

from tqdm import tqdm
import pandas as pd

from src.config import root_path, sas_folder, sas_folder2
from src.spark import start_spark, spark_write, extract_observation_units

def sas_to_parquet(filename, new_filename=None, transfer_date=None, folder=sas_folder, **kwargs):
    if new_filename is None: new_filename = filename
    if transfer_date is not None: filename = f'transfer{transfer_date}/{filename}'
        
    filepath = f'{root_path}/{folder}/{filename}.sas7bdat'
    filesize = os.path.getsize(filepath) / 1024 ** 3 # GB
    if filesize > 50: 
        return large_sas_to_parquet(filepath, new_filename, **kwargs)
    
    df = pd.read_sas(filepath, encoding='ISO-8859-1')
    df = drop_empty_rows(df)
    df = process_edge_cases(df, new_filename)
    new_filepath = f'{root_path}/data/{new_filename}.parquet.gzip'
    df.to_parquet(new_filepath, compression='gzip', index=False)
    
def large_sas_to_parquet(filepath, new_filename, chunksize=10**8):
    """Convert large sas file into multiple parquet files.
    
    Currently no easy way to convert a single large sas file (>50GB)
    into a single parquet file. Best we can do is:
    
    1. Read and write in chunks
    2. Load multiple parquet files into a single dataframe using 
       pd.read_parquet(dir_path) or spark.read.parquet(dir_path)
    """
    # os.mkdir(f'{root_path}/data/{new_filename}')
    chunks = pd.read_sas(filepath, encoding='ISO-8859-1', chunksize=chunksize)
    for i, chunk in tqdm(enumerate(chunks), desc='Converting to csv'):
        chunk = drop_empty_rows(chunk)
        new_filepath = f"{root_path}/data/{new_filename}/chunk{i}.parquet.gzip"
        chunk.to_parquet(new_filepath, compression='gzip', index=False)
        
def drop_empty_rows(df):
    empty_mask = df.iloc[:,1:].isnull().all(axis=1) # remove empty rows
    df = df[~empty_mask]
    return df

def process_edge_cases(df, filename):
    if filename == 'dad':
        # Need to take care of out of bounds timestamp error error before 
        # writing to parquet or else PyArrow throws a fit
        for col in ['indate1', 'indate2', 'indate3']:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df
                                
def main():
    spark = start_spark()
    
    # Please Refer to datadictionary.ices.on.ca
    sas_to_parquet('y3') # ALR: Cancer Activitiy Level Reporting, patient and disease database
    sas_to_parquet('esas') # ESAS: Edmonton Symptom Assessment System, symptom management database
    sas_to_parquet('ecog') # ECOG: Eastern Cooperative Oncology Group performance status, symptom management database
    sas_to_parquet('prfs') # PRFS: Patient-Reported Functional Status, symptom management database
    sas_to_parquet('systemic') # ALR: Cancer Activitiy Level Reporting, systemic therapy database
    sas_to_parquet('immigration') # CIC: IRCC Permanent Residents Database
    sas_to_parquet('dad') # DAD: Discharge Abstract Database
    sas_to_parquet('nacrs') # NACRS: National Ambulatory Care Reporting System
    sas_to_parquet('dad_transfusion', transfer_date='20210810')
    sas_to_parquet('nacrs_transfusion', transfer_date='20210810')
    sas_to_parquet('odb') # ODB - Ontario Drug Benefit
    sas_to_parquet('dialysis_ohip', transfer_date='20220509') 
    sas_to_parquet('cohort_ohip', new_filename='ohip', folder=sas_folder2) # OHIP: Ontario Health Insurance Plan database
    sas_to_parquet('cohort_incq', new_filename='income+comorbidity', folder=sas_folder2)
    sas_to_parquet('cohort_rural_eth', new_filename='rural', folder=sas_folder2) # urban area vs rural area
    sas_to_parquet('olis_new', new_filename='olis', folder=sas_folder2) # OLIS: Ontario Laboratories Information System
    extract_observation_units(spark) # Extra - extract olis units
    
    spark.stop()
    
    # put income and comorbidity into separate files
    get_path = lambda name: f'{root_path}/data/{name}.parquet.gzip'
    df = pd.read_parquet(get_path('income+comorbidity'))
    df[['ikn', 'incquint']].to_parquet(get_path('income'), compression='gzip', index=False) # neighborhood income quintile database
    df[['ikn', 'diabetes_diag_date', 'hypertension_diag_date']].to_parquet(get_path('comorbidity'), compression='gzip', index=False)
    os.remove(get_path('income+comorbidity'))
    
if __name__ == '__main__':
    main()
