"""
========================================================================
© 2021 Institute for Clinical Evaluative Sciences. All rights reserved.

TERMS OF USE:
##Not for distribution.## This code and data is provided to the user solely for its own non-commercial use by individuals and/or not-for-profit corporations. User shall not distribute without express written permission from the Institute for Clinical Evaluative Sciences.

##Not-for-profit.## This code and data may not be used in connection with profit generating activities.

##No liability.## The Institute for Clinical Evaluative Sciences makes no warranty or representation regarding the fitness, quality or reliability of this code and data.

##No Support.## The Institute for Clinical Evaluative Sciences will not provide any technological, educational or informational support in connection with the use of this code and data.

##Warning.## By receiving this code and data, user accepts these terms, and uses the code and data, solely at its own risk.
========================================================================
"""
""" Extract variables from data on cancer and diagnosis treatment (systemic), 
blood work (olis), and demographics (y3)
"""
import sys
for i, p in enumerate(sys.path):
    sys.path[i] = sys.path[i].replace("/software/anaconda/3/", "/MY/PATH/.conda/envs/myenv/")
import tqdm
import os
import itertools
import pandas as pd
import numpy as np
import multiprocessing as mp
import datetime as dt
import utilities as util
import matplotlib.pyplot as plt

from functools import partial
from collections import Counter, defaultdict

class Preprocess:
    def __init__(self):
        # regimens
        self.regimen_metadata = util.get_included_regimen(util.read_partially_reviewed_csv())
        self.regimen_name_mapping = {mapped_from: mapped_to if mapped_to != 'None' else mapped_from 
                                    for mapped_from, mapped_to in 
                                    self.regimen_metadata['mapping_to_proper_name'].to_dict().items()}
        self.cycle_lengths = self.regimen_metadata['cycle_length'].to_dict()
        self.regimens = self.regimen_metadata.index 

        # columns
        self.din_exclude = ['02441489', '02454548', '01968017', '02485575', '02485583', '02485656', 
                            '02485591', '02484153', '02474565', '02249790', '02506238', '02497395'] # for neutrophil
        self.y3_cols = ['ikn', 'sex', 'bdate',
                        'lhin_cd', # local health integration network
                        'curr_morph_cd', # cancer type
                        'curr_topog_cd', # cancer location
                        ]

        self.systemic_cols = ['ikn', 'regimen', 'visit_date', 
                              'body_surface_area',#m^2
                              'intent_of_systemic_treatment', # A - after surgery, C - chemo, N - before surgery, P - incurable
                              'line_of_therapy', # switch from one regimen to another
                             ] 
                             # 'din', 'cco_drug_code', 'dose_administered', 'measurement_unit']
    
        self.olis_cols = ['ikn', 'ObservationCode', 'ObservationDateTime', 
                          'ObservationReleaseTS', 'value_recommended_d']

        # chemotherapy dataframe
        if os.path.exists('data/chemo_processed.csv'):
            chemo_df = pd.read_csv('data/chemo_processed.csv', dtype={'ikn': str})
            chemo_df['visit_date'] = pd.to_datetime(chemo_df['visit_date'])
            chemo_df['prev_visit'] = pd.to_datetime(chemo_df['prev_visit'])
            chemo_df['chemo_interval'] = pd.to_timedelta(chemo_df['chemo_interval'])
            self.chemo_df = chemo_df

        # multiprocessing
        self.processes = 50
        self.manager = mp.Manager()
        self.shared_dict = manager.dict()

    def parallelize(self, df, worker, processes=16):
        # splits dataframe into x number of partitions, where x is number of processes
        generator = np.array_split(df, processes)
        pool = mp.Pool(processes=processes)
        result = pool.map(worker, generator)
        pool.close()
        pool.join() # wait for all threads
        return result

    def sas_to_csv(self):
        """Convert sas to csv
        """
        util.sas_to_csv('y3')
        util.sas_to_csv('systemic')
        util.sas_to_csv('olis')
        util.sas_to_csv('esas')
        util.sas_to_csv('ecog')
        util.sas_to_csv('olis_blood_count')


class CancerandDemographic(Preprocess):
    def __init__(self):
        super(CancerandDemographic, self).__init__()

    def filter_systemic_data(self, chunk):
        """Filter cancer data
        """
        # remove first two characters "b'" and last character "'"
        for col in ['ikn', 'din']:
            chunk[col] = chunk[col].str[2:-1]
        
        # keep only selected reigments
        chunk = chunk[chunk['regimen'].isin(self.regimens)]
        
        # remove dins in din_exclude
        chunk = chunk[~chunk['din'].isin(self.din_exclude)]
        
        # keep only selected columns
        chunk = chunk[self.systemic_cols]
        chunk = chunk.drop_duplicates()
        
        return chunk

    def create_ikn_chemo_mapping(self):
        """Create mapping between patient id and their chemo sessions
        """
        ikn_chemo_dict = {}
        chunks = pd.read_csv('data/systemic.csv', chunksize=10**6)
        for i, chunk in tqdm.tqdm(enumerate(chunks), total=14):
            chunk = self.filter_systemic_data(chunk)
            # combine all patients from all the chunks
            for ikn, group in chunk.groupby('ikn'):
                if ikn in ikn_chemo_dict:
                    ikn_chemo_dict[ikn] = pd.concat([ikn_chemo_dict[ikn], group])
                else:
                    ikn_chemo_dict[ikn] = group
        print("Number of patients =", len(ikn_chemo_dict.keys()))
        return ikn_chemo_dict

    def merge_intervals(self, df):
        """Merges small chemo intervals into a 4 day cycle, 
        or to the row below/above that has interval greater than 4 days
        """
        df = df.reset_index(drop=True)
        remove_indices = []
        for i in range(len(df)):
            if df.loc[i, 'chemo_interval'] > pd.Timedelta('3 days') or pd.isnull(df.loc[i, 'chemo_interval']):
                continue
            if i == len(df)-1:
                # merge with the row above if last entry of a regimen
                if i == 0: # if its the very first entry of the whole dataframe, leave it as is
                    continue 
                df.loc[i-1, 'visit_date'] = df.loc[i, 'visit_date']
                df.loc[i-1, 'chemo_interval'] = df.loc[i, 'chemo_interval'] + df.loc[i-1, 'chemo_interval'] 
            elif df.loc[i, 'regimen'] != df.loc[i+1, 'regimen']:
                # merge with the row above if last entry of an old regimen
                if i == 0: # if its the very first entry of the whole dataframe, leave it as is
                    continue 
                df.loc[i-1, 'visit_date'] = df.loc[i, 'visit_date']
                df.loc[i-1, 'chemo_interval'] = df.loc[i, 'chemo_interval'] + df.loc[i-1, 'chemo_interval'] 
            else:
                # merge with the row below
                df.loc[i+1, 'prev_visit'] = df.loc[i, 'prev_visit']
                df.loc[i+1, 'chemo_interval'] = df.loc[i, 'chemo_interval'] + df.loc[i+1, 'chemo_interval']
            remove_indices.append(i)
        df = df.drop(index=remove_indices)
        return df 

    def create_chemo_df(self, ikn_chemo_dict):
        """Create a dataframe of extracted variables on cancer diagnosis and treatment data
        """
        chemo = []
        num_chemo_sess_elimniated = 0
        for ikn, df in tqdm.tqdm(ikn_chemo_dict.items()):
            # change regimen name to the correct mapping
            df['regimen'] = df['regimen'].map(self.regimen_name_mapping)
            
            # order the dataframe by date
            df = df.drop_duplicates()
            df['visit_date'] = pd.to_datetime(df['visit_date'])
            df = df.sort_values(by='visit_date')
            
            # include prev visit and chemo interval
            df['prev_visit'] = df['visit_date'].shift()
            df.loc[~df['regimen'].eq(df['regimen'].shift()), 'prev_visit'] = pd.NaT # break off when chemo regimen changes
            df['chemo_interval'] = df['visit_date'] - df['prev_visit']
            
            # Merges small intervals into a 4 day cycle, or to the row below/above that has interval greater than 4 days
            # NOTE: for patient X (same with Y), they have single chemo sessions that gets eliminated. 
            num_chemo_sess_elimniated += len(df[df['prev_visit'].isnull() & ~df['regimen'].eq(df['regimen'].shift(-1))])
            df = df[~df['chemo_interval'].isnull()]
            if df.empty:
                ikn_chemo_dict[ikn] = df
                continue
            df = merge_intervals(df)
            if df.empty:
                # most likely patient (e.g. Z) had consecutive 1 day interval chemo sessions 
                # that totaled less than 5 days
                ikn_chemo_dict[ikn] = df
                continue
            
            # identify chemo cycle number (resets when patient undergoes new chemo regimen or there is a 60 day gap)
            mask = df['regimen'].eq(df['regimen'].shift()) & (df['chemo_interval'].shift() < pd.Timedelta('60 days'))
            group = (mask==False).cumsum()
            df['chemo_cycle'] = mask.groupby(group).cumsum()+1
            
            # identify if this is the first chemo cycle of a new regimen immediately after the old one
            # WARNING: currently does not account for those single chemo sessios (that gets eliminated above)
            mask = ~df['regimen'].eq(df['regimen'].shift()) & (df['prev_visit']-df['visit_date'].shift() < pd.Timedelta('60 days'))
            mask.iloc[0] = False
            df['immediate_new_regimen'] = mask
            
            # convert to list and combine for faster performance
            chemo.extend(df.values.tolist())
            
        chemo_df = pd.DataFrame(chemo, columns=df.columns)
        print("Number of single chemo sessions eliminated =", num_chemo_sess_elimniated)
        return chemo_df

    def add_demographic_data(self, chemo_df):
        """Combine variables extracted from demographic data to the chemo dataframe
        """
        y3 = pd.read_csv('data/y3.csv')
        y3 = y3[self.y3_cols]
        y3['ikn'] = y3['ikn'].str[2:-1]
        y3 = y3.set_index('ikn')
        chemo_df = chemo_df.join(y3, on='ikn')
        chemo_df['bdate'] = pd.to_datetime(chemo_df['bdate'])
        chemo_df['age'] = chemo_df['prev_visit'].dt.year - chemo_df['bdate'].dt.year
        return chemo_df

    def clean_up_features(self, chemo_df):
        """Clean up the entires
        Remove/replace erroneous entries
        Replace cancer location and cancer type entries appearing less than 6 times as 'Other'
        Rearrange the columns
        """
        # clean up some features
        chemo_df['body_surface_area'] = chemo_df['body_surface_area'].replace(0, np.nan)
        chemo_df['body_surface_area'] = chemo_df['body_surface_area'].replace(-99, np.nan)
        chemo_df['curr_morph_cd'] = chemo_df['curr_morph_cd'].replace('*U*', np.nan)
        # remove first two characters "b'" and last character "'"
        for col in ['intent_of_systemic_treatment', 'lhin_cd', 'curr_morph_cd', 'curr_topog_cd', 'sex']:
            chemo_df[col] = chemo_df[col].str[2:-1]
        # clean up morphology and topography code features
        for col in ['curr_morph_cd', 'curr_topog_cd']:
            chemo_df[col] = chemo_df[col].replace('*U*', np.nan)
            # replace code with number of rows less than 6 to 'Other'
            counts = chemo_df[col].value_counts() 
            replace_code = counts.index[counts < 6]
            chemo_df.loc[chemo_df[col].isin(replace_code), col] = 'Other'

        # rearrange the columns
        chemo_df = chemo_df[['ikn', 'regimen', 'visit_date', 'prev_visit', 'chemo_interval', 'chemo_cycle', 'immediate_new_regimen',
                'intent_of_systemic_treatment', 'line_of_therapy', 'lhin_cd', 'curr_morph_cd', 'curr_topog_cd',
                'age', 'sex', 'body_surface_area']]
        return chemo_df

    def preprocess_cancer_and_demographic_data(self):
        """Extract variables from cancer diagnosis and treatment and demograhic data and write to file
        """
        ikn_chemo_dict = self.create_ikn_chemo_mapping()
        chemo_df = self.create_chemo_df(ikn_chemo_dict)
        chemo_df = self.add_demographic_data(chemo_df)
        chemo_df = self.clean_up_features(chemo_df)
        chemo_df.to_csv('data/chemo_processed.csv', index=False)

        print("Number of rows now", len(chemo_df))
        print("Number of patients now =", len(chemo_df['ikn'].unique()))
        print("Number of rows with chemo intervals less than 4 days =",# some still remained after merging of the intervals
              len(chemo_df[chemo_df['chemo_interval'] < pd.Timedelta('4 days')]))

class BloodWork(Preprocess):
    def __init__(self):
        super(BloodWork, self).__init__()
        # create columns for 5 days before to 28 days after chemo administration
        df = self.chemo_df.copy()
        df.loc[:, range(-5, 29, 1)] = np.nan
        neutrophil_df = df.copy()
        platelet_df = df.copy()
        hemoglobin_df = df.copy()
        self.mapping = {'777-3': platelet_df, '751-8': neutrophil_df, '718-7': hemoglobin_df}
        del df

    def worker(self, partition):
        chunk = self.shared_dict['olis_chunk']
        result = []
        # loop through each row of this partition
        for chemo_idx, chemo_row in tqdm.tqdm(partition.iterrows(), total=len(partition), position=0):
            
            # see if there any blood count data within the target dates
            earliest_date = chemo_row['prev_visit'] - pd.Timedelta('5 days')
            # set limit to 28 days after chemo administration or the day of next chemo administration, 
            # whichever comes first
            latest_date = min(chemo_row['visit_date'], chemo_row['prev_visit'] + pd.Timedelta('28 days'))
            tmp = chunk[(earliest_date < chunk['ObservationDateTime']) & 
                        (chunk['ObservationDateTime'] < latest_date) & 
                        (chunk['ikn'] == chemo_row['ikn'])]
            
            # loop through the blood count data
            for blood_idx, blood_row in tmp.iterrows():
                days_after_chemo = (blood_row['ObservationDateTime'] - chemo_row['prev_visit']).days

                # place onto result
                result.append((blood_row['ObservationCode'], chemo_idx, days_after_chemo, 
                               blood_row['value_recommended_d']))
        return result

    def filter_blood_data(self, chunk):
        # only keep rows where patient ids exist in both the olis chunk and chemo_df
        filtered_chemo_df = self.chemo_df[self.chemo_df['ikn'].isin(chunk['ikn'])]
        chunk = chunk[chunk['ikn'].isin(filtered_chemo_df['ikn'])]
        
        # remove rows with blood count null values
        chunk = chunk[~chunk['value_recommended_d'].isnull()]
        
        # remove duplicate rows
        subset = ['ikn','ObservationCode', 'ObservationDateTime', 'value_recommended_d']
        chunk = chunk.drop_duplicates(subset=subset) 
        
        # if only the patient id, blood, and observation timestamp are duplicated (NOT the blood count value), 
        # keep the most recently RELEASED row
        chunk = chunk.sort_values(by='ObservationReleaseTS')
        subset = ['ikn','ObservationCode', 'ObservationDateTime']
        chunk = chunk.drop_duplicates(subset=subset, keep='last')
        
        return filtered_chemo_df, chunk

    def extract_blood_work(self):
        """Parallelize extraction of blood work data and write result to a npy file
        """
        chunks = pd.read_csv("data/olis.csv", chunksize=10**6) # (chunksize=10**5, i=653, 01:45), (chunksize=10**6, i=66, 1:45)
        result = [] # np.load('data/checkpoint/data_list_chunk15.npy').tolist()
        for i, chunk in tqdm.tqdm(enumerate(chunks), total=66):
            # if i < 16: continue
                
            # remove first two characters "b'" and last character "'"
            for col in ['ikn', 'ObservationCode']:
                chunk[col] = chunk[col].str[2:-1]
                
            # keep only selected columns
            chunk = chunk[self.olis_cols]
            
            # convert string column into timestamp column
            chunk['ObservationDateTime'] = pd.to_datetime(chunk['ObservationDateTime'])
            chunk['ObservationDateTime'] = chunk['ObservationDateTime'].dt.floor('D') # keep only the date, not time
            chunk['ObservationReleaseTS'] = pd.to_datetime(chunk['ObservationReleaseTS'])
            
            # filter out rows
            filtered_chemo_df, chunk = self.filter_blood_data(chunk)
            
            # find blood count values corresponding to each row in df concurrently in parallel processes
            shared_dict['olis_chunk'] = chunk
            chunk_result = self.parallelize(filtered_chemo_df, self.worker, processes=self.processes)
            chunk_result = list(itertools.chain(*chunk_result))
            result += chunk_result
            if i != 0:
                os.remove(f'data/checkpoint/data_list_chunk{i-1}.npy')
            print(f'OLIS chunk {i} completed: size of list', len(result))
            np.save(f"data/checkpoint/data_list_chunk{i}.npy", result)
        np.save("data/data_list.npy", result)

    def preprocess_blood_work_data(self):
        """Extract variables from blood work data, combine to chemo dataframe, write to file
        """
        data_list = np.load('data/data_list.npy') # OHH I think it converts all the ints to string to save space

        # update the neturophil/hemoglobin/platelet dataframes 
        df = pd.DataFrame(data_list, columns=['blood_type', 'chemo_idx', 'days_after_chemo', 'blood_count'])
        for blood_type, blood_group in df.groupby('blood_type'):
            for day, day_group in blood_group.groupby('days_after_chemo'):
                # print(f'Blood Type: {blood_type}, Days After Chemo: {day}, Number of Blood Samples: {len(day_group)}')
                chemo_indices = day_group['chemo_idx'].values.astype(int)
                blood_count_values = day_group['blood_count'].values.astype(float)
                self.mapping[blood_type].loc[chemo_indices, int(day)] = blood_count_values
        self.mapping['neutrophil'].to_csv('data/neutrophil.csv',index=False)
        self.mapping['platelet'].to_csv('data/platelet.csv',index=False)
        self.mapping['hemoglobin'].to_csv('data/hemoglobin.csv',index=False)

class Questionnaire(Preprocess):
    def __init__(self, chemo_df):
        super(Questionnaire, self).__init__()
        self.chemo_df = chemo_df

    def filter_questionnaire_data(self, chunk):
        chunk = chunk.rename(columns={'esas_value': 'severity', 'esas_resourcevalue': 'symptom'})
        # remove first two characters "b'" and last character "'"
        for col in ['ikn', 'severity', 'symptom']:
            chunk[col] = chunk[col].str[2:-1]
        chunk['surveydate'] = pd.to_datetime(chunk['surveydate'])
        chunk['severity'] = chunk['severity'].astype(int)
        
        # filter patients not in chemo_df
        chunk = chunk[chunk['ikn'].isin(self.chemo_df['ikn'])]
        
        # sort by date
        chunk = chunk.sort_values(by='surveydate')
        
        # filter out patients not in chunk
        filtered_chemo_df = self.chemo_df[self.chemo_df['ikn'].isin(chunk['ikn'])]
        
        return chunk, filtered_chemo_df

    def worker(self, partition):
        esas = self.shared_dict['esas_chunk']

        result = []
        for ikn, group in partition.groupby('ikn'):
            esas_specific_ikn = esas[esas['ikn'] == ikn]
            for idx, chemo_row in group.iterrows():
                visit_date = chemo_row['visit_date']
                esas_most_recent = esas_specific_ikn[esas_specific_ikn['surveydate'] < visit_date]
                if not esas_most_recent.empty:
                    symptoms = list(esas_most_recent['symptom'].unique())
                    for symptom in symptoms:
                        esas_specific_symptom = esas_most_recent[esas_most_recent['symptom'] == symptom]

                        # last item is always the last observed grade (we've already sorted by date)
                        result.append((idx, symptom, esas_specific_symptom['severity'].iloc[-1]))
        return result

    def extract_questionnaire(self):
        """Parallelize extraction of questionnaire data and write result to a npy file
        """
        chunks = pd.read_csv('data/esas.csv', chunksize=10**6) # chunksize=10**6, i=44, 1:05
        result = [] #np.load('data/checkpoint/esas_features_chunk0.npy')
        for i, chunk in tqdm.tqdm(enumerate(chunks), total=44):
            # if i < 0: continue
            chunk, filtered_chemo_df = self.filter_questionnaire_data(chunk)
            
            # get results
            self.shared_dict['esas_chunk'] = chunk
            chunk_result = self.parallelize(filtered_chemo_df, self.worker)
            chunk_result = list(itertools.chain(*chunk_result))
            result += chunk_result
            if i != 0:
                os.remove(f'data/checkpoint/esas_features_chunk{i-1}.npy')
            np.save(f'data/checkpoint/esas_features_chunk{i}.npy', result)
        np.save('data/esas_features.npy', result)

    def preprocess_questionnaire(self):
        """Extract variables from symptom questionnaires, combine to chemo dataframe, write to file
        """
        esas_features = np.load('data/esas_features.npy')
        symptoms = set(esas_features[:, 1]) 
        # clean up - remove Activities & Function as they only have 26 samples
        symptoms.remove('Activities & Function:')
        result = {symptom: [] for symptom in symptoms}
        esas_idx_mapping = {}

        # fill out the esas_idx_mapping
        # TODO: CLEAN UP THE DUPLICATES USING SURVEYDATE - xxxx out of xxxxxxx
        for idx,symptom,severity in tqdm.tqdm(esas_features):
            if symptom in esas_idx_mapping:
                esas_idx_mapping[symptom][idx] = severity
            else:
                esas_idx_mapping[symptom] = {idx: severity}
                
        # fill out the results
        for i in tqdm.tqdm(range(0, len(self.chemo_df))):
            for symptom in symptoms:
                if str(i) in esas_idx_mapping[symptom]:
                    result[symptom].append(esas_idx_mapping[symptom][str(i)])
                else:
                    result[symptom].append(np.nan)

        esas = pd.DataFrame(result)

        # mode impute the missing data
        # esas = esas.fillna(esas.mode().iloc[0])

        # put esas features in chemo_df
        chemo_df = pd.concat([self.chemo_df, esas], axis=1)
        chemo_df.to_csv('data/chemo_processed2.csv', index=False)
        return chemo_df

class ECOGStatus(Preprocess):
    def __init__(self, chemo_df):
        super(ECOGStatus, self).__init__()
        self.chemo_df = chemo_df

    def filter_ecog_data(self):
        ecog = pd.read_csv('data/ecog.csv')

        # organize and format columns
        ecog = ecog.drop(columns=['ecog_resourcevalue'])
        ecog = ecog.rename(columns={'ecog_value': 'ecog_grade'})
        # remove first two characters "b'" and last character "'"
        for col in ['ikn', 'ecog_grade']:
            ecog[col] = ecog[col].str[2:-1]
        ecog['ecog_grade'] = ecog['ecog_grade'].astype(int)
        ecog['surveydate'] = pd.to_datetime(ecog['surveydate'])

        # filter out patients not in chemo_df
        ecog = ecog[ecog['ikn'].isin(self.chemo_df['ikn'])]

        # sort by date
        ecog = ecog.sort_values(by='surveydate')

        # filter out patients not in ecog
        filtered_chemo_df = self.chemo_df[self.chemo_df['ikn'].isin(ecog['ikn'])]
        return ecog, filtered_chemo_df

    def worker(self, partition):
        ecog = self.shared_dict['ecog']
        result = []
        for ikn, group in tqdm.tqdm(partition.groupby('ikn'), position=0):
            ecog_specific_ikn = ecog[ecog['ikn'] == ikn]
            for idx, chemo_row in group.iterrows():
                visit_date = chemo_row['visit_date']
                ecog_most_recent = ecog_specific_ikn[ecog_specific_ikn['surveydate'] < visit_date]
                if not ecog_most_recent.empty:
                    # last item is always the last observed grade (we've already sorted by date)
                    result.append((idx, ecog_most_recent['ecog_grade'].iloc[-1]))
        return result

    def extract_ecog_status(self):
        """Parallelize extraction of ecog data and write result to a npy file
        """
        ecog, filtered_chemo_df = self.filter_ecog_data()
        self.shared_dict['ecog'] = ecog
        result = self.parallelize(filtered_chemo_df, self.worker)
        data_list = list(itertools.chain(*result))
        np.save('data/ecog_features.npy', data_list)

    def preprocess_ecog_status(self):
        """Extract ecog status, combine to chemo dataframe, write to file
        """
        ecog_features = np.load('data/ecog_features.npy')
        idx_ecog_mapping = {idx: ecog_grade for idx, ecog_grade in ecog_features}

        # fill out result
        result = []
        for i in tqdm.tqdm(range(0, len(self.chemo_df))):
            if i in idx_ecog_mapping:
                result.append(idx_ecog_mapping[i])
            else:
                result.append(np.nan)
        
        # put ecog grade in chemo_df
        self.chemo_df['ecog_grade'] = result

        # mean impute missing values
        # mode = self.chemo_df['ecog_grade'].mode()[0]
        # self.chemo_df['ecog_grade'] = self.chemo_df['ecog_grade'].fillna(mode)

        self.chemo_df.to_csv('data/chemo_processed2.csv', index=False)
        return self.chemo_df

class ExtraBloodWork(Preprocess):
    def __init__(self, chemo_df):
        super(ExtraBloodWork, self).__init__()
        self.chemo_df = chemo_df
        self.keep_blood_types = {'4544-3': 'hematocrit',
                                 '6690-2': 'leukocytes',
                                 '787-2': 'erythrocyte_mean_corpuscular_volume',
                                 '789-8': 'erythrocytes',
                                 '788-0': 'erythrocyte_distribution_width',
                                 '785-6': 'erythrocyte_mean_corpuscular_hemoglobin',
                                 '786-4': 'erythrocyte_mean_corpuscular_hemoglobin_concentration',
                                 '731-0': 'lymphocytes',
                                 '711-2': 'eosinophils',
                                 '704-7': 'basophils',
                                 '742-7': 'monocytes',
                                 '32623-1': 'platelet_mean_volume'}
        self.mapping = {blood_type: pd.DataFrame(index=self.chemo_df.index, columns=range(-5,29)) 
                        for blood_type in self.keep_blood_types}

    def filter_extra_blood_data1(self, chunk):
        # keep only selected columns
        olis_cols = ['ikn', 'ObservationCode', 'ObservationDateTime', 'ObservationReleaseTS', 
                     'ReferenceRange', 'Units','Value_recommended_d']
        chunk = chunk[olis_cols]
        # remove first two characters "b'" and last character "'"
        for col in ['ikn', 'ObservationCode', 'ReferenceRange', 'Units']:
            chunk[col] = chunk[col].str[2:-1]
        
        # filter patients not in chemo_df
        chunk = chunk[chunk['ikn'].isin(self.chemo_df['ikn'])]
        
        # filter out platelet, neutrophil, and hemoglobin (already been preprocessed)
        chunk = chunk[~chunk['ObservationCode'].isin(['718-7', '751-8', '777-3'])]
        
        # rename value recommended d to value
        chunk = chunk.rename(columns={'Value_recommended_d': 'value'})
        
        return chunk

    def clean_up_data(self):
        """Clean up extra blood work data and write to new csv file
        """
        chunks = pd.read_csv('data/olis_blood_count.csv', chunksize=10**6) # chunksize=10**6, i=331, 19:09
        for i, chunk in tqdm.tqdm(enumerate(chunks), total=331):
            chunk = self.filter_extra_blood_data1(chunk)
            # write to csv
            header = True if i == 0 else False
            chunk.to_csv(f"data/olis_blood_count2.csv", header=header, mode='a', index=False)

    def filter_extra_blood_data2(self, chunk):
        # Convert string column into timestamp column
        chunk['ObservationDateTime'] = pd.to_datetime(chunk['ObservationDateTime'])
        chunk['ObservationDateTime'] = chunk['ObservationDateTime'].dt.floor('D') # keep only the date, not time
        chunk['ObservationReleaseTS'] = pd.to_datetime(chunk['ObservationReleaseTS'])
        
        # Filter rows with excluded blood types
        chunk = chunk[chunk['ObservationCode'].isin(self.keep_blood_types)]
        
        # Filter rows with blood count null values
        chunk = chunk[~chunk['value'].isnull()]

        # Remove duplicate rows
        subset = ['ikn','ObservationCode', 'ObservationDateTime', 'value']
        chunk = chunk.drop_duplicates(subset=subset) 
        
        # If only the patient id, blood, and observation timestamp are duplicated (NOT the blood count value), 
        # keep the most recently RELEASED row
        chunk = chunk.sort_values(by='ObservationReleaseTS')
        subset = ['ikn','ObservationCode', 'ObservationDateTime']
        chunk = chunk.drop_duplicates(subset=subset, keep='last')
        
        # only keep rows where patient ids exist in olis chunk
        filtered_chemo_df = self.chemo_df[self.chemo_df['ikn'].isin(chunk['ikn'])]
        
        return chunk, filtered_chemo_df

    def worker(self, partition):
        olis = self.shared_dict['olis_chunk']
        result = []
        for ikn, chemo_group in partition.groupby('ikn'):
            olis_subset = olis[olis['ikn'] == ikn]
            for chemo_idx, chemo_row in chemo_group.iterrows():
                
                # see if there any blood count data within the target dates
                earliest_date = chemo_row['prev_visit'] - pd.Timedelta('5 days')
                # set limit to 28 days after chemo administration or the day of next chemo administration, 
                # whichever comes first
                latest_date = min(chemo_row['visit_date'], chemo_row['prev_visit'] + pd.Timedelta('28 days'))
                tmp = olis_subset[(earliest_date <= olis_subset['ObservationDateTime']) & 
                                  (latest_date >= olis_subset['ObservationDateTime'])]
                
                # loop through the blood count data
                for blood_idx, blood_row in tmp.iterrows():
                    blood_type = blood_row['ObservationCode']
                    blood_count = blood_row['value']
                    obs_date = blood_row['ObservationDateTime']
                    days_after_chemo = (obs_date - chemo_row['prev_visit']).days
                    # place onto result
                    result.append((blood_type, chemo_idx, days_after_chemo, blood_count))
                
        return result

    def extract_extra_blood_work(self):
        """Parallelize extraction of extra blood work data and write result to a npy file
        """
        self.clean_up_data()
        chunks = pd.read_csv('data/olis_blood_count2.csv', dtype={'ikn':str}, chunksize=10**6) # chunksize=10**6, i=62, 0:51
        result = [] # np.load('data/checkpoint/olis_blood_count_chunk5.npy').tolist()
        for i, chunk in tqdm.tqdm(enumerate(chunks), total=62):
            # if i < 6: continue
            chunk, filtered_chemo_df = self.filter_extra_blood_data2(chunk)
            self.shared_dict['olis_chunk'] = chunk
            chunk_result = self.parallelize(filtered_chemo_df, self.worker, processes=self.processes)
            chunk_result = list(itertools.chain(*chunk_result))
            result += chunk_result
            if i != 0:
                os.remove(f'data/checkpoint/olis_blood_count_chunk{i-1}.npy')
            np.save(f'data/checkpoint/olis_blood_count_chunk{i}.npy', result)
            print(f'OLIS blood count chunk {i} completed: size of result', len(result))
        np.save('data/olis_blood_count.npy', result)

    def preprocess_extra_blood_work_data(self):
        """Extract variables from extra blood work data, combine to chemo dataframe, write to file
        """
        olis_blood_count = np.load('data/olis_blood_count.npy') # all the ints are converted to strings
        df = pd.DataFrame(olis_blood_count, columns=['blood_type', 'chemo_idx', 'days_after_chemo', 'blood_count'])

        # fill up the blood count dataframes for each blood type
        for blood_type, blood_group in tqdm.tqdm(df.groupby('blood_type')):
            for day, day_group in blood_group.groupby('days_after_chemo'):
                # print(f'Blood Type: {blood_type}, Days After Chemo: {day}, Number of Blood Samples: {len(day_group)}')
                chemo_indices = day_group['chemo_idx'].values.astype(int)
                blood_count_values = day_group['blood_count'].values.astype(float)
                self.mapping[blood_type].loc[chemo_indices, int(day)] = blood_count_values

        # combine baseline blood count to chemo dataframe
        for blood_type, df in self.mapping.items():
            # get baseline blood counts
            # forward fill blood counts from day -5 to day 0
            values = df[range(-5,1)].ffill(axis=1)[1].values
            blood_name = self.keep_blood_types[blood_type]
            self.chemo_df.loc[df.index, f'baseline_{blood_name}_count'] = values
        self.chemo_df.to_csv('data/chemo_processed2.csv', index=False)

def main():
    parser = argparse.ArgumentParser(description='Preprocess Data')
    parser.add_argument('--sas_to_csv', action='store_true', help='Convert sas to csv')
    parser.add_argument('--cancer', action='store_true', help='Preprocess cancer and demographic data')
    parser.add_argument('--blood_work', action='store_true', help='Preprocess blood work data')
    parser.add_argument('--questionnaire', action='store_true', help='Preprocess symptom questionnaire data')
    parser.add_argument('--ecog_status', action='store_true', help='Preprocess ECOG status data')
    parser.add_argument('--extra_blood_work', action='store_true', help='Preprocess extra blood work data')
    args = parser.parse_args()
    if args.sas_to_csv:
        Pp = Preprocess()
        Pp.sas_to_csv()
    if args.cancer:
        CD = CancerandDemographic()
        CD.preprocess_cancer_and_demographic_data()
    if args.blood_work:
        BW = BloodWork()
        BW.extract_blood_work()
        BW.preprocess_blood_work_data()

    Pp = Preprocess()
    if getattr(Pp, 'chemo_df', None):
        chemo_df = Pp.chemo_df
    else:
        raise ValueError('Cancer data has not been preprocessed yet!')

    if args.questionnaire:
        Qs = Questionnaire(chemo_df)
        Qs.extract_questionnaire()
        chemo_df = Qs.preprocess_questionnaire()
    if args.ecog_status:
        ES = ECOGStatus(chemo_df)
        ES.extract_ecog_status()
        chemo_df = Qs.preprocess_ecog_status()
    if args.extra_blood_work:
        EBW = ExtraBloodWork(chemo_df)
        EBW.extract_extra_blood_work()
        EBW.preprocess_extra_blood_work_data()

if __name__ == '__main__':
    main()
