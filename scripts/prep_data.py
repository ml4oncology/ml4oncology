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
import pandas as pd
import numpy as np
from scripts.config import (root_path, cyto_folder, acu_folder, can_folder, death_folder,
                            blood_types, observation_cols, symptom_cols, event_map,
                            SCr_max_threshold, SCr_rise_threshold, SCr_rise_threshold2)
from scripts.preprocess import (numpy_ffill, replace_rare_col_entries)
from scripts.utility import (get_nmissing, get_eGFR)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s', datefmt='%I:%M:%S')

class PrepData:
    """
    Prepare the data for model training
    """
    def __init__(self):
        self.observation_cols = observation_cols
        self.symptom_cols = symptom_cols
        self.imp_obs = None # observation imputer
        self.imp_ee = None # esas ecog imputer
        self.scaler = None # normalizer
        self.bsa_mean = None # body surface area mean (imputer)
        
    def get_visit_date_feature(self, df, include_first_date=False):
        if include_first_date:
            first_visit_date = df.groupby('ikn')['visit_date'].min()
            df['first_visit_date'] = df['ikn'].map(first_visit_date)
            
        # convert to cyclical features
        month = df['visit_date'].dt.month - 1
        df['visit_month_sin'] = np.sin(2*np.pi*month/12)
        df['visit_month_cos'] = np.cos(2*np.pi*month/12)
        return df
    
    def get_missingness_feature(self, df):
        cols_with_nan = df.columns[df.isnull().any()]
        df[cols_with_nan + '_is_missing'] = df[cols_with_nan].isnull()
        return df
    
    def fill_missing_feature(self, df):
        for col in ['line_of_therapy', 'num_prior_EDs', 'num_prior_Hs']:
            if col in df.columns: df[col] = df[col].fillna(0)
        for col in ['days_since_last_chemo', 'chemo_interval']:
            if col in df.columns: df[col] = df[col].fillna(df[col].max())
        return df
    
    def drop_features_with_high_missingness(self, df, missing_thresh, verbose=False):
        nmissing = get_nmissing(df)
        exclude_cols = nmissing.index[nmissing['Missing (%)'] > missing_thresh].tolist()
        if verbose: logging.info(f'Dropping the following features for missingness over {missing_thresh}%: {exclude_cols}')
        self.observation_cols = list(set(self.observation_cols) - set(exclude_cols))
        self.symptom_cols = list(set(self.symptom_cols) - set(exclude_cols))
        return df.drop(columns=exclude_cols)
    
    def clip_outliers(self, data, cols=None, lower_percentile=0.01, upper_percentile=0.99):
        """
        clip the upper and lower percentiles for the columns indicated below
        """
        if not cols:
            cols = data.columns
            cols = cols[cols.str.contains('count') & ~cols.str.contains('is_missing')]
            cols = cols.drop([f'target_{bt}_count' for bt in blood_types], errors='ignore')
            cols = cols.tolist() + ['body_surface_area']

        thresh = data[cols].quantile([lower_percentile, upper_percentile])
        data[cols] = data[cols].clip(lower=thresh.loc[lower_percentile], upper=thresh.loc[upper_percentile], axis=1)
        return data, thresh

    def dummify_data(self, data):
        # make categorical columns into one-hot encoding
        return pd.get_dummies(data)

    def replace_missing_body_surface_area(self, data):
        # replace missing body surface area with the mean based on sex
        bsa = 'body_surface_area'
        mask_F = data['sex'] == 'F' if 'sex' in data.columns else data['sex_F'] == 1
        mask_M = data['sex'] == 'M' if 'sex' in data.columns else data['sex_M'] == 1
        
        if self.bsa_mean is None:
            self.bsa_mean = {'female': data.loc[mask_F, bsa].mean(), 'male': data.loc[mask_M, bsa].mean()}
            
        data.loc[mask_F, bsa] = data.loc[mask_F, bsa].fillna(self.bsa_mean['female'])
        data.loc[mask_M, bsa] = data.loc[mask_M, bsa].fillna(self.bsa_mean['male'])

        return data

    def impute_observations(self, data):
        # mean impute missing observation data
        if self.imp_obs is None:
            self.imp_obs = SimpleImputer() 
            data[self.observation_cols] = self.imp_obs.fit_transform(data[self.observation_cols])
        else:
            data[self.observation_cols] = self.imp_obs.transform(data[self.observation_cols])
        return data

    def impute_symptoms(self, data):
        # mode impute missing symptom data
        if self.imp_ee is None:
            self.imp_ee = SimpleImputer(strategy='most_frequent') 
            data[self.symptom_cols] = self.imp_ee.fit_transform(data[self.symptom_cols])
        else:
            data[self.symptom_cols] = self.imp_ee.transform(data[self.symptom_cols])
        return data
    
    def extra_norm_cols(self):
        """
        You can overwrite this to use custom columns
        """
        return []
        
    def normalize_data(self, data):
        norm_cols = ['age', 'body_surface_area', 'line_of_therapy', 'visit_month_sin', 'visit_month_cos', 'chemo_cycle', 
                     'days_since_starting_chemo', 'days_since_last_chemo']
        norm_cols += self.symptom_cols + self.observation_cols + self.extra_norm_cols()

        if self.scaler is None:
            self.scaler = StandardScaler()
            data[norm_cols] = self.scaler.fit_transform(data[norm_cols])
        else:
            data[norm_cols] = self.scaler.transform(data[norm_cols])
        return data
    
    def clean_feature_target_cols(self, feature_cols, target_cols):
        """
        You can overwrite this to do custom operations
        """
        feature_cols = feature_cols.drop('ikn')
        return feature_cols, target_cols
    
    def create_splits(self, data, split_date=None, verbose=True):
        if split_date is None: 
            # split data based on patient ids (60-20-20 split)
            # create training set
            gss = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
            train_idxs, test_idxs = next(gss.split(data, groups=data['ikn']))
            train_data = data.iloc[train_idxs]
            test_data = data.iloc[test_idxs]

            # crate validation and testing set
            gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
            valid_idxs, test_idxs = next(gss.split(test_data, groups=test_data['ikn']))

            valid_data = test_data.iloc[valid_idxs]
            test_data = test_data.iloc[test_idxs]
        else:
            # split data temporally based on patients first visit date
            if 'first_visit_date' not in data.columns: 
                raise ValueError('Please include first_visit_date in the data')
            mask = data['first_visit_date'] <= split_date
            data = data.drop(columns='first_visit_date') # DO NOT INCLUDE first_visit_date AS A FEATURE!
            train_data, test_data = data[mask], data[~mask]
            
            if verbose:
                logging.info(f"Development Cohort: NSessions={len(train_data)}. NPatients={train_data['ikn'].nunique()}. " + \
                             f"Contains all patients that had their first visit on or before {split_date}")
                logging.info(f"Testing Cohort:     NSessions={len(test_data)}. NPatients={test_data['ikn'].nunique()}. " + \
                             f"Contains all patients that had their first visit after {split_date}")
                logging.info(f"Number of overlapping patients={len(set(train_data['ikn']).intersection(set(test_data['ikn'])))}\n")
                
            # create validation set from train data (80-20 split)
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idxs, valid_idxs = next(gss.split(train_data, groups=train_data['ikn']))
            valid_data = train_data.iloc[valid_idxs]
            train_data = train_data.iloc[train_idxs]

        # sanity check - make sure there are no overlap of patients in the splits
        assert(sum(valid_data['ikn'].isin(train_data['ikn'])) + sum(valid_data['ikn'].isin(test_data['ikn'])) + 
               sum(train_data['ikn'].isin(valid_data['ikn'])) + sum(train_data['ikn'].isin(test_data['ikn'])) + 
               sum(test_data['ikn'].isin(train_data['ikn'])) + sum(test_data['ikn'].isin(valid_data['ikn'])) == 0)

        if verbose:
            logging.info(f'Size of splits: Train:{len(train_data)}, Val:{len(valid_data)}, Test:{len(test_data)}')
            logging.info(f"Number of patients: Train:{train_data['ikn'].nunique()}, Val:{valid_data['ikn'].nunique()}, Test:{test_data['ikn'].nunique()}")
        
        return train_data, valid_data, test_data
        

    def split_data(self, data, target_keyword='target', impute=True, normalize=True, convert_to_float=True, split_date=None, verbose=True):
        """
        Split data into training, validation and test sets based on patient ids (and optionally first visit dates)
        Impute and Normalize the datasets
        
        Args:
            split_date (string or None): split the data temporally by patient's very first chemo session date (e.g. 2017-06-30)
                                         train/val set will contain all visits on or before split_date
                                         test set will contain all visits after split_date
                                         string format: 'YYYY-MM-DD'
        """
        # convert dtype object to float
        if convert_to_float: data = data.astype(float)

        # create training, validation, testing set
        train_data, valid_data, test_data = self.create_splits(data, split_date=split_date, verbose=verbose)

        if impute:
            # IMPORTANT: always make sure train data is done first, so imputer is fit to the training set
            # mean impute the body surface area based on sex
            train_data = self.replace_missing_body_surface_area(train_data.copy())
            valid_data = self.replace_missing_body_surface_area(valid_data.copy())
            test_data = self.replace_missing_body_surface_area(test_data.copy())
            if verbose:
                logging.info(f"Body Surface Area Mean - Female:{self.bsa_mean['female'].round(4)}, Male:{self.bsa_mean['male'].round(4)}")

            # mean impute the blood work data
            train_data = self.impute_observations(train_data)
            valid_data = self.impute_observations(valid_data)
            test_data = self.impute_observations(test_data)

            # mode impute the esas and ecog data
            train_data = self.impute_symptoms(train_data)
            valid_data = self.impute_symptoms(valid_data)
            test_data = self.impute_symptoms(test_data)

        if normalize:
            # normalize the splits based on training data
            train_data = self.normalize_data(train_data.copy())
            valid_data = self.normalize_data(valid_data.copy())
            test_data = self.normalize_data(test_data.copy())

        # split into input features and target labels
        cols = train_data.columns
        feature_cols = cols[~cols.str.contains(target_keyword)]
        target_cols = cols[cols.str.contains(target_keyword)]
        feature_cols, target_cols = self.clean_feature_target_cols(feature_cols, target_cols)
        X_train, X_valid, X_test = train_data[feature_cols], valid_data[feature_cols], test_data[feature_cols]
        Y_train, Y_valid, Y_test = train_data[target_cols], valid_data[target_cols], test_data[target_cols]

        return [(X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test)]

    def get_label_distribution(self, Y_train, Y_valid=None, Y_test=None, save=False, save_path=''):
        dists, cols = [], []
        def update_distribution(Y, split):
            dists.append(Y.apply(pd.value_counts))
            cols.append(split)
        update_distribution(Y_train, 'Train')
        if Y_valid is not None: update_distribution(Y_valid, 'Valid')
        if Y_test is not None: update_distribution(Y_test, 'Test')
    
        dists = pd.concat(dists)
        total = pd.DataFrame([dists.loc[False].sum(), dists.loc[True].sum()],index=[False, True])
        dists = pd.concat([dists, total]).T
        cols = pd.MultiIndex.from_product([cols+['Total'], ['False', 'True']])
        dists.columns = cols

        if save: dists.to_csv(save_path, index=False)
        return dists
    
class PrepDataCYTO(PrepData):
    """Prepare data for cytopenia model training/prediction
    """
    def __init__(self):
        super().__init__()
        self.main_dir = f'{root_path}/{cyto_folder}'
        self.transfusion_cols = ['H_hemoglobin_transfusion_date', 'H_platelet_transfusion_date', 
                                 'ED_hemoglobin_transfusion_date', 'ED_platelet_transfusion_date']
        self.datetime_cols = self.transfusion_cols + ['visit_date', 'next_visit_date']
        self.chemo_dtypes = {col: str for col in self.transfusion_cols + ['curr_morph_cd', 'lhin_cd']}
    
    def extra_norm_cols(self):
        return ['chemo_interval', 'cycle_length']
    
    def read_main_blood_count_data(self, blood_type, chemo_df):
        df = pd.read_csv(f'{self.main_dir}/data/{blood_type}.csv')

        # turn string of numbers columns into integer column 
        df.columns = df.columns.astype(int)
        max_day = df.columns[-1]

        # get baseline and target blood counts
        df['baseline_blood_count'] = chemo_df[f'baseline_{blood_type}_count']
        df['regimen'] = chemo_df['regimen']
        df['chemo_interval'] = chemo_df['chemo_interval']
        cycle_lengths = dict(chemo_df[['regimen', 'cycle_length']].values)
        result = {}
        for regimen, regimen_group in tqdm.tqdm(df.groupby(['regimen'])):
            cycle_length = int(cycle_lengths[regimen])
            for chemo_interval, group in regimen_group.groupby(['chemo_interval']):
                # forward fill blood counts from a days before to the day after administration
                day = int(min(chemo_interval, cycle_length)) # get the min between actual vs expected days until next chemo
                if day == max_day:
                    ffill_window = range(day-2, day+1)
                    end = day
                else:  
                    ffill_window = range(day-1,day+2)
                    end = day+1
                group['target_blood_count'] = numpy_ffill(group[ffill_window])
                result.update(group['target_blood_count'].to_dict())
        df['target_blood_count'] = pd.Series(result)
        
        mask = df['baseline_blood_count'].notnull() & df['target_blood_count'].notnull()
        df = df[mask]
        return df

    def get_main_blood_count_data(self, chemo_df):
        data = {bt: self.read_main_blood_count_data(bt, chemo_df) for bt in blood_types}

        # keep only rows where all blood types are present
        n_indices = data['neutrophil'].index
        h_indices = data['hemoglobin'].index
        p_indices = data['platelet'].index
        keep_indices = n_indices[n_indices.isin(h_indices) & n_indices.isin(p_indices)]
        data = {blood_type: data[blood_type].loc[keep_indices] for blood_type in blood_types}
        return data
    
    def load_data(self):
        df = pd.read_csv(f'{self.main_dir}/data/model_data.csv', dtype=self.chemo_dtypes)
        for col in self.datetime_cols: df[col] = pd.to_datetime(df[col])
        return df
    
    def get_data(self, include_first_date=False, missing_thresh=None, verbose=False):
        # extract and organize data for model input and target labels
        """
        NBC - neutrophil blood count
        HBC - hemoglobin blood count
        PBC - platelet blood count

        input:                               -->        MODEL         -->            target:
        symptoms, body functionality,                                                NBC on next admin
        laboratory test, HB/PB transfusion, GF given,                                HBC on next admin
        chemo interval, chemo length,                                                PBC on next admin
        chemo cycle, days since starting chemo,
        regimen, visit month, immediate new regimen,
        line of therapy, intent of systemic treatment, 
        local health network, age, sex, immigrant, 
        english speaker, body surface area,
        cancer type/location, features missingness
        
        Args:
            include_first_date (bool): include first ever visit date for each patient
        """
        # extract data
        chemo_df = self.load_data()
        main_blood_count_data = self.get_main_blood_count_data(chemo_df)
        df = chemo_df.loc[main_blood_count_data['neutrophil'].index] # indices are same for all blood types
        for blood_type, blood_count_data in main_blood_count_data.items():
            df[f'target_{blood_type}_count'] = blood_count_data['target_blood_count']

        # impute growth factor feature per regimen 
        # if majority of sessions where patients over 65 taking regimen X takes growth factor, 
        # set all sessions of regimen X as taking growth factor
        over_65 = df[df['age'] >= 65]
        regimen_count = over_65['regimen'].value_counts()
        regimen_with_gf_count = over_65.loc[df['ODBGF_given'], 'regimen'].value_counts()
        impute_regimens = regimen_count.index[regimen_with_gf_count / regimen_count > 0.5]
        df.loc[df['regimen'].isin(impute_regimens), 'ODBGF_given'] = True
        if verbose: logging.info(f"All sessions for the regimens {impute_regimens.tolist()} will assume " +
                                 "ODB growth factors were administered")

        # get blood transfusion features
        for col in self.transfusion_cols:
            bt = col.split('_')[1]
            new_col = f"{bt}_transfusion"
            if new_col not in df: df[new_col] = False

            # if transfusion occurs from day -5 to day 3, flag feature as true
            earliest_date = df['visit_date'] - pd.Timedelta('5 days') 
            latest_date = df['visit_date'] + pd.Timedelta('3 days') 
            df[new_col] |= df[col].between(earliest_date, latest_date)
            if verbose: logging.info(f"{df[new_col].sum()} sessions with {col.replace('_', ' ').replace('date', '')}feature")

            # if transfusion occurs from day 4 to 3 days after next chemo administration
            # set target label (low blood count) as positive via setting target counts to 0
            earliest_date = df['visit_date'] + pd.Timedelta('4 days') 
            latest_date = df['next_visit_date'] + pd.Timedelta('3 days') 
            mask = df[col].between(earliest_date, latest_date)
            if verbose: logging.info(f"{mask.sum()} sessions will be labeled as positive for low {bt} blood count " +
                                     f"regardless of actual target {bt} blood count")
            df.loc[mask, f'target_{bt}_count'] = 0

        # convert visit date feature to cyclical features
        df = self.get_visit_date_feature(df, include_first_date=include_first_date)
        
        # fill null values with 0
        df = self.fill_missing_feature(df)
        
        drop_cols = ['visit_date', 'next_visit_date'] + self.transfusion_cols
        df = df.drop(columns=drop_cols)
        if missing_thresh is not None: 
            df = self.drop_features_with_high_missingness(df, missing_thresh, verbose=verbose)
        
        # create features for missing entries
        df = self.get_missingness_feature(df)

        return df
    
    def regression_to_classification(self, target):
        """
        Convert regression labels (last blood count value) to 
        classification labels (if last blood count value is below corresponding threshold)
        """
        for blood_type, blood_info in blood_types.items():
            target[blood_info['cytopenia_name']] = target[f'target_{blood_type}_count'] < blood_info['cytopenia_threshold']
            target = target.drop(columns=[f'target_{blood_type}_count'])
        return target
    
class PrepDataEDHD(PrepData):   
    """
    ED - Emergency Department visits
    H - Hospitalizations
    D - Deaths
    Prepare data for acute care use (ED/H) or death (D) model training/prediction
    """
    def __init__(self, adverse_event):
        super().__init__()
        if adverse_event not in {'acu', 'death'}: 
            raise ValueError('advese_event must be either acu (acute case use) or death')
        self.adverse_event = adverse_event
        if self.adverse_event == 'acu':
            self.main_dir = f'{root_path}/{acu_folder}'
        elif self.adverse_event == 'death':
            self.main_dir = f'{root_path}/{death_folder}'
        self.event_dates = pd.DataFrame() # store the event dates (keep it separate from the main output data)
    
    def extra_norm_cols(self):
        return ['num_prior_EDs', 'num_prior_Hs',
                'days_since_prev_H', 'days_since_prev_ED']
    
    def load_event_data(self, event='H'):
        df = pd.read_csv(f'{self.main_dir}/data/{event}_dates.csv')
        df['arrival_date'] = pd.to_datetime(df['arrival_date'])
        df['depart_date'] = pd.to_datetime(df['depart_date'])
        df = df.set_index('chemo_idx')
        return df

    def get_event_data(self, df, target_keyword, event='H', create_targets=True):
        event_dates = self.load_event_data(event=event)
        event_cause_cols = event_map[event]['event_cause_cols']

        # create the features - number of days since previous event occured, their causes, 
        #                       and number of events prior to visit
        features = event_dates[event_dates['feature_or_target'] == 'feature']
        col = f'days_since_prev_{event}'
        df.loc[features.index, col] = (df.loc[features.index, 'visit_date'] - features['arrival_date']).dt.days
        df[col] = df[col].fillna(df[col].max()) # fill rows where patients had no prev event with the max value
        col = f'num_prior_{event}s'
        df.loc[features.index, col] = features[col]
        for cause in event_cause_cols:
            df[f'prev_{cause}'] = False # initialize
            df.loc[features.index, f'prev_{cause}'] = features[cause]

        if create_targets:
            # create the targets - event within x days after visit date
            days = target_keyword.split('_')[-1] # Assuming target keyword is of the form _within_Xdays
            targets = event_dates[event_dates['feature_or_target'] == 'target']
            days_since_chemo_visit = targets['arrival_date'] - df.loc[targets.index, 'visit_date']
            # e.g. (within 14 days) if chemo visit is on Nov 1, a positive example is when event occurs between 
            #                       Nov 3rd to Nov 14th. We do not include the day of chemo visit and the day after
            targets[event] = days_since_chemo_visit.between(pd.Timedelta('1 days'), 
                                                            pd.Timedelta(days), inclusive=False)
            targets = targets[targets[event]]
            for col in event_cause_cols+[event]:
                df[col+target_keyword] = False # initialize
                df.loc[targets.index, col+target_keyword] = targets[col]

            # store the event dates
            self.event_dates.loc[targets.index, f'next_{event}_date'] = targets['arrival_date']
            
        return df
            
    def get_death_data(self, df, target_keyword):
        df['D_date'] = pd.to_datetime(df['D_date'])
        days_until_d = df['D_date'] - df['visit_date']

        # remove rows with negative days until death
        df = df[~(days_until_d < pd.Timedelta('0 days'))]

        # e.g. (within 14 days) if chemo visit is on Nov 1, a positive example is when event occurs between 
        #                       Nov 3rd to Nov 14th. We do not include the day of chemo visit and the day after
        for day in [14, 30, 90, 180, 365]:
            days = f'{day}d'
            df[f'{days} {target_keyword}'] = days_until_d.between(pd.Timedelta('1 days'), 
                                                          pd.Timedelta(days), inclusive=False)
        df[target_keyword] = days_until_d.notnull()
        
        # store the death dates
        self.event_dates['D_date'] = df['D_date']
        df = df.drop(columns=['D_date'])
        return df
            
    def remove_inpatients(self, df):
        inpatient_indices = np.load(f'{self.main_dir}/data/inpatient_indices.npy')
        df = df.drop(index=inpatient_indices)
        return df
    
    def test_visit_dates_sorted(self, df):
        # make sure chemo visit dates are sorted for each patient
        for ikn, group in tqdm.tqdm(df.groupby('ikn')):
            assert all(group['visit_date'] == group['visit_date'].sort_values())
    
    def load_data(self):
        df = pd.read_csv(f'{self.main_dir}/data/model_data.csv', dtype={'curr_morph_cd': str, 'lhin_cd': str})
        df = df.set_index('index')
        df.index.name = ''
        df['visit_date'] = pd.to_datetime(df['visit_date'])
        # self.test_visit_dates_sorted(df)  
        return df
    
    def get_data(self, target_keyword, missing_thresh=None, exclude_zero_observation_visits=False, verbose=False):
        """
        input:                               -->        MODEL         -->            target:
        symptoms, body functionality,                                                D Events within next 3-x days
        regimen, visit month, days since starting chemo,          
        chemo cycle, immediate new regimen,
        line of therapy, intent of systemic treatment,                               OR
        local health network, age, sex, immigrant,                                   
        english speaker, body surface area,                                          ED/H Events within next 3-x days
        cancer type/location, laboratory test,                                       ED/H event causes
        num of prior ED/H events
        days since prev ED/H events,
        cause for prev ED/H events,
        features missingness
        
        Args:
            missing_thresh (int or None): remove features with over ``missing_thresh``% missing values (e.g. 80% missing values)
            exclude_zero_observation_visits (bool): remove chemo visits with no blood work/labaratory test observations
        """
        df = self.load_data()
        self.event_dates['visit_date'] = df['visit_date'] # store the chemo visit dates (will be dropped from df later)
        
        if self.adverse_event == 'acu':
            # get event features and targets
            for event in ['H', 'ED']: 
                df = self.get_event_data(df, target_keyword, event=event)
            # create target column for acute care (ACU = ED + H) and treatment related acute care (TR_ACU = TR_ED + TR_H)
            df['ACU'+target_keyword] = df['ED'+target_keyword] | df['H'+target_keyword] 
            df['TR_ACU'+target_keyword] = df['TR_ED'+target_keyword] | df['TR_H'+target_keyword] 
            # remove errorneous inpatients
            df = self.remove_inpatients(df)
        elif self.adverse_event == 'death':
            # get event features
            for event in ['H', 'ED']: 
                df = self.get_event_data(df, None, event=event, create_targets=False)
            # get death targets
            df = self.get_death_data(df, target_keyword)
        
        # remove sessions where no observations were measured
        if exclude_zero_observation_visits:
            mask = df[observation_cols].isnull().all(axis=1)
            df = df[~mask]
            
        # fill null values with 0, max value, or most frequent value
        df = self.fill_missing_feature(df)

        # reduce sparse matrix by replacing rare col entries with less than 6 patients with 'Other'
        cols = ['regimen', 'curr_morph_cd', 'curr_topog_cd']
        df = replace_rare_col_entries(df, cols, verbose=verbose)
        
        # convert visit date features as cyclical features
        df = self.get_visit_date_feature(df)
        
        # Drop columns
        df = df.drop(columns=['visit_date'])
        if missing_thresh is not None: 
            df = self.drop_features_with_high_missingness(df, missing_thresh, verbose=verbose)
            
        # create features for missing entries
        df = self.get_missingness_feature(df)
            
        return df

class PrepDataCAN(PrepData):
    """
    CAN - Cisplatin-Associated Nephrotoxicity
    AKI - Acute Kdiney Injury
    CKD - Chronic Kidney Disease
    
    Prepare data for AKI or CKD model training/prediction
    """
    def __init__(self, adverse_event):
        super().__init__()
        if adverse_event not in {'aki', 'ckd'}: 
            raise ValueError('advese_event must be either aki (acute kidney injury) or ckd (chronic kidney disease)')
        self.adverse_event = adverse_event
        self.main_dir = f'{root_path}/{can_folder}'
        self.datetime_cols = ['visit_date', 'next_visit_date']
    
    def extra_norm_cols(self):
        return ['chemo_interval', 'cisplatin_dosage']
    
    def get_creatinine_data(self, df, verbose=False, remove_over_thresh=False):
        """
        Args:
            remove_over_thresh (bool): 
                remove sessions where baseline creatinine count is over SCr_max_threshold (e.g. 1.5mg/dL)
        """
        scr = pd.read_csv(f'{self.main_dir}/data/serum_creatinine.csv')
        scr.columns = scr.columns.astype(int)

        base_scr = 'baseline_creatinine_count'

        # get serum creatinine measurement taken within the month before visit date
        # if multiple values, take value closest to index date / prev visit via forward filling
        df[base_scr] = numpy_ffill(scr[range(-30,1)]) # NOTE: this overwrites the prev baseline_creatinine_count
        
        # exclude sessions where any of the baseline and peak creatinine measurements are missing
        mask = df[base_scr].notnull()
        if verbose: 
            logging.info(f'Removing {sum(~mask)} sessions where any of the baseline ' + \
                         f'creatinine measurements are missing')
        df = df[mask]
        
        if remove_over_thresh:
            # exclude sessions where baseline creatinine count is over the threshold
            mask = df[base_scr] > SCr_max_threshold
            if verbose: 
                logging.info(f'Removing {sum(mask)} sessions where baseline creatinine levels ' + \
                             f'were above {SCr_max_threshold} umol/L (1.5mg/dL)')
            df = df[~mask]

        if self.adverse_event == 'aki':
            # get highest creatinine value within 28 days after the visit date
            # or up to next chemotherapy administration, whichever comes first
            peak_scr = 'SCr_peak'
            for chemo_interval, group in df.groupby('chemo_interval'):
                index = group.index
                within_days = min(int(chemo_interval), 28)
                df.loc[index, peak_scr] = scr.loc[index, range(1, within_days+1)].max(axis=1)
        
            # exclude sessions where any of thepeak creatinine measurements are missing
            mask = df[peak_scr].notnull()
            if verbose: 
                logging.info(f'Removing {sum(~mask)} sessions where any of the ' + \
                             f'peak creatinine measurements are missing')
            df = df[mask]
        
            # get rise / fold increase in serum creatinine from baseline to peak measurements
            df['SCr_rise'] = df[peak_scr] - df[base_scr]
            df['SCr_fold_increase'] = df[peak_scr] / df[base_scr]

        elif self.adverse_event == 'ckd':
            next_scr = 'next_SCr_count'
            mask = df[next_scr].notnull()
            if verbose: 
                logging.info(f'Removing {sum(~mask)} sessions where any of the next ' + \
                             f'creatinine levels are missing')
            df = df[mask]
            
            # get estimated glomerular filtration rate (eGFR)
            df = get_eGFR(df, col=base_scr, prefix='baseline_')
            df = get_eGFR(df, col=next_scr, prefix='next_')
            df['eGFR_fold_increase'] = df['next_eGFR'] / df['baseline_eGFR']
        
        return df
    
    def load_data(self):
        df = pd.read_csv(f'{self.main_dir}/data/model_data.csv', dtype={'curr_morph_cd': str, 'lhin_cd': str})
        for col in self.datetime_cols: df[col] = pd.to_datetime(df[col])
        return df

    def get_data(self, include_first_date=False, missing_thresh=None, verbose=False):
        # extract and organize data for model input and target labels
        """
        input:                               -->        MODEL         -->            target:
        symptoms, body functionality,                                                CKD/AKI within 22 days or before next chemo admin
        laboratory test, dialysis, diabetes, hypertension,                                         
        regimen, visit month, chemo interval, chemo cycle,
        dyas since last chemo, days since starting chemo,
        line of therapy, intent of systemic treatment, 
        local health network, age, sex, immigrant, 
        english speaker, body surface area,
        cancer type/location, features missingness
        """
        df = self.load_data()
        df = self.get_creatinine_data(df, verbose=verbose)
        df = self.get_visit_date_feature(df, include_first_date=include_first_date) # convert visit date features as cyclical features
        df = self.fill_missing_feature(df) # fill null values with 0
        
        # reduce sparse matrix by replacing rare col entries with less than 6 patients with 'Other'
        # cols = ['regimen', 'curr_morph_cd', 'curr_topog_cd']
        # df = replace_rare_col_entries(df, cols, verbose=verbose)
        
        df = df.drop(columns=self.datetime_cols)
        if missing_thresh is not None: 
            df = self.drop_features_with_high_missingness(df, missing_thresh, verbose=verbose)
        
        # create features for missing entries
        df = self.get_missingness_feature(df)
            
        return df
    
    def regression_to_classification(self, target):
        """
        Convert regression labels (e.g. SCr rise, SCr fold increase) to classification labels 
        (e.g. if SCr rise is above corresponding threshold, indicating C-AKI (Cisplatin-associated Acute Kidney Injury))
        """
        cols = target.columns
        if self.adverse_event == 'aki':
            target['AKI_stage1'] = (target['SCr_rise'] >= SCr_rise_threshold) | target['SCr_fold_increase'].between(1.5, 2)
            target['AKI_stage2'] = target['SCr_fold_increase'].between(2, 3, inclusive='right')
            target['AKI_stage3'] = (target['SCr_rise'] >= SCr_rise_threshold2) | (target['SCr_fold_increase'] > 3)
        elif self.adverse_event == 'ckd':
            target['CKD_stage2'] = target['next_eGFR'].between(60, 90, inclusive='left')
            target['CKD_stage3a'] = target['next_eGFR'].between(45, 60, inclusive='left') | (target['eGFR_fold_increase'] <= 0.6)
            target['CKD_stage3b'] = target['next_eGFR'].between(30, 45, inclusive='left')
            target['CKD_stage4'] = target['next_eGFR'].between(15, 29, inclusive='left')
        target = target.drop(columns=cols)
        return target
