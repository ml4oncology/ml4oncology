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
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
import numpy as np

from src.config import (
    root_path, cyto_folder, acu_folder, can_folder, death_folder,
    INTENT, BSA,
    observation_cols, observation_change_cols, symptom_cols, 
    blood_types, event_map,
    SCr_max_threshold, SCr_rise_threshold, SCr_rise_threshold2
)
from src.utility import (
    get_nmissing, 
    get_eGFR,
    numpy_ffill, 
    replace_rare_categories, 
    split_and_parallelize
)

import logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s', 
    datefmt='%I:%M:%S'
)

class Imputer:
    """Imputate Missing Data by Mean, Mode, or Median
    """
    def __init__(self):
        self.impute_cols = {
            'mean': observation_cols.copy() + observation_change_cols.copy(), 
            'most_frequent': symptom_cols.copy(),
            'median': ['neighborhood_income_quintile']
        }
        self.imputer = {'mean': None, 'most_frequent': None, 'median': None}
        self.bsa_mean = None # body surface area mean for each sex
        
    def impute_body_surface_area(self, data):
        # replace missing body surface area with the mean based on sex
        mask = (data['sex'] == 1) | (data['sex'] == 'F')
        
        if self.bsa_mean is None:
            self.bsa_mean = {
                'female': data.loc[mask, BSA].mean(), 
                'male': data.loc[~mask, BSA].mean()
            }
            
        data.loc[mask, BSA] = data.loc[mask, BSA].fillna(self.bsa_mean['female'])
        data.loc[~mask, BSA] = data.loc[~mask, BSA].fillna(self.bsa_mean['male'])

        return data

    def impute(self, data):
        data = self.impute_body_surface_area(data)
        
        # loop through the mean, mode, and median imputer
        for strategy, imputer in self.imputer.items():
            cols = self.impute_cols[strategy]
            # use only the columns that exist in the data
            cols = list(set(cols).intersection(data.columns))
            
            if imputer is None:
                # create the imputer and impute the data
                imputer = SimpleImputer(strategy=strategy) 
                data[cols] = imputer.fit_transform(data[cols])
                self.imputer[strategy] = imputer # save the imputer
            else:
                # use existing imputer to impute the data
                data[cols] = imputer.transform(data[cols])
        return data
        
class PrepData:
    """Prepare the data for model training"""
    def __init__(self):
        self.imp = Imputer()
        # keep event dates, separate from the main output data
        self.event_dates = pd.DataFrame()
        self.datetime_cols = ['visit_date']
        self.datetime_cols += [f'{col}_survey_date' for col in symptom_cols]
        self.norm_cols = [
            'age', BSA, 'chemo_cycle', 'days_since_last_chemo',
            'days_since_starting_chemo', 'line_of_therapy', 
            'neighborhood_income_quintile', 'visit_month_cos', 'visit_month_sin',
            'years_since_immigration'
        ] 
        self.norm_cols += observation_cols + observation_change_cols + symptom_cols
        self.main_dir = None
        self.scaler = None # normalizer
        self.clip_thresh = None # outlier clippers
        
    def prepare_features(
        self, 
        df, 
        drop_cols=None, 
        missing_thresh=None, 
        reduce_sparsity=False, 
        treatment_intents=None, 
        regimens=None,
        first_course_treatment=False, 
        verbose=True
    ):
        """
        Args:
            missing_thresh (int): Remove features with over a X percentage of 
                missing values. If None, no features will be removed based on 
                missingness.
            treatment_intents (list): A sequence of characters representing 
                intent of systemic treatment to keep. Characters include:
                A - adjuvant
                C - curative
                N - neoadjuvant
                P - palliative
                If None, all intent of systemic treatment is kept.
            regimens (list): A sequence of regimens (str) to keep. If None, all
                treatment regimens are kept.
            first_course_treatment (bool): If True, keep only first course 
                treatments of a new line of therapy
        """
        if drop_cols is None: drop_cols = []
            
        # convert visit date feature to cyclical features
        df = self.get_visit_date_feature(df)
        
        # fill null values with 0 or max value
        df = self.fill_missing_feature(df)
        
        if treatment_intents:
            df = self.filter_treatment_catgories(
                df, treatment_intents, catcol=INTENT, verbose=verbose
            )
            if len(treatment_intents) == 1: drop_cols.append(INTENT)
                
        if regimens:
            df = self.filter_treatment_catgories(
                df, regimens, catcol='regimen', verbose=verbose
            )
            if len(regimens) == 1: drop_cols.append('regimen')
        
        if first_course_treatment: 
            df = self.get_first_course_treatments(df, verbose=verbose)

        # save patient's first visit date 
        # NOTE: It may be the case that a patient's initial chemo sessions may 
        # be excluded by filtering procedures down the pipeline, making the 
        # date a patient first appeared in the final dataset much later than 
        # their actual first chemo visit date. This may cause discrepancies 
        # when creating cohorts that are split by patient's first visit date.
        # We have decided to use the date a patient first appeared in the 
        # final dataset instead of a patient's actual first visit date (before 
        # any filtering)
        first_visit_date = df.groupby('ikn')['visit_date'].min()
        self.event_dates['first_visit_date'] = df['ikn'].map(first_visit_date)
        
        # save the date columns before dropping them
        for col in self.datetime_cols: 
            self.event_dates[col] = pd.to_datetime(df[col])
        
        # drop features and date columns
        df = df.drop(columns=drop_cols+self.datetime_cols)
        if missing_thresh is not None: 
            df = self.drop_features_with_high_missingness(
                df, missing_thresh, verbose=verbose
            )
        
        # create features for missing entries
        df = self.get_missingness_feature(df)
        
        if reduce_sparsity:
            # reduce sparse matrix by replacing rare categories with less 
            # than 6 patients with 'Other'
            cols = ['regimen', 'curr_morph_cd', 'curr_topog_cd']
            df = replace_rare_categories(df, cols, verbose=verbose)
        
        # remove filtered rows (by intent, first course, inpatient, 
        # death dates, etc) from event dates
        self.event_dates = self.event_dates.loc[df.index]
        
        return df
    
    def split_data(
        self, 
        data, 
        target_keyword='target', 
        impute=True, 
        normalize=True, 
        clip=True, 
        split_date=None, 
        verbose=True):
        """Split data into training, validation and test sets based on patient 
        ids (and optionally split temporally based patient first visit dates)
        
        Optionally clip, impute, and normalize the datasets
        
        Args:
            split_date (str): Date on which to split the data temporally. 
                String format: 'YYYY-MM-DD' (e.g. 2017-06-30).
                Train/val set (aka development cohort) will contain all patients
                whose first chemo session date was on or before split_date.
                Test set (aka test cohort) will contain all patients whose first 
                chemo session date was after split_date. 
                If None, data won't be split temporally.
        """
        # create training, validation, testing set
        train_data, valid_data, test_data = self.create_splits(
            data, split_date=split_date, verbose=verbose
        )
        
        # IMPORTANT: always make sure train data is done first for clipping, 
        # imputing, and scaling
        if clip:
            # Clip the outliers based on the train data quantiles
            train_data = self.clip_outliers(train_data.copy())
            valid_data = self.clip_outliers(valid_data.copy())
            test_data = self.clip_outliers(test_data.copy())

        if impute:
            # Impute missing data based on the train data mode/median/mean
            train_data = self.imp.impute(train_data.copy())
            valid_data = self.imp.impute(valid_data.copy())
            test_data = self.imp.impute(test_data.copy())
            
        if normalize:
            # Scale the data based on the train data distribution
            train_data = self.normalize_data(train_data.copy())
            valid_data = self.normalize_data(valid_data.copy())
            test_data = self.normalize_data(test_data.copy())

        # split into input features and target labels
        cols = train_data.columns
        feature_cols = cols[~cols.str.contains(target_keyword)].drop('ikn')
        target_cols = cols[cols.str.contains(target_keyword)]
        X_train = train_data[feature_cols]
        X_valid = valid_data[feature_cols]
        X_test = test_data[feature_cols]
        Y_train = self.convert_labels(train_data[target_cols])
        Y_valid = self.convert_labels(valid_data[target_cols])
        Y_test = self.convert_labels(test_data[target_cols])
        
        return X_train, X_valid, X_test, Y_train, Y_valid, Y_test
    
    def get_label_distribution(
        self, 
        Y_train, 
        Y_valid=None, 
        Y_test=None, 
        save_path=''
    ):
        dists, cols = [], []
        def update_distribution(Y, split):
            dists.append(Y.apply(pd.value_counts))
            cols.append(split)
        update_distribution(Y_train, 'Train')
        if Y_valid is not None: update_distribution(Y_valid, 'Valid')
        if Y_test is not None: update_distribution(Y_test, 'Test')
    
        dists = pd.concat(dists)
        total = pd.DataFrame(
            [dists.loc[False].sum(), dists.loc[True].sum()],
            index=[False, True]
        )
        dists = pd.concat([dists, total]).T
        cols = pd.MultiIndex.from_product([cols+['Total'], ['False', 'True']])
        dists.columns = cols

        if save_path: dists.to_csv(save_path, index=False)
        return dists
    
    def clip_outliers(
        self, 
        data, 
        cols=None, 
        lower_percentile=0.001, 
        upper_percentile=0.999
    ):
        """Clip the upper and lower percentiles for the columns indicated below
        """
        if not cols:
            cols = observation_cols + observation_change_cols + [BSA]
            # use only the columns that exist in the data
            cols = list(set(cols).intersection(data.columns))
        
        if self.clip_thresh is None:
            percentiles = [lower_percentile, upper_percentile]
            self.clip_thresh = data[cols].quantile(percentiles)
            
        data[cols] = data[cols].clip(
            lower=self.clip_thresh.loc[lower_percentile], 
            upper=self.clip_thresh.loc[upper_percentile], 
            axis=1
        )
        return data

    def dummify_data(self, data):
        # convert sex from a nominal column into a binary column
        data['sex'] = data['sex'] == 'F'
        # one-hot encode categorical columns
        return pd.get_dummies(data)
        
    def normalize_data(self, data):
        # use only the columns that exist in the data
        norm_cols = list(set(self.norm_cols).intersection(data.columns))
        
        if self.scaler is None:
            self.scaler = StandardScaler()
            data[norm_cols] = self.scaler.fit_transform(data[norm_cols])
        else:
            data[norm_cols] = self.scaler.transform(data[norm_cols])
        return data
    
    def create_cohort(self, df, split_date, verbose=True):
        """Create the development and testing cohort 
        by partitioning on split_date
        """
        mask = self.event_dates['first_visit_date'] <= split_date
        dev_cohort, test_cohort = df[mask], df[~mask]
        if verbose:
            logging.info("Development Cohort: "
                         f"NSessions={len(dev_cohort)}. "
                         f"NPatients={dev_cohort['ikn'].nunique()}. "
                         "Contains all patients that had their first visit on "
                         f"or before {split_date}")
            logging.info("Testing Cohort: "
                         f"NSessions={len(test_cohort)}. "
                         f"NPatients={test_cohort['ikn'].nunique()}. "
                         "Contains all patients that had their first visit "
                         f"after {split_date}")
        return dev_cohort, test_cohort
    
    def create_splits(self, data, split_date=None, verbose=True):
        if split_date is None: 
            # split data based on patient ids (60-20-20 split)
            # create training set
            gss = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
            patient_ids = data['ikn']
            train_idxs, test_idxs = next(gss.split(data, groups=patient_ids))
            train_data = data.iloc[train_idxs]
            test_data = data.iloc[test_idxs]

            # crate validation and testing set
            gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
            patient_ids = test_data['ikn']
            valid_idxs, test_idxs = next(gss.split(test_data, groups=patient_ids))

            valid_data = test_data.iloc[valid_idxs]
            test_data = test_data.iloc[test_idxs]
        else:
            # split data temporally based on patients first visit date
            train_data, test_data = self.create_cohort(
                data, split_date, verbose=verbose
            )
                
            # create validation set from train data (80-20 split)
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            patient_ids = train_data['ikn']
            train_idxs, valid_idxs = next(gss.split(train_data, groups=patient_ids))
            valid_data = train_data.iloc[valid_idxs]
            train_data = train_data.iloc[train_idxs]

        # sanity check - make sure there are no overlap of patients in the splits
        assert(not set.intersection(set(train_data['ikn']), 
                                    set(valid_data['ikn']), 
                                    set(test_data['ikn'])))

        if verbose:
            logging.info('Size of splits: '
                         f'Train:{len(train_data)}, '
                         f'Val:{len(valid_data)}, '
                         f'Test:{len(test_data)}')
            logging.info("Number of patients: "
                         f"Train:{train_data['ikn'].nunique()}, "
                         f"Val:{valid_data['ikn'].nunique()}, "
                         f"Test:{test_data['ikn'].nunique()}")
        
        return train_data, valid_data, test_data
    
    def get_visit_date_feature(self, df):
        # convert to cyclical features
        month = df['visit_date'].dt.month - 1
        df['visit_month_sin'] = np.sin(2*np.pi*month/12)
        df['visit_month_cos'] = np.cos(2*np.pi*month/12)
        return df
    
    def get_first_course_treatments(self, df, verbose=False):
        """Keep the very first treatment session for each line of therapy for 
        each patient
        """
        keep_idxs = split_and_parallelize(df, _first_course_worker)
        if verbose:
            logging.info(f'Removing {len(df)-len(keep_idxs)} sessions that are '
                         'not first course treatment of a new line of therapy')
        df = df.loc[keep_idxs]
        return df
    
    def fill_missing_feature(self, df):
        cols = df.columns
        for col in ['num_prior_EDs', 'num_prior_Hs']:
            if col in cols: df[col] = df[col].fillna(0)
        for col in ['days_since_last_chemo', 'chemo_interval']:
            if col in cols: df[col] = df[col].fillna(df[col].max())
        return df
    
    def get_missingness_feature(self, df):
        cols_with_nan = df.columns[df.isnull().any()]
        df[cols_with_nan + '_is_missing'] = df[cols_with_nan].isnull()
        return df
    
    def drop_features_with_high_missingness(
        self, 
        df, 
        missing_thresh, 
        verbose=False
    ):
        nmissing = get_nmissing(df)
        mask = nmissing['Missing (%)'] > missing_thresh
        exclude_cols = nmissing.index[mask].tolist()
        if verbose: 
            logging.info(f'Dropping the following features for missingness over '
                         f'{missing_thresh}%: {exclude_cols}')
        return df.drop(columns=exclude_cols)
    
    def filter_treatment_catgories(self, df, keep_cats, catcol, verbose=False):
        """Keep treatment sessions belonging to a specific category of a given 
        categorial column
        
        Args:
            keep_cats (list): A sequence of categories (str) to keep
            catcol (str): Name of categorical column
        """
        mask = df[catcol].isin(keep_cats)
        if verbose: 
            logging.info(f'Removing {sum(~mask)} sessions in which {catcol} '
                         f'is not {keep_cats}')
        df = df[mask]
        return df
    
    def load_data(self, dtypes=None):
        if dtypes is None: dtypes={'curr_morph_cd': str, 'lhin_cd': str}
        df = pd.read_csv(f'{self.main_dir}/data/model_data.csv', dtype=dtypes)
        for col in self.datetime_cols: 
            df[col] = pd.to_datetime(df[col])
        # self.test_visit_dates_sorted(df)  
        return df
    
    def test_visit_dates_sorted(self, df):
        # make sure chemo visit dates are sorted for each patient
        for ikn, group in tqdm(df.groupby('ikn')):
            assert all(group['visit_date'] == group['visit_date'].sort_values())
    
    def convert_labels(self, target):
        """You can overwrite this to do custom operations"""
        return target
    
class PrepDataCYTO(PrepData):
    """Prepare data for cytopenia model training/prediction"""
    def __init__(self):
        super().__init__()
        self.main_dir = f'{root_path}/{cyto_folder}'
        self.transfusion_cols = [
            'H_hemoglobin_transfusion_date', 'H_platelet_transfusion_date', 
            'ED_hemoglobin_transfusion_date', 'ED_platelet_transfusion_date'
        ]
        self.datetime_cols += self.transfusion_cols + ['next_visit_date']
        self.norm_cols += ['cycle_length']
        
        str_dtype_cols = self.transfusion_cols + ['curr_morph_cd', 'lhin_cd']
        self.chemo_dtypes = {col: str for col in str_dtype_cols}
    
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
        result, indices = [], []
        for regimen, regimen_group in tqdm(df.groupby('regimen')):
            cycle_length = int(cycle_lengths[regimen])
            for chemo_interval, group in regimen_group.groupby('chemo_interval'):
                # get the min between actual vs expected days until next chemo
                day = int(min(chemo_interval, cycle_length))
                # forward fill blood counts from a days before to the day 
                # after expected/actual administration
                ffill_window = (range(day-2, day+1) if day == max_day
                                else range(day-1,day+2))
                result += numpy_ffill(group[ffill_window]).tolist()
                indices += group.index.tolist()
        df.loc[indices, 'target_blood_count'] = result

        mask = (df['baseline_blood_count'].notnull() & 
                df['target_blood_count'].notnull())
        df = df[mask]
        return df

    def get_main_blood_count_data(self, chemo_df):
        data = {bt: self.read_main_blood_count_data(bt, chemo_df) 
                for bt in blood_types}

        # keep only rows where all blood types are present
        n_indices = data['neutrophil'].index
        h_indices = data['hemoglobin'].index
        p_indices = data['platelet'].index
        mask = n_indices.isin(h_indices) & n_indices.isin(p_indices)
        keep_indices = n_indices[mask]
        data = {blood_type: data[blood_type].loc[keep_indices] 
                for blood_type in blood_types}
        return data
    
    def process_blood_transfusion(self, df, verbose=False):
        for col in self.transfusion_cols:
            bt = col.split('_')[1]
            new_col = f"{bt}_transfusion"
            if new_col not in df: df[new_col] = False

            # if transfusion occurs from day -5 to day 0, flag feature as true
            earliest_date = df['visit_date'] - pd.Timedelta('5 days') 
            latest_date = df['visit_date']
            df[new_col] |= df[col].between(earliest_date, latest_date)
            if verbose: 
                logging.info(f"{df[new_col].sum()} sessions with "
                             f"{col.replace('_', ' ').replace('date', '')} "
                             "feature")

            # if transfusion occurs from day 4 to 3 days after next chemo 
            # administration, set target label (low blood count) as positive 
            # via setting target counts to 0
            earliest_date = df['visit_date'] + pd.Timedelta('4 days') 
            latest_date = df['next_visit_date'] + pd.Timedelta('3 days') 
            mask = df[col].between(earliest_date, latest_date)
            if verbose: 
                logging.info(f"{mask.sum()} sessions will be labeled as positive "
                             f"for low {bt} blood count regardless of actual "
                             f"target {bt} blood count")
            df.loc[mask, f'target_{bt}_count'] = 0
        return df
    
    def get_data(self, verbose=False, **kwargs):
        """Extract and organize data to create input features and target labels
        
        input features:
        Demographic - age, sex, immigrant, english speaker, local health network,
            body surface area, neighborhood income quintile
        Treatment - regimen, visit month, days since last chemo, 
            days since starting chemo, immediate new regimen, chemo cycle, 
            line of therapy, intent of systemic treatment, 
            HB/PB transfusion given, GF given,
        Cancer - cancer type, cancer location
        Questionnaire - symptoms, body functionality
        Test - laboratory test, change since last laboratory test
        Other - features missingness
        
        target:
        Neutropenia, Thrombocytopenia, Anemia prior to next session
        
        Args:
            **kwargs (dict): the parameters of PrepData.prepare_features
        """
        # extract data
        chemo_df = self.load_data(dtypes=self.chemo_dtypes)
        main_blood_count_data = self.get_main_blood_count_data(chemo_df)
        
        # combine data
        # indices are same for all blood types
        df = chemo_df.loc[main_blood_count_data['neutrophil'].index]
        for bt, blood_count_data in main_blood_count_data.items():
            df[f'target_{bt}_count'] = blood_count_data['target_blood_count']
        df = self.process_blood_transfusion(df, verbose=verbose)
        df = self.prepare_features(
            df, drop_cols=['chemo_interval'], verbose=verbose, **kwargs
        )
        return df
    
    def convert_labels(self, target):
        """Convert regression labels (last blood count value) to classification
        labels (if last blood count value is below corresponding threshold)
        """
        for bt, blood_info in blood_types.items():
            cyto_name = blood_info['cytopenia_name']
            cyto_thresh = blood_info['cytopenia_threshold']
            target[cyto_name] = target[f'target_{bt}_count'] < cyto_thresh
            target = target.drop(columns=[f'target_{bt}_count'])
        return target
    
class PrepDataEDHD(PrepData):   
    """Prepare data for acute care use (ED/H) or death (D) model 
    training/prediction
    
    ED - Emergency Department visits
    H - Hospitalizations
    D - Deaths
    """
    def __init__(self, adverse_event, target_days=None):
        super().__init__()
        if adverse_event not in {'acu', 'death'}: 
            raise ValueError('advese_event must be either acu (acute case use) '
                             'or death')
        self.adverse_event = adverse_event
        if self.adverse_event == 'acu':
            self.main_dir = f'{root_path}/{acu_folder}'
        elif self.adverse_event == 'death':
            self.main_dir = f'{root_path}/{death_folder}'
            self.datetime_cols += ['PCCS_date', 'last_seen_date']
            self.target_days = ([14, 30, 90, 180, 365] if target_days is None 
                                else target_days)
        self.norm_cols += [
            'num_prior_EDs', 'num_prior_Hs', 'days_since_prev_H', 
            'days_since_prev_ED'
        ]
    
    def load_event_data(self, event='H'):
        df = pd.read_csv(f'{self.main_dir}/data/{event}_dates.csv')
        df['arrival_date'] = pd.to_datetime(df['arrival_date'])
        df['depart_date'] = pd.to_datetime(df['depart_date'])
        df = df.set_index('chemo_idx')
        return df

    def get_event_data(self, df, target_keyword, event='H', create_targets=True):
        if self.event_dates.empty:
            # intialize the indices of self.event_dates, which stores all dates
            # (chemo visit date, survey dates, etc)
            self.event_dates['index'] = df.index
            self.event_dates = self.event_dates.set_index('index')
            
        event_dates = self.load_event_data(event=event) # ED/H event dates
        event_cause_cols = event_map[event]['event_cause_cols']

        # create the features
        features = event_dates[event_dates['feature_or_target'] == 'feature']
        idxs = features.index
        # Feature 1: Number of days since previous event occured
        col = f'days_since_prev_{event}'
        date_diff = df.loc[idxs, 'visit_date'] - features['arrival_date']
        df.loc[idxs, col] = (date_diff).dt.days
        # fill rows where patients had no prev event with the max value
        df[col] = df[col].fillna(df[col].max())
        # Feature 2: Number of events prior to visit
        col = f'num_prior_{event}s'
        df.loc[features.index, col] = features[col]
        # Feature 3: Cause of prev event occurence
        for cause in event_cause_cols:
            df[f'prev_{cause}'] = False # initialize
            df.loc[features.index, f'prev_{cause}'] = features[cause]

        if create_targets:
            targets = event_dates[event_dates['feature_or_target'] == 'target']
            idxs = targets.index
            
            # store the event dates
            col = f'next_{event}_date'
            self.event_dates.loc[idxs, col] = targets['arrival_date']
            
            # create the event targets - event within x days after visit date
            days_until_event = targets['arrival_date'] - df.loc[idxs, 'visit_date']
            days = target_keyword.split('_')[-1] # Assumes the form target_within_Xdays
            targets[event] = days_until_event < pd.Timedelta(days)
            targets = targets[targets[event]]
            for col in event_cause_cols+[event]:
                df[col+target_keyword] = False # initialize
                df.loc[targets.index, col+target_keyword] = targets[col]
            
        return df
            
    def get_death_data(self, df, target_keyword, verbose=False):
        df['D_date'] = pd.to_datetime(df['D_date'])
        
        # Allow ample time for follow up of death by filtering out sessions 
        # after max death date minus max target days. For example, if we only 
        # collected death dates up to July 31 2021 (max death date), and max 
        # target days is 365 days, we should only consider treatments up to 
        # July 30 2020. If we considered treatments on June 2021, its
        # impossible to get the ground truth of whether patient died within 365
        # days, since we stopped collecting death dates a month later
        min_date = df['D_date'].max() - pd.Timedelta(days=max(self.target_days))
        mask = df['visit_date'] > min_date
        if verbose: 
            logging.info(f'Removing {sum(mask)} sessions whose date occured '
                         f'after {min_date} to allow ample follow up time of '
                         'death')
        df = df[~mask]

        # create the death targets
        days_until_d = df['D_date'] - df['visit_date']
        for day in self.target_days:
            days = f'{day}d'
            df[f'{days} {target_keyword}'] = days_until_d < pd.Timedelta(days)
            
        # store the death dates
        self.event_dates['D_date'] = df['D_date']
        
        df = df.drop(columns=['D_date'])
        return df
            
    def remove_inpatients(self, df, verbose=False):
        inpatient_indices = np.load(f'{self.main_dir}/data/inpatient_indices.npy')
        df = df.drop(index=inpatient_indices)
        if verbose: logging.info(f'Dropped {len(inpatient_indices)} samples of inpatients')
        return df
    
    def remove_immediate_events(self, df, event, verbose=False):
        """Remove sessions in which patients experienced events in less than 
        2 days. We do not care about model identifying immediate events
        
        E.g. (within 14 days) if chemo visit is on Nov 1, a positive example is
             when event occurs between Nov 3rd to Nov 14th. We do not include 
             the day of chemo visit and the day after
        """
        if event not in {'death', 'H', 'ED'}:
            raise ValueError('event must be either death, H or ED')
            
        col = 'D_date' if event == 'death' else f'next_{event}_date'
        days_until_event = self.event_dates.loc[df.index, col] - df['visit_date']
        mask = days_until_event < pd.Timedelta('2 days')
        df = df[~mask]
        if verbose: 
            logging.info(f'Removing {sum(mask)} sessions in which patients '
                         f'experienced {event} in less than 2 days')
            if event == 'death': 
                mask = days_until_event < pd.Timedelta('0 days')
                logging.info(f'Among those sessions, {sum(mask)} had patients '
                             f'experience {event} BEFORE the session')
                
        return df
    
    def get_data(self, target_keyword, verbose=False, **kwargs):
        """Extract and organize data to create input features and target labels
        
        input features:
        Demographic - age, sex, immigrant, world region of birth, english speaker, 
            residence area density, local health network, body surface area, 
            neighborhood income quintile
        Treatment - regimen, visit month, days since last chemo, 
            days since starting chemo, chemo cycle, immediate new regimen, 
            line of therapy, *intent of systemic treatment
        Cancer - cancer type, cancer location
        Questionnaire - symptoms, body functionality
        Test - laboratory test, change since last laboratory test
        Acute Care - num of prior ED/H events, days since prev ED/H events, 
            cause for prev ED/H events,
        Other - features missingness
        
        target:
        D or ED/H Events within next 3-x days
        
        * input features with asterik next to them could be dropped
        (in PrepData.prepare_features)
        
        Args:
            **kwargs (dict): the parameters of PrepData.prepare_features
        """
        df = self.load_data()
        if self.adverse_event == 'acu':
            # get event features and targets
            for event in ['H', 'ED']: 
                df = self.get_event_data(df, target_keyword, event=event)
            # create target column for acute care (ACU = ED + H) and treatment 
            # related acute care (TR_ACU = TR_ED + TR_H)
            add_tk = lambda x: x + target_keyword
            df[add_tk('ACU')] = df[add_tk('ED')] | df[add_tk('H')]
            df[add_tk('TR_ACU')] = df[add_tk('TR_ED')] | df[add_tk('TR_H')]
            # remove errorneous inpatients
            df = self.remove_inpatients(df, verbose=verbose)
            # remove immediate events
            for event in ['H', 'ED']: 
                df = self.remove_immediate_events(df, event, verbose=verbose)
        elif self.adverse_event == 'death':
            # get event features
            for event in ['H', 'ED']: 
                df = self.get_event_data(
                    df, None, event=event, create_targets=False
                )
            # get death targets
            df = self.get_death_data(df, target_keyword, verbose=verbose)
            # remove immediate deaths
            df = self.remove_immediate_events(
                df, self.adverse_event, verbose=verbose
            )
        
        df = self.prepare_features(df, verbose=verbose, **kwargs)
            
        return df

class PrepDataCAN(PrepData):
    """Prepare data for AKI or CKD model training/prediction
    
    CAN - Cisplatin-Associated Nephrotoxicity
    AKI - Acute Kdiney Injury
    CKD - Chronic Kidney Disease
    """
    def __init__(self, adverse_event):
        super().__init__()
        if adverse_event not in {'aki', 'ckd'}: 
            raise ValueError('advese_event must be either aki (acute kidney '
                             'injury) or ckd (chronic kidney disease)')
        self.adverse_event = adverse_event
        self.main_dir = f'{root_path}/{can_folder}'
        self.datetime_cols += ['next_visit_date']
        self.norm_cols += ['cisplatin_dosage', 'baseline_eGFR']
    
    def get_creatinine_data(self, df, verbose=False, remove_over_thresh=False):
        """
        Args:
            remove_over_thresh (bool): If True, remove sessions where baseline 
                creatinine count is over SCr_max_threshold (e.g. 1.5mg/dL)
        """
        scr = pd.read_csv(f'{self.main_dir}/data/serum_creatinine.csv')
        scr.columns = scr.columns.astype(int)

        base_scr = 'baseline_creatinine_count'

        # Get serum creatinine measurement taken within the month before visit 
        # date. If there are multiple values, take the value closest to index 
        # date / prev visit via forward filling
        # NOTE: this overwrites the prev baseline_creatinine_count
        df[base_scr] = numpy_ffill(scr[range(-30,1)])
        
        # get estimated glomerular filtration rate (eGFR) prior to treatment
        df = get_eGFR(df, col=base_scr, prefix='baseline_')
        
        # exclude sessions where any of the baseline creatinine measurements 
        # are missing
        mask = df[base_scr].notnull()
        if verbose: 
            logging.info(f'Removing {sum(~mask)} sessions where any of the '
                         'baseline creatinine measurements are missing')
        df = df[mask]
        
        if remove_over_thresh:
            # exclude sessions where baseline creatinine count is over the 
            # threshold
            mask = df[base_scr] > SCr_max_threshold
            if verbose: 
                logging.info(f'Removing {sum(mask)} sessions where baseline '
                             f'creatinine levels were above {SCr_max_threshold} '
                             'umol/L (1.5mg/dL)')
            df = df[~mask]

        if self.adverse_event == 'aki':
            # get highest creatinine value within 28 days after the visit date
            # or up to next chemotherapy administration, whichever comes first
            peak_scr = 'SCr_peak'
            for chemo_interval, group in df.groupby('chemo_interval'):
                index = group.index
                min_day = min(int(chemo_interval), 28)
                days = range(1, min_days+1)
                df.loc[index, peak_scr] = scr.loc[index, days].max(axis=1)
        
            # exclude sessions where any of the peak creatinine measurements 
            # are missing
            mask = df[peak_scr].notnull()
            if verbose: 
                logging.info(f'Removing {sum(~mask)} sessions where any of the '
                             'peak creatinine measurements are missing')
            df = df[mask]
        
            # get rise / fold increase in serum creatinine from baseline to 
            # peak measurements
            df['SCr_rise'] = df[peak_scr] - df[base_scr]
            df['SCr_fold_increase'] = df[peak_scr] / df[base_scr]

        elif self.adverse_event == 'ckd':
            next_scr = 'next_SCr_count'
            mask = df[next_scr].notnull()
            if verbose: 
                logging.info(f'Removing {sum(~mask)} sessions where any of the '
                             'next creatinine levels are missing')
            df = df[mask]
            
            # get estimated glomerular filtration rate (eGFR) after treatment
            df = get_eGFR(df, col=next_scr, prefix='next_')
        
        return df

    def get_data(self, verbose=False, **kwargs):
        """Extract and organize data to create input features and target labels
        
        input features:
        Demographic - age, sex, immigrant, english speaker, 
            local health network, body surface area, 
            neighborhood income quintile, diabetes, hypertension
        Treatment - regimen, visit month, days since last chemo, 
            days since starting chemo, chemo cycle, immediate new regimen, 
            line of therapy, intent of systemic treatment
        Cancer - cancer type, cancer location
        Questionnaire - symptoms, body functionality
        Test - laboratory test, change since last laboratory test, eGFR, 
            cisplatin dosage
        Other - features missingness
        
        target:
        CKD/AKI within 22 days or before next chemo admin
        
        Args:
            **kwargs (dict): the parameters of PrepData.prepare_features
        """
        df = self.load_data()
        df = self.get_creatinine_data(df, verbose=verbose)
        df = self.prepare_features(
            df, drop_cols=['chemo_interval'], verbose=verbose, **kwargs
        )
        return df
    
    def convert_labels(self, target):
        """Convert regression labels (e.g. SCr rise, SCr fold increase) to 
        classification labels (e.g. if SCr rise is above corresponding 
        threshold, then patient has C-AKI (Cisplatin-associated Acute Kidney 
        Injury))
        """
        cols = target.columns
        if self.adverse_event == 'aki':
            rise = target['SCr_rise']
            growth = target['SCr_fold_increase']
            target['AKI'] = (rise >= SCr_rise_threshold) | (growth >= 1.5)
            # target['AKI_stage1'] = (rise >= SCr_rise_threshold) | (growth >= 1.5)
            # target['AKI_stage2'] = growth >= 2
            # target['AKI_stage3'] = (rise >= SCr_rise_threshold2) | (growth >= 3)
        elif self.adverse_event == 'ckd':
            # target['CKD (stage2)'] = target['next_eGFR'] < 90
            target['CKD'] = target['next_eGFR'] < 60 # stage 3a and higher
            target['CKD (stage3b)'] = target['next_eGFR'] < 45 # stage 3b and higher
            target['CKD (stage4)'] = target['next_eGFR'] < 30 # stage 4 and higher
            
            # if dialysis was administered from 90 days to 6 months after chemo
            # visit, set all target labels as positive
            target[target['dialysis']] = True
            
        target = target.drop(columns=cols)
        return target

# Worker Helpers
def _first_course_worker(partition):
    keep_idxs = []
    for ikn, group in partition.groupby('ikn'):
        lot = group['line_of_therapy']
        new_lot = ~(lot == lot.shift())
        keep_idxs += group.index[new_lot].tolist()
    return keep_idxs
