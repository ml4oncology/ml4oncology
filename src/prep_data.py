"""
========================================================================
© 2023 Institute for Clinical Evaluative Sciences. All rights reserved.

TERMS OF USE:
##Not for distribution.## This code and data is provided to the user solely for its own non-commercial use by individuals and/or not-for-profit corporations. User shall not distribute without express written permission from the Institute for Clinical Evaluative Sciences.

##Not-for-profit.## This code and data may not be used in connection with profit generating activities.

##No liability.## The Institute for Clinical Evaluative Sciences makes no warranty or representation regarding the fitness, quality or reliability of this code and data.

##No Support.## The Institute for Clinical Evaluative Sciences will not provide any technological, educational or informational support in connection with the use of this code and data.

##Warning.## By receiving this code and data, user accepts these terms, and uses the code and data, solely at its own risk.
========================================================================
"""
from functools import partial

from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
import numpy as np

from src import logger
from src.config import (
    root_path, cyto_folder, acu_folder, can_folder, death_folder,
    DATE, INTENT, BSA, 
    observation_cols, observation_change_cols, symptom_cols, next_scr_cols,
    blood_types, cytopenia_grades, event_map,
    SCr_max_threshold, SCr_rise_threshold, SCr_rise_threshold2
)
from src.utility import (
    get_nmissing, 
    get_eGFR,
    make_log_msg,
    numpy_ffill, 
    replace_rare_categories, 
    split_and_parallelize,
)

MAXDATE, MINDATE = pd.Timestamp.max, pd.Timestamp.min

class Imputer:
    """Impute missing data by mean, mode, or median
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
    
class OneHotEncoder:
    """One-hot encode (OHE) categorical data.
    
    Create separate indicator columns for each unique category and assign binary
    values of 1 or 0 to indicate the category's presence.
    """
    def __init__(self):
        # NOTE: if you change the ordering, you will need to retrain the models
        self.encode_cols = [
            'regimen', 'cancer_topog_cd', 'cancer_morph_cd', 'lhin_cd', INTENT,
            'world_region_of_birth'
        ]
        self.final_columns = None # the final feature names after OHE
        
        # function to get the indicator columns of a categorical feature
        self.get_indcols = lambda cols, feat: cols[cols.str.startswith(feat)]
        
    def separate_regimen_sets(self, data, verbose=True):
        """For regimens, some categories represent a regimen set (different 
        treatments taken concurrently/within a short span of time). Create
        separate columns for each regimen in the set. If those columns already
        exists, combine them together (which is for most cases, except for 
        monoclonal antibody treatments e.g beva, tras, pert).
        
        E.g.
        regimen_A  regimen_B  regimen_A+B+C
        True       False      False 
        False      True       False 
        False      False      True    
        False      False      False
        
        regimen_A  regimen_B  regimen_C
        True       False      False
        False      True       False
        True       True       True
        False      False      False
        
        Examples of A+B+C
        1. folfiri+pacl(w)+ramu
        2. folfiri+mfolfox6+beva
        """
        cols = self.get_indcols(data.columns, 'regimen_')
        drop_cols = []
        for col, mask in data[cols].items():
            regimens = col.replace('regimen_', '').split('+')
            if len(regimens) == 1: 
                continue

            for regimen in regimens:
                regimen_col = f'regimen_{regimen}'
                data[regimen_col] = data.get(regimen_col, 0) | mask
            drop_cols.append(col)
        data = data.drop(columns=drop_cols)
        
        if verbose:
            mask = self.get_indcols(data.columns, 'regimen_').isin(cols)
            msg = (f'Separated and dropped {len(drop_cols)} treatment set '
                   f'indicator columns, and added {sum(~mask)} new treatment '
                   'indicator columns')
            logger.info(msg)
        
        return data
        
    def encode(self, data, separate_regimen_sets=True, verbose=True):
        """
        Args:
            separate_regimen_sets (bool): If True, convert the regimen set 
                (regimen_A+B) indicator column into its individual regimen 
                (regimen_A, regimen_B) indicator columns (combine if already 
                exists)
        """
        # convert sex from a categorical column into a binary column
        data['sex'] = data['sex'] == 'F'
        
        # one-hot encode categorical columns
        # use only the columns that exist in the data
        cols = [col for col in self.encode_cols if col in data.columns]
        data = pd.get_dummies(data, columns=cols)
        
        if separate_regimen_sets:
            data = self.separate_regimen_sets(data, verbose=verbose)
        
        if self.final_columns is None:
            self.final_columns = data.columns
            return data
        
        # reassign any indicator columns that did not exist in final columns
        # as other
        for feature in cols:
            indicator_cols = self.get_indcols(data.columns, feature)
            extra_cols = indicator_cols.difference(self.final_columns)
            if extra_cols.empty: continue
            
            if verbose:
                count = data[extra_cols].sum()
                msg = (f'Reassigning the following {feature} indicator columns '
                       f'that did not exist in train set as Other:\n{count}')
                logger.info(msg)
                
            other_col = f'{feature}_Other'
            if other_col not in data: data[other_col] = 0
            data[other_col] |= data[extra_cols].any(axis=1).astype(int)
            data = data.drop(columns=extra_cols)
            
        # fill in any missing columns
        missing_cols = self.final_columns.difference(data.columns)
        # use concat instead of data[missing_cols] = 0 to prevent perf warning
        data = pd.concat([
            data,
            pd.DataFrame(0, index=data.index, columns=missing_cols)
        ], axis=1)
        
        return data
        
class PrepData:
    """Prepare the data for model training"""
    def __init__(self, target_keyword='target'):
        self.target_keyword = target_keyword
        self.imp = Imputer()
        self.ohe = OneHotEncoder()
        # keep event dates, separate from the main output data
        self.event_dates = pd.DataFrame()
        self.norm_cols = [
            'age', BSA, 'chemo_cycle', 'days_since_last_chemo',
            'days_since_starting_chemo', 'line_of_therapy', 
            'neighborhood_income_quintile', 'visit_month_cos', 'visit_month_sin',
            'years_since_immigration'
        ] + observation_cols + observation_change_cols + symptom_cols
        self.clip_cols = [BSA] + observation_cols + observation_change_cols
        self.main_dir = None
        self.scaler = None # normalizer
        self.clip_thresh = None # outlier clippers
        
    def prepare_features(
        self, 
        df, 
        drop_cols=None, 
        missing_thresh=None, 
        reduce_sparsity=True, 
        treatment_intents=None, 
        regimens=None,
        first_course_treatment=False, 
        include_comorbidity=False,
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
            include_comorbidity (bool): If True, include features indicating if
                patients had hypertension and/or diabetes
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
        
        if include_comorbidity:
            for col in ['hypertension', 'diabetes']:
                df[col] = df[f'{col}_diag_date'] < df[DATE]

        # save patient's first visit date 
        # NOTE: It may be the case that a patient's initial chemo sessions may 
        # be excluded by filtering procedures down the pipeline, making the 
        # date a patient first appeared in the final dataset much later than 
        # their actual first chemo visit date. This may cause discrepancies 
        # when creating cohorts that are split by patient's first visit date.
        # We have decided to use the date a patient first appeared in the 
        # final dataset instead of a patient's actual first visit date (before 
        # any filtering)
        first_visit_date = df.groupby('ikn')[DATE].min()
        self.event_dates[f'first_{DATE}'] = df['ikn'].map(first_visit_date)
        
        # save the date columns before dropping them
        date_cols = df.columns[df.columns.str.endswith('_date')].tolist()
        for col in date_cols: 
            self.event_dates[col] = pd.to_datetime(df[col])
        
        # drop features and date columns
        df = df.drop(columns=drop_cols+date_cols)
        if missing_thresh is not None: 
            df = self.drop_features_with_high_missingness(
                df, missing_thresh, verbose=verbose
            )
        
        # create features for missing entries
        df = self.get_missingness_feature(df)
        
        if reduce_sparsity:
            # recategorize regimens/cancers/lhins with less than 6 patients to 
            # other
            cols = ['regimen', 'cancer_morph_cd', 'cancer_topog_cd', 'lhin_cd']
            df = replace_rare_categories(df, cols, verbose=verbose)
        
        return df
    
    def split_and_transform_data(
        self, 
        data, 
        remove_immediate_events=False,
        split_date=None, 
        verbose=True,
        **kwargs
    ):
        """Split data into training, validation and test sets based on patient 
        ids (and optionally split temporally based patient first visit dates).
        
        Optionally transform (one-hot encode, clip, impute, normalize) the data.
        
        Args:
            split_date (str): Date on which to split the data temporally. 
                String format: 'YYYY-MM-DD' (e.g. 2017-06-30).
                Train/val set (aka development cohort) will contain all patients
                whose first chemo session date was on or before split_date.
                Test set (aka test cohort) will contain all patients whose first
                chemo session date was after split_date. 
                If None, data won't be split temporally.
            **kwargs: keyword arguments fed into PrepData.transform_data
        """
        # create training, validation, testing set
        train_data, valid_data, test_data = self.create_splits(
            data, split_date=split_date, verbose=verbose
        )
        
        if remove_immediate_events:
            # Remove sessions where event occured immediately afterwards on the
            # train and valid set ONLY
            train_data = self.remove_immediate_events(train_data, verbose=verbose)
            valid_data = self.remove_immediate_events(valid_data, verbose=verbose)
        
        # IMPORTANT: always make sure train data is done first for one-hot 
        # encoding, clipping, imputing, scaling
        train_data = self.transform_data(
            train_data, data_name='training', verbose=verbose, **kwargs
        )
        valid_data = self.transform_data(
            valid_data, data_name='validation',verbose=verbose, **kwargs
        )
        test_data = self.transform_data(
            test_data, data_name='testing', verbose=verbose, **kwargs
        )
            
        # create a split column and combine the data for convenienceR
        train_data[['cohort', 'split']] = ['Development', 'Train']
        valid_data[['cohort', 'split']] = ['Development', 'Valid']
        test_data[['cohort', 'split']] = 'Test'
        data = pd.concat([train_data, valid_data, test_data])
        self.event_dates = self.event_dates.loc[data.index]
            
        # split into input features, output labels, and tags
        tag_cols = ['ikn', 'cohort', 'split']
        cols = data.columns.drop(tag_cols)
        mask = cols.str.contains(self.target_keyword)
        target_cols, feature_cols = cols[mask], cols[~mask]
        
        X = data[feature_cols]
        Y = self.convert_labels(data[target_cols].copy())
        tag = data[tag_cols]
        
        if verbose:
            df = pd.DataFrame({
                'Number of sessions': tag.groupby('split').apply(len), 
                'Number of patients': tag.groupby('split')['ikn'].nunique()}
            ).T
            df['Total'] = df.sum(axis=1)
            logger.info(f'\n{df.to_string()}')
        
        return X, Y, tag
    
    def transform_data(
        self, 
        data,
        one_hot_encode=True,
        clip=True, 
        impute=True, 
        normalize=True, 
        ohe_kwargs=None,
        data_name=None,
        verbose=True
    ):
        """Transform (one-hot encode, clip, impute, normalize) the data.
        
        Args:
            ohe_kwargs (dict): a mapping of keyword arguments fed into 
                OneHotEncoder.encode
                
        IMPORTANT: always make sure train data is done first before valid
        or test data
        """
        if ohe_kwargs is None: ohe_kwargs = {}
        if data_name is None: data_name = 'the'
        
        if one_hot_encode:
            # One-hot encode categorical data
            if verbose: logger.info(f'One-hot encoding {data_name} data')
            data = self.ohe.encode(data, **ohe_kwargs)
            
        if clip:
            # Clip the outliers based on the train data quantiles
            data = self.clip_outliers(data)

        if impute:
            # Impute missing data based on the train data mode/median/mean
            data = self.imp.impute(data)
            
        if normalize:
            # Scale the data based on the train data distribution
            data = self.normalize_data(data)
            
        return data
    
    def get_label_distribution(
        self, 
        Y, 
        tag, 
        with_respect_to='sessions',
        save_path=''
    ):
        if with_respect_to == 'patients':
            dists = {}
            for split, group in Y.groupby(tag['split']):
                ikn = tag.loc[group.index, 'ikn']
                count = {True: group.apply(lambda mask: ikn[mask].nunique())}
                count[False] = ikn.nunique() - count[True] 
                dists[split] = pd.DataFrame(count).T
        elif with_respect_to == 'sessions':
            dists = {split: group.apply(pd.value_counts) 
                     for split, group in Y.groupby(tag['split'])}
        dists['Total'] = dists['Train'] + dists['Valid'] + dists['Test']
        dists = pd.concat(dists).T
        if save_path: dists.to_csv(save_path, index=False)
        return dists
    
    def clip_outliers(
        self, 
        data, 
        lower_percentile=0.001, 
        upper_percentile=0.999
    ):
        """Clip the upper and lower percentiles for the columns indicated below
        """
        # use only the columns that exist in the data
        cols = [col for col in self.clip_cols if col in data.columns]
        
        if self.clip_thresh is None:
            percentiles = [lower_percentile, upper_percentile]
            self.clip_thresh = data[cols].quantile(percentiles)
            
        data[cols] = data[cols].clip(
            lower=self.clip_thresh.loc[lower_percentile], 
            upper=self.clip_thresh.loc[upper_percentile], 
            axis=1
        )
        return data

    def normalize_data(self, data):
        # use only the columns that exist in the data
        norm_cols = [col for col in self.norm_cols if col in data.columns]
        
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
        mask = self.event_dates[f'first_{DATE}'] <= split_date
        dev_cohort, test_cohort = df[mask].copy(), df[~mask].copy()
        mask = self.event_dates.loc[dev_cohort.index, DATE] <= split_date
        if verbose:
            context = f' that occured after {split_date} in the development set'
            logger.info(make_log_msg(dev_cohort, mask, context=context))
        dev_cohort = dev_cohort[mask]
        if verbose:
            disp = lambda x: f"NSessions={len(x)}. NPatients={x.ikn.nunique()}"
            msg = (f"Development Cohort: {disp(dev_cohort)}. Contains all "
                   f"patients whose first visit was on or before {split_date}")
            logger.info(msg)
            msg = (f"Test Cohort: {disp(test_cohort)}. Contains all patients "
                   f"whose first visit was after {split_date}")
            logger.info(msg)
        return dev_cohort, test_cohort
    
    def create_split(self, df, test_size, random_state=42):
        """Split data based on patient ids"""
        gss = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        patient_ids = df['ikn']
        train_idxs, test_idxs = next(gss.split(df, groups=patient_ids))
        train_data = df.iloc[train_idxs].copy()
        test_data = df.iloc[test_idxs].copy()
        return train_data, test_data
    
    def create_splits(self, data, split_date=None, verbose=True):
        if split_date is None: 
            # create training, validation, and testing set (60-20-20 split)
            train_data, test_data = self.create_split(data, test_size=0.4)
            valid_data, test_data = self.create_split(test_data, test_size=0.5)
        else:
            # split data temporally based on patients first visit date
            train_data, test_data = self.create_cohort(
                data, split_date, verbose=verbose
            )
            # create validation set from train data (80-20 split)
            train_data, valid_data = self.create_split(train_data, test_size=0.2)

        # sanity check - make sure there are no overlap of patients in the splits
        assert(not set.intersection(set(train_data['ikn']), 
                                    set(valid_data['ikn']), 
                                    set(test_data['ikn'])))
        
        return train_data, valid_data, test_data
    
    def get_visit_date_feature(self, df):
        # convert to cyclical features
        month = df[DATE].dt.month - 1
        df['visit_month_sin'] = np.sin(2*np.pi*month/12)
        df['visit_month_cos'] = np.cos(2*np.pi*month/12)
        return df
    
    def get_first_course_treatments(self, df, verbose=False):
        """Keep the very first treatment session for each line of therapy for 
        each patient
        """
        keep_idxs = split_and_parallelize(df, _first_course_worker)
        if verbose:
            mask = df.index.isin(keep_idxs)
            context = (' that are not first course treatment of a new line of '
                       'therapy')
            logger.info(make_log_msg(df, mask, context=context))
        df = df.loc[keep_idxs]
        return df
    
    def fill_missing_feature(self, df):
        cols = df.columns
        for event in ['H', 'ED']:
            col = f'num_prior_{event}s_within_5_years'
            if col in cols: df[col] = df[col].fillna(0)
        for col in ['days_since_last_chemo']:
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
            msg = (f'Dropping the following features for missingness over '
                   f'{missing_thresh}%: {exclude_cols}')
            logger.info(msg)
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
            context = f' in which {catcol} is not {keep_cats}'
            logger.info(make_log_msg(df, mask, context=context))
        df = df[mask].copy()
        return df
    
    def load_data(self, dtypes=None):
        df = pd.read_parquet(f'{self.main_dir}/data/final_data.parquet.gzip')
        df['ikn'] = df['ikn'].astype(int)
        # self.test_visit_dates_sorted(df)  
        return df
    
    def test_visit_dates_sorted(self, df):
        # make sure chemo visit dates are sorted for each patient
        for ikn, group in tqdm(df.groupby('ikn')):
            assert all(group[DATE] == group[DATE].sort_values())
    
    def convert_labels(self, target):
        """You can overwrite this to do custom operations"""
        return target
    
    def remove_immediate_events(self, df, verbose=False):
        """You can overwrite this to do custom operations"""
        logger.warning('remove_immeidate_events have not been implemented yet!')
        return df
    
class PrepDataCYTO(PrepData):
    """Prepare data for cytopenia model training/prediction"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.main_dir = f'{root_path}/{cyto_folder}'
        self.norm_cols += ['cycle_length']
        
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
        df = self.load_data()
        df = self.get_target_date(df)
        df = self.get_target_blood_measurements(df, verbose=verbose)
        df = self.get_blood_transfusion_data(df, verbose=verbose)
        df = self.prepare_features(df, verbose=verbose, **kwargs)
        return df
    
    def get_target_date(self, df):
        """Compute the minimum between the anticipated and actual next visit 
        date
        """
        cycle_length = pd.to_timedelta(df['cycle_length'], unit='d')
        expect_next_date = df[DATE] + cycle_length
        actual_next_date = df[f'next_{DATE}']
        dates = [expect_next_date, actual_next_date]
        df['target_date'] = pd.concat(dates, axis=1).min(axis=1)
        return df

    def get_target_blood_measurements(self, df, verbose=True):
        missing_baseline_mask, missing_target_mask = False, False
        for bt in blood_types:
            df = self._get_target_blood_measurements(df, blood_type=bt)
            missing_baseline_mask |= df[f'baseline_{bt}_value'].isnull()
            missing_target_mask |= df[f'target_{bt}_value'].isnull()
        
        # keep only rows in which both baseline and target measurements for all
        # blood types are present
        # NOTE: I separated the masks in case I want to know each of its 
        # numbers in the future
        mask = missing_baseline_mask | missing_target_mask
        if verbose:
            context = (' that did not have baseline/target blood measurements '
                       'for the 3 blood types')
            logger.info(make_log_msg(df, mask, context=context))
        df = df[~mask]
        return df
    
    def _get_target_blood_measurements(self, df, blood_type='neutrophil'):
        bm = pd.read_parquet(f'{self.main_dir}/data/{blood_type}.parquet.gzip')
        bm.columns = bm.columns.astype(int)
        max_day = bm.columns[-1]
        
        bm['regimen'] = df['regimen']
        bm['days_until_next_visit'] = (df['target_date'] - df[DATE]).dt.days

        result, idxs = [], []
        for day, group in bm.groupby('days_until_next_visit'):
            # forward fill values from a day before to the day after target date
            adjustment = int(day == max_day) # prevent out of bound error
            ffill_window = range(day - 1, day + 2 - adjustment)
            result += numpy_ffill(group[ffill_window]).tolist()
            idxs += group.index.tolist()
        df.loc[idxs, f'target_{blood_type}_value'] = result
        return df
    
    def get_blood_transfusion_data(
        self, 
        df, 
        days_prior=5, 
        days_after=3, 
        adjust_targets=True, 
        verbose=True
    ):
        days_prior = pd.Timedelta(days=days_prior)
        days_after = pd.Timedelta(days=days_after)
        bt = pd.read_parquet(f'{self.main_dir}/data/blood_transfusion.parquet.gzip')
        for col in [DATE, 'target_date']: bt[col] = bt['chemo_idx'].map(df[col])

        # create the features
        feats = bt.query('feat_or_targ == "feature"')
        # Feature 1: Whether blood transfusion occured between x days before to
        # the day of treatment date
        mask = feats['transfusion_date'] >= feats[DATE] - days_prior
        feats = feats[mask]
        for blood_type, group in feats.groupby('type'):
            idxs = group['chemo_idx'].unique()
            new_col = f"{blood_type}_transfusion"
            df[new_col] = False
            df.loc[idxs, new_col] = True

        if not adjust_targets:
            return df

        # adjust the targets
        targs = bt.query('feat_or_targ == "target"')
        # if blood transfusion occured between 4 days after treatment date to 
        # x days after target date, set target label (low blood count) as
        # positive by setting target blood measurement to 0
        mask = targs['transfusion_date'] <= targs['target_date'] + days_after
        targs = targs[mask]
        for blood_type, group in targs.groupby('type'):
            idxs = group['chemo_idx'].unique()
            if verbose: 
                msg = (f"{len(idxs)} sessions will be labeled as positive for "
                       f"low {blood_type} count regardless of actual target "
                       f"{blood_type} count")
                logger.info(msg)
            df.loc[idxs, f'target_{blood_type}_value'] = 0
        return df
    
    def convert_labels(self, target, grade=2):
        """Convert regression labels (target blood value) to classification
        labels (if target blood value is below a threshold)
        """
        for bt, blood_info in blood_types.items():
            cyto_name = blood_info['cytopenia_name']
            cyto_thresh = cytopenia_grades[f'Grade {grade}'][cyto_name]
            target[cyto_name] = target.pop(f'target_{bt}_value') < cyto_thresh
        return target
    
class PrepDataEDHD(PrepData):   
    """Prepare data for acute care use (ED/H) or death (D) model 
    training/prediction
    
    ED - Emergency Department visits
    H - Hospitalizations
    D - Deaths
    """
    def __init__(self, adverse_event, target_days=None, **kwargs):
        super().__init__(**kwargs)
        if adverse_event not in ['acu', 'death']: 
            raise ValueError('advese_event must be either acu (acute case use) '
                             'or death')
        self.adverse_event = adverse_event
        if self.adverse_event == 'acu':
            self.main_dir = f'{root_path}/{acu_folder}'
        elif self.adverse_event == 'death':
            self.main_dir = f'{root_path}/{death_folder}'
            self.target_days = ([14, 30, 90, 180, 365] if target_days is None 
                                else target_days)
        self.norm_cols += [
            'num_prior_EDs_within_5_years', 'num_prior_Hs_within_5_years', 
            'days_since_prev_H', 'days_since_prev_ED'
        ]
        
        self.add_tk = lambda x: f'{x} {self.target_keyword}'
        
    def get_data(self, verbose=False, **kwargs):
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
                df = self.get_event_data(df, event=event)
            # create target column for acute care (ACU = ED + H) and treatment 
            # related acute care (TR_ACU = TR_ED + TR_H)
            acu = df[self.add_tk('ED')] | df[self.add_tk('H')]
            tr_acu = df[self.add_tk('TR_ED')] | df[self.add_tk('TR_H')]
            df[self.add_tk('ACU')], df[self.add_tk('TR_ACU')] = acu, tr_acu
            # remove errorneous inpatients
            df = self.remove_inpatients(df, verbose=verbose)
        elif self.adverse_event == 'death':
            # get event features
            for event in ['H', 'ED']: 
                df = self.get_event_data(df, event=event, create_targets=False)
            # get death targets
            df = self.get_death_data(df, verbose=verbose)
        
        df = self.prepare_features(df, verbose=verbose, **kwargs)
        return df

    def get_event_data(self, df, event='H', create_targets=True):
        event_df = self.load_event_data(event=event)
        event_cause_cols = event_map[event]['event_cause_cols']

        # create the features
        features = event_df.query('feat_or_targ == "feature"')
        idxs = features.index
        # Feature 1: Number of days since previous event occured
        col = f'days_since_prev_{event}'
        date_diff = df.loc[idxs, DATE] - features['date']
        df.loc[idxs, col] = (date_diff).dt.days
        # fill rows where patients had no prev event with the max value
        df[col] = df[col].fillna(df[col].max())
        # Feature 2: Number of events prior to visit
        col = f'num_prior_{event}s_within_5_years'
        df.loc[idxs, col] = features[col]
        # Feature 3: Cause of prev event occurence
        for cause in event_cause_cols:
            df[f'prev_{cause}'] = False # initialize
            df.loc[idxs, f'prev_{cause}'] = features[cause]

        if not create_targets:
            return df
        
        # create the targets
        targets = event_df.query('feat_or_targ == "target"').copy()
        idxs = targets.index

        # store the event dates
        col = f'next_{event}_date'
        df.loc[idxs, col] = targets['date']

        # create the event targets - event within x days after visit date
        days_until_event = targets['date'] - df.loc[idxs, DATE]
        days = self.target_keyword.split('_')[-2] # assumes the form within_X_days
        targets[event] = days_until_event < pd.Timedelta(days=int(days))
        targets = targets[targets[event]]
        for col in event_cause_cols+[event]:
            df[self.add_tk(col)] = False # initialize
            df.loc[targets.index, self.add_tk(col)] = targets[col]

        return df
            
    def get_death_data(self, df, verbose=False):
        # Allow ample time for follow up of death by filtering out sessions 
        # after max death date minus max target days. For example, if we only 
        # collected death dates up to July 31 2021 (max death date), and max 
        # target days is 365 days, we should only consider treatments up to 
        # July 30 2020. If we considered treatments on June 2021, its
        # impossible to get the ground truth of whether patient died within 365
        # days, since we stopped collecting death dates a month later
        min_date = df['death_date'].max() - pd.Timedelta(days=max(self.target_days))
        mask = df[DATE] > min_date
        if verbose: 
            context = (f' whose date occured after {min_date} to allow ample '
                       'follow up time of death')
            logger.info(make_log_msg(df, ~mask, context=context))
        df = df[~mask].copy()
        
        # remove ghost sessions (visits occured after death date)
        mask = df[DATE] > df['death_date']
        if verbose: 
            context = f' whose date occured AFTER death'
            logger.info(make_log_msg(df, ~mask, context=context))
        df = df[~mask].copy()

        # create the death targets
        days_until_d = df['death_date'] - df[DATE]
        for day in self.target_days:
            days = f'{day}d'
            df[self.add_tk(days)] = days_until_d < pd.Timedelta(days)
            
        return df
            
    def remove_inpatients(self, df, verbose=False):
        idxs = np.load(f'{self.main_dir}/data/inpatient_idxs.npy')
        if verbose: 
            mask = ~df.index.isin(idxs)
            logger.info(make_log_msg(df, mask, context=' of inpatients')) 
        df = df.drop(index=idxs)
        return df
    
    def load_event_data(self, event='H'):
        df = pd.read_parquet(f'{self.main_dir}/data/{event}.parquet.gzip')
        df = df.set_index('chemo_idx')
        return df
    
    def remove_immediate_events(self, df, verbose=False):
        """Remove sessions in which patients experienced events in less than 
        2 days. We do not care about model identifying immediate events
        
        E.g. (within 14 days) if chemo visit is on Nov 1, a positive example is
             when event occurs between Nov 3rd to Nov 14th. We do not include 
             the day of chemo visit and the day after
        """ 
        if self.adverse_event == 'death':
            df = self._remove_immediate_events(df, 'death', verbose=verbose)
        elif self.adverse_event == 'acu':
            for event in ['H', 'ED']:
                df = self._remove_immediate_events(df, event, verbose=verbose)
        return df
    
    def _remove_immediate_events(self, df, event, verbose=False):
        event_dates = self.event_dates.loc[df.index]
        col = 'death_date' if event == 'death' else f'next_{event}_date'
        days_until_event = event_dates[col] - event_dates[DATE]
        mask = days_until_event < pd.Timedelta('2 days')
        if verbose: 
            context = (f' in which patients experienced {event} in less than '
                       '2 days')
            logger.info(make_log_msg(df, ~mask, context=context)) 
        df = df[~mask]
        return df

class PrepDataCAN(PrepData):
    """Prepare data for AKI or CKD model training/prediction
    
    CAN - Cisplatin-Associated Nephrotoxicity
    AKI - Acute Kdiney Injury
    CKD - Chronic Kidney Disease
    """
    def __init__(self, adverse_event, **kwargs):
        super().__init__(**kwargs)
        if adverse_event not in ['aki', 'ckd']: 
            raise ValueError('advese_event must be either aki (acute kidney '
                             'injury) or ckd (chronic kidney disease)')
        self.adverse_event = adverse_event
        self.main_dir = f'{root_path}/{can_folder}'
        self.norm_cols += ['cisplatin_dosage', 'baseline_eGFR', 'cycle_length']
        self.clip_cols += ['cisplatin_dosage', 'baseline_eGFR']
        
    def get_data(
        self, 
        include_rechallenges=True,
        use_target_average=False,
        verbose=False, 
        **kwargs
    ):
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
            include_rechallenges (bool): If True, include cisplatin 
                rechallenges (treatment sessions after a 90+ day gap) for CKD
            use_target_average (bool): If True, use the average of the next two
                creatinine measurement when converting labels for CKD. If 
                second measurement does not exist, remove the sample.
            **kwargs (dict): the parameters of PrepData.prepare_features
        """
        df = self.load_data()
        df = self.get_creatinine_data(
            df, include_rechallenges=include_rechallenges, 
            use_target_average=use_target_average, verbose=verbose
        )
        df = self.prepare_features(df, verbose=verbose, **kwargs)
        return df
    
    def get_creatinine_data(
        self, 
        df, 
        include_rechallenges=True, 
        use_target_average=False,
        remove_over_thresh=False,
        verbose=False, 
    ):
        """
        Args:
            include_rechallenges (bool): If True, include cisplatin 
                rechallenges (treatment sessions after a 90+ day gap) for CKD
            use_target_average (bool): If True, use the average of the next two
                creatinine measurement when converting labels for CKD. If 
                second measurement does not exist, remove the sample.
            remove_over_thresh (bool): If True, remove sessions where baseline 
                creatinine is over SCr_max_threshold (e.g. 1.5mg/dL)
        """
        scr = pd.read_parquet(f'{self.main_dir}/data/creatinine.parquet.gzip')
        scr.columns = scr.columns.astype(int)
        base_scr = 'baseline_creatinine_value'
        
        # Get serum creatinine measurement taken within the month before visit 
        # date. If there are multiple values, take the value closest to index 
        # date / prev visit via forward filling
        # NOTE: this overwrites the prev baseline_creatinine_value
        df[base_scr] = numpy_ffill(scr[range(-30,1)])
        
        # get estimated glomerular filtration rate (eGFR) prior to treatment
        df = get_eGFR(df, col=base_scr, prefix='baseline_')

        if self.adverse_event == 'aki':
            # get highest creatinine value within 28 days after the visit date
            # or up to next chemotherapy administration, whichever comes first
            peak_scr = 'SCr_peak'
            intervals = (df[f'next_{DATE}'] - df[DATE]).dt.days
            intervals = intervals.clip(upper=28)
            for chemo_interval, group in intervals.groupby(intervals):
                idx = group.index
                days = range(1, int(chemo_interval)+1)
                df.loc[idx, peak_scr] = scr.loc[idx, days].max(axis=1)
        
            # exclude sessions without peak creatinine measurements
            mask = df[peak_scr].notnull()
            if verbose: 
                context = ' without peak creatinine values'
                logger.info(make_log_msg(df, mask, context=context))
            df = df[mask].copy()
        
            # get rise / fold increase in serum creatinine from baseline to 
            # peak measurements
            df['SCr_rise'] = df[peak_scr] - df[base_scr]
            df['SCr_fold_increase'] = df[peak_scr] / df[base_scr]

        elif self.adverse_event == 'ckd':
            next_scr = 'next_SCr_value'
            
            # exclude session without a valid future creatinine value and 
            # reassign the next creatinine value with the valid ones
            df = self.reassign_next_creatinine(
                df, include_rechallenges=include_rechallenges, verbose=verbose
            )
            
            if use_target_average:
                # exclude sessions without the seoncd future creatinine value
                mask = df[f'next_{next_scr}'].notnull()
                if verbose: 
                    context = ' without second future creatinine value'
                    logger.info(make_log_msg(df, mask, context=context))
                df = df[mask].copy()
                # take the average of the two future creatinine values
                df[next_scr] = df[[next_scr, f'next_{next_scr}']].mean(axis=1)
            
            # get estimated glomerular filtration rate (eGFR) after treatment
            df = get_eGFR(df, col=next_scr, prefix='next_')
            
            # avoid confusions
            del df[f'next_{next_scr}']
        
        # exclude sessions without baseline creatinine measurements
        mask = df[base_scr].notnull()
        if verbose: 
            context = ' without baseline creatinine values'
            logger.info(make_log_msg(df, mask, context=context))
        df = df[mask].copy()
        
        if remove_over_thresh:
            # exclude sessions where baseline creatinine is over the threshold
            mask = df[base_scr] > SCr_max_threshold
            if verbose: 
                context = (' where baseline creatinine levels were above '
                           f'{SCr_max_threshold} umol/L (1.5mg/dL)')
                logger.info(make_log_msg(df, mask, context=context))
            df = df[~mask].copy()
        
        return df
    
    def reassign_next_creatinine(
        self, 
        df, 
        include_rechallenges=True, 
        verbose=False
    ):
        """Reassign the next creatinine value (NCV)

        NCV is valid if there were no more cisplatin (treatment) administered 
        between the current session date and the NCV measurement date. Assign the 
        valid NCV to the prior sessions.
        
        If a 90+ day gap exist between sessions, and the earlier session does 
        not have valid NCV, that means there were no creatinine measurement
        within 2 years after that session, in which case target label would be
        unclear. So the earlier session and all sessions prior without a 90+ 
        day gap will be removed.

        NOTE: NCV is the closest creatinine measurement between 90 days to 2 years
        after treatment session

        Examples 1)
        treatment_date | NCV_date   | NCV  | valid_NCV
        2014-03-21       2014-06-01   50     60
        2014-04-01       2015-10-01   60     60
        2014-04-11       2015-10-01   60     60

        2015-11-21       2016-03-01   40     80
        2015-12-11       2016-06-01   80     80

        Examples 2)
        treatment_date | NCV_date   | NCV  | valid_NCV
        2014-03-21       2014-06-01   50     NaN
        2014-04-01       NaT          NaN    NaN
        2014-04-11       NaT          NaN    NaN

        2015-11-21       2016-03-01   40     80
        2015-12-11       2016-06-01   80     80

        Args:
            include_rechallenges (bool): If True, include sessions after the first 
                valid NCV
        """        
        worker = partial(
            _reassign_next_creatinine_worker, 
            include_rechallenges=include_rechallenges
        )
        result = split_and_parallelize(df, worker, processes=8)
        result = pd.DataFrame(result, columns=['index'] + next_scr_cols)
        result = result.set_index('index')
        if verbose: 
            mask = df.index.isin(result.index)
            context = ' during next creatinine reassignment'
            logger.info(make_log_msg(df, mask, context=context))
        df = df.loc[result.index]
        df[next_scr_cols] = result[next_scr_cols]
        df = df.sort_values(by=['ikn', DATE])
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
            next_eGFR = target['next_eGFR']
            # target['CKD (stage2)'] = next_eGFR < 90
            target['CKD'] = next_eGFR < 60 # stage 3a and higher
            target['CKD (stage3b)'] = next_eGFR < 45 # stage 3b and higher
            target['CKD (stage4)'] = next_eGFR < 30 # stage 4 and higher
            
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

def _reassign_next_creatinine_worker(partition, include_rechallenges=True):
    result = []
    for ikn, group in partition.groupby('ikn'):
        idxs = group.index
        mask = (group[f'next_{DATE}'].fillna(MAXDATE) > 
                group['next_SCr_obs_date'].fillna(MINDATE))
        valid_idxs = idxs[mask]

        prev_idx = idxs[0] - 1
        for cur_idx in valid_idxs:
            tmp = group.loc[cur_idx, next_scr_cols]
            if not np.isnan(tmp['next_SCr_value']):
                for i in range(prev_idx+1, cur_idx+1):
                    result.append([i]+tmp.tolist())
            if not include_rechallenges: break
            prev_idx = cur_idx
            
    return result
