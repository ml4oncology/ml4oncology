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
Module to create various summary tables
"""
import os

from tqdm import tqdm
import numpy as np
import pandas as pd

from src.conf_int import get_confidence_interval, compute_bootstrap_nadir_days
from src.config import (
    cancer_code_mapping,
    cytopenia_grades, 
    symptom_cols, 
    variable_groupings_by_keyword
)
from src.utility import (
    get_clean_variable_name, 
    get_clean_variable_names, 
    get_cyto_rates,
    get_units, 
    most_common_categories,
    twolevel, 
)

###############################################################################
# Base Class
###############################################################################
class SubgroupSummary:
    """Base class for summarizing / analyzing different population subgroups
    """
    def __init__(self, data, top_categories=None):
        """
        Args:
            top_categories (dict): mapping of categorical columns (str) and 
                their most common categories (list of str)
        """
        self.data = data
        if top_categories is None: top_categories = {}
        self.top_categories = top_categories
        self.N = self.data['ikn'].nunique()
        
    def age_summary(self, *args):
        for (low, high) in [(18, 64), (65, np.inf)]:
            mask = self.data['age'].between(low, high)
            interval = f'{low}+' if high == np.inf else f'{low}-{high}'
            name = ('Age',  interval)
            self._summary(*args, mask=mask, subgroup='Age', category=interval)
        
    def sex_summary(self, *args):
        mask = self.data['sex'] == 'F'
        self._summary(*args, mask=mask, subgroup='Sex', category='Female')
        self._summary(*args, mask=~mask, subgroup='Sex', category='Male')
            
    def income_summary(self, *args):
        income_quintile = self.data['neighborhood_income_quintile']
        for q in sorted(income_quintile.dropna().unique()):
            mask =  income_quintile == q
            self._summary(*args, mask=mask, subgroup='Neighborhood Income Quintile', category=f'Q{q:.0f}')
            
    def area_density_summary(self, *args):
        mask = self.data['rural']
        self._summary(*args, mask=mask, subgroup='Area of Residence', category='Rural')
        self._summary(*args, mask=~mask, subgroup='Area of Residence', category='Urban')
                
    def immigration_summary(self, *args):
        mask = self.data['recent_immigrant']
        self._summary(*args, mask=mask, subgroup='Immigration', category='Recent Immigrant')
        self._summary(*args, mask=~mask, subgroup='Immigration', category='Long-Term Resident')
        
    def language_summary(self, *args):
        mask = self.data['speaks_english']
        self._summary(*args, mask=mask, subgroup='Language', category='English Speaker')
        self._summary(*args, mask=~mask, subgroup='Language', category='Non-English Speaker')
        
    def arrival_summary(self, *args):
        mask = self.data['years_since_immigration'] < 10
        self._summary(*args, mask=mask, subgroup='Immigration Arrival', category='Arrival < 10 years')
        self._summary(*args, mask=~mask, subgroup='Immigration Arrival', category='Arrival >= 10 years')
        
    def world_region_of_birth_summary(self, *args):
        birth_region = self.data['world_region_of_birth']
        for region in sorted(birth_region.unique()):
            if region in ['Unknown', 'Other']: continue
            mask = birth_region == region
            self._summary(*args, mask, subgroup='Immigrant World Region of Birth', category=region)
        
    def days_since_starting_regimen_summary(self, *args):
        for (low, high) in [(0, 30), (31, 90), (91, np.inf)]:
            mask = self.data['days_since_starting_chemo'].between(low, high)
            interval = f'{low}+' if high == np.inf else f'{low}-{high}'
            self._summary(*args, mask=mask, subgroup='Days Since Starting Regimen', category=interval)
    
    def cycle_length_summary(self, *args):
        for cycle_length in self.data['cycle_length'].unique():
            mask = self.data['cycle_length'] == cycle_length
            self._summary(*args, mask=mask, subgroup='Cycle Length', category=str(cycle_length))
        
    def comorbidity_summary(self, *args):
        mask = self.data['diabetes']
        self._summary(*args, mask=mask, subgroup='Comorbidity', category='Diabetes')
        self._summary(*args, mask=~mask, subgroup='Comorbidity', category='Non-Diabetes')
        
        mask = self.data['hypertension']
        self._summary(*args, mask=mask, subgroup='Comorbidity', category='Hypertension')
        self._summary(*args, mask=~mask, subgroup='Comorbidity', category='Non-Hypertension')
    
    def gcsf_summary(self, *args):
        mask = (self.data['age'] >= 65) & self.data['GF_given']
        self._summary(*args, mask=mask, subgroup='GCSF Administered', category='Age 65 Years or Older')
            
    def ckd_summary(self, *args):
        mask = self.data['baseline_eGFR'] < 60
        self._summary(*args, mask=mask, subgroup='CKD Prior to Treatment', category='Y')
        self._summary(*args, mask=~mask, subgroup='CKD Prior to Treatment', category='N')
    
    def dialysis_summary(self, *args):
        mask = self.data['dialysis']
        self._summary(*args, mask=mask, subgroup='Dialysis After Treatment', category='Y')
        self._summary(*args, mask=~mask, subgroup='Dialysis After Treatment', category='N')
            
    def most_common_category_summary(
        self,
        *args, 
        catcol='regimen', 
        mapping=None,
        transform=None, 
        top=5
    ):
        subgroup = get_clean_variable_name(catcol)
        
        if catcol not in self.top_categories:
            top_cats = most_common_categories(self.data, catcol=catcol, top=top)
            top_cats = list(top_cats)
            self.top_categories[catcol] = top_cats
        else:
            top_cats = self.top_categories[catcol]
        
        for category in top_cats:
            mask = self.data[catcol] == category
            if mapping is not None: category = mapping[category]
            if transform is not None: category = transform(category)
            self._summary(*args, mask, subgroup=subgroup, category=category)
            
    def _summary(self, *args, mask=None, subgroup='', category=''):
        raise NotImplementedError

###############################################################################
# Data Characteristics
###############################################################################
class CharacteristicsSummary(SubgroupSummary):
    """Computes total number of patients and treatment sessions in different 
    population subgroups
    """
    def __init__(self, X, Y, top=3, subgroups=None, **kwargs):
        """
        Args:
            X (pd.DataFrame): table of the original data partition (not one-hot
                encoded, normalized, clipped, etc)
            Y (pd.DataFrame): table of target labels
            top (int): the number of most common categories. We only
                analyze populations subgroups belonging to those top categories
            **kwargs: keyword arguments fed into SubgroupSummary
        """
        super().__init__(X, **kwargs)
        self.Y = Y.loc[X.index]
        self.top = top
        if subgroups is None:
            subgroups = [
                'sex', 'immigration', 'language', 'income', 'area_density', 
                'regimen', 'cancer_type', 'cancer_location', 'target'
            ]
        self.subgroups = subgroups
        
        self.patient_idxs = X['ikn'].drop_duplicates(keep='last').index
        p = 'patients'
        s = 'sessions'
        self.total = {p: len(self.patient_idxs), s: len(X)}
        self.col = {
            p: f'Patients (N={self.total[p]})',
            s: f'Treatment Sessions (N={self.total[s]})'
        }
        self.display = lambda x, total: f"{x} ({x/total*100:.1f})"
        
        if self.total[p] == self.total[s]:
            # number of patients and sessions are same, only keep patients
            del self.total[s], self.col[s]
        
    def get_summary(self):
        df = pd.DataFrame(index=twolevel)
        
        self.num_sessions_per_patient_summary(df)
        self.median_age_summary(df)
        if 'sex' in self.subgroups: 
            self.sex_summary(df)
        if 'immigration' in self.subgroups: 
            self.immigration_summary(df)
        if 'birth_region' in self.subgroups: 
            self.world_region_of_birth_summary(df)
        if 'language' in self.subgroups: 
            self.language_summary(df)
        if 'arrival' in self.subgroups: 
            self.arrival_summary(df)
        if 'income' in self.subgroups: 
            self.income_summary(df)
        if 'area_density' in self.subgroups: 
            self.area_density_summary(df)
        if 'regimen' in self.subgroups: 
            self.most_common_category_summary(
                df, catcol='regimen', top=self.top, 
                transform=str.upper
            )
        if 'cancer_location' in self.subgroups: 
            self.most_common_category_summary(
                df, catcol='cancer_topog_cd', top=self.top, 
                mapping=cancer_code_mapping
            )
        if 'cancer_type' in self.subgroups: 
            self.most_common_category_summary(
                df, catcol='cancer_morph_cd', top=self.top, 
                mapping=cancer_code_mapping
            )
        if 'target' in self.subgroups: 
            self.target_summary(df)
        if 'comorbidity' in self.subgroups:
            self.comorbidity_summary(df)
        if 'gcsf' in self.subgroups: 
            self.gcsf_summary(df)
        if 'ckd' in self.subgroups: 
            self.ckd_summary(df)
        if 'dialysis' in self.subgroups: 
            self.dialysis_summary(df)
            
        return df
        
    def num_sessions_per_patient_summary(self, df):
        num_sessions = self.data.groupby('ikn').apply(len)
        median = int(num_sessions.median())
        q25, q75 = np.percentile(num_sessions, [25, 75]).astype(int)
        row = ('Number of Treatments, Median (IQR)', '')
        df.loc[row, self.col['patients']] = f'{median} ({q25}-{q75})'

    def median_age_summary(self, df):
        for with_respect_to, col in self.col.items():
            if with_respect_to == 'patients':
                age = self.data.loc[self.patient_idxs, 'age']
            elif with_respect_to == 'sessions':
                age = self.data['age']
            median = int(age.median())
            q25, q75 = np.percentile(age, [25, 75]).astype(int)
            row = ('Age, Median (IQR)', '')
            df.loc[row, col] = f"{median} ({q25}-{q75})"
            
    def target_summary(self, *args):
        for target, Y in self.Y.items():
            self._summary(*args, mask=Y, subgroup='Target Event', category=target)
            
    def _summary(self, df, mask=None, subgroup='', category=''):
        if mask is None: 
            raise ValueError('Please provide a mask')
            
        row = (subgroup, f'{category}, No. (%)')
        for with_respect_to, col in self.col.items():
            total = self.total[with_respect_to]
            if with_respect_to == 'patients':
                num_events = self.data.loc[mask, 'ikn'].nunique()
            elif with_respect_to == 'sessions':
                num_events = mask.sum()
            df.loc[row, col] = self.display(num_events, total)
            
def data_description_summary(
    X, 
    Y,
    tag,
    save_dir=None, 
    partition_method='split', 
    target_event=None,
    **kwargs
):
    """Describe the data. Get the characteristics of patients and treatments 
    for each data split (Train-Valid-Test) or cohort (Development-Test).
    
    Development cohort refers to Training and Validation split.
    
    Args:
        X (pd.DataFrame): table of the original data partition (not one-hot
            encoded, normalized, clipped, etc)
        Y (pd.DataFrame): table of target labels
        tag (pd.DataFrame): table containing patient id and partition names
        partition_method (str): method to partition the data for summarization,
            either by split or cohort
        target_event (str): get summarization for the portion of data in which 
            target event has occured
        **kwargs: keyword arguments fed into CharacteristicsSummary
    """
    result = {}
    
    # Full summary
    cs = CharacteristicsSummary(X, Y, **kwargs)
    result['All'] = cs.get_summary()
    top_cats = cs.top_categories
    
    # Target event occured summary
    if target_event is not None:
        mask = Y[target_event]
        cs = CharacteristicsSummary(
            X[mask], Y[mask], top_categories=top_cats, **kwargs
        )
        result[target_event] = cs.get_summary()
    
    # Partition summary
    for partition, group in tag.groupby(partition_method):
        cs = CharacteristicsSummary(
            X.loc[group.index], Y.loc[group.index], 
            top_categories=top_cats,**kwargs
        )
        result[partition] = cs.get_summary()
        
    summary = pd.concat({k: df.T for k, df in result.items()}).T
    if save_dir is not None: 
        summary.to_csv(f'{save_dir}/data_characteristic_summary.csv')
    return summary

def top_cancer_regimen_summary(data, top=10):
    result = pd.DataFrame()
    top_cancers = most_common_categories(
        data, catcol='cancer_topog_cd', top=top # cancer locations
    ) 
    for cancer_code, nsessions in top_cancers.items():
        cancer_name = cancer_code_mapping[cancer_code]
        mask = data['cancer_topog_cd'] == cancer_code
        top_regimens = most_common_categories(
            data[mask], catcol='regimen', top=top
        )
        top_regimens = [f'{k} ({v})' for k, v in top_regimens.items()]
        result[f'{cancer_name} (N={nsessions})'] = top_regimens
    result.index += 1
    return result

###############################################################################
# Input Features
###############################################################################
def feature_summary(
    X_train,
    save_dir, 
    deny_old_survey=False,
    event_dates=None
):
    """
    Args:
        X_train (pd.DataFrame): table of the original data (not one-hot 
            encoded, normalized, clipped, etc) for the training set
        deny_old_survey (bool): If True, considers survey (symptom 
            questionnaire) response more than 5 days prior to chemo session as
            "missing" data
        event_dates (pd.DataFrame): table of relevant event dates associated 
            with each session (i.e. survey date, visit date)
    """
    N = len(X_train)

    # remove missingness features
    cols = X_train.columns
    drop_cols = cols[cols.str.contains('is_missing')]
    X_train = X_train.drop(columns=drop_cols)

    # get number of missing values, mean, and standard deviation for each 
    # feature for the training set
    summary = X_train.astype(float).describe()
    summary = summary.loc[['count', 'mean', 'std']].T
    summary = summary.round(6)
    summary['count'] = N - summary['count']
    # mask small cells less than 6
    summary['count'] = summary['count'].replace({i:6 for i in range(1,6)})
    
    if deny_old_survey:
        if event_dates is None:
            raise ValueError('Please provide event_dates')
        for col in symptom_cols:
            if col not in summary.index: continue
            survey_date = event_dates[f'{col}_survey_date']
            days_before_chemo = (event_dates['visit_date'] - survey_date).dt.days
            summary.loc[col, 'count'] = sum(days_before_chemo > 5)
    
    summary['Missingness (%)'] = (summary['count'] / N * 100).round(1)
    
    format_arr = lambda arr: arr.round(3).astype(str)
    mean, std = format_arr(summary['mean']), format_arr(summary['std'])
    summary['Mean (SD)'] = mean + ' (' + std + ')'
    
    name_map = {
        'count': f'Train (N={N}) - Missingness Count', 
        'mean': 'Train - Mean', 
        'std': 'Train - SD'
    }
    summary = summary.rename(columns=name_map)

    # assign the groupings for each feature
    features = summary.index
    for group, keyword in variable_groupings_by_keyword.items():
        summary.loc[features.str.contains(keyword), 'Group'] = group
    
    # insert units
    rename_map = {name: f'{name} ({unit})' for name, unit in get_units().items()}
    summary = summary.rename(index=rename_map)
    
    summary.index = get_clean_variable_names(summary.index)
    summary.to_csv(f'{save_dir}/feature_summary.csv')
    return summary

###############################################################################
# Nadir
###############################################################################
def nadir_summary(
    df, 
    output_path, 
    cyto='Neutropenia', 
    load_ci=False,
    n_bootstraps=1000,
):
    """
    Args:
        load_ci (bool): If True, loads the bootstrapped nadir days to compute 
            confidence interval
    """
    cytos = ['Neutropenia', 'Anemia', 'Thrombocytopenia']
    if cyto not in cytos: raise ValueError(f'cyto must be one of {cytos}')
    save_path = f'{output_path}/analysis/tables'
    if not os.path.exists(save_path): os.makedirs(save_path)
    
    if load_ci:
        filepath = f'{save_path}/nadir_{cyto}_bootstraps.csv'
        ci_df = pd.read_csv(filepath, index_col='regimen')
    else:
        ci_df = pd.DataFrame()
        
    regimen_cycle_map = dict(df[['regimen', 'cycle_length']].to_numpy())
    result = {}
    for regimen, cycle_length in tqdm(regimen_cycle_map.items()):
        group = df.query('regimen == @regimen')
        result[regimen] = {
            'NSessions': len(group), 
            'Cycle Length': cycle_length
        }
        
        for grade, thresholds in cytopenia_grades.items():
            if cyto not in thresholds: 
                continue
            
            thresh = thresholds[cyto]   
            cyto_rates_per_day = get_cyto_rates(
                group[range(cycle_length)], thresh
            )
            name = f'{grade} {cyto} Rate (<{thresh})'
            
            if all(cyto_rates_per_day == 0):
                # if no cytopenia was observed for all days
                result[regimen][name] = '0 (0-1)'
                continue
                
            nadir_day = np.argmax(cyto_rates_per_day)
            nadir_day_measurements = group[nadir_day].dropna()
            mask = nadir_day_measurements < thresh
            if mask.sum() < 5: 
                # can't allow small cells less than 5 according to ICES 
                # privacy policy
                result[regimen][name] = '0 (0-1)'
                continue

            worst_cyto_rate = cyto_rates_per_day[nadir_day]
            # binomial confidence interval for cytopenia rate (since we are
            # working with binomial distribution, i.e. cytopenia - 1, not 
            # cytopenia - 0)
            lower, upper = get_confidence_interval(mask, method='binomial')
            worst_cyto_rate = f'{worst_cyto_rate:.3f} ({lower:.3f}-{upper:.3f})'
            result[regimen][name] = worst_cyto_rate
                
            if grade != 'Grade 2':
                continue
                
            if regimen not in ci_df.index:
                # get 95% confidence interval for nadir day using bootstrap
                # technique
                nadir_days = compute_bootstrap_nadir_days(
                    group, cycle_length, thresh, n_bootstraps=n_bootstraps
                )
                ci_df.loc[regimen, range(n_bootstraps)] = nadir_days
            nadir_days = ci_df.loc[regimen].to_numpy()
            lower_ci, upper_ci = get_confidence_interval(
                nadir_days, method='basic'
            )
            # set day 1 as day of administration (not day 0)
            nadir_day = f'{nadir_day+1} ({lower_ci:.0f}-{upper_ci:.0f})'
            result[regimen]['Nadir Day'] = nadir_day

            # col = 'NMeasurements at Nadir Day'
            # result[regimen][col] = len(nadir_day_measurements)
        
    summary_df = pd.DataFrame(result).T
    
    # write the results
    summary_df.to_csv(f'{save_path}/nadir_{cyto}_summary.csv')
    ci_df.to_csv(f'{save_path}/nadir_{cyto}_bootstraps.csv', index_label='regimen')
    
    return summary_df
